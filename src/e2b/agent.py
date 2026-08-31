"""DQN agent: Double + Dueling + n-step + prioritized replay.

This is "Rainbow minus NoisyNets minus distributional".

**NoisyNets is deliberately excluded.** It is itself an exploration mechanism --
it injects parameter noise whose scale is learned -- and it is known to render
epsilon-greedy redundant. Including it would mean every arm of this study
carried a second, uncontrolled exploration strategy underneath the one being
measured, and differences between epsilon-greedy and Boltzmann would be damped
by whatever NoisyNets was doing. Since the object of study *is* the exploration
strategy, the base agent must have no exploration of its own.

Distributional (C51) is excluded for cost, not principle: it roughly doubles CPU
time per step, and the return-distribution spread it provides is *aleatoric*
uncertainty, whereas the extension here needs *epistemic* uncertainty (what the
data has not pinned down). The ensemble heads supply that directly.

Everything else about the agent is held fixed across arms. The only thing that
varies between experimental conditions is the exploration policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from e2b.config import AgentConfig
from e2b.envs.make import EnvSpec
from e2b.nets import build_qnetwork
from e2b.replay import Batch, ReplayBuffer, build_replay
from e2b.replay.nstep import NStepAccumulator


class DQNAgent:
    """Value-based agent with a pluggable exploration policy.

    The agent owns the value function and the replay; it does *not* own the
    exploration strategy. It exposes :meth:`q_values` and :meth:`q_heads` so the
    training loop can hand the Q-row to whichever policy is under test.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        spec: EnvSpec,
        device: torch.device,
        rng: np.random.Generator,
        total_steps: int,
    ) -> None:
        self.cfg = cfg
        self.spec = spec
        self.device = device
        self.total_steps = total_steps
        self.num_heads = cfg.net.num_heads

        self.online = build_qnetwork(
            obs_shape=spec.obs_shape,
            num_actions=spec.num_actions,
            hidden_sizes=cfg.net.hidden_sizes,
            dueling=cfg.net.dueling,
            num_heads=cfg.net.num_heads,
            kind=spec.net_kind,
            cnn_channels=cfg.net.cnn_channels,
            cnn_kernels=cfg.net.cnn_kernels,
            cnn_strides=cfg.net.cnn_strides,
            cnn_hidden=cfg.net.cnn_hidden,
        ).to(device)
        self.target = build_qnetwork(
            obs_shape=spec.obs_shape,
            num_actions=spec.num_actions,
            hidden_sizes=cfg.net.hidden_sizes,
            dueling=cfg.net.dueling,
            num_heads=cfg.net.num_heads,
            kind=spec.net_kind,
            cnn_channels=cfg.net.cnn_channels,
            cnn_kernels=cfg.net.cnn_kernels,
            cnn_strides=cfg.net.cnn_strides,
            cnn_hidden=cfg.net.cnn_hidden,
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=cfg.lr, eps=cfg.adam_eps
        )

        self.replay: ReplayBuffer = build_replay(
            cfg.replay,
            obs_shape=spec.obs_shape,
            obs_dtype=spec.obs_dtype,
            num_heads=cfg.net.num_heads,
            bootstrap_prob=cfg.net.bootstrap_prob,
            rng=rng,
        )
        self.nstep = NStepAccumulator(cfg.replay.n_step, cfg.gamma)
        self.updates = 0

    # ------------------------------------------------------------------ acting

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        arr = np.asarray(obs)
        if arr.dtype == np.uint8:
            tensor = torch.as_tensor(arr, dtype=torch.uint8, device=self.device)
        else:
            tensor = torch.as_tensor(
                np.asarray(arr, dtype=np.float32), device=self.device
            )
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def q_heads(self, obs: np.ndarray) -> np.ndarray:
        """Per-head Q-values for one observation: ``(num_heads, num_actions)``.

        Used by the ensemble uncertainty estimator. Returned even for
        single-head networks (as a 1 x A array) so callers need no branching.
        """
        self.online.eval()
        out = self.online(self._to_tensor(obs))[0]
        self.online.train()
        return out.cpu().numpy()

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Head-averaged Q-values for one observation: ``(num_actions,)``."""
        self.online.eval()
        out = self.online.q_values(self._to_tensor(obs))[0]
        self.online.train()
        return out.cpu().numpy()

    # ------------------------------------------------------------- experience

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Feed a 1-step transition; n-step transitions land in replay."""
        for tr in self.nstep.push(obs, action, reward, next_obs, terminated, truncated):
            self.replay.add(
                tr.obs, tr.action, tr.reward, tr.next_obs, tr.terminated, tr.discount
            )

    # ---------------------------------------------------------------- learning

    def beta(self, step: int) -> float:
        """Importance-sampling exponent, annealed to 1 over training.

        Annealing beta up (rather than holding it at 1) is the standard PER
        recipe: the bias from non-uniform sampling matters most near
        convergence, while early on the variance reduction is worth more.
        """
        r = self.cfg.replay
        if self.total_steps <= 0:
            return r.beta_end
        frac = min(1.0, max(0.0, step / self.total_steps))
        return r.beta_start + frac * (r.beta_end - r.beta_start)

    def _batch_to_torch(self, batch: Batch) -> dict[str, torch.Tensor]:
        def obs_tensor(arr: np.ndarray) -> torch.Tensor:
            if arr.dtype == np.uint8:
                return torch.as_tensor(arr, dtype=torch.uint8, device=self.device)
            return torch.as_tensor(
                np.asarray(arr, dtype=np.float32), device=self.device
            )

        return {
            "obs": obs_tensor(batch.obs),
            "next_obs": obs_tensor(batch.next_obs),
            "actions": torch.as_tensor(batch.actions, device=self.device),
            "rewards": torch.as_tensor(
                batch.rewards, dtype=torch.float32, device=self.device
            ),
            "terminated": torch.as_tensor(
                batch.terminated, dtype=torch.float32, device=self.device
            ),
            "discounts": torch.as_tensor(
                batch.discounts, dtype=torch.float32, device=self.device
            ),
            "masks": torch.as_tensor(
                batch.masks, dtype=torch.float32, device=self.device
            ),
            "weights": torch.as_tensor(
                batch.weights, dtype=torch.float32, device=self.device
            ),
        }

    @torch.no_grad()
    def _compute_targets(self, t: dict[str, torch.Tensor]) -> torch.Tensor:
        """n-step (Double) DQN targets, shaped ``(batch, num_heads)``.

        Double DQN: the *online* network selects the bootstrap action and the
        *target* network evaluates it, which removes the max-operator's
        systematic overestimation. With an ensemble, selection is done per head
        so each head bootstraps off its own greedy policy rather than off a
        consensus -- otherwise the heads collapse towards each other and the
        disagreement signal the extension depends on vanishes.
        """
        next_target = self.target(t["next_obs"])  # (B, H, A)
        if self.cfg.double_dqn:
            next_online = self.online(t["next_obs"])  # (B, H, A)
            best = next_online.argmax(dim=-1, keepdim=True)  # (B, H, 1)
        else:
            best = next_target.argmax(dim=-1, keepdim=True)
        next_q = next_target.gather(-1, best).squeeze(-1)  # (B, H)

        not_done = (1.0 - t["terminated"]).unsqueeze(-1)
        # `discounts` is gamma ** k with the *actual* k, which is shorter than
        # n for transitions flushed at an episode boundary.
        discounts = t["discounts"].unsqueeze(-1)
        return t["rewards"].unsqueeze(-1) + discounts * not_done * next_q

    def learn(self, step: int) -> dict[str, float] | None:
        """One gradient step. Returns training stats, or None if not ready."""
        if len(self.replay) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return None

        batch = self.replay.sample(self.cfg.batch_size, beta=self.beta(step))
        t = self._batch_to_torch(batch)

        q_all = self.online(t["obs"])  # (B, H, A)
        actions = t["actions"].view(-1, 1, 1).expand(-1, self.num_heads, 1)
        q_taken = q_all.gather(-1, actions).squeeze(-1)  # (B, H)

        targets = self._compute_targets(t)
        td_errors = targets - q_taken  # (B, H)

        # Huber loss: quadratic near zero, linear in the tail. With PER already
        # up-weighting large errors, an unclipped squared loss would compound
        # the emphasis and destabilise training.
        elementwise = F.huber_loss(
            q_taken, targets, reduction="none", delta=self.cfg.huber_kappa
        )
        masks = t["masks"]  # (B, H)
        # Per-sample loss = mean over the heads that own this sample. Dividing
        # by the *live* head count keeps the gradient scale independent of how
        # many heads the bootstrap mask happened to select.
        live = masks.sum(dim=1).clamp(min=1.0)
        per_sample = (elementwise * masks).sum(dim=1) / live
        loss = (per_sample * t["weights"]).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.cfg.max_grad_norm
        )
        self.optimizer.step()

        # Priorities use the head-mean absolute TD error so a transition's
        # priority does not depend on which heads happen to own it.
        with torch.no_grad():
            abs_td = ((td_errors.abs() * masks).sum(dim=1) / live).cpu().numpy()
        self.replay.update_priorities(batch.indices, abs_td)

        self.updates += 1
        if self.updates % max(1, self.cfg.target_update_interval) == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {
            "loss": float(loss.detach().cpu()),
            "td_error": float(np.abs(abs_td).mean()),
            "grad_norm": float(grad_norm),
            "q_taken": float(q_taken.detach().mean().cpu()),
            "beta": self.beta(step),
        }

    # -------------------------------------------------------------- checkpoint

    def state_dict(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
