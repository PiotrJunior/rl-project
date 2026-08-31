"""The unified epsilon-tau-k exploration family.

Almost every variant in this study is a *path* through the parameter space of a
single policy::

    pi(a|s) = eps_t * Uniform(A)
            + (1 - eps_t) * Softmax_{a in TopK_t(Q)}( Qtilde(s, .) / tau_t )

with three scheduled knobs and a Q-scaling map.  The two ideas in the project
brief are two different paths:

* **temperature path** -- hold ``k = |A|`` and drive ``tau`` up from ~0 (which
  is exactly greedy) to a target temperature while ``eps`` falls to ~0. This is
  "anneal from epsilon-greedy to Boltzmann by changing the temperature".
* **support path** -- hold ``tau`` fixed and grow ``k`` from 1 (again exactly
  greedy) to ``|A|``. This is "sample Boltzmann over the top few actions".

Both endpoints of both paths are *exactly* representable, which is what makes
the equivalence tests in ``tests/test_policies.py`` pass to machine precision
rather than approximately:

===========================  =================================================
configuration                equivalent to
===========================  =================================================
``k = 1`` (any tau)          epsilon-greedy
``tau -> 0``, ``k = |A|``    epsilon-greedy
``eps = 0``, ``k = |A|``     pure Boltzmann
===========================  =================================================
"""

from __future__ import annotations

from typing import Any

import numpy as np

from e2b.policies.base import ExplorationPolicy
from e2b.policies.core import compose_probs, top_k_support, top_p_support
from e2b.policies.scaling import QScaler
from e2b.utils.schedules import Schedule, make_schedule


class UnifiedPolicy(ExplorationPolicy):
    """Scheduled epsilon / temperature / support-size exploration.

    Parameters
    ----------
    epsilon, temperature, top_k
        Schedule specifications (see :func:`e2b.utils.schedules.make_schedule`).
        A bare number means a constant.
    top_p
        If set, nucleus sampling replaces the fixed top-k support: the support is
        the smallest set of highest-Q actions whose softmax mass reaches
        ``top_p``. ``top_k`` is then ignored.
    q_scaling
        ``none`` | ``per_state`` | ``running`` -- see
        :class:`e2b.policies.scaling.QScaler`.
    """

    name = "unified"

    def __init__(
        self,
        num_actions: int,
        epsilon: Any = 0.05,
        temperature: Any = 1.0,
        top_k: Any = None,
        top_p: float | None = None,
        q_scaling: str = "running",
        q_scaling_decay: float = 0.999,
    ) -> None:
        super().__init__(num_actions)
        self.epsilon_schedule: Schedule = make_schedule(epsilon)
        self.temperature_schedule: Schedule = make_schedule(temperature)
        # `top_k = None` means "all actions", the pure-Boltzmann support.
        self.top_k_schedule: Schedule = make_schedule(
            num_actions if top_k is None else top_k
        )
        self.top_p = top_p
        self.scaler = QScaler(q_scaling, decay=q_scaling_decay)

    def observe(self, q: np.ndarray) -> None:
        self.scaler.observe(q)

    def knobs(self, step: int) -> tuple[float, float, int]:
        """The (epsilon, temperature, k) actually in force at ``step``."""
        eps = float(np.clip(self.epsilon_schedule(step), 0.0, 1.0))
        tau = max(float(self.temperature_schedule(step)), 0.0)
        k = int(np.clip(round(self.top_k_schedule(step)), 1, self.num_actions))
        return eps, tau, k

    def action_probs(
        self, q: np.ndarray, step: int, uncertainty: float | None = None
    ) -> np.ndarray:
        eps, tau, k = self.knobs(step)
        q_tilde = self.scaler(q)
        if self.top_p is not None:
            support = top_p_support(q_tilde, self.top_p, tau)
        else:
            support = top_k_support(q_tilde, k)
        self._last.update(
            eps=eps,
            temperature=tau,
            top_k=int(support.sum()),
            q_scale=self.scaler.scale,
        )
        return compose_probs(q_tilde, eps, tau, support)

    def state_dict(self) -> dict[str, Any]:
        return {"scaler": self.scaler.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


class MixturePolicy(ExplorationPolicy):
    """Literal interpolation of two policies (project idea 1, variant a)::

        pi_t = (1 - beta_t) * pi_epsilon_greedy + beta_t * pi_boltzmann

    This is the most direct reading of "anneal from epsilon-greedy to
    Boltzmann": rather than deforming one distribution into the other through a
    shared parameterisation, it mixes the two distributions outright.

    The distinction from the temperature path is not cosmetic. A mixture at
    ``beta = 0.5`` puts real mass on *both* behaviours -- it will sometimes take
    a uniformly random action even when Boltzmann would never do so -- whereas
    the temperature path passes through intermediate distributions that are
    neither. Which of the two transfers better is one of the questions this
    project measures.
    """

    name = "mixture"

    def __init__(
        self,
        num_actions: int,
        first: ExplorationPolicy,
        second: ExplorationPolicy,
        beta: Any = 0.0,
    ) -> None:
        super().__init__(num_actions)
        self.first = first
        self.second = second
        self.beta_schedule: Schedule = make_schedule(beta)

    def observe(self, q: np.ndarray) -> None:
        self.first.observe(q)
        self.second.observe(q)

    def action_probs(
        self, q: np.ndarray, step: int, uncertainty: float | None = None
    ) -> np.ndarray:
        beta = float(np.clip(self.beta_schedule(step), 0.0, 1.0))
        p_first = self.first.action_probs(q, step, uncertainty)
        p_second = self.second.action_probs(q, step, uncertainty)
        probs = (1.0 - beta) * p_first + beta * p_second
        info: dict[str, Any] = {"beta": beta}
        # Surface the second (Boltzmann) component's knobs; the first is
        # epsilon-greedy whose only knob is eps, reported alongside.
        info.update(
            {f"first_{k}": v for k, v in self.first.diagnostics().items() if k == "eps"}
        )
        info.update(
            {
                k: v
                for k, v in self.second.diagnostics().items()
                if k in ("temperature", "top_k", "q_scale")
            }
        )
        self._last.update(info)
        return probs / probs.sum()

    def state_dict(self) -> dict[str, Any]:
        return {"first": self.first.state_dict(), "second": self.second.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.first.load_state_dict(state.get("first", {}))
        self.second.load_state_dict(state.get("second", {}))
