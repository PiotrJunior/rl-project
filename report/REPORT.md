# From ε-greedy to Boltzmann: annealing exploration in DQN

**RL course project — first iteration, OpenAI Gym environments.**

---

## 1. Problem

ε-greedy is the default exploration strategy for discrete-action value-based
agents. It is *uninformed*: with probability ε it draws uniformly from the whole
action set, spending most of its exploration budget on actions the Q-function
already ranks last. Boltzmann (softmax) exploration is *informed*: it samples in
proportion to `exp(Q/τ)`, so exploration concentrates on actions that look good.

The catch is that Boltzmann inherits whatever prior the Q-function encodes, and
early in training that prior is noise. A softmax over a randomly initialised
network is not a uniform policy — it is a policy *biased by initialisation
noise*, which is strictly worse than the honest uniform draw ε-greedy makes. So
the sensible thing is to start with ε-greedy and move to Boltzmann as Q becomes
trustworthy.

This project asks **how** to make that transfer, and measures whether it helps.

## 2. Method: one policy family, several paths through it

The design decision that organises everything here is to refuse to treat
ε-greedy and Boltzmann as two separate strategies to be blended ad hoc. Instead,
every variant is a point in a single parameterised family:

```
π(a|s) = ε_t · Uniform(A)  +  (1 − ε_t) · Softmax_{a ∈ TopK_t(Q)}( Q̃(s,·) / τ_t )
```

with three scheduled knobs — `ε_t` (uniform floor), `τ_t` (temperature), `k_t`
(support size) — and a Q-scaling map `Q̃` discussed in §3.

Both classic strategies are **exact** points in this family:

| configuration | is exactly |
|---|---|
| `k = 1`, any `τ` | ε-greedy (the softmax support is the argmax set) |
| `τ → 0`, `k = \|A\|` | ε-greedy (non-maximal exponents underflow to 0) |
| `ε = 0`, `k = \|A\|` | pure Boltzmann |

This is not a convenient approximation: `tests/test_policies.py` asserts these
as exact array equalities against independently written reference
implementations, across a grid of ε and τ, including the tie-breaking behaviour
(both limits must spread mass uniformly over tied maxima — and ties are the norm
at initialisation, when every action has near-identical Q).

Having exact endpoints matters because it makes "annealing between them" a
well-defined operation: a *path through (ε, τ, k) space*. The two ideas in the
brief are then two different paths.

**The temperature path** (`eps_boltzmann`, brief idea 1). Hold `k = |A|` and
start at `τ = 10⁻⁴`, which makes the softmax an exact argmax, so the policy *is*
ε-greedy. Then raise `τ` geometrically to 0.3 while `ε` falls from 1.0 to 0.01.
The policy deforms continuously into pure Boltzmann without ever leaving the
family.

**The support path** (`anneal_k`, brief ideas 1+2). Hold `τ` fixed and grow `k`
from 1 (again exactly ε-greedy) to `|A|` (full Boltzmann). The handover happens
by *widening the support* rather than by raising the temperature.

**The mixture path** (`mixture_anneal`, brief idea 1a). Interpolate the two
action *distributions* outright: `π = (1−β)·π_ε-greedy + β·π_Boltzmann`. This is
genuinely different from the temperature path, not a reparameterisation: at
β = 0.5 the policy retains a real chance of a uniformly random action inherited
from the ε-greedy component, even in states where Boltzmann is confident,
whereas the temperature path passes through distributions that are neither
endpoint. Which behaves better is an empirical question, and one of the things
measured here.

**Fixed small support** (`topk_boltzmann`, `topk3_boltzmann`, brief idea 2).
Boltzmann over the top 2 or 3 actions with a decaying ε floor. The rationale
from the brief: a noisy Q-function often ranks the truly-best action second, so
spreading probability over the top few actions explores precisely the plausible
alternatives. The ε floor is retained (decaying to 0.02) so that an action
wrongly excluded from the top-k is not unreachable forever.

**State-adaptive support** (`topp_boltzmann`). Nucleus sampling: the support is
the smallest set of highest-Q actions whose softmax mass reaches `p = 0.9`.
Where Q is peaked the support collapses towards greedy; where Q is flat — which
is exactly where the agent has no basis for a preference — it widens. This
spends exploration where the Q-function is undecided rather than uniformly in
time.

## 3. The detail that decides whether any of this works: Q-value scale

A temperature is only meaningful relative to the spread of the values it
divides, and that spread is not a constant:

- across environments it differs by orders of magnitude (CartPole returns
  ~10–500, LunarLander ~−400–300, clipped Atari ~0–50);
- **within a single run it grows steadily**, as the value function inflates from
  its near-zero initialisation towards the true return scale.

So a fixed τ that produces healthy exploration at 20k steps is effectively
greedy at 300k steps, through no change in the schedule. Any comparison of
"ε-greedy vs Boltzmann" that ignores this is really measuring an uncontrolled,
environment-specific annealing schedule that the experimenter did not choose.

Three modes are implemented and ablated:

- **`none`** — `Q̃ = Q`. The temperature carries raw Q units. Included to
  demonstrate the failure mode rather than assert it.
- **`per_state`** — `Q̃ = (Q − mean_a Q)/(std_a Q + δ)`, computed independently
  at each state. Dimensionless, but it *destroys information*: a state where
  every action is genuinely equally good has its tiny Q spread inflated to unit
  variance, so the policy becomes confidently peaked on what is really numerical
  noise. That is exactly backwards from what exploration should do.
- **`running`** (default) — `Q̃ = (Q − mean_a Q)/σ̂`, where `σ̂` is an EMA of the
  across-action Q spread over recently visited states. Dimensionless *and* the
  normaliser is shared across states, so a flat Q-row stays flat and yields a
  near-uniform policy while a peaked row stays peaked.

`tests/test_scaling.py` verifies that under `running` two Q-rows with identical
shape but magnitudes differing by 1000× produce the *same* action distribution
to within floating-point error, and that under `none` they do not (one is
meaningfully stochastic, the other has collapsed to greedy).

## 4. Extension: uncertainty-gated handover

A step-count schedule is a guess about when Q becomes trustworthy. The extension
replaces it with a measurement.

Given a confidence `c ∈ [0,1]`, the knobs are interpolated between an uncertain
endpoint (`ε = 1`, `k = 1` — i.e. ε-greedy) and a confident endpoint
(`ε = 0.01`, `k = |A|`, `τ = 0.3` — i.e. Boltzmann). Temperature is interpolated
geometrically, since it spans orders of magnitude.

Two signals:

- **`ensemble`** — 5 bootstrapped Q-heads on a shared torso (Bernoulli(0.8)
  masks stored per transition), `u(s) = mean_a std_k Q_k(s,a)`. This is
  **per-state**: the agent can be Boltzmann in a well-visited region while still
  ε-greedy in an unfamiliar one. No time-based schedule can express that. It
  measures epistemic uncertainty — what the data has not pinned down — rather
  than the intrinsic randomness of the return.
- **`td_error`** — an EMA of |TD error|, which prioritized replay already
  computes, so it is free. It is a single *global* scalar, so it can only
  produce a time-varying schedule — albeit one driven by measured learning
  progress rather than by step count.

Both normalise the raw signal against a running high-quantile of its own
history, so confidence is scale-free. This is essential: uncertainty falls by
orders of magnitude over training, so any fixed threshold would trip once and
never move again, collapsing the adaptive scheme into a step function.

Two guards proved necessary in practice. **Warm-up**: before the reference
quantile has data, confidence is pinned to 0, otherwise the first states report
spurious high confidence and the run starts in near-greedy Boltzmann on a random
network — the worst possible combination. **Smoothing**: `confidence_smoothing`
interpolates between the raw per-state signal (1.0) and a global EMA (0.0),
which doubles as the ablation isolating "per-state resolution" from "adaptive
timing".

### Why the control arm is the crux

Bootstrapped ensemble heads change *value learning*, not just uncertainty
measurement. Comparing `uncertainty_gated`-with-ensemble against single-head
ε-greedy would confound the gating with the architecture. The uncertainty sweep
therefore includes `eps_greedy_ensemble` — plain ε-greedy running the **same**
5-head architecture — as the arm that actually answers the research question,
plus a single-head ε-greedy arm to quantify what the architecture change alone
does.

## 5. Base agent

**Double + Dueling + n-step(3) + prioritized replay** — Rainbow minus NoisyNets
minus distributional.

**NoisyNets is deliberately excluded.** It is itself an exploration mechanism —
learned parameter noise — and is known to make ε-greedy redundant. Including it
would give every arm a second, uncontrolled exploration strategy underneath the
one being measured, damping exactly the differences this study exists to detect.
When the exploration strategy *is* the object of study, the base agent must have
none of its own.

**C51 is excluded** for cost (roughly 2× CPU time per step) and because the
spread it provides is aleatoric, whereas the extension needs epistemic
uncertainty — which the ensemble heads supply directly.

Every other component is held identical across arms. `configs/base.yaml` is
shared by every arm and the config loader **rejects unknown keys** rather than
ignoring them, so a typo in an experiment config fails loudly instead of
silently producing a run that tested the wrong thing.

## 6. Protocol

- **Evaluation is always greedy**, in a separate environment with its own seed
  stream, 10 episodes per point. This is the only way the comparison is
  meaningful: a Boltzmann behaviour policy scores differently from an ε-greedy
  one for reasons unrelated to how well it learned, so measuring the behaviour
  policy's own return would conflate exploration cost with learning quality.
  Argmax ties are broken uniformly at random, not by index — index tie-breaking
  biases evaluation toward low-numbered actions for an under-trained network.
- **Aggregation** uses **IQM** (interquartile mean) with **stratified bootstrap**
  95% CIs over seeds, following Agarwal et al. (2021). The mean over 3–5 seeds
  is the wrong statistic for DQN, which produces catastrophic-failure seeds
  often enough to move it substantially; IQM is far more robust and much less
  noisy than the median. `probability_of_improvement` is also reported: it asks
  the question a practitioner actually has ("if I run this once, will it beat
  the baseline?") and is robust to a single outlying seed.
- **Two headline metrics.** *Final return* averages the last 3 evaluation points
  (a single point is close to meaningless on a noisy DQN curve). *AUC* is the
  mean return over all evaluation points — a sample-efficiency proxy. For an
  exploration study the path matters as much as the destination: two arms can
  finish equal with very different costs along the way.
- **Exploration diagnostics.** ε, τ and k are *not* comparable across
  strategies — an ε of 0.05 and a τ of 0.3 are not "the same amount" of
  exploration, and how much exploration a given τ buys drifts as the Q-scale
  grows. So the comparable measures are **behaviour-policy entropy** and
  **P(action ≠ argmax Q)**, both averaged over each logging interval. These are
  what let the report distinguish "the schedule moved" from "the behaviour
  changed".

### 6.1 Compute budget and what it constrains

All experiments ran on a **4-core CPU box with no GPU**. Measured throughput of
the identical agent differs sharply by environment:

| environment | actions | steps/s per worker | why |
|---|---|---|---|
| CartPole-v1 | 2 | ~400 | trivial physics |
| Acrobot-v1 | 3 | ~400 | trivial physics |
| LunarLander-v3 | 4 | **~50** | Box2D contact solving dominates |

LunarLander is therefore ~8× more expensive per step than the classic-control
tasks, which is what set the step budgets below. This is a real constraint on
the conclusions, and §10 says so plainly rather than presenting the numbers as
though they came from a full-budget study.

| study | environment | steps | arms | seeds |
|---|---|---|---|---|
| main comparison | CartPole-v1 | 60k | 7 | 3 |
| main comparison | Acrobot-v1 | 100k | 8 | 3 |
| main comparison | LunarLander-v3 | 120k | 8 | 3 |
| Q-scaling ablation | Acrobot-v1 | 100k | 3 | 3 |
| uncertainty extension | Acrobot-v1 | 100k | 5 | 3 |

