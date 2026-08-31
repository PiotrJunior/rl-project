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

The measured answer (§7.1) is that the premise above is **conditional, and its
sign flips between environments**. Dropping the ε floor entirely is decisively
worse than ε-greedy on CartPole (`P = 0.04 [0.00, 0.24]`) and decisively
*better* on LunarLander (`P = 0.92 [0.68, 1.00]`) — the two intervals sit either
side of 0.5 and neither contains it. The ε floor is insurance against an
uninformative Q, and it is worth its premium only where the premium — the cost
of a uniformly random action — is low. On CartPole a random action is nearly
free; on LunarLander it fires thrusters at random and crashes the lander.

### Summary of findings

| | result |
|---|---|
| **Resolved** (interval excludes 0.5) | The ε-floor's value reverses sign between CartPole and LunarLander (§7.1). The support path `anneal_k` beats ε-greedy on Acrobot final return, `P = 0.96 [0.76, 1.00]` (§7.2). |
| **Resolved** (replicated on two machines) | Q-scale normalisation controls where the behaviour policy's entropy settles, and `none` never settles at all (§8.1). The uncertainty-gated arms collapse, `P = 0.00`, attributable to the gate and not the ensemble architecture (§8.2). |
| **Diagnosed** | The uncertainty gate never fired because confidence was normalised against a running quantile *of its own signal*, which tracks any trend and cancels it. Freezing the reference makes the handover fire as designed — confidence 0.46 → 0.71, ε 1.00 → 0.30 — and separates the epistemic signal (ensemble disagreement, falls 2×) from the unsuitable one (TD error, grows 3×). It still loses to the control (§8.2). |
| **Not resolved** | Any ranking among the ε-floor variants on CartPole or LunarLander. Whether `per_state` Q-scaling costs performance. §6.2 shows a 3-seed ranking here does not survive a change of BLAS library. |

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
history, so confidence is scale-free. The reasoning was that uncertainty falls
by orders of magnitude over training, so any fixed threshold would trip once and
never move again, collapsing the adaptive scheme into a step function.

> **This design decision is the one that broke the extension, and §8.2 works
> out why.** Normalising a signal against a running estimate drawn from *that
> same signal* removes exactly the trend the gate exists to detect: the
> reference follows the signal, the ratio goes to a constant, and confidence is
> pinned there for the whole run. The premise was wrong too — measured on
> Acrobot, neither raw signal falls over training; both grow, because both are
> denominated in Q-values, which inflate as the value function learns.
> `reference_freeze_step` (off by default) is the fix, and §8.2 shows the
> handover firing once it is on.

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

### 6.1 Compute budget and what it was spent on

Every number in this report was produced on an **Apple M1 Pro (10 cores, 32 GB,
CPU only)**, `--workers 8`. Throughput of the identical agent is close across
all three environments:

| environment | actions | steps/s per worker (8 workers) |
|---|---|---|
| CartPole-v1 | 2 | 1607-1858 (median 1735) |
| Acrobot-v1 | 3 | 1423-1650 (median 1590) |
| LunarLander-v3 | 4 | 1391-1671 (median 1534) |

LunarLander is **not** meaningfully slower than classic control -- within 12% of
CartPole per step. An earlier draft recorded it at ~50 steps/s and rescoped the
whole study around that number; it was an artifact of orphaned worker processes
from a killed sweep competing for CPU at measurement time. The lesson is worth
keeping: a throughput measurement taken while unrelated load is on the machine
is not a measurement of your program.

| study | environments | steps | arms | seeds | runs |
|---|---|---|---|---|---|
| **full study** (`full_gym`) | CartPole / Acrobot / LunarLander | 100k / 150k / 400k | 8 | **5** | 120 |
| reduced (`reduced_gym`) | CartPole | 60k | 7 | 3 | 21 |
| reduced (`acrobot_gym`) | Acrobot | 100k | 8 | 3 | 24 |
| reduced (`main_gym`) | LunarLander | 400k | 8 | 3 | 24 |
| Q-scaling ablation | Acrobot | 100k | 3 | 3 | 9 |
| uncertainty extension | Acrobot | 100k | 9 | 3 | 27 |

The full study is 26.0M environment steps: 4.6 h of summed worker time, 35 min
wall-clock. **§7 reports the 5-seed full study.** The 3-seed sweeps are retained
because §8's ablation and extension run at that size -- and because comparing
the two tiers is itself the report's most useful methodological result.

### 6.2 Three seeds do not resolve a ranking -- demonstrated, not asserted

The 3-seed sweeps were run twice on different machines: first on a 4-core Linux
box, then re-run here. Same code, same seeds, same configs. Only the torch/BLAS
build differs, which perturbs floating-point arithmetic enough for trajectories
to diverge chaotically.

| CartPole arm | 4-core Linux | M1 Pro |
|---|---|---|
| ε-greedy | 344.1 [303.0, 378.6] | 275.1 [218.8, 317.0] |
| Boltzmann (no ε floor) | 157.3 [108.2, 221.8] | 211.9 [113.9, 393.5] |

| Acrobot arm | 4-core Linux | M1 Pro |
|---|---|---|
| top-3 Boltzmann | −110.4 (**1st** of 8) | −173.7 (**7th** of 8) |
| ε-greedy⊕Boltzmann mix | −140.3 (5th of 8) | −89.7 (**1st** of 8) |

The ordering of the ε-floor arms is not stable under a change of linear-algebra
library, let alone under a change of seed. **Anything read out of a 3-seed
ranking in this family is noise.** Two things did survive the machine change
unchanged -- the Q-scaling entropy mechanism (§8.1) and the uncertainty-gating
failure (§8.2) -- and the difference between those and the rankings is exactly
the line this report draws between a result and a point estimate.

## 7. Results: the main comparison (5 seeds, full budgets)

IQM with stratified-bootstrap 95% CIs, `n = 5` seeds. Figures:
`report/figures/full_gym_*`.

### CartPole-v1 (100k steps, |A| = 2)

| Variant | Final return | AUC | P(beats ε-greedy) |
|---|---|---|---|
| **ε-greedy (baseline)** | 319.8 [266.0, 370.2] | 295.9 [251.6, 329.0] | — |
| ε-greedy⊕Boltzmann mix | 309.1 [264.9, 364.8] | 276.1 [253.1, 327.7] | 0.44 [0.08, 0.84] |
| ε→Boltzmann (τ path) | 287.5 [267.3, 299.4] | 274.7 [254.8, 313.3] | 0.24 [0.00, 0.64] |
| ε→Boltzmann (k path) | 283.0 [254.4, 324.0] | 274.2 [250.9, 309.1] | 0.28 [0.00, 0.64] |
| top-2 Boltzmann | 272.7 [240.0, 330.5] | 259.6 [237.7, 299.2] | 0.24 [0.00, 0.60] |
| top-3 Boltzmann | 272.7 [240.0, 330.5] | 259.6 [237.7, 299.2] | 0.24 [0.00, 0.60] |
| top-p Boltzmann | 268.3 [231.4, 333.1] | 257.0 [225.4, 300.4] | 0.24 [0.00, 0.60] |
| Boltzmann (no ε floor) | 209.1 [121.4, 426.3] | 134.4 [107.7, 251.3] | 0.24 [0.00, 0.60] |

*(`top-2` and `top-3` are **bit-identical on all five seeds** — on a 2-action
environment both clamp to full support, so they are the same policy. This is an
unplanned end-to-end check that the top-k clamp is exact and the pipeline is
per-seed deterministic.)*

### Acrobot-v1 (150k steps, |A| = 3)

| Variant | Final return | AUC | P(beats ε-greedy) |
|---|---|---|---|
| ε→Boltzmann (k path) | **−79.6 [−81.4, −78.1]** | −256.8 [−285.3, −231.6] | **0.96 [0.76, 1.00]** |
| top-2 Boltzmann | −81.9 [−86.2, −79.0] | −281.6 [−309.1, −215.7] | 0.80 [0.44, 1.00] |
| top-p Boltzmann | −82.5 [−92.9, −80.0] | −233.7 [−281.3, −213.7] | 0.60 [0.20, 1.00] |
| ε-greedy⊕Boltzmann mix | −82.6 [−87.8, −78.3] | −235.9 [−275.5, −165.3] | 0.64 [0.24, 1.00] |
| Boltzmann (no ε floor) | −83.2 [−214.1, −77.8] | −240.3 [−321.8, −228.4] | 0.56 [0.16, 1.00] |
| **ε-greedy (baseline)** | −84.9 [−86.8, −81.4] | −250.3 [−294.8, −199.0] | — |
| top-3 Boltzmann | −90.7 [−101.0, −78.4] | −231.3 [−258.3, −174.6] | 0.32 [0.00, 0.72] |
| ε→Boltzmann (τ path) | −98.3 [−128.2, −80.9] | −236.7 [−330.9, −184.1] | 0.32 [0.00, 0.72] |

### LunarLander-v3 (400k steps, |A| = 4)

| Variant | Final return | AUC | P(beats ε-greedy) |
|---|---|---|---|
| **ε-greedy (baseline)** | 165.4 [76.2, 217.1] | 44.8 [−11.0, 79.8] | — |
| ε→Boltzmann (τ path) | 162.6 [122.8, 207.0] | 36.1 [8.0, 77.2] | 0.48 [0.12, 0.88] |
| top-2 Boltzmann | 152.8 [138.6, 178.8] | 34.0 [5.8, 78.1] | 0.48 [0.08, 0.88] |
| Boltzmann (no ε floor) | 148.3 [119.2, 174.7] | **114.5 [70.1, 122.2]** | 0.40 [0.04, 0.80] |
| top-3 Boltzmann | 122.8 [46.1, 168.5] | 32.8 [12.0, 56.2] | 0.28 [0.00, 0.64] |
| ε-greedy⊕Boltzmann mix | 117.2 [7.8, 165.4] | 62.2 [24.1, 73.1] | 0.28 [0.00, 0.64] |
| top-p Boltzmann | 104.7 [51.1, 150.8] | 12.8 [−22.0, 59.8] | 0.24 [0.00, 0.64] |
| ε→Boltzmann (k path) | 100.5 [95.3, 123.4] | 26.4 [12.1, 37.0] | 0.20 [0.00, 0.60] |

### 7.1 The one clearly resolved finding: the ε-floor's value reverses sign

The project's premise is that a softmax over an untrained Q is *worse* than
uniform exploration. Measured as sample efficiency (AUC) against the ε-greedy
baseline, the no-floor Boltzmann arm gives:

| environment | actions | P(Boltzmann AUC beats ε-greedy AUC) |
|---|---|---|
| CartPole-v1 | 2 | **0.04 [0.00, 0.24]** — decisively worse |
| Acrobot-v1 | 3 | 0.48 [0.12, 0.80] — indistinguishable |
| LunarLander-v3 | 4 | **0.92 [0.68, 1.00]** — decisively better |

Both extremes exclude 0.5 from their intervals. This is the only effect in the
whole main comparison large enough to resolve at 5 seeds, and it is a **sign
reversal**, not a magnitude difference: dropping the ε floor is the worst thing
you can do on CartPole and the best thing you can do on LunarLander.

The behaviour-policy entropy traces (mean over 5 seeds) show the mechanism:

| | step 0 | 2k | 10k | late |
|---|---|---|---|---|
| **CartPole**, max entropy ln 2 = 0.69 | | | | *(100k)* |
| ε-greedy | 0.693 | 0.692 | 0.647 | 0.117 |
| Boltzmann | **0.569** | 0.559 | 0.458 | 0.163 |
| **LunarLander**, max entropy ln 4 = 1.39 | | | | *(400k)* |
| ε-greedy | 1.386 | 1.386 | 1.379 | 0.201 |
| Boltzmann | 1.261 | 1.261 | 1.147 | **0.882** |

On both environments the Boltzmann arm starts *below* maximum entropy — the
bias from softmaxing an untrained Q, visible in the very first evaluated
policy, exactly as §1 predicts. What differs is what that bias costs:

- On **CartPole** the two actions are symmetric and a uniform random action is
  nearly free, so the early bias is pure loss and the arm never recovers its
  sample-efficiency deficit.
- On **LunarLander** a uniform random action is *expensive* — firing thrusters
  at random crashes the lander — so ε-greedy's uniform floor spends its
  exploration budget on actions already known to be catastrophic. Boltzmann
  spends it proportionally instead, and holds a genuinely informative 0.88 nats
  of exploration at 400k where ε-greedy has collapsed to 0.20.

**The right way to state the project's premise is therefore conditional**: the
ε floor is insurance against a Q-function that is not yet informative, and like
any insurance it is worth buying only when the premium — the cost of a uniformly
random action — is low relative to the risk. That is a sharper claim than "anneal
from one to the other", and it predicts *where* the handover idea should pay off:
environments with many actions of widely differing cost.

### 7.2 The one resolved comparison among the ε-floor arms

On Acrobot, the **support path** (`anneal_k`: k grows 1 → |A| at fixed τ) beats
ε-greedy on final return with `P = 0.96 [0.76, 1.00]` — the interval excludes
0.5, so this one is real. Its AUC is *not* better (`P = 0.48`): it reaches a
better final policy without learning faster.

This is idea 2 from the brief working as intended, and Acrobot is where it
should first become visible — |A| = 3 is the smallest action set where growing
the support has more than one step to take. Note the temperature path is
simultaneously the *worst* arm on the same environment (−98.3, `P = 0.32`), so
this is specifically about the support path, not about annealing in general.

### 7.3 What is not resolved

Everything else. On CartPole and LunarLander every ε-floor arm's
`P(beats ε-greedy)` interval contains 0.5, and every final-return point estimate
sits inside its neighbours' CIs. ε-greedy is the top point estimate on both, and
that is **also** not a finding — its CIs overlap the field just as thoroughly.

At 5 seeds this study can resolve a sign reversal in sample efficiency and one
support-path win on one environment. It cannot rank eight exploration schedules,
and §6.2 shows what happens when you try to at three.

## 8. Results: Q-scaling ablation and the uncertainty extension

### 8.1 Q-scaling ablation (Acrobot-v1, `eps_boltzmann` schedule, 100k, 3 seeds)

| `q_scaling` mode | Final return | P(beats `running`) |
|---|---|---|
| `per_state` | −151.4 [−237.9, −75.5] | 0.56 [0.00, 1.00] |
| `running` (default) | −154.5 [−207.1, −86.1] | — (baseline) |
| `none` | −307.7 [−500.0, −106.9] | 0.22 [0.00, 0.67] |

`per_state` and `running` are indistinguishable (point estimates 3 return apart,
`P = 0.56` with an interval spanning [0, 1]). Only `none` separates, and even
that is not resolved at 3 seeds (`P = 0.22 [0.00, 0.67]`).

The *performance* column is therefore weak. The **entropy** column is not — and
it is the reason this ablation is in the report. Running the identical configs
on two different machines (§6.2) reproduces the entropy trace almost exactly
while scrambling the performance ranking:

| mode | entropy @100k (4-core Linux → M1 Pro) | P(non-greedy) @100k |
|---|---|---|
| `none` | 0.93 → **0.90** | 0.46 → **0.45** |
| `running` | 0.38 → **0.34** | 0.15 → **0.13** |
| `per_state` | 0.13 → **0.14** | 0.05 → **0.04** |

This is exactly what §3 predicts, and it is machine-independent:

- **`none` never settles.** Entropy stays near 0.9 nats for the entire run: as Q
  inflates from its near-zero initialisation, a fixed τ keeps meaning something
  different, so the policy never converges to a stable exploration level. The
  single number that is supposed to control Boltzmann exploration is not
  actually under the experimenter's control.
- **`per_state` is ~2.5× more confidently peaked than `running`** (0.14 vs 0.34
  nats; 4% vs 13% non-greedy actions), because forcing every Q-row to unit
  spread manufactures a confident preference on rows where the actions are
  genuinely equivalent.

**The honest split: the mechanism is confirmed, its performance consequence is
not.** Going in, the expectation from §3 was that `per_state`'s false confidence
should *hurt*. On Acrobot it does not — reward is dense (−1 per step until the
goal) and early exploitation is not obviously wrong, so the same mechanism that
manufactures false confidence also cuts wasted exploration. Whether it costs
performance should depend on how many states have a genuinely flat Q-row —
plausibly more on LunarLander, where several actions can be similarly reasonable
mid-flight, than on Acrobot's more decisive dynamics. This dataset does not
test that; `running` remains the default because it is the mode whose behaviour
matches its stated intent, not because it measurably won.

### 8.2 Uncertainty-gated extension (Acrobot-v1, 100k steps, 3 seeds)

The extension went through **three iterations**. The first failed, the second
falsified the obvious explanation for why, and the third identified and fixed
the actual defect. All nine arms are in one sweep so they share seeds and
architecture.

| Variant | Final return | P(beats ε-greedy, ensemble) |
|---|---|---|
| ε→Boltzmann, ensemble architecture | −84.1 [−92.1, −77.2] | 0.56 [0.00, 1.00] |
| **ε-greedy, ensemble (control)** | −93.2 [−117.7, −80.3] | — |
| ε-greedy, single head | −105.4 [−138.3, −81.1] | 0.33 [0.00, 0.89] |
| gated, ensemble, **frozen ref** (v3) | −296.1 [−500.0, −93.4] | 0.11 [0.00, 0.44] |
| gated, TD error, scaled (v2) | −312.4 [−461.2, −97.9] | 0.11 [0.00, 0.44] |
| gated, TD error, **frozen ref** (v3) | −357.1 [−452.9, −241.0] | 0.00 [0.00, 0.00] |
| gated, ensemble, scaled (v2) | −375.1 [−500.0, −226.8] | 0.00 [0.00, 0.00] |
| gated, ensemble (v1) | −385.8 [−500.0, −290.1] | 0.00 [0.00, 0.00] |
| gated, TD error (v1) | −398.5 [−500.0, −335.9] | 0.00 [0.00, 0.00] |

**Every gated arm loses to the control.** The extension does not work at this
budget, and the honest headline is that it failed. What follows is why, because
the failure turned out to be diagnosable and partly fixable, and the diagnosis
is more interesting than the ranking.

The matched-architecture control (§4) does its job throughout: ensemble
ε-greedy (−93.2) beats single-head ε-greedy (−105.4), so the extra heads are
mildly *helpful* and the collapse is attributable to the gating, not the
architecture. This also replicated across both machines (§6.2) — v1's arms
scored −330/−347 on the Linux box and −386/−399 here, `P = 0.00` in both.

#### v1: the gate never fires

Confidence is pinned to 0 during warm-up (10k steps), then jumps and flatlines:

| step | 5k | 10k | 12k | 20k | 50k | 100k |
|---|---|---|---|---|---|---|
| confidence (ensemble) | 0.00 | 0.00 | **0.33** | 0.34 | 0.31 | 0.26 |
| ε (ensemble) | 1.00 | 1.00 | 0.67 | 0.67 | 0.69 | 0.74 |
| confidence (TD error) | 0.00 | 0.00 | **0.83** | 0.28 | 0.04 | 0.04 |
| ε (TD error) | 1.00 | 1.00 | 0.18 | 0.73 | **0.96** | 0.96 |

Both arms sit between `ε ≈ 0.67` and `ε ≈ 0.96` for essentially the whole run,
while every fixed-schedule arm reaches `ε ≤ 0.05` by 30k. With
`eps_uncertain = 1.0, eps_confident = 0.01`, confidence 0.3 interpolates to
`ε = 0.70` exactly as observed: the gate is doing precisely what it was told,
and what it was told is wrong. **The handover the extension exists to trigger
never happens**, which is a sufficient explanation for the collapse on its own.

#### v2: the obvious explanation, tested and falsified

Both signals are measured in **raw Q units** — ensemble disagreement is a
standard deviation of Q-values, the TD error a difference of them — and Q
magnitudes inflate during training. Measured over 100k steps the *raw* signals
do not decay, they grow: ensemble disagreement 0.076 → 0.136, TD error
0.27 → 1.45. This looked like §3's failure mode appearing in a place §3 was
never applied, so `normalise_uncertainty` divides the signal by the same running
Q-scale the policy uses for its temperature (`policy/uncertainty_gated_scaled`).

It made the signal dimensionless and **changed nothing**:

| `u / reference` | 11k | 40k | 100k |
|---|---|---|---|
| ensemble, v1 (raw units) | 0.699 | 0.671 | 0.741 |
| ensemble, v2 (scale-normalised) | 0.691 | 0.712 | 0.831 |
| TD error, v1 | 0.142 | 0.954 | 0.964 |
| TD error, v2 | 0.327 | 0.958 | 0.963 |

Final returns moved from −385.8 to −375.1 (ensemble) and −398.5 to −312.4 (TD),
both well inside the noise. **The units were a real defect and not the binding
one.**

#### v3: the actual defect is structural

The ratio is unmoved because of how confidence is defined:

```
confidence = 1 − u / quantile(u's own recent history)
```

The reference is estimated **from the very stream it normalises**. Whatever `u`
does — grow, shrink, or hold — the running quantile follows it, `u/ref`
converges to a constant, and confidence is pinned there. This is not a
calibration error that a better learning rate would fix; **no choice of units,
quantile or step size can make a self-normalising signal detect its own trend.**
It was designed to be scale-free and is accidentally trend-free as well.

v3 freezes the reference at the end of warm-up (`reference_freeze_step`), making
it a fixed calibration constant measured during early training. Confidence then
rises if and only if uncertainty falls below its early-training level, which is
what the gate was always supposed to mean. The ensemble arm's trace:

| `gated_ensemble_frozen` | 11k | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|---|
| uncertainty (frozen ref = 5.25) | 3.24 | 2.81 | 1.84 | 1.95 | 1.66 | **1.53** |
| confidence | 0.00 | 0.46 | 0.64 | 0.62 | 0.68 | **0.71** |
| ε | 1.00 | 0.55 | 0.37 | 0.39 | 0.33 | **0.30** |
| k | 1.0 | 1.9 | 2.3 | 2.2 | 2.4 | **2.3** |

**The handover fires.** Uncertainty genuinely falls, confidence rises
monotonically, ε decays and the Boltzmann support widens — the mechanism
described in §4, working, for the first time across three iterations. Final
return improves from −385.8 to −296.1, and the per-seed spread is the
interesting part: seed 1 finishes at **−93.4**, level with the control, while
seed 0 finishes at −500. The mechanism can work; it is not yet reliable.

Two things still stop it winning, and both are now specific rather than
mysterious:

1. **It is far too slow.** ε reaches only 0.30 by 100k where the fixed
   schedules reach 0.05 by 30k. The arm spends the entire budget more
   exploratory than every competitor, so it is still measuring a
   less-trained network. The confidence→ε map, not the signal, is what needs
   re-tuning now.
2. **There is a feedback trap.** High ε produces a worse value function, which
   produces higher disagreement, which holds ε high. A measurement-driven gate
   closes a loop that a time-based schedule does not, and nothing in the design
   accounts for that.

#### The TD-error signal is unsuitable, and now demonstrably so

With the reference frozen, the TD-error arm's uncertainty **grows** — 8.9 →
24.9 — and confidence collapses to 0, holding `ε ≈ 0.95`. This is not a
calibration problem either: `|TD error|` reflects reward magnitude, bootstrap
noise and the target network's staleness, none of which shrink just because the
agent has learned. It is an *aleatoric-flavoured* signal being asked to do an
epistemic job. The ensemble's disagreement, which is genuinely epistemic, falls
by a factor of 2 over the same window. **This comparison was the point of
including both signals** (§4), and freezing the reference is what finally made
it visible: only when the reference stops chasing the signal can you see which
signal has a trend worth chasing.

#### Status

The extension is **implemented, diagnosed across three iterations, and still
losing to the control**. The defect that made it lose is understood and fixed;
what remains is a mis-tuned confidence→ε map and a feedback trap, both of which
are ordinary tuning problems rather than design flaws. Given §6.2's finding that
3 seeds cannot resolve a ranking in this family, the next step is not more
tuning against these numbers but a 5-seed run of the frozen-reference ensemble
arm against the control with a faster confidence→ε map. `normalise_uncertainty`
and `reference_freeze_step` both default to **off**, so the v1 results above
remain exactly reproducible.

## 9. What the design buys, independent of the numbers

Three things in this project are worth keeping regardless of how the arms
ranked, because they are the parts that make the comparison *possible*:

**The unified family makes the endpoints exact.** Because `k = 1` and `τ → 0`
both reduce to ε-greedy *to machine precision*, and `ε = 0` with full support is
exactly softmax, an annealing schedule genuinely starts at one classic strategy
and ends at the other. A hand-rolled blend would start and end near them, and
every claim about "how the handover affects learning" would be contaminated by
the difference. `tests/test_policies.py` pins this down as exact array equality
against independently written reference implementations.

**The Q-scale normaliser makes a temperature portable.** Without it, "τ = 0.3"
means something different on every environment and at every point in training,
so the single number that defines Boltzmann exploration is not actually under
the experimenter's control. §8.1 shows what happens without it: an entropy
trace that never settles, versus one that does.

**Explicit action distributions make exploration measurable.** Computing
`π(a|s)` rather than sampling procedurally costs microseconds and yields
behaviour-policy entropy and P(action ≠ argmax) for free — the only two
quantities that are comparable across strategies whose knobs are not. Without
them, the report could say the schedule moved but not that the behaviour did.

## 10. Limitations

Stated plainly, because they bound what the results above support.

- **Five seeds is enough for a sign reversal and not much else.** §7.1's
  ε-floor result and §7.2's support-path win are the only two comparisons whose
  probability-of-improvement intervals exclude 0.5. Every other ranking in §7 is
  a point estimate, and §6.2 demonstrates empirically what happens if you trust
  one at 3 seeds: the ordering does not survive a change of BLAS library. Ten to
  thirty seeds is what the deep-RL evaluation literature asks for; this study
  has five.
- **The §8 studies are still at 3 seeds.** The Q-scaling ablation and the
  uncertainty extension were not re-run at 5. Their *performance* columns are
  correspondingly weak — §8.1 says so explicitly. Their load-bearing claims are
  the entropy mechanism and the gating collapse, both of which replicated across
  two machines (§6.2), which is a different and stronger kind of evidence than a
  3-seed ranking.
- **No Atari results.** The ALE code path is implemented, unit-tested against
  real ALE environments, and verified to complete a training run end-to-end —
  but not trained. At ~10M frames per run, 8 arms × 5 seeds is on the order of
  a GPU-month. This matters directly for §7.1's conclusion: the mechanism it
  identifies — the ε floor is worth its premium only when a uniformly random
  action is cheap — predicts that the effect should *grow* with the action set,
  and Atari's up-to-18 actions is where that prediction would be tested. It has
  not been.
- **Small action sets limit idea 2.** On CartPole (|A| = 2) `top-2` and `top-3`
  Boltzmann are provably the same policy, and the §7 table confirms it to the
  last decimal on all 5 seeds. `anneal_k` has one step to take. Acrobot (3) and
  LunarLander (4) are better but still narrow — and §7.2's single resolved
  support-path win is on Acrobot, the environment where |A| first makes the
  variant non-trivial. That is suggestive, not conclusive.
- **Three environments, all dense-reward and low-dimensional.** The §7.1
  mechanism is about the *cost of a random action*, which these three happen to
  span usefully (free on CartPole, fatal on LunarLander). They do not span
  sparse reward, long horizons, or pixel observations at all.
- **The extension is left mid-repair.** §8.2 fixes the defect that stopped the
  uncertainty gate firing, but the frozen-reference arm is still slow (ε only
  reaches 0.30 by 100k) and still loses to its control. The remaining problems
  are named — a mis-tuned confidence→ε map and a value/uncertainty feedback
  trap — but not solved, and re-tuning against 3-seed numbers that §6.2 shows
  are unreliable would be the wrong move.
- **One hyperparameter setting per variant.** Each variant's schedule endpoints
  and durations were chosen once, on reasoning rather than a search. A variant
  that looks worse may simply be mis-tuned. §8.1 shows how sensitive this family
  is to a single such choice (`q_scaling`), and §8.2's failure is *itself* a
  calibration bug in one specific normalisation scheme rather than a refutation
  of uncertainty-gated exploration.
- **Evaluation is greedy only.** Every number is the greedy policy's return.
  This is the right choice for comparing learning quality across arms (§6), but
  it means the report says nothing about the online return an agent would
  actually collect while exploring — which for a deployed system is often the
  quantity that matters.

## 11. Reproduction

```bash
make setup                     # swig BEFORE gymnasium[box2d] -- see README
make test                      # 194 tests

make experiment-full           # THE study in §7: 3 envs x 8 arms x 5 seeds
make experiment-ablation       # §8.1  Acrobot, Q-scaling
make experiment-uncertainty    # §8.2  Acrobot, uncertainty-gated extension

make plots SWEEP=full_gym      # regenerate every figure and table for one sweep
```

`make experiment-full WORKERS=8` reproduces §7 end to end: 120 runs, 26.0M
environment steps, **35 minutes wall-clock on an M1 Pro** (4.6 h of summed
worker time). The three reduced 3-seed sweeps (`experiment-reduced`,
`experiment-acrobot`, `experiment-main`) are retained for §6.2's cross-machine
comparison and run in under 20 minutes together.

Every figure in this report is regenerated from run outputs by
`scripts/make_plots.py`; nothing here was drawn by hand. Sweeps are resumable —
a cell whose `result.json` exists is skipped — so an interrupted run is
restarted with the same command.

**On exact reproducibility:** individual runs are deterministic per seed *on a
fixed machine and library stack* — §7's CartPole table shows `top-2` and `top-3`
agreeing to the last decimal across all 5 seeds. They are **not** reproducible
across torch/BLAS builds; §6.2 quantifies how far the aggregate rankings move.
This is normal for deep RL and is the reason this report reports intervals and
probabilities rather than point estimates.
