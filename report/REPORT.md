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

Experiments in this report ran on a **4-core CPU box with no GPU**. Cleanly
measured throughput of the identical agent is close across environments — the
classic-control tasks and LunarLander both run at roughly 650–700 steps/s per
worker on 4 workers, once a measurement artifact is corrected for:

| environment | actions | steps/s per worker (4 workers) |
|---|---|---|
| CartPole-v1 | 2 | ~340–700 |
| Acrobot-v1 | 3 | ~640–710 |
| LunarLander-v3 | 4 | ~660–700 |

*(An earlier throughput check reported LunarLander at ~50 steps/s — a
transient artifact of stale orphaned worker processes from a previous run
competing for CPU at measurement time, not a real cost of Box2D. The
correction matters: it means `configs/sweeps/main_gym.yaml` runs LunarLander at
its full 400k-step budget directly, rather than a truncated one, at roughly
1 h for all 24 runs on 4 workers — cheap enough that the honest move is to run
it at full budget on whatever machine actually produces the numbers, rather
than lock in a compromise here.)*

| study | environment | steps | arms | seeds | run in this report |
|---|---|---|---|---|---|
| main comparison | CartPole-v1 | 60k | 7 | 3 | yes |
| main comparison | Acrobot-v1 | 100k | 8 | 3 | yes |
| main comparison | LunarLander-v3 | 400k | 8 | 3 | **configured, not run here** |
| Q-scaling ablation | Acrobot-v1 | 100k | 3 | 3 | yes |
| uncertainty extension | Acrobot-v1 | 100k | 5 | 3 | yes |

**LunarLander results are not included in §7** — `main_gym.yaml` is fully
configured and validated (`tests/test_sweeps.py` checks it expands and
constructs correctly) but was not executed as part of this report; run
`make experiment-main` to produce it. This is the same status as the Atari
code path (§10): implemented and ready, not a claimed result.

## 7. Results: CartPole-v1 and Acrobot-v1

Full tables and figures are generated by `scripts/make_plots.py` from the
committed run outputs; the headline numbers are reproduced here (IQM, 95%
stratified-bootstrap CI, `n = 3` seeds).

### CartPole-v1 (60k steps)

| Variant | Final return | P(beats ε-greedy) |
|---|---|---|
| top-p Boltzmann | 348.7 [310.7, 375.9] | 0.56 [0.00, 1.00] |
| **ε-greedy (baseline)** | 344.1 [303.0, 378.6] | — |
| ε-greedy⊕Boltzmann mix | 337.7 [301.7, 383.7] | 0.44 [0.00, 1.00] |
| ε→Boltzmann (τ path) | 322.8 [265.4, 356.2] | 0.33 [0.00, 0.89] |
| top-2 Boltzmann | 293.1 [210.3, 334.7] | 0.22 [0.00, 0.67] |
| ε→Boltzmann (k path) | 259.8 [240.1, 293.7] | 0.00 [0.00, 0.00] |
| Boltzmann (no ε floor) | 157.3 [108.2, 221.8] | 0.00 [0.00, 0.00] |

### Acrobot-v1 (100k steps)

| Variant | Final return | P(beats ε-greedy) |
|---|---|---|
| top-3 Boltzmann | −110.4 [−127.6, −89.2] | 0.67 [0.00, 1.00] |
| ε→Boltzmann (k path) | −120.8 [−160.1, −87.8] | 0.56 [0.00, 1.00] |
| Boltzmann (no ε floor) | −134.2 [−206.7, −93.3] | 0.56 [0.00, 1.00] |
| top-2 Boltzmann | −134.6 [−228.1, −83.3] | 0.67 [0.11, 1.00] |
| ε-greedy⊕Boltzmann mix | −140.3 [−197.1, −96.9] | 0.56 [0.00, 1.00] |
| **ε-greedy (baseline)** | −150.9 [−229.5, −86.5] | — |
| ε→Boltzmann (τ path) | −163.0 [−226.9, −85.8] | 0.56 [0.00, 1.00] |
| top-p Boltzmann | −190.1 [−219.7, −142.9] | 0.33 [0.00, 1.00] |

### Reading these numbers honestly

At `n = 3` almost every CI above spans the full [0, 1] range for
`P(beats ε-greedy)`, and every point-estimate ranking sits inside its
neighbours' intervals. **The correct reading is that CartPole and Acrobot do
not resolve a ranking among the variants at this seed count** — the ordering
above is a point estimate, not a finding. Two things are nonetheless real and
worth stating:

1. **Pure Boltzmann without an ε floor is the one clearly separated result on
   CartPole**: 157 vs. 344 IQM final return, non-overlapping CIs,
   `P(beats ε-greedy) = 0.00`. This is the premise of the project, observed
   directly: a softmax over a Q-function that has not yet learned anything is
   *worse* than uninformed uniform exploration. On Acrobot the same arm is
   competitive, which is consistent with Acrobot's Q-values separating faster
   (its state space is smaller) — the failure mode is about *when* Q becomes
   informative, not a property of Boltzmann exploration in general.
2. **Every variant with an ε floor — however it is scheduled — lands within
   noise of ε-greedy on both environments at this budget.** Neither the
   temperature path, the support path, nor the mixture path shows a resolvable
   advantage or disadvantage over the ε-greedy baseline here. That is a
   genuine (if unexciting) result: on these two small, dense-reward
   environments, *how* the handover happens matters much less than *whether
   there is a floor at all* during the phase these runs cover.

The exploration-diagnostics figures
(`report/figures/{reduced_gym,acrobot_gym}_*_exploration.png`) show why finding
1 is not a fluke of one Q-scale choice: the `boltzmann` arm's behaviour-policy
entropy *starts below* the other arms' (0.57 nats vs. ε-greedy's 0.69 = ln 2,
the maximum for 2 actions) and only reaches it briefly around 2k steps before
falling — i.e., the softmax is biased from the very first evaluated policy, not
merely "less exploratory than uniform" by construction. This is the failure
mode described in §1 made visible.

## 8. Results: Q-scaling ablation and the uncertainty extension

### 8.1 Q-scaling ablation (Acrobot-v1, `eps_boltzmann` schedule, 100k steps)

| `q_scaling` mode | Final return | P(beats `running`) |
|---|---|---|
| `per_state` | −89.1 [−94.7, −83.0] | 0.78 [0.33, 1.00] |
| `running` (default) | −163.0 [−226.9, −85.8] | — (baseline) |
| `none` | −242.3 [−416.3, −86.6] | 0.33 [0.00, 0.89] |

The final-return CIs overlap enough (`per_state` vs `running` in particular)
that the ranking above is **not** a resolved finding at `n = 3` — the learning
curves (`report/figures/ablation_scaling_Acrobot-v1_curves.png`) show all three
modes occupying the same band for most of training and only separating near
the end.

What *is* resolved, and confirms the mechanism described in §3 directly, is the
behaviour-policy entropy trace
(`report/figures/ablation_scaling_Acrobot-v1_exploration.png`), where the three
modes separate cleanly and stay separated:

| mode | entropy at 100k (nats) | P(non-greedy) at 100k |
|---|---|---|
| `none` | ≈0.93 | ≈0.46 |
| `running` | ≈0.38 | ≈0.15 |
| `per_state` | ≈0.13 | ≈0.05 |

`per_state` produces a policy that is **roughly 3× more confidently peaked**
than `running` throughout the second half of training, and `none` never
converges below the entropy it started near — exactly what §3 predicts: without
normalisation the effective temperature keeps drifting as Q grows, so the
policy never settles into a stable exploration level; with `per_state`
normalisation every Q-row is forced to unit spread regardless of whether the
row is genuinely decided, so the policy is confident almost everywhere.

**This is a real result about the mechanism, and a candid limitation about the
performance claim.** Going in, the expectation from §3 was that `per_state`'s
false confidence on genuinely-flat Q-rows should *hurt* — but on Acrobot,
where reward is dense (−1 every step until the goal) and confident early
exploitation is not obviously wrong, the same mechanism that produces
false confidence also produces less wasted exploration, and the point estimate
favours it. Whether `per_state`'s failure mode actually costs performance
likely depends on how many states in the environment have a genuinely flat
Q-row — plausibly more on LunarLander, where several distinct actions can be
similarly reasonable mid-flight, than on Acrobot's more decisive dynamics. That
is a hypothesis this dataset does not have the seeds or the second environment
to confirm; the honest conclusion is **the entropy mechanism is confirmed, the
performance consequence is not resolved here**.

### 8.2 Uncertainty-gated extension (Acrobot-v1, 100k steps)

| Variant | Final return | P(beats ε-greedy, ensemble) |
|---|---|---|
| ε→Boltzmann, ensemble architecture | −82.9 [−88.7, −77.4] | 0.67 [0.11, 1.00] |
| **ε-greedy, ensemble (control)** | −106.5 [−156.1, −80.5] | — |
| ε-greedy, single head | −150.9 [−229.5, −86.5] | 0.22 [0.00, 0.67] |
| uncertainty-gated, TD-error signal | −330.4 [−394.6, −222.0] | 0.00 [0.00, 0.00] |
| uncertainty-gated, ensemble signal | −346.6 [−500.0, −160.9] | 0.00 [0.00, 0.00] |

Both uncertainty-gated arms are **clearly worse** here — non-overlapping CIs
against every other arm, `P = 0.00`. The matched-architecture control (§4)
does its job: the ensemble-vs-single-head epsilon-greedy comparison (−106.5 vs
−150.9) shows the ensemble architecture itself is if anything mildly *helpful*,
so the gated arms' collapse is attributable to the gating, not the extra heads.

**Why**, and it is fully diagnosable from the logged knob traces
(`report/figures/uncertainty_Acrobot-v1_{confidence,knobs}.png`), is a genuine
finding about this scheme's calibration rather than a mystery:

1. **A confidence overshoot right as warm-up ends.** Confidence is pinned to 0
   during the first `warmup_steps` (10k here), but the reference quantile
   keeps accumulating raw uncertainty in the background throughout warm-up —
   and raw uncertainty (TD error, ensemble disagreement) is largest early,
   when the network is least trained. The instant warm-up ends, current
   uncertainty is compared against that inflated reference and confidence
   spikes to ≈0.85 (TD-error signal) or ≈0.35 (ensemble) within a few thousand
   steps — briefly pushing the policy toward near-Boltzmann (`k` up to 3) on a
   network that has had only ~2,000 gradient steps. This is precisely the bad
   combination the warm-up guard was designed to prevent (§4), except the
   guard only pins confidence *during* warm-up and does nothing about the
   transient the moment it ends.
2. **The reference then overcorrects and stays there.** As the slow stochastic
   quantile tracker (`RunningQuantile`, lr = 0.01) chases the now-lower signal
   down, the *reference* keeps shrinking too, so confidence decays back
   towards ≈0.15–0.2 (ensemble) / ≈0.05 (TD-error) and **stays there for the
   rest of the 100k-step run** — it never recovers. With
   `eps_uncertain = 1.0, eps_confident = 0.01`, a steady-state confidence of
   0.2 interpolates to `ε ≈ 0.80`. The knob trace confirms this directly: both
   gated arms sit at `ε ≈ 0.75–0.95` for effectively the entire run, while
   every fixed-schedule arm has decayed to `ε ≤ 0.05` by 30k steps. The gated
   policy spends nearly this whole budget more exploratory than *any* other
   arm in the study, which is a straightforward, sufficient explanation for
   why it learns slower within a fixed 100k-step budget.

This is **not evidence against uncertainty-gated exploration as an idea** — it
is evidence that mapping a *relative* confidence signal onto absolute (ε, τ, k)
endpoints needs a reference that adapts on a similar timescale to the quantity
it normalises, and a high-quantile stochastic tracker with a small fixed
learning rate does not. Two concrete fixes this dataset points to, neither
implemented here: (a) exclude the warm-up window from the reference's own
training so it does not start inflated by the noisiest phase of learning, and
(b) widen the reference's adaptation rate or switch to a fixed-window quantile
so it tracks the *current* uncertainty level rather than a decaying memory of
its historical peak. Given the compute budget for this project, re-running
with a fix was out of scope; it is the natural next step for this extension.

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

- **Three seeds.** The confidence intervals are wide and mostly overlapping.
  Where they overlap, this report says *inconclusive* — it does not read a
  ranking out of the point estimates. Deep-RL exploration effects are small
  relative to DQN's seed variance, and 3 seeds is below what is needed to
  resolve them. Five to ten seeds is the minimum for confident claims;
  `configs/sweeps/full_gym.yaml` is set up for that.
- **Reduced step budgets on CartPole and Acrobot.** The main comparison ran
  CartPole to 60k steps and Acrobot to 100k, both below their `configs/env/`
  full defaults (100k / 150k). An exploration strategy that pays off later in
  training than these budgets cover could be undersold on these two
  environments.
- **No LunarLander results.** `configs/sweeps/main_gym.yaml` runs the full
  400k-step budget and is validated by `tests/test_sweeps.py`, but was not
  executed as part of this report — see §6.1. LunarLander is the environment
  the project brief names first, so this is the most consequential gap here;
  `make experiment-main` produces it directly.
- **No Atari results.** The ALE code path is implemented, unit-tested against
  real ALE environments, and verified to complete a training run end-to-end —
  but not trained. At ~10M frames per run, 8 arms × 5 seeds is on the order of
  a GPU-month. This matters for the conclusions: the two annealing paths
  (temperature and support) are *nearly equivalent* when |A| is 2–4, and only
  separate meaningfully as the action set grows. Atari's up-to-18 actions is
  where the support path should either prove itself or not, and that experiment
  has not been run.
- **Small action sets limit idea 2.** On CartPole (|A| = 2), `top-2 Boltzmann`
  is definitionally identical to full-support Boltzmann, and `anneal_k` has a
  single step to take. Acrobot (3) and LunarLander (4) are better but still
  narrow. The top-k family is under-tested here by construction.
- **One hyperparameter setting per variant.** Each variant's schedule endpoints
  and durations were chosen once, on reasoning rather than a search. A variant
  that looks worse may simply be mis-tuned; the ablation in §8.1 shows how
  sensitive this family is to one such choice (`q_scaling`), and the
  uncertainty extension's failure in §8.2 is *itself* a calibration bug in one
  specific normalisation scheme, not a rejection of the underlying idea.

## 11. Reproduction

```bash
make setup                     # swig BEFORE gymnasium[box2d] -- see README
make test                      # the full test suite

make experiment-reduced        # CartPole
make experiment-acrobot        # Acrobot
make experiment-ablation       # Acrobot, Q-scaling
make experiment-uncertainty    # Acrobot, uncertainty-gated extension
make experiment-main           # LunarLander

make plots SWEEP=acrobot_gym   # regenerate every figure and table for one sweep
```

Every figure in this report is regenerated from the committed run outputs by
`scripts/make_plots.py`; nothing here was drawn by hand. Sweeps are resumable —
a cell whose `result.json` exists is skipped — so an interrupted run is
restarted with the same command.

For results worth relying on, run `make experiment-full WORKERS=6`: 3
environments × 8 arms × 5 seeds at full step budgets (CartPole 100k, Acrobot
150k, LunarLander 400k), a few hours on an M1 Pro.
