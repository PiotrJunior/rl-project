# From ε-greedy to Boltzmann: annealing exploration in DQN

An RL course project studying **how to gradually transfer exploration from the
ε-greedy strategy to the Boltzmann (softmax) strategy** during DQN training.

The motivation: early in training the Q-estimate is noise, so a softmax over it
is worse than an honest uniform random action — it is *biased* by the noise.
Later, once Q is informative, Boltzmann exploration is better than ε-greedy
because it spends the exploration budget in proportion to how good each action
looks, instead of drawing uniformly from actions the agent already knows are
bad. The question is how to get from the first regime to the second.

---

## The central idea: one policy family, several paths through it

Rather than implementing ε-greedy and Boltzmann as two separate strategies and
blending them ad hoc, every variant here is a point in **one** parameterised
family:

```
π(a|s) = ε_t · Uniform(A)  +  (1 − ε_t) · Softmax_{a ∈ TopK_t(Q)}( Q̃(s,·) / τ_t )
```

with three scheduled knobs — `ε` (uniform floor), `τ` (temperature), `k`
(support size) — and a Q-scaling map `Q̃`.

Both classic strategies are **exact** points in this family, not approximations:

| configuration | is exactly |
|---|---|
| `k = 1` (any `τ`) | ε-greedy |
| `τ → 0`, `k = |A|` | ε-greedy |
| `ε = 0`, `k = |A|` | pure Boltzmann |

so "annealing from ε-greedy to Boltzmann" becomes *a path through (ε, τ, k)
space*, and the two ideas in the project brief are two different paths:

- **temperature path** (`eps_boltzmann`) — hold `k = |A|`, raise `τ` from ~0 to a
  target while `ε` falls. This is the brief's "simple annealing … with gradually
  changing the temperature".
- **support path** (`anneal_k`) — hold `τ` fixed, grow `k` from 1 to `|A|`. This is
  the brief's "sample from the Boltzmann distribution on the top few actions",
  expressed as an anneal.

These equivalences are enforced by tests (`tests/test_policies.py`) as exact
array comparisons, so the endpoints really are the textbook strategies.

## Variants implemented

| Variant | Idea | What it does |
|---|---|---|
| `eps_greedy` | baseline | ε: 1 → 0.05, `k = 1` |
| `boltzmann` | baseline | `ε = 0`, τ annealed, full support |
| `eps_boltzmann` | **1b** | temperature path: ε ↓, τ ↑ together |
| `mixture_anneal` | **1a** | π = (1−β)·π_ε-greedy + β·π_Boltzmann, β: 0 → 1 |
| `topk_boltzmann` | **2** | Boltzmann over the top-2 actions, fixed |
| `topk3_boltzmann` | **2** | …top-3 |
| `anneal_k` | **1+2** | support path: k: 1 → \|A\| |
| `topp_boltzmann` | **2** | nucleus: support adapts per state to Q's peakedness |
| `uncertainty_gated` | **extension** | knobs driven by measured Q-uncertainty, per state |
| `uncertainty_gated_td` | **extension** | same, driven by the (free) TD-error signal |

`mixture_anneal` is genuinely different from `eps_boltzmann`, not a
reparameterisation: at β = 0.5 it retains a real chance of a *uniformly random*
action inherited from the ε-greedy component, whereas the temperature path
passes through distributions that are neither endpoint.

## The detail that decides whether any of this works: Q-value scale

A Boltzmann temperature is meaningless unless you say *what units* the Q-values
are in. Raw Q magnitudes differ by orders of magnitude between environments
(CartPole ~10–500, LunarLander ~−400–300) **and grow substantially during a
single run** as the value function inflates from its near-zero initialisation.
A temperature that gives healthy exploration at 20k steps is effectively greedy
at 300k steps, with no change to the schedule.

Three modes are implemented (`src/e2b/policies/scaling.py`) and ablated:

- `none` — raw Q. Kept so the report can *show* the failure mode.
- `per_state` — `(q − mean_a q) / (std_a q + δ)`. Dimensionless, but it forces
  every Q-row to unit spread, so a state where all actions are genuinely
  equivalent becomes a confident preference over numerical noise. Backwards from
  what exploration should do.
- `running` (**default**) — `(q − mean_a q) / σ̂`, with `σ̂` an EMA of the
  across-action Q spread over recently visited states. Dimensionless *and*
  keeps a flat Q-row flat.

## Base agent

**Double + Dueling + n-step + prioritized replay** — "Rainbow minus NoisyNets
minus distributional".

**NoisyNets is deliberately excluded.** It is itself an exploration mechanism,
and including it would give every arm a second, uncontrolled exploration
strategy underneath the one being measured. Since the exploration strategy *is*
the object of study, the base agent must have none of its own. C51 is excluded
for cost, and because the spread it provides is aleatoric, whereas the
uncertainty extension needs epistemic uncertainty — which the ensemble heads
supply directly.

Everything about the agent is held identical across arms; **only the policy
block differs**, which is enforced by the config layout (`configs/base.yaml` is
shared, and unknown keys are rejected rather than ignored).

---

## Install

```bash
make setup
```

`swig` **must** be installed before `gymnasium[box2d]` — `box2d-py` has no
prebuilt wheels for CPython 3.11+ on linux-x86_64 or macOS-arm64, so it compiles
from source and needs `swig` on `PATH`. The Makefile does this in the right
order; installing both in one `pip install` fails.

On Apple Silicon this is the same story (`pip install swig` works; `brew install
swig` also works).

## Reproduce

```bash
make test                      # 140 tests
make smoke                     # 3k-step end-to-end check

make experiment-reduced        # CartPole, 7 variants, 3 seeds  (~15 min, 4 cores)
make experiment-main           # LunarLander, 8 variants, 3 seeds
make experiment-ablation       # Q-scaling ablation
make experiment-uncertainty    # uncertainty-gated extension

make plots SWEEP=main_gym      # regenerate figures + tables
```

Sweeps are **resumable**: a run whose `result.json` already exists is skipped,
so an interrupted sweep is restarted by re-running the same command. Each worker
is pinned to one torch thread — without that, N worker processes each grab every
core and the sweep runs slower than serial.

### Full-scale run

`make experiment-full WORKERS=6` runs 3 environments × 8 variants × 5 seeds at
full step budgets (CartPole 100k, Acrobot 150k, LunarLander 400k). On an M1 Pro
with 6 workers this is a few hours. The reported numbers in `report/REPORT.md`
come from the reduced 3-seed sweeps that fit this project's compute budget; the
full sweep is the configuration to use for conclusions you want to rely on.

Device note: `run.device: auto` resolves to **CPU** for the MLP experiments on
purpose. With two-layer MLPs at batch 64, per-kernel dispatch overhead dominates
and MPS/CUDA are measurably slower — while also serialising a parallel sweep
onto one accelerator. Pass `run.device=mps` (or `cuda`) for the convolutional
Atari path, where it does pay off.

## Atari

The ALE code path is **implemented and unit-tested but not trained**
(`src/e2b/envs/atari_wrappers.py`, `configs/env/atari.yaml`,
`tests/test_atari_wrappers.py` — including an end-to-end env → replay → CNN →
gradient-step contract test against a real ALE environment).

It is not run because a Rainbow-class Atari result is ~10M frames per run; at
8 variants × 5 seeds × 1 game that is on the order of a GPU-month, and this
project has 4 CPU cores. The same agent and the same exploration policies point
at ALE unchanged on suitable hardware:

```bash
PYTHONPATH=src python -m e2b.train --config env/atari --set run.device=cuda
```

Preprocessing follows Machado et al. (2018): sticky actions (p = 0.25) rather
than random no-ops, no loss-of-life termination by default, frame-skip 4 with
max-pooling, 84×84 grayscale, 4-frame stack, uint8 replay.

The `anneal_k` and `topk_boltzmann` variants are the ones most worth running
there: with |A| up to 18, the support path and the temperature path stop being
nearly equivalent, which they largely are on 2–4 action Gym tasks.

## Layout

```
configs/          base + env/ + policy/ + sweeps/  (composable YAML)
src/e2b/
  policies/       core.py (the action-distribution algebra), unified.py,
                  scaling.py (Q-scale normalisation), uncertainty_gated.py
  uncertainty/    ensemble disagreement + TD-error signals
  replay/         segment tree, n-step accumulator, uniform + prioritized
  nets/           MLP / Nature-CNN torsos, dueling heads, ensemble heads
  agent.py        DQN (double, dueling, n-step, PER)
  train.py        single run loop + diagnostics
  analysis.py     IQM + stratified bootstrap CIs
  plotting.py     figures
scripts/          run_sweep.py, make_plots.py
tests/            140 tests
report/REPORT.md  the write-up
```

## Results

See **[`report/REPORT.md`](report/REPORT.md)**.
