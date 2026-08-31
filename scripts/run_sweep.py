#!/usr/bin/env python
"""Run a grid of (environment x exploration policy x seed) training runs.

Usage::

    python scripts/run_sweep.py --sweep reduced_gym --workers 4

A sweep file lists the axes; this script expands them, runs each cell in its own
process, and collects the per-run summaries into one CSV under
``results/summaries/``.

Two properties matter for using this on a laptop:

* **Resumable.** A cell whose ``result.json`` already exists is skipped, so an
  interrupted sweep can simply be re-run. This is the resilience mechanism for
  long sweeps -- there is deliberately no mid-run checkpointing, because at
  these run lengths (minutes) restarting a cell is cheaper than the complexity
  of saving and restoring replay buffers.
* **One torch thread per worker.** Torch defaults to using every core *per
  process*; with N worker processes that oversubscribes the machine by N x and
  runs slower than serial. The environment variables must be set before torch
  is imported in the child, which is why they are set at module import time
  here and the workers use the 'spawn'-safe top-level function below.
"""

from __future__ import annotations

import argparse
import csv
import os

# Must precede any torch import in this process or its children.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import itertools  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2b.config import Config, config_from_dict, load_yaml_tree  # noqa: E402

SWEEP_ROOT = ROOT / "configs" / "sweeps"


def build_cells(spec: dict) -> list[dict]:
    """Expand a sweep spec into one config dict per run."""
    base = spec.get("base", "base")
    seeds = spec.get("seeds", [0])
    common = spec.get("overrides", {}) or {}
    cells: list[dict] = []

    # `arms` lets a sweep pair a policy with arm-specific agent overrides --
    # needed because the uncertainty arms require num_heads > 1 while their
    # control arms must use the identical architecture.
    arms = spec.get("arms")
    if arms is None:
        arms = [{"policy": p} for p in spec["policies"]]

    for env_name, arm, seed in itertools.product(spec["envs"], arms, seeds):
        parts = [base, env_name, arm["policy"]]
        data: dict = {}
        for part in parts:
            data = _merge(data, load_yaml_tree(part))
        data = _merge(data, common)
        data = _merge(data, arm.get("overrides", {}) or {})
        data = _merge(data, {"run": {"seed": seed}})
        # Label from the arm, else from the policy config's FILE stem -- not
        # from `policy.name`. Several config files share a policy name (e.g.
        # topk_boltzmann.yaml and topk3_boltzmann.yaml are both the
        # `topk_boltzmann` policy with different k), and labelling by name
        # would silently collapse them into one arm.
        data["_label"] = arm.get("label") or Path(arm["policy"]).stem
        cells.append(data)
    return cells


def _merge(base: dict, override: dict) -> dict:
    from e2b.config import _deep_merge

    return _deep_merge(base, override)


def cell_run_dir(cfg: Config, label: str, out_root: Path) -> Path:
    return out_root / cfg.env.id.replace("/", "_") / label / f"seed{cfg.run.seed}"


def run_cell(payload: tuple[dict, str, str]) -> dict:
    """Worker entry point. Runs one training run and returns its summary."""
    data, label, out_root = payload
    data = dict(data)
    data.pop("_label", None)
    from e2b.train import train

    cfg = config_from_dict(data)
    run_dir = cell_run_dir(cfg, label, Path(out_root))
    result_path = run_dir / "result.json"
    if result_path.exists():
        import json

        summary = json.loads(result_path.read_text())
        summary["status"] = "skipped"
        summary["label"] = label
        return summary

    started = time.time()
    try:
        summary = train(cfg, run_dir)
        summary["status"] = "ok"
    except Exception:  # pragma: no cover - surfaced in the sweep CSV
        summary = {
            "run_id": cfg.run_id,
            "env": cfg.env.id,
            "policy": cfg.policy.get("name"),
            "seed": cfg.run.seed,
            "status": "failed",
            "error": traceback.format_exc(limit=3),
            "wall_time_s": time.time() - started,
        }
    summary["label"] = label
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, help="name under configs/sweeps/")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default=None, help="override results root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    spec_path = SWEEP_ROOT / f"{args.sweep}.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    out_root = Path(args.out or spec.get("out_dir", f"results/runs/{args.sweep}"))
    out_root.mkdir(parents=True, exist_ok=True)

    cells = build_cells(spec)
    payloads = []
    for data in cells:
        label = data.get("_label") or data["policy"]["name"]
        payloads.append((data, label, str(out_root)))

    print(f"sweep '{args.sweep}': {len(payloads)} runs, {args.workers} workers")
    for data, label, _ in payloads:
        cfg = config_from_dict({k: v for k, v in data.items() if k != "_label"})
        print(f"  {cfg.env.id:20s} {label:28s} seed={cfg.run.seed} "
              f"steps={cfg.train.total_steps}")
    if args.dry_run:
        return 0

    started = time.time()
    if args.workers <= 1:
        summaries = [run_cell(p) for p in payloads]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            summaries = []
            for i, summary in enumerate(pool.imap_unordered(run_cell, payloads), 1):
                summaries.append(summary)
                print(
                    f"[{i}/{len(payloads)}] {summary.get('status'):8s} "
                    f"{summary.get('env')} {summary.get('label')} "
                    f"seed={summary.get('seed')} "
                    f"final={summary.get('final_eval_return', float('nan')):.1f}",
                    flush=True,
                )

    summary_dir = ROOT / "results" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f"{args.sweep}.csv"
    fields = [
        "env", "label", "policy", "seed", "status", "total_steps",
        "final_eval_return", "best_eval_return", "auc_eval_return",
        "episodes", "wall_time_s", "steps_per_second",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for s in sorted(summaries, key=lambda r: (r.get("env", ""), r.get("label", ""), r.get("seed", 0))):
            writer.writerow(s)

    failed = [s for s in summaries if s.get("status") == "failed"]
    print(f"\ndone in {time.time() - started:.0f}s -> {csv_path}")
    if failed:
        print(f"{len(failed)} FAILED:")
        for s in failed:
            print(f"  {s['env']} {s['label']} seed={s['seed']}\n{s.get('error', '')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
