"""Loading run outputs back into arrays for analysis and plotting."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ArmResults:
    """All seeds of one (environment, exploration variant) cell."""

    env: str
    label: str
    steps: np.ndarray                      # (num_eval_points,)
    curves: np.ndarray                     # (num_seeds, num_eval_points)
    seeds: list[int] = field(default_factory=list)
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    summaries: list[dict] = field(default_factory=list)

    @property
    def final(self) -> np.ndarray:
        """Per-seed final performance (mean of the last 3 eval points).

        A single last evaluation point is close to meaningless on a DQN curve;
        averaging a short tail is the cheapest way to stop reading noise.
        """
        tail = self.curves[:, -3:] if self.curves.shape[1] >= 3 else self.curves
        return tail.mean(axis=1)

    @property
    def auc(self) -> np.ndarray:
        """Per-seed area under the learning curve -- a sample-efficiency proxy.

        Two arms can finish at the same score with very different costs along
        the way, and for an exploration study the path matters as much as the
        destination.
        """
        return self.curves.mean(axis=1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_run(run_dir: Path) -> dict | None:
    """Load one run directory. Returns None if the run did not complete."""
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        return None
    summary = json.loads(result_path.read_text())
    evals = _read_csv(run_dir / "eval.csv")
    if not evals:
        return None
    steps = np.array([_to_float(r["step"]) for r in evals])
    returns = np.array([_to_float(r["eval_return_mean"]) for r in evals])
    diagnostics = {}
    for row in _read_csv(run_dir / "diagnostics.csv"):
        for key, value in row.items():
            diagnostics.setdefault(key, []).append(_to_float(value))
    return {
        "summary": summary,
        "steps": steps,
        "returns": returns,
        "diagnostics": {k: np.array(v) for k, v in diagnostics.items()},
    }


def load_sweep(root: Path) -> list[ArmResults]:
    """Load every completed run under ``root`` into per-arm bundles.

    Layout is ``<root>/<env>/<label>/seed<k>/``. Seeds with different numbers of
    evaluation points (possible if a sweep mixed step budgets) are truncated to
    the shortest, so the stacked array is rectangular and every seed contributes
    to every plotted step.
    """
    root = Path(root)
    arms: list[ArmResults] = []
    if not root.is_dir():
        return arms

    for env_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
            runs = []
            for seed_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                run = load_run(seed_dir)
                if run is not None:
                    runs.append(run)
            if not runs:
                continue
            n = min(len(r["steps"]) for r in runs)
            steps = runs[0]["steps"][:n]
            curves = np.stack([r["returns"][:n] for r in runs])

            diag_keys = set.intersection(*(set(r["diagnostics"]) for r in runs))
            diagnostics = {}
            for key in sorted(diag_keys):
                m = min(len(r["diagnostics"][key]) for r in runs)
                if m == 0:
                    continue
                diagnostics[key] = np.stack(
                    [r["diagnostics"][key][:m] for r in runs]
                )

            arms.append(
                ArmResults(
                    env=runs[0]["summary"]["env"],
                    label=label_dir.name,
                    steps=steps,
                    curves=curves,
                    seeds=[int(r["summary"]["seed"]) for r in runs],
                    diagnostics=diagnostics,
                    summaries=[r["summary"] for r in runs],
                )
            )
    return arms


def group_by_env(arms: list[ArmResults]) -> dict[str, list[ArmResults]]:
    out: dict[str, list[ArmResults]] = {}
    for arm in arms:
        out.setdefault(arm.env, []).append(arm)
    return out
