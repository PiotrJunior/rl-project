#!/usr/bin/env python
"""Regenerate every figure and results table from run outputs.

    python scripts/make_plots.py --sweep reduced_gym

Reads ``results/runs/<sweep>/``, writes PNGs to ``report/figures/`` and a
markdown results table to ``results/summaries/<sweep>_table.md``.

The tables are not decoration: three of the palette's hues sit below 3:1
contrast on a light surface, so the figures alone would rest identity partly on
colour. The tables are the accessible view of the same numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2b.analysis import bootstrap_interval, iqm, probability_of_improvement  # noqa: E402
from e2b.plotting import (  # noqa: E402
    plot_confidence_trace,
    plot_exploration_diagnostics,
    plot_final_performance,
    plot_knob_traces,
    plot_learning_curves,
    pretty,
)
from e2b.results import group_by_env, load_sweep  # noqa: E402

TITLES = {
    "reduced_gym": "CartPole-v1 — exploration variants (reduced study)",
    "main_gym": "LunarLander-v3 — exploration variants",
    "ablation_scaling": "LunarLander-v3 — Q-value scale normalisation ablation",
    "uncertainty": "LunarLander-v3 — uncertainty-gated handover",
    "full_gym": "Full study",
}

BASELINE = {"reduced_gym": "eps_greedy", "main_gym": "eps_greedy",
            "ablation_scaling": "scaling_running", "uncertainty": "eps_greedy_ensemble",
            "full_gym": "eps_greedy"}


def results_table(arms, baseline_label: str | None) -> str:
    """Markdown table: IQM final return, IQM AUC, and P(improvement) vs baseline."""
    rows = []
    baseline = next((a for a in arms if a.label == baseline_label), None)
    for arm in sorted(arms, key=lambda a: -iqm(a.final)):
        final = bootstrap_interval(arm.final, statistic=iqm)
        auc = bootstrap_interval(arm.auc, statistic=iqm)
        if baseline is not None and arm.label != baseline.label:
            poi = probability_of_improvement(arm.final, baseline.final)
            poi_str = f"{poi.point:.2f} [{poi.low:.2f}, {poi.high:.2f}]"
        else:
            poi_str = "—" if baseline is None else "(baseline)"
        rows.append(
            f"| {pretty(arm.label)} | {final.point:.1f} "
            f"[{final.low:.1f}, {final.high:.1f}] | {auc.point:.1f} "
            f"[{auc.low:.1f}, {auc.high:.1f}] | {poi_str} | {len(arm.seeds)} |"
        )
    header = (
        "| Variant | Final return (IQM, 95% CI) | AUC (IQM, 95% CI) "
        f"| P(beats {pretty(baseline_label) if baseline_label else 'baseline'}) | seeds |\n"
        "|---|---|---|---|---|"
    )
    return "\n".join([header, *rows])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--fig-dir", default=str(ROOT / "report" / "figures"))
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root or ROOT / "results" / "runs" / args.sweep)
    arms = load_sweep(runs_root)
    if not arms:
        print(f"no completed runs under {runs_root}", file=sys.stderr)
        return 1

    fig_dir = Path(args.fig_dir)
    title = TITLES.get(args.sweep, args.sweep)
    baseline = BASELINE.get(args.sweep)
    written: list[Path] = []

    for env, env_arms in group_by_env(arms).items():
        tag = f"{args.sweep}_{env.replace('/', '_')}"
        written.append(plot_learning_curves(
            env_arms, fig_dir / f"{tag}_curves.png", f"{title} — learning curves"))
        written.append(plot_final_performance(
            env_arms, fig_dir / f"{tag}_final.png", f"{title} — final performance"))
        written.append(plot_exploration_diagnostics(
            env_arms, fig_dir / f"{tag}_exploration.png",
            f"{title} — behaviour-policy exploration"))
        written.append(plot_knob_traces(
            env_arms, fig_dir / f"{tag}_knobs.png", f"{title} — exploration knobs"))
        conf = plot_confidence_trace(
            env_arms, fig_dir / f"{tag}_confidence.png",
            f"{title} — measured Q-confidence")
        if conf.exists():
            written.append(conf)

    table_dir = ROOT / "results" / "summaries"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / f"{args.sweep}_table.md"
    sections = []
    for env, env_arms in group_by_env(arms).items():
        sections.append(f"### {env}\n\n{results_table(env_arms, baseline)}")
    table_path.write_text("\n\n".join(sections) + "\n")

    print(f"figures -> {fig_dir}")
    for path in sorted(set(written)):
        if path.exists():
            print(f"  {path.relative_to(ROOT)}")
    print(f"table   -> {table_path.relative_to(ROOT)}")
    print()
    print(table_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
