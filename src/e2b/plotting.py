"""Figures for the report.

Design rules followed here (see the project's data-visualisation guidance):

* **Categorical hues assigned in fixed order, never cycled.** An arm keeps its
  colour across every figure, so the reader learns the mapping once. The order
  is a validated colourblind-safe sequence (worst adjacent CVD Delta E 9.1,
  normal-vision 19.6 on the light surface).
* **One y-axis per panel, never two.** Where two quantities of different scale
  need showing (return and entropy, epsilon and temperature), they get stacked
  panels sharing an x-axis rather than a dual axis.
* **Uncertainty is always shown.** A learning curve without a seed band invites
  reading noise as signal, which at 3-5 seeds is the single easiest way to
  reach a wrong conclusion in deep RL.
* **Direct labels plus a legend.** Three palette slots sit below 3:1 contrast on
  a light surface, so identity never rests on colour alone; the report also
  carries the numbers as tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from e2b.analysis import curve_interval, iqm  # noqa: E402
from e2b.results import ArmResults  # noqa: E402

# Validated categorical palette, light surface. Fixed slot order.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Panel titles are off by default: in the paper and the report every figure is
# introduced by a caption, and a title above the axes then says the same thing
# twice. `scripts/make_plots.py --titles` puts them back for standalone viewing.
SHOW_TITLES = False

# Legends sit bottom-right on an opaque white box, so they stay readable even
# where a curve passes underneath.
LEGEND_STYLE = dict(
    frameon=True, facecolor="white", framealpha=0.92, edgecolor="none",
    fontsize=8.5, labelcolor=INK_SECONDARY,
)

# Fraction of the axes width occupied by data in plot_learning_curves, which
# reserves a right-hand band for direct end-labels via ax.margins(x=X_MARGIN).
# The legend is anchored to the right edge of the DATA, not of the axes, so it
# never lands on top of an end-label.
X_MARGIN = 0.26
DATA_RIGHT = (1.0 + X_MARGIN) / (1.0 + 2.0 * X_MARGIN)

# A stable arm -> slot mapping, so a variant is the same colour in every figure
# even when a sweep contains a different subset of arms.
ARM_ORDER = [
    "eps_greedy", "eps_greedy_single", "eps_greedy_ensemble",
    "boltzmann",
    "eps_boltzmann", "eps_boltzmann_ensemble", "scaling_running",
    "mixture_anneal",
    "topk_boltzmann",
    "topk3_boltzmann",
    "anneal_k",
    "topp_boltzmann",
    "gated_ensemble", "gated_td_error",
    "scaling_none", "scaling_per_state",
]

PRETTY = {
    "eps_greedy": "ε-greedy",
    "eps_greedy_single": "ε-greedy (1 head)",
    "eps_greedy_ensemble": "ε-greedy (ensemble)",
    "boltzmann": "Boltzmann",
    "eps_boltzmann": "ε→Boltzmann (τ path)",
    "eps_boltzmann_ensemble": "ε→Boltzmann (ensemble)",
    "mixture_anneal": "ε-greedy⊕Boltzmann mix",
    "topk_boltzmann": "top-2 Boltzmann",
    "topk3_boltzmann": "top-3 Boltzmann",
    "anneal_k": "ε→Boltzmann (k path)",
    "topp_boltzmann": "top-p Boltzmann",
    "gated_ensemble": "uncertainty-gated (ensemble)",
    "gated_td_error": "uncertainty-gated (TD error)",
    "scaling_running": "q_scaling: running",
    "scaling_none": "q_scaling: none",
    "scaling_per_state": "q_scaling: per_state",
}


def pretty(label: str) -> str:
    return PRETTY.get(label, label.replace("_", " "))


def colour_for(label: str, present: list[str]) -> str:
    """Fixed-order hue assignment.

    Falls back to position within the sweep for labels not in ``ARM_ORDER``,
    but never cycles past the palette: a 9th arm would repeat a hue, so callers
    facet instead.
    """
    known = [a for a in ARM_ORDER if a in present]
    if label in known:
        return PALETTE[known.index(label) % len(PALETTE)]
    rest = [p for p in present if p not in known]
    return PALETTE[(len(known) + rest.index(label)) % len(PALETTE)]


def _style_axes(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if title and SHOW_TITLES:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)


def _thousands(ax) -> None:
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"
        )
    )


def _bottom_right_legend(fig, ax, ncol_max: int = 4) -> None:
    """Legend below the panels, flush right -- outside the data area.

    Inside the axes a bottom-right legend lands on the curves in these stacked
    figures: on the Q-scaling ablation it covered exactly the end-of-training
    entropy levels the figure exists to show. Space for it is reserved from the
    figure instead, so it can never occlude anything.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        fig.tight_layout()
        return
    ncol = min(ncol_max, len(handles))
    rows = -(-len(handles) // ncol)
    reserve = 0.042 * rows + 0.015
    fig.tight_layout(rect=(0.0, reserve, 1.0, 1.0))
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.995, 0.006),
               ncol=ncol, columnspacing=1.6, **LEGEND_STYLE)


def _spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Push overlapping direct-label positions apart, preserving their order.

    Converging learning curves would otherwise stack their end-labels on top of
    each other -- exactly where the reader most needs to tell the arms apart.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for rank, idx in enumerate(order):
        if rank == 0:
            continue
        prev = placed[order[rank - 1]]
        if placed[idx] - prev < min_gap:
            placed[idx] = prev + min_gap
    return placed


def plot_learning_curves(
    arms: list[ArmResults], out_path: Path, title: str, ylabel: str = "Greedy eval return"
) -> Path:
    """IQM learning curves with 95% stratified-bootstrap bands."""
    present = [a.label for a in arms]
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=SURFACE)

    ends: list[tuple[float, str, str]] = []
    for arm in arms:
        colour = colour_for(arm.label, present)
        point, low, high = curve_interval(arm.curves, seed=0)
        ax.fill_between(arm.steps, low, high, color=colour, alpha=0.13, linewidth=0)
        ax.plot(arm.steps, point, color=colour, linewidth=2.0,
                label=f"{pretty(arm.label)} (n={arm.curves.shape[0]})", zorder=3)
        ends.append((float(point[-1]), pretty(arm.label), colour))

    _style_axes(ax, "Environment steps", ylabel, title)
    _thousands(ax)
    ax.margins(x=X_MARGIN)

    # Direct end-labels: identity must never rest on colour alone. Placed after
    # the axes are scaled so the minimum gap is in real data units.
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    positions = _spread_labels([e[0] for e in ends], min_gap=0.045 * span)
    x_end = max(arm.steps[-1] for arm in arms)
    for y, (_, label, colour) in zip(positions, ends):
        ax.annotate(label, xy=(x_end, y), xytext=(8, 0), textcoords="offset points",
                    color=colour, fontsize=8.5, va="center")

    ax.legend(loc="lower right", bbox_to_anchor=(DATA_RIGHT, 0.0),
              bbox_transform=ax.transAxes, ncol=2, columnspacing=1.2,
              **LEGEND_STYLE)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_final_performance(
    arms: list[ArmResults], out_path: Path, title: str, metric: str = "final"
) -> Path:
    """Dot plot of IQM final (or AUC) return with bootstrap CIs.

    A dot-with-interval rather than a bar chart: the quantity of interest is an
    estimate with uncertainty, and a bar implies a precision that 3-5 seeds do
    not support.
    """
    from e2b.analysis import bootstrap_interval

    present = [a.label for a in arms]
    values = {a.label: (a.final if metric == "final" else a.auc) for a in arms}
    intervals = {k: bootstrap_interval(v, statistic=iqm) for k, v in values.items()}
    order = sorted(arms, key=lambda a: intervals[a.label].point)

    fig, ax = plt.subplots(figsize=(8.5, 0.52 * len(arms) + 2.0), facecolor=SURFACE)
    for row, arm in enumerate(order):
        colour = colour_for(arm.label, present)
        ci = intervals[arm.label]
        ax.plot([ci.low, ci.high], [row, row], color=colour, linewidth=2.0,
                solid_capstyle="round", zorder=2)
        ax.plot([ci.point], [row], "o", color=colour, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        # Per-seed points, so the reader sees the actual spread behind the IQM.
        ax.plot(values[arm.label], [row] * len(values[arm.label]), "o",
                color=colour, markersize=4, alpha=0.45, zorder=1)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([pretty(a.label) for a in order], fontsize=9.5, color=INK)
    label = "Final greedy eval return" if metric == "final" else "Mean return over training (AUC)"
    _style_axes(ax, f"{label}  ·  IQM with 95% bootstrap CI", "", title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_exploration_diagnostics(
    arms: list[ArmResults], out_path: Path, title: str
) -> Path:
    """Behaviour-policy entropy and non-greedy action mass over training.

    These two are the only exploration measures comparable *across* strategies.
    An epsilon of 0.05 and a temperature of 0.3 are not "the same amount" of
    exploration, and how much exploration a temperature buys drifts as the
    Q-scale grows -- so the knobs cannot be compared directly, but their effect
    on the action distribution can.

    Stacked panels sharing an x-axis, never a dual y-axis.
    """
    present = [a.label for a in arms]
    panels = [
        ("entropy", "Policy entropy (nats)"),
        ("non_greedy", "P(action ≠ argmax Q)"),
    ]
    available = [(k, lbl) for k, lbl in panels if any(k in a.diagnostics for a in arms)]
    if not available:
        return out_path

    fig, axes = plt.subplots(
        len(available), 1, figsize=(9, 3.0 * len(available) + 0.8),
        sharex=True, facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes)

    for ax, (key, ylabel) in zip(axes, available):
        for arm in arms:
            if key not in arm.diagnostics:
                continue
            data = arm.diagnostics[key]
            colour = colour_for(arm.label, present)
            steps = np.linspace(0, arm.steps[-1], data.shape[1])
            point = np.array([iqm(data[:, t]) for t in range(data.shape[1])])
            ax.plot(steps, point, color=colour, linewidth=2.0,
                    label=pretty(arm.label), zorder=3)
        _style_axes(ax, "", ylabel)
        _thousands(ax)

    if SHOW_TITLES:
        axes[0].set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    axes[-1].set_xlabel("Environment steps", color=INK_SECONDARY, fontsize=10)
    _bottom_right_legend(fig, axes[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_knob_traces(arms: list[ArmResults], out_path: Path, title: str) -> Path:
    """The exploration knobs actually in force: epsilon, temperature, support size.

    One panel per knob (never a dual axis); temperature on a log scale because
    it is annealed geometrically across orders of magnitude.

    This figure is what separates "the schedule moved" from "the behaviour
    changed" -- with Q-scale drift the two are not the same thing.
    """
    present = [a.label for a in arms]
    panels = [("eps", "ε (uniform floor)", False),
              ("temperature", "τ (temperature)", True),
              ("top_k", "k (support size)", False)]
    available = [p for p in panels if any(p[0] in a.diagnostics for a in arms)]
    if not available:
        return out_path

    fig, axes = plt.subplots(
        len(available), 1, figsize=(9, 2.6 * len(available) + 0.8),
        sharex=True, facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes)

    for ax, (key, ylabel, log) in zip(axes, available):
        for arm in arms:
            if key not in arm.diagnostics:
                continue
            data = arm.diagnostics[key]
            colour = colour_for(arm.label, present)
            steps = np.linspace(0, arm.steps[-1], data.shape[1])
            ax.plot(steps, np.nanmedian(data, axis=0), color=colour,
                    linewidth=2.0, label=pretty(arm.label), zorder=3)
        if log:
            ax.set_yscale("log")
        _style_axes(ax, "", ylabel)
        _thousands(ax)

    if SHOW_TITLES:
        axes[0].set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    axes[-1].set_xlabel("Environment steps", color=INK_SECONDARY, fontsize=10)
    _bottom_right_legend(fig, axes[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_confidence_trace(arms: list[ArmResults], out_path: Path, title: str) -> Path:
    """Measured Q-confidence and the support size it induces, over training.

    The extension's central claim is that the handover should be *driven by the
    agent's own uncertainty*. This figure shows whether the measured signal
    actually moved, and whether it moved the policy.
    """
    gated = [a for a in arms if "confidence" in a.diagnostics]
    if not gated:
        return out_path
    present = [a.label for a in arms]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.0), sharex=True, facecolor=SURFACE)
    for arm in gated:
        colour = colour_for(arm.label, present)
        conf = arm.diagnostics["confidence"]
        steps = np.linspace(0, arm.steps[-1], conf.shape[1])
        point, low, high = curve_interval(conf, resamples=500, seed=0)
        axes[0].fill_between(steps, low, high, color=colour, alpha=0.13, linewidth=0)
        axes[0].plot(steps, point, color=colour, linewidth=2.0,
                     label=pretty(arm.label), zorder=3)
        if "top_k" in arm.diagnostics:
            k = arm.diagnostics["top_k"]
            axes[1].plot(np.linspace(0, arm.steps[-1], k.shape[1]),
                         np.nanmedian(k, axis=0), color=colour, linewidth=2.0,
                         label=pretty(arm.label), zorder=3)

    _style_axes(axes[0], "", "Measured confidence  c ∈ [0,1]", title)
    _style_axes(axes[1], "Environment steps", "Induced support size k")
    for ax in axes:
        _thousands(ax)
    _bottom_right_legend(fig, axes[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
