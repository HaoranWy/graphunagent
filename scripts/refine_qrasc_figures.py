#!/usr/bin/env python3
"""Render the refined Q-RASC method and motivation figures.

The motivation panels read only the frozen, traceable examples recorded in
``figures/q-rasc-motivation-sidecar.json``.  Both figures are first emitted as
editable SVG and vector PDF sources, then converted to outlined-font PDFs for
anonymous submission compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qrasc-refined-matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1784304000")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures/q-rasc-motivation-sidecar.json"
SOURCE_DIR = ROOT / "figures/refined-qrasc"
METHOD_TARGET = ROOT / "AuthorKit27/Figures/method_overivew.pdf"
MOTIVATION_TARGET = ROOT / "AuthorKit27/Figures/motivation.pdf"
MANIFEST_TARGET = SOURCE_DIR / "figure-manifest.json"

INK = "#18252B"
MUTED = "#62727A"
LIGHT = "#D8E0E3"
SOFT = "#F6F8F8"
CORAL = "#D85C50"
CORAL_DARK = "#B8473D"
CORAL_PALE = "#FCF1EF"
TEAL = "#087F79"
TEAL_DARK = "#056863"
TEAL_PALE = "#EDF7F6"
GOLD = "#C38A18"
GOLD_DARK = "#96680E"
GOLD_PALE = "#FFF8E9"
SLATE = "#95A6AE"
SLATE_DARK = "#647780"
WHITE = "#FFFFFF"


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Arial", "DejaVu Sans"],
            "font.monospace": ["Nimbus Mono PS", "DejaVu Sans Mono"],
            "mathtext.fontset": "stixsans",
            "font.size": 7.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_examples() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    with SOURCE.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    local = payload["examples"]["local_quality"]
    structure = payload["examples"]["task_structure"]

    expected_positions = [71, 161, 164, 218, 265]
    observed_positions = sorted(int(item["token_index"]) for item in local["weakest_tokens"])
    if observed_positions != expected_positions:
        raise RuntimeError(f"frozen weakest-token positions drifted: {observed_positions}")
    if int(local["trace_token_count"]) != 1738 or int(local["weakest_k"]) != 5:
        raise RuntimeError("frozen local-quality example no longer matches the figure claim")
    if not math.isclose(float(local["full_trace_confidence"]), 0.7397684380590452):
        raise RuntimeError("frozen full-trace confidence drifted")
    if not math.isclose(float(local["weakest_5_confidence"]), 0.005577968668098307):
        raise RuntimeError("frozen weakest-5 confidence drifted")
    if structure["public_probe"]["call"] != "rolling_max([])":
        raise RuntimeError("frozen public probe drifted")
    if structure["anchor_program"]["excerpt"] != [
        "result = [numbers[0]]",
        "for num in numbers[1:]:",
        "    ...",
    ]:
        raise RuntimeError("frozen anchor-program excerpt drifted")
    if structure["peer_program"]["excerpt"] != [
        "rolling_max_list = []",
        "for num in numbers:",
        "    ...",
    ]:
        raise RuntimeError("frozen peer-program excerpt drifted")
    return local, structure


def _canvas(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=figsize, facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str = WHITE,
    edge: str = LIGHT,
    accent: str | None = None,
    radius: float = 0.010,
    linewidth: float = 0.55,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(patch)
    if accent:
        ax.add_patch(
            Rectangle(
                (x, y + h - 0.010),
                w,
                0.010,
                facecolor=accent,
                edgecolor="none",
                zorder=3,
            )
        )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = SLATE_DARK,
    linewidth: float = 0.75,
    dashed: bool = False,
    connectionstyle: str = "arc3,rad=0",
    zorder: float = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=6.5,
            linewidth=linewidth,
            linestyle=(0, (2.4, 2.2)) if dashed else "-",
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=zorder,
        )
    )


def _orthogonal_arrow(
    ax: plt.Axes,
    points: Sequence[tuple[float, float]],
    *,
    color: str = SLATE_DARK,
    linewidth: float = 0.75,
    zorder: float = 1,
) -> None:
    """Draw a grid-aligned connector with the arrowhead on its final segment."""
    if len(points) < 3:
        raise ValueError("an orthogonal connector needs at least three points")
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
        if not (math.isclose(x0, x1) or math.isclose(y0, y1)):
            raise ValueError("orthogonal connector segments must be axis-aligned")
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder,
        )
    final_start, final_end = points[-2], points[-1]
    if not (
        math.isclose(final_start[0], final_end[0])
        or math.isclose(final_start[1], final_end[1])
    ):
        raise ValueError("orthogonal connector segments must be axis-aligned")
    _arrow(
        ax,
        final_start,
        final_end,
        color=color,
        linewidth=linewidth,
        zorder=zorder,
    )


def _section_label(
    ax: plt.Axes,
    x: float,
    y: float,
    number: str,
    label: str,
    color: str,
    *,
    size: float = 5.8,
) -> None:
    ax.add_patch(Rectangle((x, y - 0.008), 0.0045, 0.036, facecolor=color, edgecolor="none", zorder=4))
    ax.text(
        x + 0.011,
        y + 0.010,
        f"{number}  {label}",
        color=color,
        fontsize=size,
        fontweight="bold",
        va="center",
        zorder=5,
    )


def _matrix(
    ax: plt.Axes,
    x: float,
    y: float,
    size: float,
    values: Sequence[Sequence[float]],
    color: str,
    *,
    gap: float = 0.0014,
) -> None:
    array = np.asarray(values, dtype=float)
    rows, cols = array.shape
    cell_w = (size - gap * (cols - 1)) / cols
    cell_h = (size - gap * (rows - 1)) / rows
    rgb = mpl.colors.to_rgb(color)
    for row in range(rows):
        for col in range(cols):
            value = float(np.clip(array[row, col], 0, 1))
            mixed = tuple((1 - value * 0.88) + value * 0.88 * channel for channel in rgb)
            ax.add_patch(
                Rectangle(
                    (x + col * (cell_w + gap), y + (rows - row - 1) * (cell_h + gap)),
                    cell_w,
                    cell_h,
                    facecolor=mixed,
                    edgecolor=WHITE,
                    linewidth=0.20,
                    zorder=5,
                )
            )


def _tokens(ax: plt.Axes, x: float, y: float, *, anchor: bool, n: int = 5) -> None:
    for index in range(n):
        fill = CORAL_PALE if index != 2 else (CORAL if anchor else "#E4E9EB")
        edge = CORAL if index != 2 else (CORAL_DARK if anchor else SLATE)
        ax.add_patch(
            FancyBboxPatch(
                (x + index * 0.0130, y),
                0.0102,
                0.026,
                boxstyle="round,pad=0,rounding_size=0.0025",
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.38,
                zorder=4,
            )
        )


def _weight_bars(
    ax: plt.Axes,
    x: float,
    y: float,
    values: Sequence[float],
    colors: Sequence[str],
    *,
    width: float = 0.045,
    height: float = 0.072,
) -> None:
    gap = width * 0.08
    bar_w = (width - gap * (len(values) - 1)) / len(values)
    for index, (value, color) in enumerate(zip(values, colors)):
        bar_h = max(0.008, height * float(value))
        ax.add_patch(
            FancyBboxPatch(
                (x + index * (bar_w + gap), y),
                bar_w,
                bar_h,
                boxstyle="round,pad=0,rounding_size=0.002",
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
        )


def _spectral_bars(
    ax: plt.Axes,
    x: float,
    y: float,
    values: Sequence[float],
    colors: Sequence[str],
    *,
    width: float = 0.080,
    height: float = 0.085,
) -> None:
    ax.plot([x - 0.004, x + width + 0.004], [y, y], color=INK, linewidth=0.45, zorder=5)
    gap = width * 0.075
    bar_w = (width - gap * (len(values) - 1)) / len(values)
    for index, (value, color) in enumerate(zip(values, colors)):
        ax.add_patch(
            FancyBboxPatch(
                (x + index * (bar_w + gap), y),
                bar_w,
                height * value,
                boxstyle="round,pad=0,rounding_size=0.002",
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
        )


def _draw_method() -> plt.Figure:
    # Match the AAAI 7.0-inch text block so TeX can place the PDF at 100%.
    fig, ax = _canvas((7.00, 2.60))

    # Column headings and guide rules.
    headings = (
        (0.035, 0.924, "SAMPLED RESPONSES", INK),
        (0.202, 0.924, "BUILD TWO SIGNALS", INK),
        (0.432, 0.924, "Q-RASC", GOLD_DARK),
        (0.807, 0.924, "OUTPUTS", INK),
    )
    for x, y, label, color in headings:
        ax.text(x, y, label, color=color, fontsize=6.15, fontweight="bold", va="center")
    ax.text(0.432, 0.875, "Response-Anchored Spectral Risk", color=INK, fontsize=9.6, fontweight="bold", va="center")
    ax.plot([0.025, 0.975], [0.848, 0.848], color=LIGHT, linewidth=0.55, zorder=0)

    # Sampled paths.
    path_rows = (
        (0.710, r"$y_0$", "anchor", True),
        (0.535, r"$y_1$", "peer", False),
        (0.245, r"$y_{N-1}$", "peer", False),
    )
    for y, symbol, role, anchor in path_rows:
        ax.add_patch(
            Rectangle(
                (0.040, y - 0.010),
                0.006,
                0.046,
                facecolor=CORAL if anchor else SLATE,
                edgecolor="none",
                zorder=4,
            )
        )
        ax.text(0.053, y + 0.014, symbol, fontsize=8.0, color=INK, va="center")
        ax.text(0.126, y + 0.014, role, fontsize=6.1, color=CORAL_DARK if anchor else MUTED, ha="right", va="center")
        _tokens(ax, 0.053, y - 0.048, anchor=anchor)
    ax.text(0.083, 0.395, r"$\vdots$", color=MUTED, fontsize=10.5, ha="center", va="center")

    # Signal cards.
    _panel(ax, 0.198, 0.510, 0.190, 0.306, face=CORAL_PALE, edge="#E8B7B0", accent=CORAL)
    _section_label(ax, 0.211, 0.775, "1A", "LOCAL QUALITY", CORAL_DARK)
    ax.text(0.211, 0.712, "weakest-5 chosen tokens", color=INK, fontsize=6.4, va="center")
    _tokens(ax, 0.211, 0.648, anchor=True)
    ax.text(0.285, 0.661, r"$\longrightarrow\; c_i$", color=CORAL_DARK, fontsize=7.1, va="center")
    ax.text(0.211, 0.600, r"validity filter $v_i$", color=MUTED, fontsize=6.25, va="center")
    ax.plot([0.211, 0.371], [0.569, 0.569], color="#EDC7C2", linewidth=0.45)
    ax.text(0.291, 0.535, r"$q_i=c_i v_i$", color=CORAL_DARK, fontsize=8.1, fontweight="bold", ha="center", va="center")

    _panel(ax, 0.198, 0.105, 0.190, 0.330, face=TEAL_PALE, edge="#ADD8D4", accent=TEAL)
    _section_label(ax, 0.211, 0.394, "1B", "TASK GEOMETRY", TEAL_DARK)
    for y, text_value in ((0.333, "Math equivalence"), (0.277, "Logic labels"), (0.221, "Code behavior")):
        ax.add_patch(Rectangle((0.213, y - 0.010), 0.009, 0.020, facecolor=TEAL, edgecolor="none"))
        ax.text(0.229, y, text_value, color=INK, fontsize=6.25, va="center")
    _matrix(
        ax,
        0.324,
        0.244,
        0.051,
        [[1.0, 0.23, 0.72], [0.23, 1.0, 0.38], [0.72, 0.38, 1.0]],
        TEAL,
    )
    ax.text(0.349, 0.201, r"$K=\Phi\Phi^\mathsf{T}$", color=TEAL_DARK, fontsize=7.2, ha="center", va="center")
    ax.text(0.211, 0.146, "semantic / behavioral structure", color=MUTED, fontsize=5.75, va="center")

    # Q-RASC core.
    _panel(ax, 0.428, 0.645, 0.326, 0.171, face=GOLD_PALE, edge="#E6CC91", accent=GOLD)
    _section_label(ax, 0.442, 0.778, "2", "NORMALIZE QUALITY MASS", GOLD_DARK)
    weights = (0.82, 0.44, 0.64, 0.25, 0.52)
    colors = (GOLD, "#8BC9C4", "#69B7B1", "#B5DCD8", "#8BC9C4")
    _weight_bars(ax, 0.443, 0.675, weights, colors, width=0.073, height=0.078)
    ax.text(0.529, 0.708, r"$Q=q_0+Q_-$", color=INK, fontsize=7.0, va="center")
    ax.text(0.627, 0.708, r"$w_i=q_i/Q$", color=INK, fontsize=6.75, va="center")
    ax.text(0.627, 0.671, r"$\pi_i=q_i/Q_-$", color=INK, fontsize=6.75, va="center")

    _panel(ax, 0.428, 0.355, 0.326, 0.245, face=TEAL_PALE, edge="#AED7D3", accent=TEAL)
    _section_label(ax, 0.442, 0.560, "3", "WEIGHTED GRAM MATRICES", TEAL_DARK)
    rows = (
        (0.477, "All", GOLD, r"$D(w,K)$", [[1, .25, .7], [.25, 1, .35], [.7, .35, 1]]),
        (0.393, "Peers", TEAL, r"$D(\pi,K_{-0,-0})$", [[1, .35], [.35, 1]]),
    )
    for y, label, color, formula, matrix_values in rows:
        ax.text(0.443, y, label, color=color, fontsize=6.5, fontweight="bold", va="center")
        _weight_bars(ax, 0.485, y - 0.026, (0.85, 0.50, 0.68), (color, "#9FD0CC", "#C8E5E2"), width=0.034, height=0.052)
        ax.text(0.527, y, r"$\times$", color=MUTED, fontsize=6.6, ha="center", va="center")
        _matrix(ax, 0.539, y - 0.026, 0.052, matrix_values, TEAL if label == "Peers" else GOLD)
        ax.text(0.601, y, r"$\times$", color=MUTED, fontsize=6.6, ha="center", va="center")
        _weight_bars(ax, 0.611, y - 0.026, (0.85, 0.50, 0.68), (color, "#9FD0CC", "#C8E5E2"), width=0.034, height=0.052)
        ax.text(0.654, y, r"$\longrightarrow$", color=MUTED, fontsize=6.8, va="center")
        ax.text(0.713, y, formula, color=INK, fontsize=6.8, ha="center", va="center")

    _panel(ax, 0.428, 0.080, 0.326, 0.225, face=SOFT, edge="#CBD5D9", accent=INK)
    _section_label(ax, 0.442, 0.265, "4", "PAIRED SPECTRAL CONTRAST", INK)
    _spectral_bars(ax, 0.448, 0.139, (0.90, 0.58, 0.34, 0.18), (GOLD, "#E6C36A", "#BBD6D3", "#DCE8E7"), width=0.070, height=0.080)
    ax.text(0.483, 0.116, r"$S_{\mathrm{mix}}$", color=GOLD_DARK, fontsize=6.6, ha="center", va="top")
    ax.text(0.541, 0.173, r"$-$", color=MUTED, fontsize=8.0, ha="center", va="center")
    _spectral_bars(ax, 0.563, 0.139, (0.78, 0.48, 0.29, 0.13), (TEAL, "#55AAA4", "#A9D3D0", "#DCE8E7"), width=0.070, height=0.080)
    ax.text(0.598, 0.116, r"$\beta S_{\mathrm{peer}}$", color=TEAL_DARK, fontsize=6.6, ha="center", va="top")
    ax.text(0.648, 0.173, r"$\longrightarrow$", color=MUTED, fontsize=7.0, ha="center", va="center")
    ax.text(0.701, 0.173, r"$\chi_0$", color=INK, fontsize=9.0, fontweight="bold", ha="center", va="center")
    ax.text(0.701, 0.125, r"$\max(0,\,S_{mix}-\beta S_{peer})$", color=MUTED, fontsize=5.55, ha="center", va="center")

    # Outputs: main risk and a deliberately separate auxiliary branch.
    _panel(ax, 0.802, 0.445, 0.173, 0.371, face=GOLD_PALE, edge="#E3C476", accent=GOLD)
    _section_label(ax, 0.815, 0.777, "5", "ANCHOR RISK", GOLD_DARK)
    ax.scatter([0.834], [0.691], s=115, facecolor=GOLD, edgecolor="none", zorder=5)
    ax.text(0.834, 0.691, r"$y_0$", color=WHITE, fontsize=6.2, fontweight="bold", ha="center", va="center", zorder=6)
    ax.plot([0.859, 0.894], [0.691, 0.691], color=GOLD, linewidth=0.8, zorder=4)
    for dx, dy in ((0.0, 0.0), (0.022, 0.025), (0.024, -0.026)):
        ax.scatter([0.915 + dx], [0.691 + dy], s=30, facecolor="#D9EFED", edgecolor=TEAL, linewidth=0.55, zorder=5)
    ax.text(0.834, 0.642, "anchor", color=MUTED, fontsize=5.75, ha="center")
    ax.text(0.929, 0.642, "peer mix", color=MUTED, fontsize=5.75, ha="center")
    ax.plot([0.816, 0.962], [0.616, 0.616], color="#E7D39F", linewidth=0.45)
    ax.text(0.888, 0.574, r"$\alpha=q_0/Q,\quad\beta=Q_-/Q$", color=INK, fontsize=6.4, ha="center", va="center")
    ax.text(0.888, 0.521, r"$R_{\mathrm{RASC}}=\mathrm{clip}_{[0,1]}\!\left(\chi_0/h_2(\alpha)\right)$", color=GOLD_DARK, fontsize=6.65, fontweight="bold", ha="center", va="center")
    ax.text(0.888, 0.474, "risk of the designated response", color=MUTED, fontsize=5.55, ha="center", va="center")

    _panel(ax, 0.802, 0.080, 0.173, 0.285, face=TEAL_PALE, edge="#AFD8D4", accent=TEAL)
    _section_label(ax, 0.815, 0.326, "AUX", "ANSWER SELECTION", TEAL_DARK, size=5.45)
    for index, (y, value, color) in enumerate(((0.260, 0.52, "#AED8D4"), (0.216, 0.88, TEAL), (0.172, 0.35, "#AED8D4"))):
        ax.text(0.817, y, f"{index}", color=MUTED, fontsize=5.8, va="center")
        ax.plot([0.834, 0.912], [y, y], color="#D5E9E7", linewidth=4.1, solid_capstyle="round", zorder=3)
        ax.plot([0.834, 0.834 + 0.078 * value], [y, y], color=color, linewidth=4.1, solid_capstyle="round", zorder=4)
    ax.text(0.887, 0.123, r"$i^*=\arg\max_i\,q_i(Kw)_i$", color=TEAL_DARK, fontsize=6.6, fontweight="bold", ha="center", va="center")
    ax.text(0.887, 0.099, "from q,K  |  risk unchanged", color=MUTED, fontsize=5.1, ha="center", va="center")

    # Reading path.
    _arrow(ax, (0.152, 0.663), (0.192, 0.663))
    _arrow(ax, (0.152, 0.270), (0.192, 0.270))
    _orthogonal_arrow(
        ax,
        ((0.391, 0.663), (0.409, 0.663), (0.409, 0.721), (0.423, 0.721)),
        color=CORAL_DARK,
    )
    _orthogonal_arrow(
        ax,
        ((0.391, 0.270), (0.409, 0.270), (0.409, 0.393), (0.423, 0.393)),
        color=TEAL_DARK,
    )
    _arrow(ax, (0.591, 0.641), (0.591, 0.607), color=GOLD_DARK)
    _arrow(ax, (0.591, 0.351), (0.591, 0.311), color=TEAL_DARK)
    _orthogonal_arrow(
        ax,
        ((0.756, 0.188), (0.779, 0.188), (0.779, 0.630), (0.798, 0.630)),
        color=GOLD_DARK,
    )

    return fig


def _confidence_track(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    value: float,
    color: str,
    *,
    label: str,
    value_text: str,
) -> None:
    ax.text(x, y, label, color=INK, fontsize=6.2, va="center")
    start = x + 0.33 * width
    track_width = 0.55 * width
    ax.plot([start, start + track_width], [y, y], color="#E4E9EB", linewidth=3.4, solid_capstyle="round", zorder=2)
    ax.plot([start, start + track_width * value], [y, y], color=color, linewidth=3.4, solid_capstyle="round", zorder=3)
    ax.scatter([start + track_width * value], [y], s=10, facecolor=color, edgecolor=WHITE, linewidth=0.4, zorder=4)
    ax.text(x + width, y, value_text, color=color, fontsize=6.5, fontweight="bold", ha="right", va="center")


def _draw_motivation(local: Mapping[str, Any], structure: Mapping[str, Any]) -> plt.Figure:
    # Match one AAAI column: (7.0 - 0.375) / 2 = 3.3125 inches.
    fig, ax = _canvas((3.3125, 4.18))

    # A. Local quality from the frozen GSM8K trace.
    _section_label(ax, 0.048, 0.957, "A", "LOCAL QUALITY SIGNAL", CORAL_DARK, size=6.0)
    ax.text(0.048, 0.913, "Sequence means can hide local failures", color=INK, fontsize=8.7, fontweight="bold", va="center")
    ax.text(0.048, 0.880, "Real incorrect GSM8K response  |  0-300 of 1,738 tokens", color=MUTED, fontsize=5.85, va="center")

    plot_ax = fig.add_axes([0.145, 0.688, 0.807, 0.162])
    profile = np.asarray(local["trace_nll_profile"], dtype=float)
    mask = profile[:, 0] <= 300
    positions = profile[mask, 0]
    confidences = np.exp(-profile[mask, 1])
    plot_ax.plot(positions, confidences, color=SLATE, linewidth=0.52, zorder=2)
    weakest = sorted(local["weakest_tokens"], key=lambda item: int(item["token_index"]))
    weak_x = [int(item["token_index"]) for item in weakest]
    weak_y = [float(item["confidence"]) for item in weakest]
    plot_ax.scatter(weak_x, weak_y, s=9.0, facecolor=CORAL, edgecolor=WHITE, linewidth=0.38, zorder=5)
    for x_value in weak_x:
        plot_ax.vlines(x_value, 0.00155, 0.00205, color=CORAL, linewidth=0.65, zorder=4)
    plot_ax.set_xlim(-12, 308)
    plot_ax.set_yscale("log")
    plot_ax.set_ylim(0.00145, 1.08)
    plot_ax.set_xticks((0, 100, 200, 300))
    plot_ax.set_yticks((0.01, 0.1, 1.0), (".01", ".1", "1"))
    plot_ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    plot_ax.grid(axis="y", color=LIGHT, linewidth=0.42)
    plot_ax.spines["top"].set_visible(False)
    plot_ax.spines["right"].set_visible(False)
    plot_ax.spines["left"].set_visible(False)
    plot_ax.spines["bottom"].set_color(LIGHT)
    plot_ax.tick_params(axis="both", colors=MUTED, labelsize=5.8, length=1.8, width=0.45, pad=1.5)
    plot_ax.set_ylabel("token confidence\n(log scale)", color=MUTED, fontsize=5.5, labelpad=2)
    plot_ax.set_xlabel("token position", color=MUTED, fontsize=5.6, labelpad=1.5)
    plot_ax.scatter([0.66], [1.125], s=8, color=CORAL, transform=plot_ax.transAxes, clip_on=False, zorder=6)
    plot_ax.text(0.68, 1.125, "weakest-5 tokens", color=CORAL_DARK, fontsize=5.55, transform=plot_ax.transAxes, va="center")

    ax.text(0.048, 0.646, "AGGREGATED CONFIDENCE", color=MUTED, fontsize=5.25, fontweight="bold", va="center")
    _confidence_track(
        ax,
        0.048,
        0.604,
        0.904,
        float(local["full_trace_confidence"]),
        SLATE_DARK,
        label="Full-trace mean",
        value_text=f"{float(local['full_trace_confidence']):.3f}",
    )
    _confidence_track(
        ax,
        0.048,
        0.559,
        0.904,
        float(local["weakest_5_confidence"]),
        CORAL,
        label="Weakest-5 bottleneck",
        value_text=f"{float(local['weakest_5_confidence']):.3f}",
    )
    ax.text(0.048, 0.521, "Same response; the local bottleneck reveals the hidden fragility.", color=CORAL_DARK, fontsize=5.65, va="center")

    # A quiet divider separates the two independent motivations.
    ax.plot([0.048, 0.952], [0.492, 0.492], color=LIGHT, linewidth=0.65)

    # B. Structure from the frozen HumanEvalPlus response pair.
    _section_label(ax, 0.048, 0.459, "B", "PROGRAM STRUCTURE SIGNAL", TEAL_DARK, size=6.0)
    ax.text(0.048, 0.415, "Similar confidence can mask different execution", color=INK, fontsize=8.45, fontweight="bold", va="center")
    ax.text(0.048, 0.382, "Real HumanEvalPlus pair  |  public probe: rolling_max([])", color=MUTED, fontsize=5.75, va="center")

    anchor_conf = float(structure["anchor_confidence"])
    peer_conf = float(structure["peer_confidence"])
    gap = float(structure["anchor_peer_confidence_gap"])
    scale_min, scale_max = 0.270, 0.300
    scale_x0, scale_x1, scale_y = 0.275, 0.724, 0.343
    ax.plot([scale_x0, scale_x1], [scale_y, scale_y], color="#DDE5E7", linewidth=1.2, zorder=2)
    for value, color in ((anchor_conf, CORAL), (peer_conf, TEAL)):
        x_value = scale_x0 + (value - scale_min) / (scale_max - scale_min) * (scale_x1 - scale_x0)
        ax.scatter([x_value], [scale_y], s=18, facecolor=color, edgecolor=WHITE, linewidth=0.55, zorder=5)
    ax.text(0.048, scale_y, f"conf. {anchor_conf:.4f}", color=CORAL_DARK, fontsize=6.1, fontweight="bold", va="center")
    ax.text(0.952, scale_y, f"{peer_conf:.4f} conf.", color=TEAL_DARK, fontsize=6.1, fontweight="bold", ha="right", va="center")
    ax.text(0.500, 0.319, f"gap = {gap:.4f}", color=MUTED, fontsize=5.4, ha="center", va="center")

    card_y, card_h, card_w = 0.166, 0.133, 0.432
    cards = (
        (0.048, CORAL, CORAL_PALE, "PATH 0", structure["anchor_program"]["excerpt"]),
        (0.520, TEAL, TEAL_PALE, "PATH 1", structure["peer_program"]["excerpt"]),
    )
    for x, color, pale, title, lines in cards:
        _panel(ax, x, card_y, card_w, card_h, face=pale, edge=color, radius=0.008, linewidth=0.62)
        ax.add_patch(Rectangle((x, card_y), 0.006, card_h, facecolor=color, edgecolor="none", zorder=4))
        ax.text(x + 0.019, card_y + card_h - 0.028, title, color=color, fontsize=5.35, fontweight="bold", va="center", zorder=5)
        for index, line in enumerate(lines):
            ax.text(
                x + 0.019,
                card_y + card_h - 0.060 - index * 0.029,
                line,
                color=INK,
                fontsize=5.55,
                fontfamily="monospace",
                va="center",
                zorder=5,
            )

    # One shared probe branches to two observably different outcomes.
    probe_x0, probe_x1, probe_y = 0.386, 0.614, 0.128
    _panel(ax, probe_x0, 0.108, probe_x1 - probe_x0, 0.043, face=WHITE, edge=SLATE, radius=0.007, linewidth=0.52)
    ax.text(0.500, 0.129, "rolling_max([])", color=INK, fontsize=5.9, fontfamily="monospace", ha="center", va="center", zorder=5)
    _arrow(ax, (0.435, 0.108), (0.264, 0.077), color=CORAL_DARK, connectionstyle="arc3,rad=0.07")
    _arrow(ax, (0.565, 0.108), (0.736, 0.077), color=TEAL_DARK, connectionstyle="arc3,rad=-0.07")
    ax.text(0.264, 0.061, "ERROR  IndexError", color=CORAL_DARK, fontsize=6.4, fontweight="bold", ha="center", va="center")
    ax.text(0.736, 0.061, "OK  returns []", color=TEAL_DARK, fontsize=6.4, fontweight="bold", ha="center", va="center")
    ax.text(0.500, 0.021, "Token confidence is similar; executable structure is not.", color=INK, fontsize=5.65, ha="center", va="center")

    return fig


def _save_source(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = SOURCE_DIR / f"{stem}.pdf"
    svg_path = SOURCE_DIR / f"{stem}.svg"
    metadata = {
        "Title": stem.replace("-", " ").title(),
        "Author": "",
        "Subject": "Q-RASC publication figure",
        "Keywords": "Q-RASC",
        "Creator": "Matplotlib",
    }
    fig.savefig(pdf_path, format="pdf", bbox_inches=None, pad_inches=0, metadata=metadata)
    fig.savefig(svg_path, format="svg", bbox_inches=None, pad_inches=0, metadata={"Date": "2026-07-26"})
    plt.close(fig)
    return pdf_path, svg_path


def _outline_pdf(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".outlined.tmp.pdf")
    command = [
        "gs",
        "-q",
        "-dSAFER",
        "-dNOPAUSE",
        "-dBATCH",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        "-dAutoRotatePages=/None",
        "-dNoOutputFonts",
        f"-sOutputFile={temporary}",
        str(source),
    ]
    subprocess.run(command, check=True)
    temporary.replace(target)


def _write_manifest(
    local: Mapping[str, Any],
    structure: Mapping[str, Any],
    sources: Sequence[Path],
) -> None:
    manifest = {
        "schema_version": 1,
        "rendered_at": "2026-07-26",
        "generator": "scripts/refine_qrasc_figures.py",
        "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": _sha256(SOURCE)},
        "claims": {
            "local_quality": {
                "task": local["task"],
                "sample_id": local["sample_id"],
                "trace_token_count": local["trace_token_count"],
                "displayed_window": [0, 300],
                "weakest_token_positions": sorted(int(item["token_index"]) for item in local["weakest_tokens"]),
                "full_trace_confidence": local["full_trace_confidence"],
                "weakest_5_confidence": local["weakest_5_confidence"],
            },
            "program_structure": {
                "task": structure["task"],
                "sample_id": structure["sample_id"],
                "anchor_confidence": structure["anchor_confidence"],
                "peer_confidence": structure["peer_confidence"],
                "confidence_gap": structure["anchor_peer_confidence_gap"],
                "public_probe": structure["public_probe"],
                "anchor_excerpt": structure["anchor_program"]["excerpt"],
                "peer_excerpt": structure["peer_program"]["excerpt"],
            },
        },
        "editable_sources": {
            str(path.relative_to(ROOT)): _sha256(path) for path in sources
        },
        "submission_pdfs": {
            str(METHOD_TARGET.relative_to(ROOT)): _sha256(METHOD_TARGET),
            str(MOTIVATION_TARGET.relative_to(ROOT)): _sha256(MOTIVATION_TARGET),
        },
    }
    MANIFEST_TARGET.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    _configure_matplotlib()
    local, structure = _load_examples()
    method_pdf, method_svg = _save_source(_draw_method(), "method-overview")
    motivation_pdf, motivation_svg = _save_source(_draw_motivation(local, structure), "motivation")
    _outline_pdf(method_pdf, METHOD_TARGET)
    _outline_pdf(motivation_pdf, MOTIVATION_TARGET)
    _write_manifest(local, structure, (method_pdf, method_svg, motivation_pdf, motivation_svg))
    print(METHOD_TARGET)
    print(MOTIVATION_TARGET)
    print(MANIFEST_TARGET)


if __name__ == "__main__":
    main()
