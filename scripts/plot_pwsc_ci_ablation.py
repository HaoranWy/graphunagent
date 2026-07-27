#!/usr/bin/env python3
"""Plot traceable PWSC-CI component evidence from the independent Llama run.

The figure intentionally excludes retired PWSC variants. It compares the
canonical PWSC-CI score with input-preserving non-PWSC controls from the same
sample CSVs: mean NLL, weakest-k NLL (canonical k=5), and task-kernel target
consistency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pwsc-ci-matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1784304000")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from diagnostics.analyzer import ScientificSignalAnalyzer  # noqa: E402


DEFAULT_FREEZE = (
    ROOT / "code/diagnostics/pwsc_ci_frozen/pwsc_ci_results_20260717.json"
)
TASKS = ("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO")
ENSEMBLE_SIZES = (1, 4, 8, 16)
PWSC_COLUMN = "pwsc_ci_risk"
METHODS = OrderedDict(
    (
        ("pwsc_ci", ("PWSC-CI", PWSC_COLUMN)),
        ("quality_only", ("Worst-k only (k=5)", "worst_5_nll")),
        (
            "structure_only",
            ("Task structure only", "target_consistency_risk"),
        ),
        ("mean_nll", ("Mean NLL", "mean_nll")),
    )
)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260732


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} hash mismatch for {path}: expected {expected}, got {observed}"
        )


def _load_task_rows(
    freeze: Mapping[str, Any], task: str
) -> tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, Any]]:
    scope = freeze["evidence_scope"]
    dataset = freeze["datasets"][task]
    provenance = freeze["provenance"]["manifests"][task]
    manifest_path = ROOT / provenance["manifest_path"]
    csv_path = ROOT / provenance["sample_csv_path"]
    _require_hash(manifest_path, provenance["manifest_sha256"], f"{task} manifest")
    _require_hash(csv_path, provenance["sample_csv_sha256"], f"{task} sample CSV")
    manifest = _load_json(manifest_path)

    expected_pairs = {
        "task": task,
        "seed": scope["seed"],
        "model_identifier": scope["model"],
        "model_revision": scope["model_root"],
        "input_sample_sha256": dataset["input_sha256"],
        "generation_cache_sha256": provenance["cache_sha256"],
        "csv_sha256": provenance["sample_csv_sha256"],
    }
    for key, expected in expected_pairs.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"{task} manifest {key} mismatch: expected {expected!r}, "
                f"got {manifest.get(key)!r}"
            )
    sampling = manifest.get("sampling", {})
    if sampling.get("ensemble_sizes") != list(ENSEMBLE_SIZES):
        raise RuntimeError(f"{task} manifest does not bind N={ENSEMBLE_SIZES}")
    if sampling.get("python_hash_seed") != scope["python_hash_seed"]:
        raise RuntimeError(f"{task} Python hash seed mismatch")
    if manifest.get("dataset_fingerprint", {}).get("files", [{}])[0].get(
        "sha256"
    ) != dataset["file_sha256"]:
        raise RuntimeError(f"{task} dataset fingerprint mismatch")

    grouped: Dict[int, list[Dict[str, str]]] = {size: [] for size in ENSEMBLE_SIZES}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            size = int(row["ensemble_size"])
            if size in grouped:
                grouped[size].append(row)

    expected_count = int(dataset["question_count"])
    output: Dict[int, Dict[str, np.ndarray]] = {}
    reference_ids: list[str] | None = None
    reference_labels: np.ndarray | None = None
    for size in ENSEMBLE_SIZES:
        rows = sorted(grouped[size], key=lambda row: row["sample_id"])
        if len(rows) != expected_count:
            raise RuntimeError(
                f"{task}/N={size} has {len(rows)} rows; expected {expected_count}"
            )
        sample_ids = [row["sample_id"] for row in rows]
        if len(set(sample_ids)) != expected_count:
            raise RuntimeError(f"{task}/N={size} contains duplicate sample IDs")
        labels = np.asarray(
            [int(not _parse_bool(row["base_is_correct"])) for row in rows],
            dtype=np.int64,
        )
        if reference_ids is None:
            reference_ids = sample_ids
            reference_labels = labels
        elif sample_ids != reference_ids or not np.array_equal(labels, reference_labels):
            raise RuntimeError(f"{task} sample order or fixed-first labels drift across N")

        arrays: Dict[str, np.ndarray] = {
            "error": labels,
            "sample_id": np.asarray(sample_ids, dtype=object),
        }
        for method, (_label, column) in METHODS.items():
            if method == "structure_only" and size == 1:
                continue
            try:
                values = np.asarray([float(row[column]) for row in rows], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid {task}/N={size} column {column}") from exc
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"non-finite {task}/N={size} column {column}")
            arrays[method] = values
        output[size] = arrays

    provenance_record = {
        "question_count": expected_count,
        "dataset_id": dataset["dataset_id"],
        "dataset_revision": dataset["revision"],
        "dataset_file_sha256": dataset["file_sha256"],
        "input_sha256": dataset["input_sha256"],
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": provenance["manifest_sha256"],
        "sample_csv_path": _relative(csv_path),
        "sample_csv_sha256": provenance["sample_csv_sha256"],
        "generation_cache_path": provenance["cache_path"],
        "generation_cache_sha256": provenance["cache_sha256"],
    }
    return output, provenance_record


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(ScientificSignalAnalyzer.calculate_auroc(labels, scores))


def _point_estimates(
    arrays: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]]
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    macro: Dict[str, Dict[str, float]] = {}
    per_task: Dict[str, Any] = {}
    for size in ENSEMBLE_SIZES:
        size_key = f"N={size}"
        macro[size_key] = {}
        per_task[size_key] = {}
        for method in METHODS:
            if method == "structure_only" and size == 1:
                continue
            values: Dict[str, float] = {}
            for task in TASKS:
                task_values = arrays[task][size]
                values[task] = _auroc(task_values["error"], task_values[method])
            per_task[size_key][method] = values
            macro[size_key][method] = float(np.mean(list(values.values())))
    return macro, per_task


def _bootstrap_macro(
    arrays: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    draws: int,
    seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    sampled = {
        f"N={size}": {
            method: np.empty(draws, dtype=np.float64)
            for method in METHODS
            if not (method == "structure_only" and size == 1)
        }
        for size in ENSEMBLE_SIZES
    }
    rng = np.random.default_rng(seed)
    for draw in range(draws):
        indices_by_task = {
            task: rng.integers(
                0,
                arrays[task][1]["error"].size,
                arrays[task][1]["error"].size,
            )
            for task in TASKS
        }
        for size in ENSEMBLE_SIZES:
            size_key = f"N={size}"
            for method in sampled[size_key]:
                task_values = []
                for task in TASKS:
                    values = arrays[task][size]
                    indices = indices_by_task[task]
                    score = _auroc(values["error"][indices], values[method][indices])
                    if not math.isfinite(score):
                        raise RuntimeError(
                            f"non-finite bootstrap AUROC at draw {draw}, "
                            f"{task}, N={size}, {method}"
                        )
                    task_values.append(score)
                sampled[size_key][method][draw] = float(np.mean(task_values))
    return sampled


def _intervals(
    sampled: Mapping[str, Mapping[str, np.ndarray]]
) -> Dict[str, Dict[str, list[float]]]:
    return {
        size: {
            method: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for method, values in methods.items()
        }
        for size, methods in sampled.items()
    }


def _contrast(
    point: Mapping[str, Mapping[str, float]],
    sampled: Mapping[str, Mapping[str, np.ndarray]],
    left: str,
    right: str,
    size: int = 16,
) -> Dict[str, float]:
    size_key = f"N={size}"
    values = sampled[size_key][left] - sampled[size_key][right]
    return {
        "estimate": float(point[size_key][left] - point[size_key][right]),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "mathtext.fontset": "stix",
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.7,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 6.8,
            "lines.linewidth": 1.55,
            "lines.markersize": 4.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def _draw_figure(
    point: Mapping[str, Mapping[str, float]],
    intervals: Mapping[str, Mapping[str, Sequence[float]]],
    contrasts: Mapping[str, Mapping[str, float]],
    output_dir: Path,
) -> tuple[Path, Path]:
    _configure_matplotlib()
    colors = {
        "pwsc_ci": "#006D77",
        "quality_only": "#D55E00",
        "structure_only": "#3A6F5C",
        "mean_nll": "#6B7280",
    }
    markers = {"pwsc_ci": "o", "quality_only": "s", "structure_only": "^"}
    x = np.arange(len(ENSEMBLE_SIZES), dtype=float)
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(7.02, 2.75),
        gridspec_kw={"width_ratios": [1.08, 0.92], "wspace": 0.48},
    )

    for method in ("pwsc_ci", "structure_only", "quality_only"):
        y = []
        low = []
        high = []
        for size in ENSEMBLE_SIZES:
            key = f"N={size}"
            if method not in point[key]:
                y.append(np.nan)
                low.append(np.nan)
                high.append(np.nan)
                continue
            y.append(point[key][method])
            low.append(intervals[key][method][0])
            high.append(intervals[key][method][1])
        y_values = np.asarray(y, dtype=float)
        low_values = np.asarray(low, dtype=float)
        high_values = np.asarray(high, dtype=float)
        ax_left.plot(
            x,
            y_values,
            color=colors[method],
            marker=markers[method],
            markerfacecolor="white" if method != "pwsc_ci" else colors[method],
            markeredgewidth=1.0,
            zorder=3,
        )
        ax_left.fill_between(
            x,
            low_values,
            high_values,
            color=colors[method],
            alpha=0.11,
            linewidth=0,
            zorder=1,
        )

        direct_label = {
            "pwsc_ci": "PWSC-CI",
            "structure_only": "Structure only (TC)",
            "quality_only": "Worst-k only (k=5)",
        }[method]
        finite = np.flatnonzero(np.isfinite(y_values))
        last = int(finite[-1])
        ax_left.annotate(
            direct_label,
            xy=(x[last], y_values[last]),
            xytext=(5, 0),
            textcoords="offset points",
            color=colors[method],
            va="center",
            ha="left",
            fontsize=6.9,
            fontweight="bold" if method == "pwsc_ci" else "normal",
        )

    ax_left.set_xticks(x, [str(size) for size in ENSEMBLE_SIZES])
    ax_left.set_xlim(-0.15, 3.78)
    ax_left.set_xlabel("Number of sampled paths, $N$")
    ax_left.set_ylabel("Task-macro AUROC")
    ax_left.set_ylim(0.58, 0.90)
    ax_left.set_yticks(np.arange(0.60, 0.91, 0.05))
    ax_left.grid(axis="y", color="#D5D9DE", linewidth=0.5, alpha=0.85)
    ax_left.set_axisbelow(True)
    ax_left.spines[["top", "right"]].set_visible(False)
    ax_left.set_title("More peers expose structured disagreement", loc="left", pad=5)
    ax_left.text(
        -0.16,
        1.08,
        "a",
        transform=ax_left.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="top",
    )
    contrast_order = (
        ("bottleneck", "Bottleneck quality ($k=5$)\nWorst-$k$ $-$ Mean NLL", "#C44E00"),
        ("structured_peers", "Structured peers\nPWSC-CI $-$ Worst-$k$ ($k=5$)", "#006D77"),
        (
            "confidence_fusion",
            "Confidence fusion\nPWSC-CI $-$ structure only",
            "#3A6F5C",
        ),
    )
    y_positions = np.arange(len(contrast_order) - 1, -1, -1, dtype=float)
    all_lows = []
    all_highs = []
    for y_pos, (name, _label, color) in zip(y_positions, contrast_order):
        result = contrasts[name]
        estimate = result["estimate"]
        low = result["ci95_low"]
        high = result["ci95_high"]
        all_lows.append(low)
        all_highs.append(high)
        ax_right.errorbar(
            estimate,
            y_pos,
            xerr=np.asarray([[estimate - low], [high - estimate]]),
            fmt="o",
            color=color,
            ecolor=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.65,
            elinewidth=1.6,
            capsize=2.4,
            capthick=1.0,
            zorder=3,
        )
        ax_right.annotate(
            f"{estimate:+.3f}",
            xy=(high, y_pos),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=7.0,
            color="#252A31",
        )

    ax_right.axvline(0.0, color="#7B818A", linewidth=0.8, linestyle=(0, (2, 2)))
    ax_right.set_yticks(y_positions, [label for _name, label, _color in contrast_order])
    ax_right.set_xlabel("Paired $\Delta$ task-macro AUROC at $N=16$")
    x_min = min(-0.012, min(all_lows) - 0.012)
    x_max = max(all_highs) + 0.045
    ax_right.set_xlim(x_min, x_max)
    ax_right.grid(axis="x", color="#D5D9DE", linewidth=0.5, alpha=0.85)
    ax_right.set_axisbelow(True)
    ax_right.spines[["top", "right", "left"]].set_visible(False)
    ax_right.tick_params(axis="y", length=0, pad=4)
    ax_right.set_title("Each design choice adds ranking signal", loc="left", pad=5)
    ax_right.text(
        -0.19,
        1.08,
        "b",
        transform=ax_right.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="top",
    )

    fig.subplots_adjust(left=0.085, right=0.988, bottom=0.22, top=0.88)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "pwsc-ci-ablation.pdf"
    svg_path = output_dir / "pwsc-ci-ablation.svg"
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Title": "PWSC-CI independent component evidence",
            "Author": "Anonymous",
            "Creator": "scripts/plot_pwsc_ci_ablation.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Title": "PWSC-CI independent component evidence",
            "Creator": "scripts/plot_pwsc_ci_ablation.py",
            "Date": "2026-07-18",
        },
    )
    plt.close(fig)
    return pdf_path, svg_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the independent PWSC-CI component-evidence figure."
    )
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "figures")
    parser.add_argument(
        "--tex_figure_dir", type=Path, default=ROOT / "AuthorKit27/Figures"
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("bootstrap must be at least 100 draws")

    freeze_path = args.freeze.expanduser().resolve()
    freeze = _load_json(freeze_path)
    if freeze.get("method") != "PWSC-CI-v1.0":
        raise RuntimeError("the selected freeze is not canonical PWSC-CI-v1.0")
    if freeze.get("contract") != "pwsc-ci-normalized-holevo-v1":
        raise RuntimeError("PWSC-CI contract mismatch")
    if freeze.get("score_column") != PWSC_COLUMN:
        raise RuntimeError("freeze does not admit exactly pwsc_ci_risk")
    if tuple(freeze["evidence_scope"]["tasks"]) != TASKS:
        raise RuntimeError("freeze task order/scope mismatch")

    arrays: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    provenance: Dict[str, Any] = {}
    for task in TASKS:
        arrays[task], provenance[task] = _load_task_rows(freeze, task)

    point, per_task = _point_estimates(arrays)
    frozen_macro = float(freeze["macro_metrics"]["PWSC-CI"]["AUROC"])
    if not math.isclose(point["N=16"]["pwsc_ci"], frozen_macro, abs_tol=1e-12):
        raise RuntimeError("N=16 macro AUROC does not match the active freeze")
    for task in TASKS:
        observed = per_task["N=16"]["pwsc_ci"][task]
        expected = float(freeze["task_metrics"][task]["PWSC-CI"]["AUROC"])
        if not math.isclose(observed, expected, abs_tol=1e-12):
            raise RuntimeError(f"{task} N=16 AUROC does not match the active freeze")

    sampled = _bootstrap_macro(arrays, draws=args.bootstrap, seed=args.seed)
    ci95 = _intervals(sampled)
    contrasts = {
        "bottleneck": _contrast(point, sampled, "quality_only", "mean_nll"),
        "structured_peers": _contrast(point, sampled, "pwsc_ci", "quality_only"),
        "confidence_fusion": _contrast(
            point, sampled, "pwsc_ci", "structure_only"
        ),
    }
    pdf_path, svg_path = _draw_figure(point, ci95, contrasts, args.output_dir)

    tex_dir = args.tex_figure_dir.expanduser().resolve()
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_pdf = tex_dir / pdf_path.name
    tex_svg = tex_dir / svg_path.name
    shutil.copyfile(pdf_path, tex_pdf)
    shutil.copyfile(svg_path, tex_svg)

    statistics = {
        "schema": "pwsc-ci-independent-component-figure-v1",
        "method": freeze["method"],
        "contract": freeze["contract"],
        "evidence_scope": freeze["evidence_scope"],
        "analysis": {
            "metric": "AUROC",
            "aggregation": "equal-weight macro average over four tasks",
            "bootstrap_contract": (
                "paired within-task resampling followed by four-task macro mean"
            ),
            "bootstrap_draws": args.bootstrap,
            "bootstrap_seed": args.seed,
            "ensemble_sizes": list(ENSEMBLE_SIZES),
            "controls": {
                "mean_nll": "mean chosen-token NLL; local likelihood control",
                "quality_only": (
                    "worst-k chosen-token NLL with k=5; bottleneck-quality control"
                ),
                "structure_only": (
                    "task-kernel target-consistency adaptation; structure-only control"
                ),
            },
        },
        "source_freeze": {
            "path": _relative(freeze_path),
            "sha256": _sha256(freeze_path),
        },
        "provenance": provenance,
        "macro_auroc": point,
        "macro_auroc_ci95": ci95,
        "per_task_auroc": per_task,
        "n16_paired_macro_contrasts": contrasts,
        "artifacts": {
            "pdf": {"path": _relative(pdf_path), "sha256": _sha256(pdf_path)},
            "svg": {"path": _relative(svg_path), "sha256": _sha256(svg_path)},
            "tex_pdf": {"path": _relative(tex_pdf), "sha256": _sha256(tex_pdf)},
            "tex_svg": {"path": _relative(tex_svg), "sha256": _sha256(tex_svg)},
        },
    }
    stats_path = args.output_dir / "pwsc-ci-ablation-stats.json"
    stats_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "pdf": _relative(pdf_path),
                "svg": _relative(svg_path),
                "statistics": _relative(stats_path),
                "n16_macro_auroc": point["N=16"],
                "n16_contrasts": contrasts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
