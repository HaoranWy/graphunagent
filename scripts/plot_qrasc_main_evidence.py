#!/usr/bin/env python3
"""Generate traceable main-paper figures for results and mechanism evidence.

The result figures read the frozen Llama/Qwen metric aggregates and the Llama
risk-coverage CSV. The ablation figures read the independent component freeze
plus the generation-free Llama control manifests. Every input is hash-checked
before plotting. Main Figures 3 and 4 retain native vector geometry and
embedded editable fonts; the appendix-only component copy remains outlined.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qrasc-main-matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1784736000")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    ROOT
    / ".pipeline/archive/pwsc_retired_results_20260717"
    / "supplemental_experiments_20260718"
)
LLAMA_METRICS = ARCHIVE / "aggregate/Llama-3.1-8B-Instruct/metrics.csv"
QWEN_METRICS = (
    ARCHIVE
    / "qwen_main_20260721/aggregate/Qwen2.5-7B-Instruct/metrics.csv"
)
LLAMA70B_METRICS = (
    ARCHIVE
    / "llama70b_main_20260723/aggregate/Llama-3.1-70B-Instruct/metrics.csv"
)
LLAMA_RISK_COVERAGE = (
    ARCHIVE / "aggregate/Llama-3.1-8B-Instruct/risk_coverage.csv"
)
CONTROLS = (
    ARCHIVE / "controls/Llama-3.1-8B-Instruct/seed_20260716"
)
PATH_COUNT_CONTROLS = {
    4: CONTROLS / "path_count_n4",
    8: CONTROLS / "path_count_n8",
}
COMPONENT_STATS = ROOT / "figures/pwsc-ci-ablation-stats.json"

SOURCE_HASHES = {
    LLAMA_METRICS: "6b3ccace512a18b6cb50a6c1beae62470c4258e7e22bd37a6f0927dad3679375",
    QWEN_METRICS: "a382391d97f01778ec6a0d172ca337e76560a70c83924e5a4dc240b9d7eb7cb6",
    LLAMA70B_METRICS: "d4f4af0c25d0e717d5e0ee888aa4e769c1311430860ca695c12ff97015a4826b",
    LLAMA_RISK_COVERAGE: "391769659bd1c7ded0d28fea816471b22fa64e51e23bc63925bcfb1fc8ccc36f",
    COMPONENT_STATS: "c8ed5d7c2a7907d8d7dfda7ab42160b07a708f06b0d1dcbc4a1a7ea7a59bf8a8",
}

TASKS = ("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO")
TASK_COUNTS = {
    "GSM8K": 1319,
    "MATH-500": 500,
    "HumanEvalPlus": 164,
    "FOLIO": 204,
}
PATH_COUNT_LABELS = OrderedDict(
    (
        (1, "1 path ($N=1$)"),
        (4, "4 paths ($N=4$)"),
        (8, "8 paths ($N=8$)"),
    )
)
MODELS = OrderedDict(
    (
        (
            "Llama",
            {
                "path": LLAMA_METRICS,
                "seeds": (20260716, 20260717, 20260718),
                "group_prefix": "meta-llama/Llama-3.1-8B-Instruct",
            },
        ),
        (
            "Qwen",
            {
                "path": QWEN_METRICS,
                "seeds": (20260721, 20260722, 20260723),
                "group_prefix": "Qwen/Qwen2.5-7B-Instruct",
            },
        ),
    )
)
COMPLETE_TASKWISE_MODELS = OrderedDict(
    (
        *MODELS.items(),
        (
            "Llama-70B",
            {
                "path": LLAMA70B_METRICS,
                "seeds": (20260716, 20260717, 20260718),
                "group_prefix": "meta-llama/Llama-3.1-70B-Instruct",
            },
        ),
    )
)
METHODS = OrderedDict(
    (
        ("Q-RASC", "PWSC-CI"),
        ("Q-RASC-KD", "Target Consistency (task-kernel adaptation)"),
        ("CoCoA-MSP", "CoCoA-MSP (outcome-kernel adaptation)"),
    )
)
ALL_METHODS = OrderedDict(
    (
        ("Q-RASC", "PWSC-CI"),
        ("Q-RASC-KD", "Target Consistency (task-kernel adaptation)"),
        ("CoCoA-MSP", "CoCoA-MSP (outcome-kernel adaptation)"),
        ("CoCoA-PPL", "CoCoA-PPL (outcome-kernel adaptation)"),
        ("KLE", "KLE (task-kernel adaptation)"),
        ("Weighted MBR", "Weighted MBR"),
        ("Semantic entropy", "Semantic Entropy"),
        ("Self-consistency", "Self-Consistency"),
        ("Worst-$k$ NLL", "Worst-k NLL (k=5)"),
        ("Mean NLL", "Mean NLL"),
    )
)
METRICS = OrderedDict(
    (
        ("AUROC", "AUROC $\\uparrow$"),
        ("AUPRC", "AUPRC $\\uparrow$"),
        ("AURC", "AURC $\\downarrow$"),
        ("SelectiveAcc@50", "Selective accuracy @ 50% $\\uparrow$"),
    )
)

CONTROL_METHODS = OrderedDict(
    (
        ("Q-RASC", "pwsc_ci_risk"),
        ("Raw $\\chi_0$", "control_raw_holevo"),
        ("Uniform peers", "control_ci_uniform_peer_mass"),
        ("Task kernel", "control_ci_task_kernel"),
    )
)
TARGET_MODES = OrderedDict(
    (
        ("Path-0", {"directory": "fixed", "contract": "fixed"}),
        ("Rotation", {"directory": "all_target", "contract": "all"}),
    )
)
# Paper-facing finite-window diagnostic. This deliberately restricted one-seed
# view is not a cross-domain ablation of the archived all-token condition.
K_VALUES = ("1", "3", "5", "10")
PROBE_COUNTS = (4, 8, 16)

COLORS = {
    "Q-RASC": "#007C83",
    "Q-RASC-KD": "#4B5563",
    "CoCoA-MSP": "#D55E00",
    "AUROC": "#007C83",
    "AURC": "#3B6FB6",
    "neutral": "#68717D",
    "grid": "#D5D9DE",
    "shade": "#F3F5F7",
}
RISK_METHODS = OrderedDict(
    (
        ("Q-RASC", "PWSC-CI"),
        ("Q-RASC-KD", "Target Consistency (task-kernel adaptation)"),
        ("Semantic Entropy", "Semantic Entropy"),
        ("Worst-$k$ NLL", "Worst-k NLL (k=5)"),
        ("Mean NLL", "Mean NLL"),
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} hash mismatch for {path}: expected {expected}, got {observed}"
        )


def _configure_matplotlib(*, publication_figure: bool = False) -> None:
    if publication_figure:
        font_size = 9.5
        title_size = 9.6
        label_size = 9.5
        tick_size = 9.3
        legend_size = 9.3
    else:
        font_size = 8.8
        title_size = 9.7
        label_size = 8.8
        tick_size = 8.2
        legend_size = 8.0
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "mathtext.fontset": "stix",
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "axes.linewidth": 0.65,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": legend_size,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def _load_taskwise_results() -> dict[str, Any]:
    output: dict[str, Any] = {}
    seed_pattern = re.compile(r"^(.+)\|seed=(\d+)\|([^|]+)$")
    for model, contract in MODELS.items():
        path = contract["path"]
        expected_seeds = tuple(contract["seeds"])
        prefix = str(contract["group_prefix"])
        values: dict[str, dict[str, dict[str, list[float]]]] = {
            metric: {
                label: {task: [] for task in TASKS} for label in METHODS
            }
            for metric in METRICS
        }
        seen: set[tuple[int, str, str]] = set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("scope") != "task":
                    continue
                match = seed_pattern.fullmatch(row.get("group", ""))
                if not match or match.group(1) != prefix:
                    continue
                seed = int(match.group(2))
                task = match.group(3)
                if seed not in expected_seeds or task not in TASKS:
                    continue
                labels = [
                    label
                    for label, source_name in METHODS.items()
                    if row.get("method") == source_name
                ]
                if not labels:
                    continue
                label = labels[0]
                key = (seed, task, label)
                if key in seen:
                    raise RuntimeError(f"duplicate task result row: {model}/{key}")
                seen.add(key)
                for metric in METRICS:
                    value = float(row[metric])
                    if not math.isfinite(value):
                        raise RuntimeError(
                            f"non-finite {model}/{seed}/{task}/{label}/{metric}"
                        )
                    values[metric][label][task].append(value)

        expected_count = len(expected_seeds) * len(TASKS) * len(METHODS)
        if len(seen) != expected_count:
            raise RuntimeError(
                f"{model} selected result count is {len(seen)}; expected {expected_count}"
            )
        model_stats: dict[str, Any] = {}
        for metric, method_values in values.items():
            model_stats[metric] = {}
            for label, task_values in method_values.items():
                model_stats[metric][label] = {}
                for task, raw in task_values.items():
                    if len(raw) != len(expected_seeds):
                        raise RuntimeError(
                            f"{model}/{metric}/{label}/{task} has {len(raw)} seeds"
                        )
                    array = np.asarray(raw, dtype=float)
                    model_stats[metric][label][task] = {
                        "mean": float(np.mean(array)),
                        "std": float(np.std(array, ddof=1)),
                        "seed_values": [float(value) for value in array],
                    }
        output[model] = {
            "seeds": list(expected_seeds),
            "statistics": model_stats,
        }
    return output


def _load_complete_taskwise_results() -> dict[str, Any]:
    output: dict[str, Any] = {}
    seed_pattern = re.compile(r"^(.+)\|seed=(\d+)\|([^|]+)$")
    complete_metrics = ("AUROC", "AUPRC", "AURC", "SelectiveAcc@50")
    for model, contract in COMPLETE_TASKWISE_MODELS.items():
        path = contract["path"]
        expected_seeds = tuple(contract["seeds"])
        prefix = str(contract["group_prefix"])
        values = {
            metric: {
                label: {task: [] for task in TASKS} for label in ALL_METHODS
            }
            for metric in complete_metrics
        }
        source_to_label = {
            source_name: label for label, source_name in ALL_METHODS.items()
        }
        seen: set[tuple[int, str, str]] = set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("scope") != "task":
                    continue
                match = seed_pattern.fullmatch(row.get("group", ""))
                if not match or match.group(1) != prefix:
                    continue
                seed = int(match.group(2))
                task = match.group(3)
                label = source_to_label.get(row.get("method", ""))
                if seed not in expected_seeds or task not in TASKS or label is None:
                    continue
                key = (seed, task, label)
                if key in seen:
                    raise RuntimeError(f"duplicate complete task row: {model}/{key}")
                seen.add(key)
                for metric in complete_metrics:
                    value = float(row[metric])
                    if not math.isfinite(value):
                        raise RuntimeError(
                            f"non-finite {model}/{seed}/{task}/{label}/{metric}"
                        )
                    values[metric][label][task].append(value)

        expected_count = len(expected_seeds) * len(TASKS) * len(ALL_METHODS)
        if len(seen) != expected_count:
            raise RuntimeError(
                f"{model} complete result count is {len(seen)}; expected {expected_count}"
            )
        statistics: dict[str, Any] = {}
        for metric, method_values in values.items():
            statistics[metric] = {}
            for label, task_values in method_values.items():
                statistics[metric][label] = {}
                for task, raw in task_values.items():
                    if len(raw) != len(expected_seeds):
                        raise RuntimeError(
                            f"{model}/{metric}/{label}/{task} has {len(raw)} seeds"
                        )
                    array = np.asarray(raw, dtype=float)
                    statistics[metric][label][task] = {
                        "mean": float(np.mean(array)),
                        "std": float(np.std(array, ddof=1)),
                        "seed_values": [float(value) for value in array],
                    }
        output[model] = {"seeds": list(expected_seeds), "statistics": statistics}
    return output


def _load_risk_coverage() -> list[dict[str, Any]]:
    source_to_label = {source: label for label, source in RISK_METHODS.items()}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    with LLAMA_RISK_COVERAGE.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            label = source_to_label.get(raw.get("method", ""))
            if label is None:
                continue
            model_key = raw.get("model_key", "")
            if model_key != "meta-llama/Llama-3.1-8B-Instruct":
                raise RuntimeError(f"unexpected risk-coverage model: {model_key}")
            coverage = float(raw["coverage"])
            key = (label, coverage)
            if key in seen:
                raise RuntimeError(f"duplicate risk-coverage row: {key}")
            seen.add(key)
            row = {
                "label": label,
                "source_method": raw["method"],
                "coverage": coverage,
                "selective_error": float(raw["selective_error"]),
                "ci95_low": float(raw["ci95_low"]),
                "ci95_high": float(raw["ci95_high"]),
            }
            if not all(
                math.isfinite(float(row[field]))
                for field in ("coverage", "selective_error", "ci95_low", "ci95_high")
            ):
                raise RuntimeError(f"non-finite risk-coverage row: {key}")
            if not row["ci95_low"] <= row["selective_error"] <= row["ci95_high"]:
                raise RuntimeError(f"invalid risk-coverage interval: {key}")
            rows.append(row)

    expected = len(RISK_METHODS) * 20
    if len(rows) != expected:
        raise RuntimeError(f"risk-coverage row count is {len(rows)}; expected {expected}")
    for label in RISK_METHODS:
        coverages = sorted(row["coverage"] for row in rows if row["label"] == label)
        if len(coverages) != 20 or not np.allclose(
            coverages, np.linspace(0.05, 1.0, 20), atol=1e-12, rtol=0.0
        ):
            raise RuntimeError(f"risk-coverage grid mismatch for {label}")
    return rows


def _control_paths(directory: str) -> dict[str, Path]:
    return {task: CONTROLS / directory / f"{task}.json" for task in TASKS}


def _require_path_count_source_match(
    source: Mapping[str, Any],
    reference: Mapping[str, Any],
    label: str,
) -> None:
    for field in (
        "manifest_sha256",
        "cache_sha256",
        "csv_sha256",
        "input_sample_sha256",
        "model_identifier",
        "model_revision",
        "seed",
    ):
        if source.get(field) != reference.get(field):
            raise RuntimeError(f"path-count source drift for {label}/{field}")


def _load_single_path_control(
    reference_payloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_metrics: dict[str, dict[str, float]] = {}
    provenance: dict[str, Any] = {"Path-0": {}, "Rotation": {}}
    for task in TASKS:
        reference = reference_payloads["Path-0"][task].get("provenance", {}).get(
            task, {}
        )
        manifest_path = Path(str(reference.get("manifest_path", "")))
        manifest_hash = str(reference.get("manifest_sha256", ""))
        if not manifest_hash or not manifest_path.is_file():
            raise RuntimeError(f"missing N=1 source manifest for {task}")
        _require_hash(manifest_path, manifest_hash, f"{task}/N=1 source manifest")
        manifest = _load_json(manifest_path)
        source = {
            "manifest_sha256": manifest_hash,
            "cache_sha256": manifest.get("generation_cache_sha256"),
            "csv_sha256": manifest.get("csv_sha256"),
            "input_sample_sha256": manifest.get("input_sample_sha256"),
            "model_identifier": manifest.get("model_identifier"),
            "model_revision": manifest.get("model_revision"),
            "seed": manifest.get("seed"),
        }
        _require_path_count_source_match(source, reference, f"Path-0/{task}/N=1")
        sampling = manifest.get("sampling", {})
        if sampling.get("ensemble_sizes") != list(PATH_COUNT_LABELS) + [16]:
            raise RuntimeError(f"{task} source manifest lacks the frozen N prefixes")

        report_path = Path(str(manifest.get("report_path", "")))
        report_hash = str(manifest.get("report_sha256", ""))
        csv_path = Path(str(manifest.get("csv_path", "")))
        if not report_hash or not report_path.is_file() or not csv_path.is_file():
            raise RuntimeError(f"incomplete N=1 source bundle for {task}")
        _require_hash(report_path, report_hash, f"{task}/N=1 source report")
        _require_hash(csv_path, source["csv_sha256"], f"{task}/N=1 source CSV")
        report = _load_json(report_path)
        if report.get("task") != task or report.get("n_scored_questions") != TASK_COUNTS[task]:
            raise RuntimeError(f"N=1 source report scope mismatch for {task}")
        row = report.get("Methods by N", {}).get("N=1", {}).get("PWSC-CI", {})
        task_metrics[task] = {
            metric: float(row[metric]) for metric in ("AUROC", "AURC")
        }
        if not all(math.isfinite(value) for value in task_metrics[task].values()):
            raise RuntimeError(f"non-finite N=1 source metric for {task}")
        provenance["Path-0"][task] = {
            "report": _relative(report_path),
            "report_sha256": report_hash,
            "sample_csv": _relative(csv_path),
            "sample_csv_sha256": source["csv_sha256"],
            "manifest": _relative(manifest_path),
            "manifest_sha256": manifest_hash,
            "generation_cache": _relative(Path(str(manifest["generation_cache_path"]))),
            "generation_cache_sha256": source["cache_sha256"],
        }

    fixed = {
        metric: float(np.mean([task_metrics[task][metric] for task in TASKS]))
        for metric in ("AUROC", "AURC")
    }
    return {
        "Path-0": fixed,
        "Rotation": {"AUROC": None, "AURC": None},
    }, provenance


def _load_multi_path_count_control(
    reference_payloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ensemble_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    control_root = PATH_COUNT_CONTROLS[ensemble_size]
    for target_label, mode in TARGET_MODES.items():
        task_payloads: dict[str, dict[str, Any]] = {}
        provenance[target_label] = {}
        for task in TASKS:
            path = control_root / mode["directory"] / f"{task}.json"
            manifest_path = path.with_name(f"{path.stem}.manifest.json")
            manifest = _load_json(manifest_path)
            if manifest.get("schema") != "pwsc-ci-supplemental-controls-manifest-v1":
                raise RuntimeError(
                    f"unexpected path-count manifest schema: {manifest_path}"
                )
            output_hash = manifest.get("output", {}).get("sha256")
            csv_hash = manifest.get("sample_csv", {}).get("sha256")
            csv_path = path.with_suffix(".csv")
            if not output_hash or not csv_hash:
                raise RuntimeError(f"incomplete path-count manifest: {manifest_path}")
            _require_hash(
                path, output_hash, f"{target_label}/{task}/N={ensemble_size} control"
            )
            _require_hash(
                csv_path, csv_hash, f"{target_label}/{task}/N={ensemble_size} CSV"
            )

            payload = _load_json(path)
            if payload.get("target_mode") != mode["contract"]:
                raise RuntimeError(f"path-count target mode mismatch: {path}")
            if list(payload.get("tasks", [])) != [task]:
                raise RuntimeError(f"path-count task scope mismatch: {path}")
            if payload.get("ensemble_size") != ensemble_size:
                raise RuntimeError(f"path-count ensemble size mismatch: {path}")
            expected_rows = TASK_COUNTS[task] * (
                ensemble_size if mode["contract"] == "all" else 1
            )
            if payload.get("row_count") != expected_rows:
                raise RuntimeError(
                    f"path-count row mismatch for {target_label}/{task}: "
                    f"expected {expected_rows}, got {payload.get('row_count')}"
                )

            source = payload.get("provenance", {}).get(task, {})
            reference = reference_payloads["Path-0"][task].get("provenance", {}).get(
                task, {}
            )
            _require_path_count_source_match(
                source, reference, f"{target_label}/{task}/N={ensemble_size}"
            )

            task_payloads[task] = payload
            provenance[target_label][task] = {
                "result": _relative(path),
                "result_sha256": output_hash,
                "sample_csv": _relative(csv_path),
                "sample_csv_sha256": csv_hash,
                "manifest": _relative(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
            }

        metrics[target_label] = {
            metric: float(
                np.mean(
                    [
                        task_payloads[task]["task_metrics"][task]["methods"]
                        ["pwsc_ci_risk"][metric]
                        for task in TASKS
                    ]
                )
            )
            for metric in ("AUROC", "AURC")
        }
    return metrics, provenance


def _load_path_count_controls(
    reference_payloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[int, Any], dict[str, Any]]:
    metrics: dict[int, Any] = {}
    provenance: dict[str, Any] = {}
    metrics[1], provenance["Path count N=1"] = _load_single_path_control(
        reference_payloads
    )
    for ensemble_size in (4, 8):
        size_metrics, size_provenance = _load_multi_path_count_control(
            reference_payloads, ensemble_size
        )
        metrics[ensemble_size] = size_metrics
        provenance[f"Path count N={ensemble_size}"] = size_provenance
    return metrics, provenance


def _load_controls() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for label, mode in TARGET_MODES.items():
        payloads[label] = {}
        provenance[label] = {}
        for task, path in _control_paths(mode["directory"]).items():
            manifest_path = path.with_name(f"{path.stem}.manifest.json")
            manifest = _load_json(manifest_path)
            if manifest.get("schema") != "pwsc-ci-supplemental-controls-manifest-v1":
                raise RuntimeError(f"unexpected control manifest schema: {manifest_path}")
            expected_hash = manifest.get("output", {}).get("sha256")
            if not expected_hash:
                raise RuntimeError(f"missing output hash: {manifest_path}")
            _require_hash(path, expected_hash, f"{label}/{task} control")
            payload = _load_json(path)
            if payload.get("target_mode") != mode["contract"]:
                raise RuntimeError(f"target mode mismatch: {path}")
            if list(payload.get("tasks", [])) != [task]:
                raise RuntimeError(f"task scope mismatch: {path}")
            payloads[label][task] = payload
            provenance[label][task] = {
                "result": _relative(path),
                "result_sha256": expected_hash,
                "manifest": _relative(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
            }

    central: dict[str, Any] = {}
    weakest_k: dict[str, Any] = {}
    for target_label in TARGET_MODES:
        central[target_label] = {}
        weakest_k[target_label] = {}
        for display_name, method_key in CONTROL_METHODS.items():
            central[target_label][display_name] = {}
            for metric in ("AUROC", "AURC"):
                raw = [
                    payloads[target_label][task]["task_metrics"][task]["methods"]
                    [method_key][metric]
                    for task in TASKS
                ]
                central[target_label][display_name][metric] = float(np.mean(raw))
        for k in K_VALUES:
            method_key = f"control_ci_weakest_k_{k}"
            weakest_k[target_label][k] = {}
            for metric in ("AUROC", "AURC"):
                raw = [
                    payloads[target_label][task]["task_metrics"][task]["methods"]
                    [method_key][metric]
                    for task in TASKS
                ]
                weakest_k[target_label][k][metric] = float(np.mean(raw))

    path_counts, path_count_provenance = _load_path_count_controls(payloads)
    for ensemble_size, label in PATH_COUNT_LABELS.items():
        for target_label in TARGET_MODES:
            central[target_label][label] = path_counts[ensemble_size][target_label]
    provenance.update(path_count_provenance)

    fixed_human = payloads["Path-0"]["HumanEvalPlus"]
    probes: dict[str, Any] = {}
    for count in PROBE_COUNTS:
        probes[str(count)] = {}
        for recovery in ("off", "on"):
            method_key = f"control_ci_behavior_p{count}_recovery_{recovery}"
            probes[str(count)][recovery] = {
                metric: float(
                    fixed_human["task_metrics"]["HumanEvalPlus"]["methods"]
                    [method_key][metric]
                )
                for metric in ("AUROC", "AURC")
            }
    return {"central": central, "weakest_k": weakest_k, "probes": probes}, provenance, payloads


def _load_component_stats() -> dict[str, Any]:
    stats = _load_json(COMPONENT_STATS)
    if stats.get("schema") != "pwsc-ci-independent-component-figure-v1":
        raise RuntimeError("unexpected independent component-statistics schema")
    if stats.get("source_freeze", {}).get("sha256") != (
        "2d81f827b6cae21ba345d27c871a246bbc25a2d5e743eee5a95955fd81a0971d"
    ):
        raise RuntimeError("independent component freeze hash mismatch")
    return stats


def _save_figure(fig: mpl.figure.Figure, stem: str, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Title": stem.replace("-", " ").title(),
            "Author": "Anonymous",
            "Creator": "scripts/plot_qrasc_main_evidence.py",
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
            "Title": stem.replace("-", " ").title(),
            "Creator": "scripts/plot_qrasc_main_evidence.py",
            "Date": "2026-07-23",
        },
    )
    plt.close(fig)
    return pdf_path, svg_path


def _outline_pdf(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(".outlined.tmp.pdf")
    command = [
        "gs",
        "-q",
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        "-dAutoRotatePages=/None",
        "-dNoOutputFonts",
        f"-sOutputFile={temporary}",
        str(source),
    ]
    subprocess.run(command, check=True)
    temporary.replace(destination)


def _copy_editable_vector_pdf(source: Path, destination: Path) -> None:
    shutil.copyfile(source, destination)


def _plot_risk_coverage(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> tuple[Path, Path]:
    _configure_matplotlib(publication_figure=True)
    fig, ax = plt.subplots(figsize=(3.17, 2.55))
    styles = OrderedDict(
        (
            ("Q-RASC", ("#007C83", "-", "o")),
            ("Q-RASC-KD", ("#4B5563", "--", "s")),
            ("Semantic Entropy", ("#D55E00", "-.", "^")),
            ("Worst-$k$ NLL", ("#3B6FB6", ":", "D")),
            ("Mean NLL", ("#111827", (0, (5, 2)), "v")),
        )
    )
    legend_labels = {
        "Q-RASC": "Q-RASC",
        "Q-RASC-KD": "Q-RASC-KD",
        "Semantic Entropy": "Sem. Ent.",
        "Worst-$k$ NLL": "Worst-$k$",
        "Mean NLL": "Mean NLL",
    }
    for label, (color, linestyle, marker) in styles.items():
        values = sorted(
            (row for row in rows if row["label"] == label),
            key=lambda row: float(row["coverage"]),
        )
        x = np.asarray([row["coverage"] for row in values], dtype=float)
        y = np.asarray([row["selective_error"] for row in values], dtype=float)
        low = np.asarray([row["ci95_low"] for row in values], dtype=float)
        high = np.asarray([row["ci95_high"] for row in values], dtype=float)
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=4,
            markerfacecolor="white" if label != "Q-RASC" else color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            label=legend_labels[label],
            zorder=3,
        )
        ax.fill_between(x, low, high, color=color, alpha=0.09, linewidth=0)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective error")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0.08, 0.61)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Patch(
            facecolor="#7C8790",
            edgecolor="none",
            alpha=0.18,
            label="95% CI",
        )
    )
    labels.append("95% CI")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=0.55,
        handlelength=1.45,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.18, top=0.68)
    return _save_figure(fig, "q-rasc-llama-risk-coverage", output_dir)


def _plot_taskwise_results(results: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    _configure_matplotlib()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.05, 4.18),
        sharey=True,
        gridspec_kw={"hspace": 0.27, "wspace": 0.12},
    )
    axes_flat = axes.flatten()
    task_labels = (
        "Llama: GSM8K",
        "Llama: MATH-500",
        "Llama: HumanEval+",
        "Llama: FOLIO",
        "Qwen: GSM8K",
        "Qwen: MATH-500",
        "Qwen: HumanEval+",
        "Qwen: FOLIO",
    )
    y_base = np.arange(len(task_labels), dtype=float)[::-1]
    offsets = {"Q-RASC": 0.20, "Q-RASC-KD": 0.0, "CoCoA-MSP": -0.20}
    markers = {"Q-RASC": "o", "Q-RASC-KD": "s", "CoCoA-MSP": "D"}
    all_handles = []
    all_labels = []

    for panel_index, ((metric, title), ax) in enumerate(zip(METRICS.items(), axes_flat)):
        panel_values = []
        row_index = 0
        for model in MODELS:
            stats = results[model]["statistics"][metric]
            for task in TASKS:
                for method in METHODS:
                    entry = stats[method][task]
                    panel_values.extend(
                        [entry["mean"] - entry["std"], entry["mean"] + entry["std"]]
                    )
                row_index += 1
        span = max(panel_values) - min(panel_values)
        padding = max(0.025, span * 0.09)
        ax.set_xlim(min(panel_values) - padding, max(panel_values) + padding * 1.75)
        ax.axhspan(-0.5, 3.5, color=COLORS["shade"], zorder=0)
        ax.axhline(3.5, color="#AEB5BD", linewidth=0.75, zorder=1)

        for method in METHODS:
            means = []
            stds = []
            for model in MODELS:
                stats = results[model]["statistics"][metric][method]
                for task in TASKS:
                    means.append(stats[task]["mean"])
                    stds.append(stats[task]["std"])
            y = y_base + offsets[method]
            handle = ax.errorbar(
                means,
                y,
                xerr=stds,
                fmt=markers[method],
                color=COLORS[method],
                ecolor=COLORS[method],
                markerfacecolor=(COLORS[method] if method == "Q-RASC" else "white"),
                markeredgecolor=COLORS[method],
                markeredgewidth=1.0,
                elinewidth=1.05,
                capsize=1.8,
                capthick=0.8,
                linestyle="none",
                zorder=3,
                label=method,
            )
            if panel_index == 0:
                all_handles.append(handle)
                all_labels.append(method)
            if method == "Q-RASC":
                x_left, x_right = ax.get_xlim()
                for value, y_value in zip(means, y):
                    use_left = value > x_left + 0.79 * (x_right - x_left)
                    ax.annotate(
                        f"{value:.3f}".lstrip("0"),
                        xy=(value, y_value),
                        xytext=(-5 if use_left else 5, 0),
                        textcoords="offset points",
                        ha="right" if use_left else "left",
                        va="center",
                        color=COLORS["Q-RASC"],
                        fontsize=8.2,
                        fontweight="bold",
                    )

        ax.set_title(f"({chr(97 + panel_index)}) {title}", loc="left", pad=4)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.48, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_yticks(y_base, task_labels)
        if panel_index % 2 == 1:
            ax.tick_params(labelleft=False)
        ax.set_ylim(-0.55, 7.55)

    fig.legend(
        all_handles,
        all_labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.006),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.195, right=0.994, bottom=0.075, top=0.915)
    return _save_figure(fig, "q-rasc-taskwise-results", output_dir)


def _plot_complete_taskwise_results(
    results: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    _configure_matplotlib()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.05, 4.62),
        sharex=True,
        gridspec_kw={"hspace": 0.29},
    )
    columns = [(model, task) for model in MODELS for task in TASKS]
    rows = list(ALL_METHODS.keys())
    cmap = LinearSegmentedColormap.from_list(
        "qrasc_complete", ("#E5A01A", "#FAFAFA", "#007C83")
    )

    for panel_index, (metric, ax) in enumerate(zip(("AUROC", "AURC"), axes)):
        absolute = np.asarray(
            [
                [
                    results[model]["statistics"][metric][method][task]["mean"]
                    for model, task in columns
                ]
                for method in rows
            ],
            dtype=float,
        )
        benefit = absolute if metric == "AUROC" else -absolute
        low = float(np.min(benefit))
        high = float(np.max(benefit))
        normalized = (benefit - low) / max(high - low, 1e-12)
        color_values = 2.0 * normalized - 1.0
        ax.imshow(color_values, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)

        for row_index, method in enumerate(rows):
            for col_index in range(len(columns)):
                value = absolute[row_index, col_index]
                intensity = abs(color_values[row_index, col_index])
                text_color = "white" if intensity > 0.66 else "#1F2933"
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.3f}".lstrip("0"),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7.4,
                    fontweight="bold" if method == "Q-RASC" else "normal",
                )

        for col_index in range(len(columns)):
            best_row = int(
                np.argmax(absolute[:, col_index])
                if metric == "AUROC"
                else np.argmin(absolute[:, col_index])
            )
            ax.add_patch(
                Rectangle(
                    (col_index - 0.48, best_row - 0.48),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor="#17202A",
                    linewidth=1.05,
                )
            )

        ax.axvline(3.5, color="#17202A", linewidth=1.0)
        ax.set_yticks(np.arange(len(rows)), rows)
        ax.tick_params(axis="both", length=0, pad=3)
        ax.set_title(
            f"({chr(97 + panel_index)}) {metric} "
            + ("$\\uparrow$" if metric == "AUROC" else "$\\downarrow$"),
            loc="left",
            pad=4,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        for label in ax.get_yticklabels():
            if label.get_text() == "Q-RASC":
                label.set_fontweight("bold")
                label.set_color(COLORS["Q-RASC"])

    axes[0].text(
        1.5,
        -0.95,
        "Llama-3.1-8B",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        clip_on=False,
    )
    axes[0].text(
        5.5,
        -0.95,
        "Qwen2.5-7B",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        clip_on=False,
    )
    axes[-1].set_xticks(
        np.arange(len(columns)),
        ("GSM8K", "MATH", "HumanEval+", "FOLIO") * 2,
    )
    axes[-1].tick_params(axis="x", labelrotation=0)
    fig.subplots_adjust(left=0.20, right=0.995, bottom=0.095, top=0.945)
    return _save_figure(fig, "q-rasc-all-method-taskwise", output_dir)


def _benefit_delta(value: float, baseline: float, metric: str) -> float:
    direction = -1.0 if metric == "AURC" else 1.0
    return 100.0 * direction * (value - baseline)


def _plot_component_evidence(
    component: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    _configure_matplotlib()
    fig, ax_component = plt.subplots(figsize=(3.55, 2.05))
    component_order = (
        ("Q-RASC", "pwsc_ci", COLORS["Q-RASC"]),
        ("Q-RASC-KD", "structure_only", COLORS["neutral"]),
        ("Worst-$k$ NLL", "quality_only", "#D55E00"),
        ("Mean NLL", "mean_nll", "#9AA1AA"),
    )
    point = component["macro_auroc"]["N=16"]
    intervals = component["macro_auroc_ci95"]["N=16"]
    y = np.arange(len(component_order), dtype=float)[::-1]
    for y_value, (label, key, color) in zip(y, component_order):
        estimate = float(point[key])
        low, high = (float(value) for value in intervals[key])
        ax_component.errorbar(
            estimate,
            y_value,
            xerr=np.asarray([[estimate - low], [high - estimate]]),
            fmt="o",
            color=color,
            ecolor=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.45,
            capsize=2.0,
            zorder=3,
        )
        ax_component.annotate(
            f"{estimate:.3f}".lstrip("0"),
            xy=(high, y_value),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.3,
            color="#22272E",
        )
    ax_component.set_yticks(y, [entry[0] for entry in component_order])
    ax_component.set_xlim(0.52, 0.90)
    ax_component.set_xlabel("Task-macro AUROC (95% bootstrap CI)")
    ax_component.set_title("Components add complementary signal", loc="left", pad=4)
    ax_component.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    ax_component.spines[["top", "right", "left"]].set_visible(False)
    ax_component.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.31, right=0.91, bottom=0.22, top=0.88)
    return _save_figure(fig, "q-rasc-ablation-components", output_dir)


def _plot_sensitivity_sweeps(
    controls: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    _configure_matplotlib(publication_figure=True)
    fig, (ax_k, ax_probe) = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.46),
        gridspec_kw={"wspace": 0.38},
    )
    x_k = np.arange(len(K_VALUES), dtype=float)
    k_styles = (
        ("Path-0", "AUROC", "P0 AUROC", COLORS["AUROC"], "-", "o"),
        ("Rotation", "AUROC", "Rot. AUROC", COLORS["AUROC"], "--", "o"),
        ("Path-0", "AURC", "P0 AURC", COLORS["AURC"], "-", "s"),
        ("Rotation", "AURC", "Rot. AURC", COLORS["AURC"], "--", "s"),
    )
    for target, metric, label, color, linestyle, marker in k_styles:
        baseline = controls["weakest_k"][target]["5"][metric]
        values = [
            _benefit_delta(
                controls["weakest_k"][target][k][metric], baseline, metric
            )
            for k in K_VALUES
        ]
        ax_k.plot(
            x_k,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="white" if linestyle == "--" else color,
            markeredgecolor=color,
            markeredgewidth=0.9,
        )
    ax_k.axhline(0.0, color="#7C848D", linewidth=0.75, linestyle=(0, (2, 2)))
    ax_k.set_xticks(x_k, K_VALUES)
    ax_k.set_xlabel("Weakest-token window $k$")
    ax_k.set_ylabel("AUROC/AURC difference\nfrom $k=5$")
    ax_k.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax_k.spines[["top", "right"]].set_visible(False)
    ax_k.legend(
        loc="best",
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.8,
        handletextpad=0.35,
    )

    probe_styles = (
        ("off", "AUROC", "AUROC, off", COLORS["AUROC"], "--", "o"),
        ("on", "AUROC", "AUROC, on", COLORS["AUROC"], "-", "o"),
        ("off", "AURC", "AURC, off", COLORS["AURC"], "--", "s"),
        ("on", "AURC", "AURC, on", COLORS["AURC"], "-", "s"),
    )
    for recovery, metric, label, color, linestyle, marker in probe_styles:
        baseline = controls["probes"]["16"]["on"][metric]
        values = [
            _benefit_delta(
                controls["probes"][str(count)][recovery][metric], baseline, metric
            )
            for count in PROBE_COUNTS
        ]
        ax_probe.plot(
            PROBE_COUNTS,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="white" if recovery == "off" else color,
            markeredgecolor=color,
            markeredgewidth=0.9,
        )
    ax_probe.axhline(0.0, color="#7C848D", linewidth=0.75, linestyle=(0, (2, 2)))
    ax_probe.set_xticks(PROBE_COUNTS)
    ax_probe.set_xlabel("Behavior probes, $p$")
    ax_probe.set_ylabel("AUROC/AURC difference\nfrom canonical $p=16$")
    ax_probe.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax_probe.spines[["top", "right"]].set_visible(False)
    ax_probe.legend(
        loc="best",
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.35,
    )

    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.31, top=0.97)
    panel_captions = (
        (ax_k, "(a) Weakest-token window sensitivity"),
        (ax_probe, "(b) Behavior-probe sensitivity"),
    )
    for axis, caption in panel_captions:
        bounds = axis.get_position()
        fig.text(
            bounds.x0 + bounds.width / 2,
            0.015,
            caption,
            ha="center",
            va="bottom",
            fontsize=9.3,
            fontweight="semibold",
        )
    return _save_figure(fig, "q-rasc-ablation-sensitivity", output_dir)


def main() -> int:
    for path, expected in SOURCE_HASHES.items():
        _require_hash(path, expected, "frozen figure source")

    results = _load_taskwise_results()
    complete_results = _load_complete_taskwise_results()
    risk_coverage = _load_risk_coverage()
    controls, control_provenance, _payloads = _load_controls()
    component = _load_component_stats()
    output_dir = ROOT / "figures"
    tex_dir = ROOT / "AuthorKit27/Figures"

    result_pdf, result_svg = _plot_taskwise_results(results, output_dir)
    complete_pdf, complete_svg = _plot_complete_taskwise_results(
        complete_results, output_dir
    )
    risk_pdf, risk_svg = _plot_risk_coverage(risk_coverage, output_dir)
    component_pdf, component_svg = _plot_component_evidence(component, output_dir)
    sensitivity_pdf, sensitivity_svg = _plot_sensitivity_sweeps(
        controls, output_dir
    )
    tex_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, Any]] = {}
    for path in (component_pdf, sensitivity_pdf, risk_pdf):
        destination = tex_dir / path.name
        if path in (sensitivity_pdf, risk_pdf):
            _copy_editable_vector_pdf(path, destination)
            font_treatment = "editable_embedded_fonts"
        else:
            _outline_pdf(path, destination)
            font_treatment = "outlined_fonts"
        copied[path.name] = {
            "path": _relative(destination),
            "sha256": _sha256(destination),
            "vector_geometry": True,
            "font_treatment": font_treatment,
        }

    statistics = {
        "schema": "qrasc-main-paper-evidence-figures-v9",
        "generated_by": "scripts/plot_qrasc_main_evidence.py",
        "sources": {
            _relative(path): {"sha256": expected}
            for path, expected in SOURCE_HASHES.items()
        },
        "main_taskwise": results,
        "complete_taskwise_metrics": complete_results,
        "risk_coverage": {
            "source": {
                "path": _relative(LLAMA_RISK_COVERAGE),
                "sha256": SOURCE_HASHES[LLAMA_RISK_COVERAGE],
            },
            "row_count": len(risk_coverage),
            "methods": list(RISK_METHODS),
            "band_legend": "95% CI",
        },
        "ablation": {
            "component_source": {
                "path": _relative(COMPONENT_STATS),
                "sha256": SOURCE_HASHES[COMPONENT_STATS],
            },
            "central_controls": controls["central"],
            "path_count": {
                "canonical_ensemble_size": 16,
                "rows": {
                    f"N={ensemble_size}": {
                        "ensemble_size": ensemble_size,
                        "label": label,
                        "metrics": {
                            target_label: controls["central"][target_label][label]
                            for target_label in TARGET_MODES
                        },
                    }
                    for ensemble_size, label in PATH_COUNT_LABELS.items()
                },
                "rotation_omission": (
                    "N=1 has no peer set; rotation is intentionally unreported."
                ),
            },
            "weakest_k": controls["weakest_k"],
            "humaneval_probes": controls["probes"],
            "control_provenance": control_provenance,
        },
        "artifacts": {
            "taskwise_pdf": {
                "path": _relative(result_pdf),
                "sha256": _sha256(result_pdf),
            },
            "taskwise_svg": {
                "path": _relative(result_svg),
                "sha256": _sha256(result_svg),
            },
            "complete_taskwise_pdf": {
                "path": _relative(complete_pdf),
                "sha256": _sha256(complete_pdf),
            },
            "complete_taskwise_svg": {
                "path": _relative(complete_svg),
                "sha256": _sha256(complete_svg),
            },
            "risk_coverage_pdf": {
                "path": _relative(risk_pdf),
                "sha256": _sha256(risk_pdf),
            },
            "risk_coverage_svg": {
                "path": _relative(risk_svg),
                "sha256": _sha256(risk_svg),
            },
            "component_pdf": {
                "path": _relative(component_pdf),
                "sha256": _sha256(component_pdf),
            },
            "component_svg": {
                "path": _relative(component_svg),
                "sha256": _sha256(component_svg),
            },
            "sensitivity_pdf": {
                "path": _relative(sensitivity_pdf),
                "sha256": _sha256(sensitivity_pdf),
            },
            "sensitivity_svg": {
                "path": _relative(sensitivity_svg),
                "sha256": _sha256(sensitivity_svg),
            },
            "author_kit": copied,
        },
    }
    stats_path = output_dir / "q-rasc-main-evidence-stats.json"
    stats_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "taskwise_figure": _relative(result_pdf),
                "complete_taskwise_figure": _relative(complete_pdf),
                "risk_coverage_figure": _relative(risk_pdf),
                "component_figure": _relative(component_pdf),
                "sensitivity_figure": _relative(sensitivity_pdf),
                "statistics": _relative(stats_path),
                "artifact_hashes": {
                    result_pdf.name: _sha256(result_pdf),
                    complete_pdf.name: _sha256(complete_pdf),
                    risk_pdf.name: _sha256(risk_pdf),
                    component_pdf.name: _sha256(component_pdf),
                    sensitivity_pdf.name: _sha256(sensitivity_pdf),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
