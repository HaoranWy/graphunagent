#!/usr/bin/env python3
"""Summarize traced PWSC-CI runs across seeds without pooling model families."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pwsc-ci-matplotlib")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.risk_estimator import PWSC_CI_CONTRACT_VERSION  # noqa: E402
from diagnostics.analyzer import ScientificSignalAnalyzer  # noqa: E402
from run_pipeline import UniversalDatasetLoader, _input_sha256  # noqa: E402


DEFAULT_TASKS = ("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO")
CACHE_BOUND_SAMPLING_FIELDS = (
    "num_samples",
    "temperature",
    "top_p",
    "max_tokens",
    "requested_top_logprobs",
    "stop_token_ids",
    "sampling_top_k",
    "generation_protocol",
    "python_hash_seed",
)
MODEL_SAMPLING_FIELDS = (
    "num_samples",
    "ensemble_sizes",
    "temperature",
    "top_p",
    "max_tokens",
    "requested_top_logprobs",
    "stop_token_ids",
    "sampling_top_k",
    "generation_protocol",
    "weakest_k",
)
METHOD_COLUMNS = OrderedDict(
    [
        ("PWSC-CI", ("pwsc_ci_risk",)),
        ("Mean NLL", ("mean_nll",)),
        ("Worst-k NLL (k=5)", ("worst_5_nll",)),
        (
            "Self-Consistency",
            ("baseline_self_consistency", "self_consistency_risk"),
        ),
        (
            "Semantic Entropy",
            ("baseline_semantic_entropy", "semantic_entropy_risk"),
        ),
        ("Weighted MBR", ("baseline_mbr", "mbr_risk")),
        (
            "Target Consistency (task-kernel adaptation)",
            (
                "baseline_target_consistency_task_kernel_adaptation",
                "target_consistency_risk",
            ),
        ),
        (
            "KLE (task-kernel adaptation)",
            ("baseline_kle_task_kernel_adaptation", "kle_task_kernel_risk"),
        ),
        (
            "CoCoA-MSP (outcome-kernel adaptation)",
            (
                "baseline_cocoa_msp_outcome_kernel_adaptation",
                "cocoa_msp_risk",
            ),
        ),
        (
            "CoCoA-PPL (outcome-kernel adaptation)",
            (
                "baseline_cocoa_ppl_outcome_kernel_adaptation",
                "cocoa_ppl_risk",
            ),
        ),
    ]
)
PLOT_METHODS = (
    "PWSC-CI",
    "Target Consistency (task-kernel adaptation)",
    "Semantic Entropy",
    "Worst-k NLL (k=5)",
    "Mean NLL",
)
METRICS = OrderedDict(
    [
        ("AUROC", ScientificSignalAnalyzer.calculate_auroc),
        ("AUPRC", ScientificSignalAnalyzer.calculate_auprc),
        ("AURC", ScientificSignalAnalyzer.calculate_aurc),
        ("SelectiveAcc@50", ScientificSignalAnalyzer.calculate_selective_accuracy),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve_bound_file(
    run_dir: Path, recorded: str, expected_sha256: str, label: str
) -> Path:
    basename = Path(recorded).name
    candidates = [run_dir / basename]
    if len(run_dir.parents) >= 3:
        bundle_root = run_dir.parents[2]
        candidates.append(
            bundle_root
            / "generation_cache"
            / run_dir.parent.name
            / run_dir.name
            / basename
        )
    candidates.append(Path(recorded).expanduser())

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and _sha256(candidate) == expected_sha256:
            return candidate
    raise ValueError(f"{label} is missing or hash-mismatched")


def _single_manifest(run_dir: Path, task: str) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"run directory does not exist: {run_dir}; check --run-dir and SUP_ROOT"
        )
    paths = sorted(run_dir.glob(f"manifest_{task}_*.json"))
    if len(paths) != 1:
        raise ValueError(
            f"expected one {task} manifest under {run_dir}; found {len(paths)}"
        )
    return paths[0]


def _bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _method_column(fieldnames: Sequence[str], aliases: Sequence[str]) -> str:
    available = [name for name in aliases if name in fieldnames]
    if len(available) != 1:
        raise ValueError(
            f"expected exactly one whitelisted score column from {aliases}; "
            f"found {available}"
        )
    return available[0]


def _sampling_contract_value(values: Mapping[str, Any], field: str) -> Any:
    value = values.get(field)
    if field == "stop_token_ids" and value is None:
        return []
    return value


def _model_sampling_signature(sampling: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(
        json.dumps(_sampling_contract_value(sampling, field), sort_keys=True)
        for field in MODEL_SAMPLING_FIELDS
    )


def _validate_cache_sampling_contract(
    cache_header: Mapping[str, Any], manifest: Mapping[str, Any], task: str
) -> None:
    sampling = manifest.get("sampling") or {}
    for field in CACHE_BOUND_SAMPLING_FIELDS:
        if _sampling_contract_value(
            cache_header, field
        ) != _sampling_contract_value(sampling, field):
            raise ValueError(f"cache sampling contract mismatch for {task}: {field}")


def _load_run_task(
    run_dir: Path,
    task: str,
    ensemble_size: int,
    loader: UniversalDatasetLoader,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = _single_manifest(run_dir, task)
    manifest = _load_json(manifest_path)
    if manifest.get("task") != task:
        raise ValueError(f"task mismatch in {manifest_path}")
    if manifest.get("publication_eligible") is not True:
        raise ValueError(f"publication-ineligible run in {manifest_path}")
    if manifest.get("publication_ineligibility_reasons"):
        raise ValueError(f"publication ineligibility reasons present in {manifest_path}")
    if (
        manifest.get("evaluation", {}).get("pwsc_ci_contract")
        != PWSC_CI_CONTRACT_VERSION
    ):
        raise ValueError(f"noncanonical PWSC-CI contract in {manifest_path}")
    dataset = loader.dataset_fingerprint(task)
    samples = loader.load_any_dataset(task, 0)
    if manifest.get("dataset_fingerprint") != dataset:
        raise ValueError(f"dataset fingerprint drift for {task}")
    if manifest.get("input_sample_sha256") != _input_sha256(samples):
        raise ValueError(f"input sample hash drift for {task}")
    if str(manifest.get("seed")) != str(
        manifest.get("sampling", {}).get("python_hash_seed")
    ):
        raise ValueError(f"Python hash seed mismatch for {task}")
    cache_path = _resolve_bound_file(
        run_dir,
        str(manifest["generation_cache_path"]),
        str(manifest["generation_cache_sha256"]),
        f"{task} generation cache",
    )
    csv_path = _resolve_bound_file(
        run_dir,
        str(manifest["csv_path"]),
        str(manifest["csv_sha256"]),
        f"{task} sample CSV",
    )
    report_path = _resolve_bound_file(
        run_dir,
        str(manifest["report_path"]),
        str(manifest["report_sha256"]),
        f"{task} report",
    )
    with cache_path.open(encoding="utf-8") as handle:
        cache_header = json.loads(handle.readline())
    if cache_header.get("task") != task or cache_header.get("seed") != manifest["seed"]:
        raise ValueError(f"cache header run identity mismatch for {task}")
    if cache_header.get("dataset") != dataset:
        raise ValueError(f"cache dataset binding mismatch for {task}")
    if cache_header.get("model", {}).get("id") != manifest.get("model_identifier"):
        raise ValueError(f"model ID mismatch for {task}")
    _validate_cache_sampling_contract(cache_header, manifest, task)
    report = _load_json(report_path)
    if report.get("task") != task or report.get("publication_eligible") is not True:
        raise ValueError(f"report publication identity mismatch for {task}")
    if report.get("n_skipped") != 0:
        raise ValueError(f"report contains skipped questions for {task}")
    if report.get("n_scored_questions") != len(samples):
        raise ValueError(f"report split coverage mismatch for {task}")

    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        columns = {
            method: _method_column(reader.fieldnames, aliases)
            for method, aliases in METHOD_COLUMNS.items()
        }
        for raw in reader:
            if int(raw["ensemble_size"]) != ensemble_size:
                continue
            row: Dict[str, Any] = {
                "sample_id": str(raw["sample_id"]),
                "task": task,
                "error": int(not _bool(raw["base_is_correct"])),
            }
            for method, column in columns.items():
                value = raw.get(column, "")
                row[method] = float(value) if str(value).strip() else None
            for name in (
                "valid_path_count",
                "pwsc_ci_probe_count",
                "pwsc_ci_active_probe_count",
                "pwsc_ci_successful_calls",
                "pwsc_ci_kernel_source",
            ):
                if name in raw and raw[name] != "":
                    row[name] = raw[name]
            rows.append(row)
    expected_ids = {sample.sample_id for sample in samples}
    if {row["sample_id"] for row in rows} != expected_ids:
        raise ValueError(f"N={ensemble_size} CSV rows do not cover {task}")
    provenance = {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "csv_path": str(csv_path),
        "csv_sha256": _sha256(csv_path),
        "cache_path": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "model_identifier": manifest.get("model_identifier"),
        "model_revision": manifest.get("model_revision"),
        "seed": int(manifest["seed"]),
        "dataset_fingerprint": dataset,
        "input_sample_sha256": manifest.get("input_sample_sha256"),
        "timing_seconds": manifest.get("timing_seconds"),
        "sampling": manifest.get("sampling"),
    }
    return provenance, rows


def _metrics(rows: Sequence[Mapping[str, Any]], method: str) -> Dict[str, float]:
    values = [row for row in rows if row.get(method) is not None]
    labels = np.asarray([int(row["error"]) for row in values], dtype=np.int64)
    scores = np.asarray([float(row[method]) for row in values], dtype=np.float64)
    return {name: float(function(labels, scores)) for name, function in METRICS.items()}


def _run_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["model_key"]), int(row["seed"]), str(row["task"]))].append(row)
    task_metrics: Dict[str, Any] = {}
    macro_inputs: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (model_key, seed, task), rows in sorted(grouped.items()):
        methods = {
            method: _metrics(rows, method)
            for method in METHOD_COLUMNS
            if all(row.get(method) is not None for row in rows)
        }
        key = f"{model_key}|seed={seed}|{task}"
        task_metrics[key] = {"question_count": len(rows), "methods": methods}
        macro_inputs[(model_key, seed)][task] = methods
    run_macro: Dict[str, Any] = {}
    for (model_key, seed), tasks in sorted(macro_inputs.items()):
        methods: Dict[str, Any] = {}
        for method in METHOD_COLUMNS:
            admitted = [values[method] for values in tasks.values() if method in values]
            if admitted:
                methods[method] = {
                    metric: float(np.mean([item[metric] for item in admitted]))
                    for metric in METRICS
                }
                methods[method]["task_count"] = len(admitted)
        run_macro[f"{model_key}|seed={seed}"] = methods
    model_summary: Dict[str, Any] = {}
    model_keys = sorted({key[0] for key in macro_inputs})
    for model_key in model_keys:
        seed_keys = [key for key in run_macro if key.startswith(model_key + "|seed=")]
        methods: Dict[str, Any] = {}
        for method in METHOD_COLUMNS:
            admitted = [run_macro[key][method] for key in seed_keys if method in run_macro[key]]
            if not admitted:
                continue
            methods[method] = {
                metric: {
                    "mean": float(np.mean([item[metric] for item in admitted])),
                    "seed_std": (
                        float(np.std([item[metric] for item in admitted], ddof=1))
                        if len(admitted) > 1
                        else None
                    ),
                }
                for metric in METRICS
            }
            methods[method]["seed_count"] = len(admitted)
        model_summary[model_key] = methods
    return {
        "task_metrics": task_metrics,
        "run_macro_metrics": run_macro,
        "model_seed_summary": model_summary,
    }


def _paired_bootstrap(
    records: Sequence[Mapping[str, Any]], draws: int, seed: int
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    by_model_group: Dict[
        str, Dict[Tuple[int, str], List[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_model_group[str(row["model_key"])][
            (int(row["seed"]), str(row["task"]))
        ].append(row)
    output: Dict[str, Any] = {}
    for model_key, groups in by_model_group.items():
        model_output: Dict[str, Any] = {}
        for method in list(METHOD_COLUMNS)[1:]:
            eligible = {
                key: rows
                for key, rows in groups.items()
                if all(row.get(method) is not None for row in rows)
            }
            if not eligible:
                continue
            points = {metric: [] for metric in ("AUROC", "AURC")}
            for rows in eligible.values():
                labels = np.asarray([int(row["error"]) for row in rows])
                left = np.asarray([float(row["PWSC-CI"]) for row in rows])
                right = np.asarray([float(row[method]) for row in rows])
                points["AUROC"].append(
                    ScientificSignalAnalyzer.calculate_auroc(labels, left)
                    - ScientificSignalAnalyzer.calculate_auroc(labels, right)
                )
                points["AURC"].append(
                    ScientificSignalAnalyzer.calculate_aurc(labels, left)
                    - ScientificSignalAnalyzer.calculate_aurc(labels, right)
                )
            sampled = {metric: [] for metric in points}
            attempts = 0
            while len(sampled["AUROC"]) < draws:
                attempts += 1
                if attempts > draws * 20:
                    raise RuntimeError("too many undefined paired bootstrap draws")
                differences = {metric: [] for metric in points}
                valid = True
                for rows in eligible.values():
                    indices = rng.integers(0, len(rows), len(rows))
                    labels = np.asarray([int(rows[index]["error"]) for index in indices])
                    left = np.asarray([float(rows[index]["PWSC-CI"]) for index in indices])
                    right = np.asarray([float(rows[index][method]) for index in indices])
                    values = {
                        "AUROC": ScientificSignalAnalyzer.calculate_auroc(labels, left)
                        - ScientificSignalAnalyzer.calculate_auroc(labels, right),
                        "AURC": ScientificSignalAnalyzer.calculate_aurc(labels, left)
                        - ScientificSignalAnalyzer.calculate_aurc(labels, right),
                    }
                    if not all(np.isfinite(value) for value in values.values()):
                        valid = False
                        break
                    for metric, value in values.items():
                        differences[metric].append(value)
                if valid:
                    for metric in sampled:
                        sampled[metric].append(float(np.mean(differences[metric])))
            model_output[method] = {
                "seed_task_group_count": len(eligible),
                "paired_difference": {
                    metric: {
                        "estimate": float(np.mean(points[metric])),
                        "ci95": [
                            float(np.quantile(sampled[metric], 0.025)),
                            float(np.quantile(sampled[metric], 0.975)),
                        ],
                    }
                    for metric in sampled
                },
            }
        output[model_key] = model_output
    return output


def _selective_error_at(
    rows: Sequence[Mapping[str, Any]], method: str, coverage: float
) -> float:
    return float(_selective_error_curve(rows, method, (coverage,))[0])


def _selective_error_curve(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    coverages: Sequence[float],
) -> np.ndarray:
    """Sort once and reuse cumulative errors for every requested coverage."""
    ordered = sorted(
        rows, key=lambda row: (float(row[method]), str(row["sample_id"]))
    )
    cumulative = np.cumsum(
        np.asarray([int(row["error"]) for row in ordered], dtype=np.float64)
    )
    keep = np.clip(
        np.ceil(np.asarray(coverages, dtype=np.float64) * len(ordered)).astype(int),
        1,
        len(ordered),
    )
    return cumulative[keep - 1] / keep


def _risk_coverage(
    records: Sequence[Mapping[str, Any]], draws: int, seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    coverages = np.linspace(0.05, 1.0, 20)
    by_model_group: Dict[
        str, Dict[Tuple[int, str], List[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_model_group[str(row["model_key"])][
            (int(row["seed"]), str(row["task"]))
        ].append(row)
    curve_rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    for model_index, (model_key, groups) in enumerate(sorted(by_model_group.items())):
        model_meta: Dict[str, Any] = {}
        for method in METHOD_COLUMNS:
            eligible = {
                key: rows
                for key, rows in groups.items()
                if all(row.get(method) is not None for row in rows)
            }
            if not eligible:
                continue
            point = np.mean(
                [
                    _selective_error_curve(rows, method, coverages)
                    for rows in eligible.values()
                ],
                axis=0,
            )
            rng = np.random.default_rng(seed + model_index * 1009 + len(curve_rows))
            sampled = np.empty((draws, len(coverages)), dtype=np.float64)
            for draw in range(draws):
                group_curves = []
                for rows in eligible.values():
                    chosen = rng.integers(0, len(rows), len(rows))
                    sampled_rows = [rows[index] for index in chosen]
                    group_curves.append(
                        _selective_error_curve(sampled_rows, method, coverages)
                    )
                sampled[draw] = np.mean(group_curves, axis=0)
            low = np.quantile(sampled, 0.025, axis=0)
            high = np.quantile(sampled, 0.975, axis=0)
            for index, coverage in enumerate(coverages):
                curve_rows.append(
                    {
                        "model_key": model_key,
                        "method": method,
                        "coverage": float(coverage),
                        "selective_error": float(point[index]),
                        "ci95_low": float(low[index]),
                        "ci95_high": float(high[index]),
                    }
                )
            model_meta[method] = {"seed_task_group_count": len(eligible)}
        metadata[model_key] = model_meta
    return curve_rows, metadata


def _failure_cases(
    records: Sequence[Mapping[str, Any]], count: int
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["model_key"]), int(row["seed"]), str(row["task"]))].append(row)
    output: List[Dict[str, Any]] = []
    for (model_key, seed, task), rows in sorted(grouped.items()):
        missed = sorted(
            (row for row in rows if int(row["error"]) == 1),
            key=lambda row: (float(row["PWSC-CI"]), str(row["sample_id"])),
        )[:count]
        alarms = sorted(
            (row for row in rows if int(row["error"]) == 0),
            key=lambda row: (-float(row["PWSC-CI"]), str(row["sample_id"])),
        )[:count]
        for failure_type, selected in (
            ("low-risk error", missed),
            ("high-risk correct", alarms),
        ):
            for rank, row in enumerate(selected, start=1):
                output.append(
                    {
                        "model_key": model_key,
                        "seed": seed,
                        "task": task,
                        "failure_type": failure_type,
                        "rank": rank,
                        "sample_id": row["sample_id"],
                        "error": row["error"],
                        "pwsc_ci_risk": row["PWSC-CI"],
                    }
                )
    return output


def _diagnostics(records: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> Dict[str, Any]:
    by_model: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_model[str(row["model_key"])].append(row)
    output: Dict[str, Any] = {}
    for model_key, rows in by_model.items():
        sources = Counter(
            str(row["pwsc_ci_kernel_source"])
            for row in rows
            if row.get("pwsc_ci_kernel_source") is not None
        )
        valid = [float(row["valid_path_count"]) for row in rows if row.get("valid_path_count")]
        timing = [
            item["timing_seconds"]
            for item in provenance.values()
            if item["model_key"] == model_key and item.get("timing_seconds")
        ]
        output[model_key] = {
            "kernel_source_counts": dict(sources),
            "mean_valid_path_count": float(np.mean(valid)) if valid else None,
            "timing_records": timing,
        }
    return output


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.2,
            "axes.linewidth": 0.65,
            "legend.fontsize": 6.3,
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "pwsc-ci-supplement-v1",
        }
    )


def _plot_curves(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Tuple[Path, Path]:
    _configure_matplotlib()
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    models = sorted({str(row["model_key"]) for row in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(3.45 * len(models), 2.55), squeeze=False)
    colors = ["#006D77", "#D55E00", "#3A6F5C", "#6B7280", "#8F3D56", "#5E548E", "#2A6F97", "#7A6C2F", "#444444", "#B5651D"]
    for axis, model_key in zip(axes[0], models):
        model_rows = [row for row in rows if row["model_key"] == model_key]
        methods = [
            method
            for method in PLOT_METHODS
            if any(row["method"] == method for row in model_rows)
        ]
        for index, method in enumerate(methods):
            values = sorted(
                (row for row in model_rows if row["method"] == method),
                key=lambda row: row["coverage"],
            )
            x = np.asarray([row["coverage"] for row in values])
            y = np.asarray([row["selective_error"] for row in values])
            low = np.asarray([row["ci95_low"] for row in values])
            high = np.asarray([row["ci95_high"] for row in values])
            color = colors[index % len(colors)]
            axis.plot(x, y, color=color, label=method, zorder=3)
            axis.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
        axis.set_title(model_key.rsplit("/", 1)[-1])
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Selective error")
        axis.set_xlim(0.05, 1.0)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.45, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=1, loc="upper left")
    fig.tight_layout(pad=0.5)
    pdf_path = output_dir / "risk_coverage.pdf"
    svg_path = output_dir / "risk_coverage.svg"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Creator": "PWSC-CI supplemental summarizer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Creator": "PWSC-CI supplemental summarizer", "Date": None},
    )
    plt.close(fig)
    return pdf_path, svg_path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize(args: argparse.Namespace) -> Dict[str, str]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = UniversalDatasetLoader(args.dataset_root)
    records: List[Dict[str, Any]] = []
    provenance: Dict[str, Any] = {}
    seen_runs = set()
    model_revisions: Dict[str, str] = {}
    model_sampling: Dict[str, Tuple[Any, ...]] = {}
    for run_index, raw_dir in enumerate(args.run_dir):
        run_dir = raw_dir.expanduser().resolve()
        run_identity = None
        for task in args.tasks:
            task_provenance, task_rows = _load_run_task(
                run_dir, task, args.ensemble_size, loader
            )
            identity = (
                task_provenance["model_identifier"],
                task_provenance["model_revision"],
                task_provenance["seed"],
            )
            if run_identity is None:
                run_identity = identity
            elif identity != run_identity:
                raise ValueError(f"run directory mixes model/seed identities: {run_dir}")
            model_key = str(task_provenance["model_identifier"])
            revision = str(task_provenance["model_revision"])
            if model_key in model_revisions and model_revisions[model_key] != revision:
                raise ValueError(
                    f"model ID {model_key!r} appears with multiple revisions"
                )
            model_revisions[model_key] = revision
            sampling = task_provenance["sampling"] or {}
            sampling_signature = _model_sampling_signature(sampling)
            if (
                model_key in model_sampling
                and model_sampling[model_key] != sampling_signature
            ):
                raise ValueError(
                    f"model {model_key!r} appears with multiple sampling contracts"
                )
            model_sampling[model_key] = sampling_signature
            task_provenance["model_key"] = model_key
            provenance[f"run={run_index}|{task}"] = task_provenance
            for row in task_rows:
                row.update(
                    {
                        "run_index": run_index,
                        "model_key": model_key,
                        "model_revision": task_provenance["model_revision"],
                        "seed": task_provenance["seed"],
                    }
                )
                records.append(row)
        if run_identity in seen_runs:
            raise ValueError(f"duplicate model/revision/seed run: {run_identity}")
        seen_runs.add(run_identity)

    metrics = _run_metrics(records)
    paired = _paired_bootstrap(records, args.bootstrap, args.bootstrap_seed)
    curve_rows, curve_metadata = _risk_coverage(
        records, args.curve_bootstrap, args.bootstrap_seed + 1
    )
    failures = _failure_cases(records, args.failure_count)
    metrics_csv_rows = []
    for key, value in metrics["task_metrics"].items():
        for method, method_metrics in value["methods"].items():
            metrics_csv_rows.append(
                {"scope": "task", "group": key, "method": method, **method_metrics}
            )
    for key, methods in metrics["run_macro_metrics"].items():
        for method, method_metrics in methods.items():
            metrics_csv_rows.append(
                {"scope": "run_macro", "group": key, "method": method, **method_metrics}
            )

    metrics_csv = output_dir / "metrics.csv"
    curves_csv = output_dir / "risk_coverage.csv"
    failures_csv = output_dir / "failure_cases.csv"
    _write_csv(metrics_csv, metrics_csv_rows)
    _write_csv(curves_csv, curve_rows)
    _write_csv(failures_csv, failures)
    pdf_path, svg_path = _plot_curves(curve_rows, output_dir)
    summary_path = output_dir / "summary.json"
    summary = {
        "schema": "pwsc-ci-supplemental-summary-v1",
        "canonical_method": "PWSC-CI-v1.0",
        "canonical_contract": PWSC_CI_CONTRACT_VERSION,
        "aggregation_rule": (
            "model families are separate; task metrics are equal-weighted within "
            "each seed and seed macros are equal-weighted within each model"
        ),
        "tasks": list(args.tasks),
        "ensemble_size": args.ensemble_size,
        "bootstrap": {
            "paired_draws": args.bootstrap,
            "risk_coverage_draws": args.curve_bootstrap,
            "seed": args.bootstrap_seed,
        },
        **metrics,
        "paired_bootstrap": paired,
        "risk_coverage": curve_metadata,
        "runtime_and_stability": _diagnostics(records, provenance),
        "failure_case_count": len(failures),
        "provenance": provenance,
        "artifacts": {},
    }
    artifacts = {
        "metrics_csv": metrics_csv,
        "risk_coverage_csv": curves_csv,
        "failure_cases_csv": failures_csv,
        "risk_coverage_pdf": pdf_path,
        "risk_coverage_svg": svg_path,
    }
    summary["artifacts"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in artifacts.items()
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = output_dir / "summary.manifest.json"
    manifest = {
        "schema": "pwsc-ci-supplemental-summary-manifest-v1",
        "command": [str(item) for item in sys.argv],
        "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        "artifacts": summary["artifacts"],
        "source_provenance": provenance,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "risk_coverage_pdf": str(pdf_path),
        "risk_coverage_svg": str(svg_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--dataset-root", default="/data/newmodel/graphunagent/datasets")
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--curve-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260732)
    parser.add_argument("--failure-count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.ensemble_size, args.bootstrap, args.curve_bootstrap, args.failure_count) < 1:
        raise SystemExit("numeric arguments must be positive")
    print(json.dumps(summarize(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
