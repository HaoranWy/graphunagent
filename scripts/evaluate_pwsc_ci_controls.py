#!/usr/bin/env python3
"""Evaluate single-factor PWSC-CI controls from immutable generation caches.

This script never generates text. It validates each runner manifest, dataset,
cache, report, and sample CSV before recomputing diagnostic controls. The only
canonical method remains PWSC-CI; every other score is explicitly a control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.behavioral_kernel import build_behavior_kernel  # noqa: E402
from core.risk_estimator import (  # noqa: E402
    PWSC_CI_CONTRACT_VERSION,
    PWSCCIRiskEstimator,
    target_conditional_information_components,
)
from diagnostics.analyzer import ScientificSignalAnalyzer  # noqa: E402
from routing.baselines import SCSVBaselines  # noqa: E402
from run_pipeline import (  # noqa: E402
    UniversalDatasetLoader,
    _input_sha256,
    _sample_metadata,
    _verifier_id,
)
from routing.verifier_manager import VerifierManager  # noqa: E402


DEFAULT_TASKS = ("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO")
METRICS = {
    "AUROC": ScientificSignalAnalyzer.calculate_auroc,
    "AUPRC": ScientificSignalAnalyzer.calculate_auprc,
    "AURC": ScientificSignalAnalyzer.calculate_aurc,
    "SelectiveAcc@50": ScientificSignalAnalyzer.calculate_selective_accuracy,
}


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
    candidates = [Path(recorded).expanduser(), run_dir / Path(recorded).name]
    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == expected_sha256:
            return candidate.resolve()
    raise ValueError(
        f"{label} is missing or hash-mismatched: {recorded} / {expected_sha256}"
    )


def _single_manifest(run_dir: Path, task: str) -> Path:
    paths = sorted(run_dir.glob(f"manifest_{task}_*.json"))
    if len(paths) != 1:
        raise ValueError(
            f"expected exactly one {task} manifest under {run_dir}, found {len(paths)}"
        )
    return paths[0]


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def _parse_int_list(value: str) -> Tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def _parse_weakest_k(value: str) -> Tuple[Tuple[str, int], ...]:
    parsed: List[Tuple[str, int]] = []
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "all":
            parsed.append(("all", 2**31 - 1))
        else:
            number = int(token)
            if number < 1:
                raise argparse.ArgumentTypeError("weakest-k values must be positive")
            parsed.append((str(number), number))
    if not parsed:
        raise argparse.ArgumentTypeError("weakest-k sweep cannot be empty")
    return tuple(parsed)


def _read_fixed_rows(
    csv_path: Path, ensemble_size: int
) -> Tuple[Dict[str, bool], Dict[str, float]]:
    values: Dict[str, bool] = {}
    risks: Dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["ensemble_size"]) != ensemble_size:
                continue
            raw = str(row.get("base_is_correct", "")).strip().lower()
            if raw not in {"true", "false", "1", "0"}:
                raise ValueError(f"invalid correctness value in {csv_path}: {raw!r}")
            sample_id = str(row["sample_id"])
            values[sample_id] = raw in {"true", "1"}
            risks[sample_id] = float(row["pwsc_ci_risk"])
    if not values:
        raise ValueError(f"no N={ensemble_size} rows in {csv_path}")
    return values, risks


def _load_task_inputs(
    run_dir: Path,
    task: str,
    loader: UniversalDatasetLoader,
    ensemble_size: int,
) -> Tuple[
    Dict[str, Any],
    Path,
    Path,
    List[Any],
    Dict[str, bool],
    Dict[str, float],
]:
    manifest_path = _single_manifest(run_dir, task)
    manifest = _load_json(manifest_path)
    if manifest.get("task") != task:
        raise ValueError(f"manifest task mismatch: {manifest_path}")
    contract = manifest.get("evaluation", {}).get("pwsc_ci_contract")
    if contract != PWSC_CI_CONTRACT_VERSION:
        raise ValueError(f"manifest has a noncanonical score contract: {contract}")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("num_samples", 0)) < ensemble_size:
        raise ValueError(f"{task} cache has fewer than {ensemble_size} paths")
    if str(sampling.get("python_hash_seed")) != str(manifest.get("seed")):
        raise ValueError(f"{task} Python hash seed is not bound to the run seed")

    dataset = loader.dataset_fingerprint(task)
    if dataset != manifest.get("dataset_fingerprint"):
        raise ValueError(f"{task} local dataset fingerprint has drifted")
    samples = loader.load_any_dataset(task, 0)
    if _input_sha256(samples) != manifest.get("input_sample_sha256"):
        raise ValueError(f"{task} input sample hash has drifted")

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
    _resolve_bound_file(
        run_dir,
        str(manifest["report_path"]),
        str(manifest["report_sha256"]),
        f"{task} report",
    )
    with cache_path.open(encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    for key, expected in (
        ("task", task),
        ("seed", manifest["seed"]),
        ("dataset", dataset),
        ("input_sha256", manifest["input_sample_sha256"]),
    ):
        if header.get(key) != expected:
            raise ValueError(f"{task} cache header mismatch for {key}")
    if header.get("model", {}).get("id") != manifest.get("model_identifier"):
        raise ValueError(f"{task} model ID differs between cache and manifest")
    labels, frozen_risks = _read_fixed_rows(csv_path, ensemble_size)
    if set(labels) != {sample.sample_id for sample in samples}:
        raise ValueError(f"{task} sample CSV does not cover the complete split")
    provenance = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "cache_path": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "csv_path": str(csv_path),
        "csv_sha256": _sha256(csv_path),
        "model_identifier": manifest.get("model_identifier"),
        "model_revision": manifest.get("model_revision"),
        "seed": manifest.get("seed"),
        "dataset_fingerprint": dataset,
        "input_sample_sha256": manifest.get("input_sample_sha256"),
    }
    return provenance, cache_path, csv_path, samples, labels, frozen_risks


def _iter_cache(cache_path: Path, samples: Sequence[Any]) -> Iterable[Tuple[Any, Any]]:
    with cache_path.open(encoding="utf-8") as handle:
        next(handle)
        count = 0
        for sample, line in zip(samples, handle):
            record = json.loads(line)
            if record.get("sample_id") != sample.sample_id:
                raise ValueError(f"cache order mismatch at {sample.sample_id}")
            count += 1
            yield sample, record.get("traces", [])
        if count != len(samples) or any(line.strip() for line in handle):
            raise ValueError("generation cache row count does not match the dataset")


def _uniform_peer_mass(q: np.ndarray, target_index: int) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64).copy()
    mask = np.ones(values.size, dtype=bool)
    mask[target_index] = False
    active = mask & (values > 0.0)
    mass = float(values[active].sum())
    values[mask] = 0.0
    if np.any(active):
        values[active] = mass / int(active.sum())
    return values


def _q_for_k(
    paths: Sequence[Sequence[Any]], valid_mask: np.ndarray, weakest_k: int
) -> np.ndarray:
    estimator = PWSCCIRiskEstimator(weakest_k=weakest_k)
    confidences = np.asarray(
        [estimator.path_action(path).confidence for path in paths], dtype=np.float64
    )
    return confidences * np.asarray(valid_mask, dtype=bool)


def _score_controls(
    logprobs: Sequence[Sequence[Any]],
    texts: Sequence[str],
    task: str,
    metadata: Mapping[str, Any],
    target_index: int,
    weakest_k: Sequence[Tuple[str, int]],
    probe_counts: Sequence[int],
    recovery_modes: Sequence[bool],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    estimator = PWSCCIRiskEstimator(weakest_k=5)
    _fixed_risk, diagnostics = estimator.score(
        logprobs, texts, task, [dict(metadata)] * len(texts)
    )
    q = np.asarray(diagnostics["q"], dtype=np.float64)
    kernel = np.asarray(diagnostics["kernel"], dtype=np.float64)
    task_kernel = np.asarray(diagnostics["task_kernel"], dtype=np.float64)
    components = target_conditional_information_components(
        q, kernel, target_index=target_index
    )
    controls: Dict[str, float] = {
        "pwsc_ci_risk": float(components["normalized_information"]),
        "control_raw_holevo": float(components["holevo_information"]),
        "control_ci_task_kernel": float(
            target_conditional_information_components(
                q, task_kernel, target_index=target_index
            )["normalized_information"]
        ),
        "control_target_consistency_same_kernel_weighted": (
            SCSVBaselines.calculate_target_consistency(
                kernel, weights=q, target_index=target_index
            )
        ),
        "control_target_consistency_same_kernel_uniform": (
            SCSVBaselines.calculate_target_consistency(
                kernel, target_index=target_index
            )
        ),
        "control_ci_uniform_peer_mass": float(
            target_conditional_information_components(
                _uniform_peer_mass(q, target_index),
                kernel,
                target_index=target_index,
            )["normalized_information"]
        ),
    }
    for label, value in weakest_k:
        q_k = _q_for_k(logprobs, diagnostics["valid_mask"], value)
        controls[f"control_ci_weakest_k_{label}"] = float(
            target_conditional_information_components(
                q_k, kernel, target_index=target_index
            )["normalized_information"]
        )

    probe_diagnostics: Dict[str, Any] = {}
    if diagnostics["domain_family"] == "code":
        prompt = str(metadata.get("prompt", ""))
        entry_point = str(metadata.get("entry_point", ""))
        for probe_count in probe_counts:
            for recovery in recovery_modes:
                label = f"p{probe_count}_recovery_{'on' if recovery else 'off'}"
                behavior, info = build_behavior_kernel(
                    texts,
                    diagnostics["valid_mask"],
                    prompt,
                    entry_point,
                    max_probes=probe_count,
                    recover_single_unsupported=recovery,
                )
                selected = task_kernel if behavior is None else behavior
                controls[f"control_ci_behavior_{label}"] = float(
                    target_conditional_information_components(
                        q, selected, target_index=target_index
                    )["normalized_information"]
                )
                probe_diagnostics[label] = info
    return controls, {
        "kernel_source": diagnostics["kernel_source"],
        "probe_diagnostics": probe_diagnostics,
    }


def _metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {name: float(metric(labels, scores)) for name, metric in METRICS.items()}


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tasks = sorted({str(row["task"]) for row in rows})
    score_columns = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if (key == "pwsc_ci_risk" or key.startswith("control_"))
            and value not in {None, ""}
        }
    )
    by_task: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        labels = np.asarray([int(row["error"]) for row in task_rows], dtype=np.int64)
        methods: Dict[str, Any] = {}
        for column in score_columns:
            if not all(row.get(column) not in {None, ""} for row in task_rows):
                continue
            scores = np.asarray([float(row[column]) for row in task_rows])
            methods[column] = _metrics(labels, scores)
        by_task[task] = {
            "row_count": len(task_rows),
            "question_count": len({str(row["sample_id"]) for row in task_rows}),
            "methods": methods,
        }
    macro: Dict[str, Any] = {}
    for column in score_columns:
        admitted = [
            values["methods"][column]
            for values in by_task.values()
            if column in values["methods"]
        ]
        if admitted:
            macro[column] = {
                metric: float(np.mean([value[metric] for value in admitted]))
                for metric in METRICS
            }
            macro[column]["task_count"] = len(admitted)
    return {"task_metrics": by_task, "macro_metrics": macro}


def _paired_macro_bootstrap(
    rows: Sequence[Mapping[str, Any]], draws: int, seed: int
) -> Dict[str, Any]:
    by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    methods = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.startswith("control_") and value not in {None, ""}
        }
    )
    rng = np.random.default_rng(seed)
    output: Dict[str, Any] = {}
    for method in methods:
        eligible = {
            task: values
            for task, values in by_task.items()
            if all(row.get(method) not in {None, ""} for row in values)
        }
        if not eligible:
            continue
        samples = {metric: np.empty(draws, dtype=np.float64) for metric in ("AUROC", "AURC")}
        for draw in range(draws):
            task_differences = {metric: [] for metric in samples}
            for task_rows in eligible.values():
                grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
                for row in task_rows:
                    grouped[str(row["sample_id"])].append(row)
                ids = sorted(grouped)
                chosen = rng.integers(0, len(ids), len(ids))
                sampled_rows = [row for index in chosen for row in grouped[ids[index]]]
                labels = np.asarray([int(row["error"]) for row in sampled_rows])
                canonical = np.asarray([float(row["pwsc_ci_risk"]) for row in sampled_rows])
                control = np.asarray([float(row[method]) for row in sampled_rows])
                for metric, function in (
                    ("AUROC", ScientificSignalAnalyzer.calculate_auroc),
                    ("AURC", ScientificSignalAnalyzer.calculate_aurc),
                ):
                    difference = float(function(labels, canonical) - function(labels, control))
                    task_differences[metric].append(difference)
            for metric in samples:
                samples[metric][draw] = float(np.mean(task_differences[metric]))
        point_tasks = []
        for task_rows in eligible.values():
            labels = np.asarray([int(row["error"]) for row in task_rows])
            canonical = np.asarray([float(row["pwsc_ci_risk"]) for row in task_rows])
            control = np.asarray([float(row[method]) for row in task_rows])
            point_tasks.append(
                {
                    metric: float(function(labels, canonical) - function(labels, control))
                    for metric, function in (
                        ("AUROC", ScientificSignalAnalyzer.calculate_auroc),
                        ("AURC", ScientificSignalAnalyzer.calculate_aurc),
                    )
                }
            )
        output[method] = {
            "task_count": len(eligible),
            "paired_macro_difference": {
                metric: {
                    "estimate": float(np.mean([item[metric] for item in point_tasks])),
                    "ci95": [
                        float(np.quantile(values, 0.025)),
                        float(np.quantile(values, 0.975)),
                    ],
                }
                for metric, values in samples.items()
            },
        }
    return output


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


def evaluate(args: argparse.Namespace) -> Dict[str, str]:
    started = time.perf_counter()
    run_dir = args.run_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    manifest_path = output_path.with_name(output_path.stem + ".manifest.json")
    loader = UniversalDatasetLoader(args.dataset_root)
    verifier_manager = VerifierManager()
    weakest_k = _parse_weakest_k(args.weakest_k)
    rows: List[Dict[str, Any]] = []
    provenance: Dict[str, Any] = {}
    kernel_sources: Dict[str, Counter[str]] = defaultdict(Counter)
    probe_totals: Dict[str, Counter[str]] = defaultdict(Counter)

    for task in args.tasks:
        task_started = time.perf_counter()
        (
            task_provenance,
            cache_path,
            _csv_path,
            samples,
            fixed_labels,
            frozen_risks,
        ) = (
            _load_task_inputs(run_dir, task, loader, args.ensemble_size)
        )
        provenance[task] = task_provenance
        verifier = verifier_manager.verifiers[_verifier_id(task)]
        for sample, traces in _iter_cache(cache_path, samples):
            prefix = list(traces[: args.ensemble_size])
            if len(prefix) != args.ensemble_size:
                raise ValueError(f"{task}/{sample.sample_id} has an incomplete ensemble")
            texts = [str(trace.get("text", "")) for trace in prefix]
            logprobs = [list(trace.get("logprobs", [])) for trace in prefix]
            metadata = _sample_metadata(sample, task)
            targets = (
                range(args.ensemble_size)
                if args.target_mode == "all"
                else (0,)
            )
            correctness: Dict[int, bool] = {0: fixed_labels[sample.sample_id]}
            for target_index in targets:
                if target_index not in correctness:
                    correctness[target_index] = verifier.execute(
                        texts[target_index], sample.gold_answer, metadata
                    )
                controls, diagnostics = _score_controls(
                    logprobs,
                    texts,
                    task,
                    metadata,
                    target_index,
                    weakest_k,
                    args.probe_counts,
                    args.recovery_modes,
                )
                if target_index == 0:
                    frozen = fixed_labels[sample.sample_id]
                    if frozen != correctness[target_index]:
                        raise RuntimeError("fixed-label lookup changed unexpectedly")
                    difference = abs(
                        controls["pwsc_ci_risk"] - frozen_risks[sample.sample_id]
                    )
                    if difference > args.replay_tolerance:
                        raise RuntimeError(
                            f"{task}/{sample.sample_id} canonical replay drift: "
                            f"{difference}"
                        )
                kernel_sources[task][str(diagnostics["kernel_source"])] += 1
                for label, info in diagnostics["probe_diagnostics"].items():
                    source = str(info.get("kernel_source", "unknown"))
                    probe_totals[label][source] += 1
                rows.append(
                    {
                        "task": task,
                        "sample_id": sample.sample_id,
                        "target_index": target_index,
                        "target_mode": args.target_mode,
                        "is_correct": correctness[target_index],
                        "error": int(not correctness[target_index]),
                        **controls,
                    }
                )
        task_provenance["control_elapsed_seconds"] = time.perf_counter() - task_started

    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows)
    result = {
        "schema": "pwsc-ci-supplemental-controls-v1",
        "canonical_method": "PWSC-CI-v1.0",
        "canonical_contract": PWSC_CI_CONTRACT_VERSION,
        "scope": "diagnostic controls; not a new PWSC method or promotion gate",
        "tasks": list(args.tasks),
        "ensemble_size": args.ensemble_size,
        "target_mode": args.target_mode,
        "weakest_k_sweep": [label for label, _ in weakest_k],
        "probe_counts": list(args.probe_counts),
        "recovery_modes": list(args.recovery_modes),
        "bootstrap": {"draws": args.bootstrap, "seed": args.bootstrap_seed},
        "row_count": len(rows),
        **summary,
        "paired_macro_bootstrap": _paired_macro_bootstrap(
            rows, args.bootstrap, args.bootstrap_seed
        ),
        "kernel_source_counts": {
            task: dict(values) for task, values in kernel_sources.items()
        },
        "probe_kernel_source_counts": {
            label: dict(values) for label, values in probe_totals.items()
        },
        "provenance": provenance,
        "sample_csv": {"path": str(csv_path), "sha256": _sha256(csv_path)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "pwsc-ci-supplemental-controls-manifest-v1",
        "command": [str(item) for item in sys.argv],
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "sample_csv": {"path": str(csv_path), "sha256": _sha256(csv_path)},
        "source_provenance": provenance,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "summary": str(output_path),
        "sample_csv": str(csv_path),
        "manifest": str(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--dataset-root", default="/data/newmodel/graphunagent/datasets")
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--target-mode", choices=("fixed", "all"), default="fixed")
    parser.add_argument("--weakest-k", default="1,3,5,10,all")
    parser.add_argument("--probe-counts", type=_parse_int_list, default=(4, 8, 16))
    parser.add_argument(
        "--recovery-modes",
        type=lambda value: tuple(_parse_bool(item) for item in value.split(",")),
        default=(False, True),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260732)
    parser.add_argument("--replay-tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.ensemble_size < 2 or args.bootstrap < 1:
        raise SystemExit("ensemble-size must be >=2 and bootstrap must be positive")
    print(json.dumps(evaluate(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
