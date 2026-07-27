#!/usr/bin/env python3
"""Evaluate source-locked black-box baselines on PWSC-CI generation caches.

This entry point currently implements the published RDS formula because it
requires only sampled response text, an external sentence encoder, and
optionally chosen-token log probabilities. SeSE and CSA require additional
NLI/self-assessment inference and are intentionally not approximated here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.adapters import make_adapter  # noqa: E402
from diagnostics.analyzer import ScientificSignalAnalyzer  # noqa: E402
from run_pipeline import CORE_TASKS, UniversalDatasetLoader, _sample_metadata  # noqa: E402


RDS_SOURCE_COMMIT = "deebe5fdca88ed0d64bcc9b33f760d3275e25df9"
RDS_METHODS = (
    "RDS-L1",
    "RDS-L2",
    "RDS-PW-L1",
    "RDS-PW-L2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest(root: Path, pattern: str) -> Path:
    paths = list(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {root / pattern}")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def _as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _average_nll(logprobs: Sequence[Mapping[str, Any]]) -> float:
    values: List[float] = []
    for item in logprobs or []:
        try:
            value = float(item.get("logprob"))
        except (AttributeError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(-value)
    return float(np.mean(values)) if values else float("inf")


def _task_representations(task: str, texts: Sequence[str], metadata: Dict[str, Any]) -> List[str]:
    if task == "HumanEvalPlus":
        return [str(text) for text in texts]
    adapter = make_adapter(task, [metadata] * len(texts))
    return [str(value) for value in adapter.outcome_keys(texts)]


def _rds_scores(embeddings: np.ndarray, average_nlls: np.ndarray) -> Dict[str, float]:
    vectors = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)

    center = np.mean(vectors, axis=0)
    differences = vectors - center
    output = {
        "RDS-L1": float(np.sum(np.linalg.norm(differences, ord=1, axis=1))),
        "RDS-L2": float(np.sum(np.linalg.norm(differences, ord=2, axis=1))),
    }

    finite = np.isfinite(average_nlls)
    if not np.any(finite):
        weights = np.ones(len(vectors), dtype=np.float64) / len(vectors)
    else:
        logits = np.full(len(vectors), -np.inf, dtype=np.float64)
        logits[finite] = -average_nlls[finite]
        maximum = float(np.max(logits[finite]))
        weights = np.exp(logits - maximum)
        weights /= float(np.sum(weights))
    weighted_center = np.sum(weights[:, None] * vectors, axis=0)
    weighted_differences = vectors - weighted_center
    output.update(
        {
            "RDS-PW-L1": float(
                np.sum(weights * np.linalg.norm(weighted_differences, ord=1, axis=1))
            ),
            "RDS-PW-L2": float(
                np.sum(weights * np.linalg.norm(weighted_differences, ord=2, axis=1))
            ),
        }
    )
    return output


def _metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "AUROC": ScientificSignalAnalyzer.calculate_auroc(labels, scores),
        "AUPRC": ScientificSignalAnalyzer.calculate_auprc(labels, scores),
        "AURC": ScientificSignalAnalyzer.calculate_aurc(labels, scores),
        "SelectiveAcc@50": ScientificSignalAnalyzer.calculate_selective_accuracy(
            labels, scores, coverage=0.5
        ),
    }


def _paired_macro_bootstrap(
    task_arrays: Mapping[str, Dict[str, np.ndarray]],
    ensemble_size: int,
    seed: int,
    n_bootstrap: int,
) -> Dict[str, Dict[str, List[float]]]:
    metrics = {
        "AUROC": ScientificSignalAnalyzer.calculate_auroc,
        "AURC": ScientificSignalAnalyzer.calculate_aurc,
    }
    point: Dict[str, Dict[str, float]] = {}
    for method in RDS_METHODS:
        point[method] = {
            metric: float(
                np.mean(
                    [
                        function(values["error"], values["PWSC-CI"])
                        - function(values["error"], values[method])
                        for values in task_arrays.values()
                    ]
                )
            )
            for metric, function in metrics.items()
        }

    draws = {
        method: {metric: [] for metric in metrics} for method in RDS_METHODS
    }
    rng = np.random.default_rng(seed + ensemble_size)
    for _ in range(n_bootstrap):
        sampled = {}
        for task, values in task_arrays.items():
            indices = rng.integers(0, len(values["error"]), len(values["error"]))
            sampled[task] = {key: value[indices] for key, value in values.items()}
        for method in RDS_METHODS:
            for metric, function in metrics.items():
                difference = float(
                    np.mean(
                        [
                            function(values["error"], values["PWSC-CI"])
                            - function(values["error"], values[method])
                            for values in sampled.values()
                        ]
                    )
                )
                if math.isfinite(difference):
                    draws[method][metric].append(difference)

    output: Dict[str, Dict[str, List[float]]] = {}
    for method in RDS_METHODS:
        key = f"PWSC-CI - {method}"
        output[key] = {}
        for metric in metrics:
            values = np.asarray(draws[method][metric], dtype=np.float64)
            output[key][metric] = [
                point[method][metric],
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
    return output


def _resolve_embedding_model(model_name: str, allow_download: bool) -> Tuple[str, str]:
    path = Path(model_name).expanduser()
    if path.exists():
        resolved = str(path.resolve())
        return resolved, path.resolve().name
    from huggingface_hub import snapshot_download

    resolved = snapshot_download(repo_id=model_name, local_files_only=not allow_download)
    return resolved, Path(resolved).name


def _source_commit() -> str:
    repository = ROOT / "external_baselines" / "RDS"
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_task_inputs(
    run_dir: Path,
    task: str,
    sizes: Sequence[int],
    loader: UniversalDatasetLoader,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, np.ndarray]]]:
    manifest_path = _latest(run_dir, f"manifest_{task}_*.json")
    csv_path = _latest(run_dir, f"sample_results_{task}_*.csv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_path = Path(manifest["generation_cache_path"]).resolve()

    rows_by_n: Dict[int, List[Dict[str, str]]] = {size: [] for size in sizes}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            size = int(row["ensemble_size"])
            if size in rows_by_n:
                rows_by_n[size].append(row)
    for size, rows in rows_by_n.items():
        if not rows:
            raise ValueError(f"{csv_path}: no N={size} rows")

    max_rows = rows_by_n[max(sizes)]
    expected_ids = [row["sample_id"] for row in max_rows]
    records: List[Dict[str, Any]] = []
    with cache_path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        for line in handle:
            records.append(json.loads(line))
    record_by_id = {record["sample_id"]: record for record in records}
    if set(record_by_id) != set(expected_ids):
        raise ValueError(f"{task}: cache/sample CSV ids do not match")

    samples = loader.load_any_dataset(task, 0)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    ordered_records = []
    for sample_id in expected_ids:
        record = dict(record_by_id[sample_id])
        sample = sample_by_id[sample_id]
        metadata = _sample_metadata(sample, task)
        traces = record["traces"][: max(sizes)]
        texts = [str(trace["text"]) for trace in traces]
        record["representations"] = _task_representations(task, texts, metadata)
        record["question"] = str(sample.raw_question)
        record["average_nlls"] = np.asarray(
            [_average_nll(trace["logprobs"]) for trace in traces], dtype=np.float64
        )
        ordered_records.append(record)

    arrays_by_n: Dict[int, Dict[str, np.ndarray]] = {}
    for size, rows in rows_by_n.items():
        by_id = {row["sample_id"]: row for row in rows}
        arrays_by_n[size] = {
            "error": np.asarray(
                [0 if _as_bool(by_id[sample_id]["base_is_correct"]) else 1 for sample_id in expected_ids],
                dtype=np.int64,
            ),
            "PWSC-CI": np.asarray(
                [float(by_id[sample_id]["pwsc_ci_risk"]) for sample_id in expected_ids],
                dtype=np.float64,
            ),
        }
    provenance = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "sample_csv_path": str(csv_path.resolve()),
        "sample_csv_sha256": _sha256(csv_path),
        "cache_path": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "cache_header": header,
    }
    return provenance, ordered_records, arrays_by_n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument(
        "--dataset_root", default="/data/newmodel/graphunagent/datasets"
    )
    parser.add_argument("--tasks", default=" ".join(CORE_TASKS))
    parser.add_argument("--ensemble_sizes", default="1,4,8,16")
    parser.add_argument(
        "--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--evaluation_status",
        choices=(
            "development_blackbox_baseline_comparison",
            "independent_promotion_comparison",
        ),
        default="development_blackbox_baseline_comparison",
    )
    parser.add_argument(
        "--score_cache",
        type=Path,
        default=None,
        help="optional per-sample RDS score sidecar for paired downstream gates",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = tuple(item for item in args.tasks.replace(",", " ").split() if item)
    if not tasks or any(task not in CORE_TASKS for task in tasks):
        parser.error(f"tasks must be selected from {CORE_TASKS}")
    sizes = tuple(int(item) for item in args.ensemble_sizes.split(","))
    if not sizes or tuple(sorted(set(sizes))) != sizes:
        parser.error("ensemble_sizes must be unique and increasing")
    if args.bootstrap < 1 or args.batch_size < 1:
        parser.error("bootstrap and batch_size must be positive")

    observed_commit = _source_commit()
    if observed_commit != RDS_SOURCE_COMMIT:
        raise RuntimeError(
            f"RDS source drift: expected {RDS_SOURCE_COMMIT}, got {observed_commit}"
        )
    resolved_model, model_revision = _resolve_embedding_model(
        args.embedding_model, args.allow_download
    )
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(resolved_model, device=args.device)
    loader = UniversalDatasetLoader(args.dataset_root)
    run_dir = args.run_dir.expanduser().resolve()

    provenance: Dict[str, Any] = {}
    arrays: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {
        size: {} for size in sizes
    }
    score_records: List[Dict[str, Any]] = []
    for task in tasks:
        task_provenance, records, arrays_by_n = _load_task_inputs(
            run_dir, task, sizes, loader
        )
        flattened = [
            text for record in records for text in record["representations"]
        ]
        embeddings = embedder.encode(
            flattened,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        width = max(sizes)
        embeddings = np.asarray(embeddings).reshape(len(records), width, -1)
        task_score_records = [
            {
                "record_type": "rds-scores",
                "task": task,
                "sample_id": str(record["sample_id"]),
                "scores_by_n": {},
            }
            for record in records
        ]
        for size in sizes:
            task_arrays = arrays_by_n[size]
            method_values = {method: [] for method in RDS_METHODS}
            for index, record in enumerate(records):
                scores = _rds_scores(
                    embeddings[index, :size], record["average_nlls"][:size]
                )
                task_score_records[index]["scores_by_n"][str(size)] = scores
                for method, value in scores.items():
                    method_values[method].append(value)
            for method, values in method_values.items():
                task_arrays[method] = np.asarray(values, dtype=np.float64)
            arrays[size][task] = task_arrays
        provenance[task] = task_provenance
        score_records.extend(task_score_records)

    metrics_by_n: Dict[str, Any] = {}
    for size in sizes:
        task_metrics = {
            task: {
                method: _metrics(values["error"], values[method])
                for method in ("PWSC-CI", *RDS_METHODS)
            }
            for task, values in arrays[size].items()
        }
        macro_metrics = {
            method: {
                metric: float(
                    np.mean([task_metrics[task][method][metric] for task in tasks])
                )
                for metric in next(iter(task_metrics.values()))[method]
            }
            for method in ("PWSC-CI", *RDS_METHODS)
        }
        metrics_by_n[f"N={size}"] = {
            "task_metrics": task_metrics,
            "macro_metrics": macro_metrics,
            "paired_macro_differences": _paired_macro_bootstrap(
                arrays[size], size, args.seed, args.bootstrap
            ),
        }

    score_cache_metadata = None
    if args.score_cache is not None:
        score_cache_path = args.score_cache.expanduser().resolve()
        score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        score_cache_payload = {
            "schema": "rds-paired-score-cache-v1",
            "source_commit": observed_commit,
            "embedding_model": args.embedding_model,
            "embedding_model_path": resolved_model,
            "embedding_model_revision": model_revision,
            "tasks": list(tasks),
            "ensemble_sizes": list(sizes),
            "seed": args.seed,
            "inputs": {
                task: {
                    key: provenance[task][key]
                    for key in (
                        "manifest_sha256",
                        "sample_csv_sha256",
                        "cache_sha256",
                    )
                }
                for task in tasks
            },
            "records": score_records,
        }
        score_cache_path.write_text(
            json.dumps(score_cache_payload, indent=2), encoding="utf-8"
        )
        score_cache_metadata = {
            "path": str(score_cache_path),
            "sha256": _sha256(score_cache_path),
            "records": len(score_records),
        }

    output = {
        "status": args.evaluation_status,
        "tasks": list(tasks),
        "ensemble_sizes": list(sizes),
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "evaluation_target": "fixed first-path error labels from the PWSC sample CSV",
        "representation": (
            "canonical final outcomes for math/logic; full generated program for code"
        ),
        "baseline": {
            "name": "Radial Dispersion Score",
            "source_repository": "https://github.com/manhitv/RDS",
            "source_commit": observed_commit,
            "embedding_model": args.embedding_model,
            "embedding_model_path": resolved_model,
            "embedding_model_revision": model_revision,
            "formula": (
                "sum of L1/L2 distances from the unit-embedding centroid; "
                "PW variants use normalized exp(-average-token-NLL) weights"
            ),
        },
        "inputs": provenance,
        "paired_score_cache": score_cache_metadata,
        "Metrics by N": metrics_by_n,
        "caveat": (
            "RDS is source-formula-compatible but uses the task-output extraction "
            "needed for math/logic reasoning. SeSE and CSA require separate official "
            "NLI/self-assessment runs and are not approximated in this file."
        ),
    }
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
