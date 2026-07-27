#!/usr/bin/env python3
"""Replay the source-locked 2026 SeSE short-form pipeline on PWSC caches.

The released SeSE code and paper describe different graph contracts.  This
evaluator follows the executable short-form source at commit 8d4c6c5: raw
responses are used through its documented enhancement fallback, clustering
mixes entailment and cosine similarity, the graph is symmetric and repaired to
be connected, and the released K=2 structural-entropy implementation is called
unchanged.  The source returns the negative of the entropy it accumulates, so
the evaluator reports both that literal value and an explicitly named
sign-corrected adaptation.  Neither is presented as an unqualified
paper-formula reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import queue
import subprocess
import sys
import types
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from diagnostics.analyzer import ScientificSignalAnalyzer  # noqa: E402
from run_pipeline import CORE_TASKS, UniversalDatasetLoader  # noqa: E402
from scripts.evaluate_blackbox_2026 import (  # noqa: E402
    _load_task_inputs,
    _metrics,
    _sha256,
)


SESE_SOURCE_COMMIT = "8d4c6c5ceab61e6d973129785b82480fb0c572c3"
SESE_NLI_MODEL_ID = "microsoft/deberta-v2-xlarge-mnli"
SESE_NLI_REVISION = "2b65bccdedcead4bc685988bcc30153d4922a789"
SESE_EMBEDDING_MODEL_ID = (
    "sentence-transformers/static-similarity-mrl-multilingual-v1"
)
SESE_EMBEDDING_REVISION = "7264ea07c5365a11d7e6d87dbb6195889a13054f"
GRAPH_SOURCE = (
    ROOT
    / "external_baselines/SeSE/sentence_structural_entropy/src/"
    "uncertainty_measures/construct_semantic_graph.py"
)
ENTROPY_SOURCE = (
    ROOT
    / "external_baselines/SeSE/sentence_structural_entropy/src/"
    "uncertainty_measures/structural_entropy.py"
)
ENHANCEMENT_CONTRACT = "released-three-retry-raw-response-fallback-no-api-call"
SESE_METHODS = ("SeSE-source-raw", "SeSE-source-sign-corrected")
SESE_CACHE_FIELDS = {
    "SeSE-source-raw": "structural_entropy_source_raw",
    "SeSE-source-sign-corrected": "structural_entropy_sign_corrected",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_commit() -> str:
    repository = ROOT / "external_baselines" / "SeSE"
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _model_revision(path: Path) -> str:
    metadata = path / ".hfd" / "repo_metadata.json"
    if metadata.exists():
        value = json.loads(metadata.read_text(encoding="utf-8"))
        return str(value.get("sha") or value.get("lastModified") or path.name)
    return path.name


def _load_cache(path: Path, header: Mapping[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    contract_sha = _value_sha256(header)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _canonical_json(
                {
                    "record_type": "header",
                    "contract_sha256": contract_sha,
                    **dict(header),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {}

    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        try:
            observed = json.loads(next(handle))
        except StopIteration as exc:
            raise ValueError(f"empty SeSE cache: {path}") from exc
        if observed.get("contract_sha256") != contract_sha:
            raise ValueError(f"SeSE cache contract mismatch: {path}")
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            item = json.loads(line)
            key = (str(item["task"]), str(item["sample_id"]))
            if key in records:
                raise ValueError(f"duplicate SeSE cache key at {path}:{line_number}: {key}")
            records[key] = item
    return records


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(record)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parse_devices(device: str, devices: str | None) -> Tuple[str, ...]:
    values = tuple(
        item
        for item in (devices if devices is not None else device).replace(",", " ").split()
        if item
    )
    if not values:
        raise ValueError("at least one SeSE device is required")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate SeSE devices: {values}")
    return values


def _ordered_runtime_map(
    items: Sequence[Any],
    runtimes: Sequence[Any],
    evaluate: Callable[[Any, Any], Any],
) -> Iterator[Any]:
    """Evaluate concurrently while yielding results in input order."""
    if not runtimes:
        raise ValueError("at least one SeSE runtime is required")
    if not items:
        return
    if len(runtimes) == 1:
        for item in items:
            yield evaluate(item, runtimes[0])
        return

    runtime_pool: queue.Queue[Any] = queue.Queue()
    for runtime in runtimes:
        runtime_pool.put(runtime)

    def run(index: int) -> Tuple[int, Any]:
        runtime = runtime_pool.get()
        try:
            return index, evaluate(items[index], runtime)
        finally:
            runtime_pool.put(runtime)

    executor = ThreadPoolExecutor(max_workers=len(runtimes))
    in_flight = {}
    completed: Dict[int, Any] = {}
    next_submit = 0
    next_yield = 0
    try:
        while next_submit < min(len(items), len(runtimes)):
            future = executor.submit(run, next_submit)
            in_flight[future] = next_submit
            next_submit += 1

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                submitted_index = in_flight.pop(future)
                result_index, result = future.result()
                if result_index != submitted_index:
                    raise RuntimeError("SeSE worker returned the wrong result index")
                completed[result_index] = result
                if next_submit < len(items):
                    new_future = executor.submit(run, next_submit)
                    in_flight[new_future] = next_submit
                    next_submit += 1

            while next_yield in completed:
                yield completed.pop(next_yield)
                next_yield += 1
    finally:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _upper_triangle(matrix: np.ndarray) -> List[float]:
    values = np.asarray(matrix, dtype=np.float64)
    return [
        float(values[i, j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]


def _load_models(nli_path: Path, embedding_path: Path, device: str):
    # The host combines an older sentencepiece-generated protobuf module with
    # protobuf>=4.  The documented Python implementation preserves tokenizer
    # semantics and avoids mutating the shared environment.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(nli_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(nli_path), local_files_only=True
    ).to(torch.device(device)).eval()
    embedder = SentenceTransformer(
        str(embedding_path), device=device, local_files_only=True
    )
    return tokenizer, model, embedder


def _entailment_matrix(
    responses: Sequence[str], tokenizer: Any, model: Any, device: str, batch_size: int
) -> np.ndarray:
    import torch

    n = len(responses)
    matrix = np.zeros((n, n), dtype=np.float64)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    torch_device = torch.device(device)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [responses[i] for i, _ in batch],
            [responses[j] for _, j in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(torch_device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
        entailment = probabilities[:, 2].detach().cpu().numpy()
        for (left, right), value in zip(batch, entailment):
            matrix[left, right] = matrix[right, left] = float(value)
    return matrix


def _cosine_matrix(responses: Sequence[str], embedder: Any) -> np.ndarray:
    embeddings = embedder.encode(
        list(responses),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float64) @ np.asarray(
        embeddings, dtype=np.float64
    ).T


def _source_clusters(
    cosine: np.ndarray,
    entailment: np.ndarray,
    similarity_threshold: float = 0.3,
    entailment_weight: float = 0.65,
) -> np.ndarray:
    similarity = np.clip(
        entailment_weight * entailment + (1.0 - entailment_weight) * cosine,
        0.0,
        1.0,
    )
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    non_diagonal = distance[~np.eye(len(distance), dtype=bool)]
    if len(distance) <= 1 or np.all(non_diagonal == 0.0):
        return np.zeros(len(distance), dtype=np.int64)
    if np.all(non_diagonal == 1.0):
        return np.arange(len(distance), dtype=np.int64)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - similarity_threshold,
        metric="precomputed",
        linkage="average",
    )
    return np.asarray(clusterer.fit_predict(distance), dtype=np.int64)


def _components(adjacency: np.ndarray) -> List[List[int]]:
    n = len(adjacency)
    parent = list(range(n))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in np.argwhere(adjacency > 0.0):
        union(int(left), int(right))
    grouped: Dict[int, List[int]] = {}
    for index in range(n):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())


def _page_rank(adjacency: np.ndarray) -> np.ndarray:
    n = len(adjacency)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    out_degree = np.asarray(adjacency.sum(axis=0), dtype=np.float64)
    out_degree[out_degree == 0.0] = 1.0
    transition = adjacency / out_degree
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(100):
        updated = (1.0 - 0.85) / n + 0.85 * transition.T @ rank
        if np.linalg.norm(updated - rank, ord=1) < 1e-6:
            rank = updated
            break
        rank = updated
    return rank


def _source_adjacency(entailment: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    n = len(entailment)
    adjacency = np.zeros((n, n), dtype=np.float64)
    for left in range(n):
        for right in range(left + 1, n):
            if labels[left] == labels[right]:
                adjacency[left, right] = adjacency[right, left] = entailment[left, right]

    components = _components(adjacency)
    if len(components) <= 1:
        return adjacency
    representatives = [
        component[
            int(np.argmax(_page_rank(adjacency[np.ix_(component, component)])))
        ]
        for component in components
    ]

    # Source make_connected is a maximum-weight Kruskal tree over component reps.
    edges = sorted(
        (
            (float(entailment[left, right]), left, right)
            for index, left in enumerate(representatives)
            for right in representatives[index + 1 :]
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    parent = {value: value for value in representatives}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for weight, left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        adjacency[left, right] = adjacency[right, left] = weight
    return adjacency


def _load_source_compute_se():
    # cdlib is imported but unused by the released structural-entropy module.
    if "cdlib" not in sys.modules:
        stub = types.ModuleType("cdlib")
        stub.NodeClustering = object
        sys.modules["cdlib"] = stub
    spec = importlib.util.spec_from_file_location("sese_structural_entropy_locked", ENTROPY_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SeSE entropy source: {ENTROPY_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_se


def _source_entropy_variants(source_value: float) -> Dict[str, float]:
    value = float(source_value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite SeSE source value: {value}")
    return {
        "SeSE-source-raw": value,
        "SeSE-source-sign-corrected": -value,
    }


def _prepare_source_item(
    task: str, record: Mapping[str, Any], ensemble_size: int
) -> Dict[str, Any]:
    responses = [str(trace["text"]) for trace in record["traces"][:ensemble_size]]
    return {
        "task": task,
        "sample_id": str(record["sample_id"]),
        "responses": responses,
        "responses_sha256": _value_sha256(responses),
    }


def _evaluate_source_item(
    item: Mapping[str, Any],
    runtime: Tuple[str, Any, Any, Any],
    nli_batch_size: int,
    compute_se: Callable[..., float],
) -> Dict[str, Any]:
    device, tokenizer, nli_model, embedder = runtime
    responses = list(item["responses"])
    entailment = _entailment_matrix(
        responses, tokenizer, nli_model, device, nli_batch_size
    )
    cosine = _cosine_matrix(responses, embedder)
    labels = _source_clusters(cosine, entailment)
    adjacency = _source_adjacency(entailment, labels)
    source_entropy = float(compute_se(adjacency, tree_depth=2))
    try:
        entropy_variants = _source_entropy_variants(source_entropy)
    except ValueError as exc:
        key = (item["task"], item["sample_id"])
        raise RuntimeError(
            f"invalid SeSE structural entropy for {key}: {source_entropy}"
        ) from exc
    return {
        "record_type": "sese-source-v2",
        "task": item["task"],
        "sample_id": item["sample_id"],
        "responses_sha256": item["responses_sha256"],
        "cluster_ids": labels.tolist(),
        "cluster_count": int(len(np.unique(labels))),
        "entailment_upper": _upper_triangle(entailment),
        "cosine_upper": _upper_triangle(cosine),
        "adjacency_sha256": _value_sha256(adjacency.tolist()),
        "structural_entropy_source_raw": entropy_variants["SeSE-source-raw"],
        "structural_entropy_sign_corrected": entropy_variants[
            "SeSE-source-sign-corrected"
        ],
    }


def _paired_macro_bootstrap(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    comparator: str,
    seed: int,
    draws: int,
) -> Dict[str, List[float]]:
    functions = {
        "AUROC": ScientificSignalAnalyzer.calculate_auroc,
        "AURC": ScientificSignalAnalyzer.calculate_aurc,
    }
    output: Dict[str, List[float]] = {}
    rng = np.random.default_rng(seed)
    for metric, function in functions.items():
        point = float(
            np.mean(
                [
                    function(values["error"], values["PWSC-CI"])
                    - function(values["error"], values[comparator])
                    for values in arrays.values()
                ]
            )
        )
        samples: List[float] = []
        for _ in range(draws):
            differences = []
            for values in arrays.values():
                indices = rng.integers(0, len(values["error"]), len(values["error"]))
                differences.append(
                    function(values["error"][indices], values["PWSC-CI"][indices])
                    - function(values["error"][indices], values[comparator][indices])
                )
            value = float(np.mean(differences))
            if math.isfinite(value):
                samples.append(value)
        sampled = np.asarray(samples, dtype=np.float64)
        output[metric] = [
            point,
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ]
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", default="/data/newmodel/graphunagent/datasets")
    parser.add_argument("--tasks", default=" ".join(CORE_TASKS))
    parser.add_argument("--ensemble_size", type=int, default=16)
    parser.add_argument(
        "--nli_model", type=Path, default=ROOT / "external_models/deberta-v2-xlarge-mnli"
    )
    parser.add_argument(
        "--embedding_model",
        type=Path,
        default=ROOT / "external_models/static-similarity-mrl-multilingual-v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--devices",
        default=None,
        help="comma- or space-separated devices; overrides --device",
    )
    parser.add_argument("--nli_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--evaluation_status",
        choices=("development_blackbox_baseline_comparison", "independent_promotion_comparison"),
        default="development_blackbox_baseline_comparison",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = tuple(item for item in args.tasks.replace(",", " ").split() if item)
    if not tasks or any(task not in CORE_TASKS for task in tasks):
        parser.error(f"tasks must be selected from {CORE_TASKS}")
    if args.ensemble_size < 2 or args.nli_batch_size < 1 or args.bootstrap < 1:
        parser.error("ensemble size, batch size, and bootstrap must be positive")
    try:
        devices = _parse_devices(args.device, args.devices)
    except ValueError as exc:
        parser.error(str(exc))

    observed_commit = _source_commit()
    if observed_commit != SESE_SOURCE_COMMIT:
        raise RuntimeError(
            f"SeSE source drift: expected {SESE_SOURCE_COMMIT}, got {observed_commit}"
        )
    nli_path = args.nli_model.expanduser().resolve()
    embedding_path = args.embedding_model.expanduser().resolve()
    if not nli_path.exists() or not embedding_path.exists():
        raise FileNotFoundError("SeSE evaluator model checkpoint is missing")
    observed_nli_revision = _model_revision(nli_path)
    observed_embedding_revision = _model_revision(embedding_path)
    if observed_nli_revision != SESE_NLI_REVISION:
        raise RuntimeError(
            "SeSE NLI revision mismatch: expected "
            f"{SESE_NLI_MODEL_ID}@{SESE_NLI_REVISION}, observed "
            f"{observed_nli_revision} at {nli_path}"
        )
    if observed_embedding_revision != SESE_EMBEDDING_REVISION:
        raise RuntimeError(
            "SeSE embedding revision mismatch: expected "
            f"{SESE_EMBEDDING_MODEL_ID}@{SESE_EMBEDDING_REVISION}, observed "
            f"{observed_embedding_revision} at {embedding_path}"
        )

    loader = UniversalDatasetLoader(args.dataset_root)
    run_dir = args.run_dir.expanduser().resolve()
    records_by_task: Dict[str, Sequence[Mapping[str, Any]]] = {}
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    provenance: Dict[str, Any] = {}
    for task in tasks:
        task_provenance, records, task_arrays = _load_task_inputs(
            run_dir, task, (args.ensemble_size,), loader
        )
        provenance[task] = task_provenance
        records_by_task[task] = records
        arrays[task] = task_arrays[args.ensemble_size]

    input_fingerprint = {
        task: {
            key: provenance[task][key]
            for key in ("manifest_sha256", "sample_csv_sha256", "cache_sha256")
        }
        for task in tasks
    }
    header = {
        "schema": "sese-source-replay-cache-v2",
        "source_commit": observed_commit,
        "graph_source_sha256": _sha256(GRAPH_SOURCE),
        "entropy_source_sha256": _sha256(ENTROPY_SOURCE),
        "nli_model_id": SESE_NLI_MODEL_ID,
        "nli_model_revision": observed_nli_revision,
        "embedding_model_id": SESE_EMBEDDING_MODEL_ID,
        "embedding_model_revision": observed_embedding_revision,
        "enhancement_contract": ENHANCEMENT_CONTRACT,
        "entailment_weight": 0.65,
        "embedding_weight": 0.35,
        "similarity_threshold": 0.3,
        "tree_depth": 2,
        "source_sign_contract": (
            "locked PartitionTree.entropy accumulates conventional nonnegative entropy "
            "then returns -ent; cache stores literal source output and its named negation"
        ),
        "ensemble_size": args.ensemble_size,
        "tasks": list(tasks),
        "inputs": input_fingerprint,
    }
    args.cache = args.cache.expanduser().resolve()
    cache_records = _load_cache(args.cache, header)

    total = sum(len(records) for records in records_by_task.values())
    progress = tqdm(total=total, desc="SeSE source replay")
    pending: List[Dict[str, Any]] = []
    for task in tasks:
        for record in records_by_task[task]:
            item = _prepare_source_item(task, record, args.ensemble_size)
            key = (item["task"], item["sample_id"])
            cached = cache_records.get(key)
            if cached is None:
                pending.append(item)
                continue
            if cached.get("responses_sha256") != item["responses_sha256"]:
                raise ValueError(f"stale SeSE source record: {key}")
            progress.update(1)

    try:
        if pending:
            runtimes = []
            for device in devices:
                print(f"Loading SeSE models on {device}", file=sys.stderr, flush=True)
                tokenizer, nli_model, embedder = _load_models(
                    nli_path, embedding_path, device
                )
                runtimes.append((device, tokenizer, nli_model, embedder))
            compute_se = _load_source_compute_se()

            def evaluate(item: Mapping[str, Any], runtime: Any) -> Dict[str, Any]:
                return _evaluate_source_item(
                    item, runtime, args.nli_batch_size, compute_se
                )

            for item in _ordered_runtime_map(pending, runtimes, evaluate):
                key = (str(item["task"]), str(item["sample_id"]))
                if key in cache_records:
                    raise RuntimeError(f"duplicate SeSE result produced: {key}")
                _append_record(args.cache, item)
                cache_records[key] = item
                progress.update(1)
    finally:
        progress.close()

    if len(cache_records) != total:
        raise RuntimeError(f"SeSE source cache incomplete: {len(cache_records)}/{total}")
    for task in tasks:
        for method in SESE_METHODS:
            arrays[task][method] = np.asarray(
                [
                    cache_records[(task, str(record["sample_id"]))][
                        SESE_CACHE_FIELDS[method]
                    ]
                    for record in records_by_task[task]
                ],
                dtype=np.float64,
            )

    task_metrics = {
        task: {
            method: _metrics(values["error"], values[method])
            for method in ("PWSC-CI", *SESE_METHODS)
        }
        for task, values in arrays.items()
    }
    macro_metrics = {
        method: {
            metric: float(np.mean([task_metrics[task][method][metric] for task in tasks]))
            for metric in task_metrics[tasks[0]][method]
        }
        for method in ("PWSC-CI", *SESE_METHODS)
    }
    output = {
        "status": args.evaluation_status,
        "tasks": list(tasks),
        "ensemble_size": args.ensemble_size,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "evaluation_target": "fixed first-path error labels from the PWSC sample CSV",
        "baseline": {
            "name": "SeSE released-code source replay (raw-response fallback)",
            "reported_variants": {
                "SeSE-source-raw": (
                    "literal return value from the locked released implementation"
                ),
                "SeSE-source-sign-corrected": (
                    "named adaptation equal to -1 times the literal source value"
                ),
            },
            "source_repository": "https://github.com/SELGroup/SeSE",
            "source_commit": observed_commit,
            "nli_model": SESE_NLI_MODEL_ID,
            "nli_model_revision": observed_nli_revision,
            "embedding_model": SESE_EMBEDDING_MODEL_ID,
            "embedding_model_revision": observed_embedding_revision,
            "formula_fidelity": (
                "released short-form .65 entailment/.35 cosine clustering, within-cluster "
                "symmetric entailment graph, connectivity repair, and K=2 source entropy"
            ),
            "protocol_adaptation": (
                "uses the released raw-response fallback because no GPT-4o enhancement "
                "credential is available; the sign-corrected row is a separately named "
                "one-operation adaptation; PWSC fixed-first labels and N=16 are retained"
            ),
        },
        "inputs": provenance,
        "sidecar_cache": {
            "path": str(args.cache),
            "sha256": _sha256(args.cache),
        },
        "task_metrics": task_metrics,
        "macro_metrics": macro_metrics,
        "paired_macro_differences": {
            f"PWSC-CI - {method}": _paired_macro_bootstrap(
                arrays,
                method,
                args.seed + args.ensemble_size,
                args.bootstrap,
            )
            for method in SESE_METHODS
        },
        "claim_boundary": (
            "This is a source-locked released-code replay with the documented raw-response "
            "fallback. The raw row preserves the released sign bug and the corrected row is "
            "an explicitly named sign-only adaptation. Neither is the paper's directed "
            "question-conditioned graph contract, and neither may be reported as an "
            "unqualified official SeSE number."
        ),
    }
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
