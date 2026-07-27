#!/usr/bin/env python3
"""Run the 2026 Clustered Self-Assessment baseline on PWSC caches.

This is a source-formula-compatible, matched-cache adaptation.  It preserves
the official bidirectional-NLI clustering rule and the full-vocabulary
probability assigned by the target model to option A.  The fixed first PWSC
path replaces CSA's separately generated greedy path so every method is scored
against exactly the same error event; that protocol difference is recorded in
the output and must remain visible in the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import requests
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.evaluate_blackbox_2026 import (  # noqa: E402
    _load_task_inputs,
    _metrics,
    _sha256,
)
from run_pipeline import CORE_TASKS, UniversalDatasetLoader  # noqa: E402


CSA_SOURCE_COMMIT = "19cadb1e0a5d67d471f6e837fee17b55d36d89d9"
CSA_NLI_MODEL_ID = "microsoft/deberta-large-mnli"
NONE_OF_THE_ABOVE = "None of the above"
REPRESENTATION_CONTRACT = (
    "canonical final outcomes for math/logic; full generated program for code"
)
PROMPT_CONTRACT = "csa-matched-cache-qa-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_commit() -> str:
    repository = ROOT / "external_baselines" / "clustered_self_assessment"
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _nli_revision(model_path: Path) -> str:
    metadata_path = model_path / ".hfd" / "repo_metadata.json"
    if not metadata_path.exists():
        return model_path.resolve().name
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return str(metadata.get("sha") or metadata.get("lastModified") or model_path.name)


def _load_jsonl_cache(
    path: Path, header: Mapping[str, Any]
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    expected_contract = _value_sha256(header)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                _canonical_json(
                    {
                        "record_type": "header",
                        "contract_sha256": expected_contract,
                        **dict(header),
                    }
                )
                + "\n"
            )
        return {}

    records: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        try:
            observed_header = json.loads(next(handle))
        except StopIteration as exc:
            raise ValueError(f"empty CSA cache: {path}") from exc
        if observed_header.get("contract_sha256") != expected_contract:
            raise ValueError(f"CSA cache contract mismatch: {path}")
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (
                str(record["task"]),
                str(record["sample_id"]),
                int(record.get("ensemble_size", 0)),
            )
            if key in records:
                raise ValueError(f"duplicate CSA cache key at {path}:{line_number}: {key}")
            records[key] = record
    return records


def _append_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(dict(record)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _unique_representations(representations: Sequence[str]) -> List[str]:
    return sorted(set(str(value).strip() for value in representations if str(value).strip()))


def _directed_pairs(values: Sequence[str]) -> List[Tuple[str, str]]:
    return [
        (left, right)
        for left_index, left in enumerate(values)
        for right_index, right in enumerate(values)
        if left_index != right_index
    ]


def _official_cluster_choices(
    representations: Sequence[str],
    unique_values: Sequence[str],
    semantic_labels: np.ndarray,
) -> List[str]:
    """Replicate the released CSA clustering and representative selection."""
    answers = [str(value).strip() for value in representations if str(value).strip()]
    if not answers:
        return [NONE_OF_THE_ABOVE]
    observed_unique = sorted(set(answers))
    if observed_unique != list(unique_values):
        raise ValueError("semantic label matrix does not match the representation prefix")
    labels = np.asarray(semantic_labels, dtype=np.int64)
    if labels.shape != (len(observed_unique), len(observed_unique)):
        raise ValueError("semantic label matrix has the wrong shape")

    semantic_set_ids = {answer: index for index, answer in enumerate(observed_unique)}
    for left_index, left in enumerate(observed_unique):
        for right_index, right in enumerate(observed_unique[left_index + 1 :], left_index + 1):
            if labels[left_index, right_index] + labels[right_index, left_index] >= 2:
                semantic_set_ids[right] = semantic_set_ids[left]

    raw_ids = [semantic_set_ids[answer] for answer in answers]
    relabel: Dict[int, int] = {}
    cluster_ids: List[int] = []
    for raw_id in raw_ids:
        if raw_id not in relabel:
            relabel[raw_id] = len(relabel)
        cluster_ids.append(relabel[raw_id])

    seen = set()
    representative_indices = []
    for index, cluster_id in enumerate(cluster_ids):
        if cluster_id not in seen:
            seen.add(cluster_id)
            representative_indices.append(index)
    return [answers[index] for index in representative_indices] + [NONE_OF_THE_ABOVE]


def _prefix_label_matrix(
    prefix: Sequence[str],
    full_unique: Sequence[str],
    full_labels: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    prefix_unique = _unique_representations(prefix)
    positions = {value: index for index, value in enumerate(full_unique)}
    try:
        indices = [positions[value] for value in prefix_unique]
    except KeyError as exc:
        raise ValueError("NLI cache is missing a prefix representation") from exc
    return prefix_unique, np.asarray(full_labels, dtype=np.int64)[np.ix_(indices, indices)]


def _choice_lines(choices: Sequence[str]) -> str:
    if len(choices) > len(string.ascii_uppercase):
        raise ValueError("CSA supports at most 26 choices")
    return "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(string.ascii_uppercase, choices)
    )


def _assessment_prompt(question: str, choices: Sequence[str]) -> str:
    none_label = string.ascii_uppercase[len(choices) - 1]
    return (
        "Task:\nSelect the one correct answer to the question from the choices "
        "provided. If none of the provided choices is correct, select the final "
        f"choice ({none_label}) {NONE_OF_THE_ABOVE}.\n\n"
        f"Question:\n{question}\n\nChoices:\n{_choice_lines(choices)}\n\n"
        "Answer:\nThe answer is ("
    )


def _load_nli_model(model_path: Path, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True
    )
    model.to(torch.device(device)).eval()
    return tokenizer, model


def _predict_nli_labels(
    tokenizer: Any,
    model: Any,
    pairs: Sequence[Tuple[str, str]],
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    predictions: List[int] = []
    torch_device = torch.device(device)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [pair[0] for pair in batch],
            [pair[1] for pair in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(torch_device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
    return np.asarray(predictions, dtype=np.int64)


def _run_nli_stage(
    records_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    cache_path: Path,
    cache_records: Dict[Tuple[str, str, int], Dict[str, Any]],
    model_path: Path,
    device: str,
    batch_size: int,
    pair_chunk_size: int,
    max_size: int,
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    pending: List[Tuple[str, Mapping[str, Any], List[str], List[Tuple[str, str]]]] = []
    pending_pair_count = 0
    tokenizer = model = None

    def flush() -> None:
        nonlocal pending, pending_pair_count, tokenizer, model
        if not pending:
            return
        all_pairs = [pair for _, _, _, pairs in pending for pair in pairs]
        if all_pairs and model is None:
            tokenizer, model = _load_nli_model(model_path, device)
        predicted = (
            _predict_nli_labels(tokenizer, model, all_pairs, batch_size, device)
            if all_pairs
            else np.zeros(0, dtype=np.int64)
        )
        offset = 0
        output_records = []
        for task, record, unique_values, pairs in pending:
            count = len(pairs)
            matrix = np.zeros((len(unique_values), len(unique_values)), dtype=np.int64)
            if count:
                values = predicted[offset : offset + count]
                cursor = 0
                for left_index in range(len(unique_values)):
                    for right_index in range(len(unique_values)):
                        if left_index != right_index:
                            matrix[left_index, right_index] = int(values[cursor])
                            cursor += 1
                offset += count
            item = {
                "record_type": "nli",
                "task": task,
                "sample_id": str(record["sample_id"]),
                "ensemble_size": 0,
                "representation_sha256": _value_sha256(unique_values),
                "unique_count": len(unique_values),
                "semantic_labels": matrix.tolist(),
            }
            output_records.append(item)
            cache_records[(task, str(record["sample_id"]), 0)] = item
        _append_records(cache_path, output_records)
        pending = []
        pending_pair_count = 0

    total = sum(len(records) for records in records_by_task.values())
    progress = tqdm(total=total, desc="CSA NLI")
    for task, records in records_by_task.items():
        for record in records:
            key = (task, str(record["sample_id"]), 0)
            unique_values = _unique_representations(record["representations"][:max_size])
            cached = cache_records.get(key)
            if cached is not None:
                if cached.get("representation_sha256") != _value_sha256(unique_values):
                    raise ValueError(f"stale CSA NLI record: {key}")
                progress.update(1)
                continue
            pairs = _directed_pairs(unique_values)
            if pending and pending_pair_count + len(pairs) > pair_chunk_size:
                flush()
            pending.append((task, record, unique_values, pairs))
            pending_pair_count += len(pairs)
            if pending_pair_count >= pair_chunk_size:
                flush()
            progress.update(1)
    flush()
    progress.close()
    return cache_records


def _server_metadata(model_url: str, api_key: str) -> Mapping[str, Any]:
    response = requests.get(
        model_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", [])
    if not models:
        raise RuntimeError(f"no model metadata returned by {model_url}")
    return models[0]


def _forced_option_logprobs(
    model_url: str,
    model_name: str,
    api_key: str,
    prompts: Sequence[str],
    timeout: float,
) -> List[float]:
    payload = {
        "model": model_name,
        "prompt": [prompt + "A" for prompt in prompts],
        "max_tokens": 0,
        "temperature": 0,
        "echo": True,
        "logprobs": 0,
    }
    response = requests.post(
        model_url.rstrip("/") + "/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    choices = sorted(response.json().get("choices", []), key=lambda item: item["index"])
    if len(choices) != len(prompts):
        raise RuntimeError("CSA assessment response count mismatch")
    values = []
    for choice in choices:
        logprobs = choice.get("logprobs") or {}
        tokens = logprobs.get("tokens") or []
        token_logprobs = logprobs.get("token_logprobs") or []
        if not tokens or not token_logprobs or tokens[-1] != "A":
            raise RuntimeError("CSA teacher-forced option A was not a single final token")
        value = float(token_logprobs[-1])
        if not math.isfinite(value) or value > 1e-6:
            raise RuntimeError(f"invalid CSA option-A logprob: {value}")
        values.append(value)
    return values


def _request_with_retries(
    endpoint: str,
    model_name: str,
    api_key: str,
    prompts: Sequence[str],
    timeout: float,
    retries: int,
) -> List[float]:
    for attempt in range(retries + 1):
        try:
            return _forced_option_logprobs(
                endpoint, model_name, api_key, prompts, timeout
            )
        except (requests.RequestException, RuntimeError):
            if attempt >= retries:
                raise
            time.sleep(min(2.0**attempt, 10.0))
    raise AssertionError("unreachable")


def _run_assessment_stage(
    tasks: Sequence[str],
    sizes: Sequence[int],
    records_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    nli_records: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    cache_path: Path,
    cache_records: Dict[Tuple[str, str, int], Dict[str, Any]],
    endpoints: Sequence[str],
    model_name: str,
    api_key: str,
    batch_size: int,
    max_workers: int,
    timeout: float,
    retries: int,
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    pending: List[Tuple[str, Mapping[str, Any], int, List[str], str]] = []
    for task in tasks:
        for record in records_by_task[task]:
            nli = nli_records[(task, str(record["sample_id"]), 0)]
            full_unique = _unique_representations(record["representations"][: max(sizes)])
            full_labels = np.asarray(nli["semantic_labels"], dtype=np.int64)
            for size in sizes:
                key = (task, str(record["sample_id"]), size)
                prefix = record["representations"][:size]
                prefix_unique, prefix_labels = _prefix_label_matrix(
                    prefix, full_unique, full_labels
                )
                choices = _official_cluster_choices(prefix, prefix_unique, prefix_labels)
                if choices[0] != str(prefix[0]).strip():
                    raise ValueError(f"CSA target is not option A: {key}")
                prompt = _assessment_prompt(str(record["question"]), choices)
                prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                cached = cache_records.get(key)
                if cached is not None:
                    if cached.get("prompt_sha256") != prompt_sha:
                        raise ValueError(f"stale CSA assessment record: {key}")
                    continue
                pending.append((task, record, size, choices, prompt))

    batches = [
        pending[start : start + batch_size]
        for start in range(0, len(pending), batch_size)
    ]
    progress = tqdm(total=len(pending), desc="CSA self-assessment")
    batch_iterator = iter(enumerate(batches))

    def submit_next(executor, future_to_batch) -> bool:
        try:
            index, batch = next(batch_iterator)
        except StopIteration:
            return False
        endpoint = endpoints[index % len(endpoints)]
        future = executor.submit(
            _request_with_retries,
            endpoint,
            model_name,
            api_key,
            [item[4] for item in batch],
            timeout,
            retries,
        )
        future_to_batch[future] = (endpoint, batch)
        return True

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {}
            for _ in range(min(max_workers, len(batches))):
                submit_next(executor, future_to_batch)
            while future_to_batch:
                future = next(as_completed(tuple(future_to_batch)))
                endpoint, batch = future_to_batch.pop(future)
                try:
                    values = future.result()
                except Exception as exc:
                    for queued in future_to_batch:
                        queued.cancel()
                    keys = [
                        f"{task}:{record['sample_id']}:N={size}"
                        for task, record, size, _, _ in batch
                    ]
                    max_prompt_chars = max(len(item[4]) for item in batch)
                    raise RuntimeError(
                        "CSA assessment request failed; complete cache records remain "
                        f"resumable. endpoint={endpoint}, batch_size={len(batch)}, "
                        f"max_prompt_chars={max_prompt_chars}, keys={keys}"
                    ) from exc
                output_records = []
                for item, logprob in zip(batch, values):
                    task, record, size, choices, prompt = item
                    confidence = float(math.exp(logprob))
                    output = {
                        "record_type": "assessment",
                        "task": task,
                        "sample_id": str(record["sample_id"]),
                        "ensemble_size": size,
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                        "choice_count": len(choices),
                        "option_a_logprob": logprob,
                        "csa_confidence": confidence,
                        "csa_risk": float(1.0 - confidence),
                        "endpoint": endpoint,
                    }
                    output_records.append(output)
                    cache_records[(task, str(record["sample_id"]), size)] = output
                _append_records(cache_path, output_records)
                progress.update(len(batch))
                submit_next(executor, future_to_batch)
    finally:
        progress.close()
    return cache_records


def _paired_macro_bootstrap(
    arrays: Mapping[str, Mapping[str, np.ndarray]], seed: int, draws: int
) -> Dict[str, List[float]]:
    metric_functions = {
        "AUROC": lambda values: _metrics(
            values["error"], values["PWSC-CI"]
        )["AUROC"]
        - _metrics(values["error"], values["CSA"])["AUROC"],
        "AURC": lambda values: _metrics(
            values["error"], values["PWSC-CI"]
        )["AURC"]
        - _metrics(values["error"], values["CSA"])["AURC"],
    }
    output: Dict[str, List[float]] = {}
    rng = np.random.default_rng(seed)
    for metric, function in metric_functions.items():
        point = float(np.mean([function(values) for values in arrays.values()]))
        sampled_values = []
        for _ in range(draws):
            differences = []
            for values in arrays.values():
                indices = rng.integers(0, len(values["error"]), len(values["error"]))
                differences.append(
                    function({key: value[indices] for key, value in values.items()})
                )
            value = float(np.mean(differences))
            if math.isfinite(value):
                sampled_values.append(value)
        samples = np.asarray(sampled_values, dtype=np.float64)
        output[metric] = [
            point,
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ]
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", default="/data/newmodel/graphunagent/datasets")
    parser.add_argument("--tasks", default=" ".join(CORE_TASKS))
    parser.add_argument("--ensemble_sizes", default="16")
    parser.add_argument(
        "--nli_model", type=Path, default=ROOT / "external_models" / "deberta-large-mnli"
    )
    parser.add_argument("--nli_device", default="cuda:0")
    parser.add_argument("--nli_batch_size", type=int, default=8)
    parser.add_argument("--nli_pair_chunk_size", type=int, default=2048)
    parser.add_argument("--model_urls", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--model_name", default="deepseek")
    parser.add_argument("--expected_model_revision", default="/data/newmodel/IE/LLM/Qwen2.5-7B")
    parser.add_argument("--api_key", default="empty")
    parser.add_argument("--assessment_batch_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--request_timeout", type=float, default=300.0)
    parser.add_argument("--request_retries", type=int, default=3)
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
    parser.add_argument("--nli_cache", type=Path, required=True)
    parser.add_argument("--assessment_cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("nli", "assessment", "all"), default="all")
    args = parser.parse_args()

    tasks = tuple(item for item in args.tasks.replace(",", " ").split() if item)
    sizes = tuple(int(item) for item in args.ensemble_sizes.split(","))
    if not tasks or any(task not in CORE_TASKS for task in tasks):
        parser.error(f"tasks must be selected from {CORE_TASKS}")
    if not sizes or tuple(sorted(set(sizes))) != sizes or min(sizes) < 2:
        parser.error("ensemble_sizes must be unique, increasing, and at least two")
    positive = (
        args.nli_batch_size,
        args.nli_pair_chunk_size,
        args.assessment_batch_size,
        args.max_workers,
        args.bootstrap,
    )
    if any(value < 1 for value in positive) or args.request_timeout <= 0:
        parser.error("batch, worker, bootstrap, and timeout values must be positive")

    observed_commit = _source_commit()
    if observed_commit != CSA_SOURCE_COMMIT:
        raise RuntimeError(
            f"CSA source drift: expected {CSA_SOURCE_COMMIT}, got {observed_commit}"
        )
    model_path = args.nli_model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"CSA NLI model is missing: {model_path}")

    loader = UniversalDatasetLoader(args.dataset_root)
    run_dir = args.run_dir.expanduser().resolve()
    records_by_task: Dict[str, Sequence[Mapping[str, Any]]] = {}
    arrays_by_size: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {
        size: {} for size in sizes
    }
    provenance: Dict[str, Any] = {}
    for task in tasks:
        task_provenance, records, task_arrays = _load_task_inputs(
            run_dir, task, sizes, loader
        )
        records_by_task[task] = records
        provenance[task] = task_provenance
        for size in sizes:
            arrays_by_size[size][task] = task_arrays[size]

    input_fingerprint = {
        task: {
            key: provenance[task][key]
            for key in ("manifest_sha256", "sample_csv_sha256", "cache_sha256")
        }
        for task in tasks
    }
    nli_header = {
        "schema": "csa-nli-cache-v1",
        "source_commit": observed_commit,
        "nli_model_id": CSA_NLI_MODEL_ID,
        "nli_model_revision": _nli_revision(model_path),
        "representation_contract": REPRESENTATION_CONTRACT,
        "tasks": list(tasks),
        "max_ensemble_size": max(sizes),
        "inputs": input_fingerprint,
    }
    args.nli_cache = args.nli_cache.expanduser().resolve()
    nli_records = _load_jsonl_cache(args.nli_cache, nli_header)
    if args.stage in {"nli", "all"}:
        nli_records = _run_nli_stage(
            records_by_task,
            args.nli_cache,
            nli_records,
            model_path,
            args.nli_device,
            args.nli_batch_size,
            args.nli_pair_chunk_size,
            max(sizes),
        )
    expected_nli = sum(len(values) for values in records_by_task.values())
    if len(nli_records) != expected_nli:
        raise RuntimeError(
            f"CSA NLI cache incomplete: {len(nli_records)}/{expected_nli}"
        )
    if args.stage == "nli":
        print(args.nli_cache)
        return 0

    endpoints = tuple(
        value for value in args.model_urls.replace(",", " ").split() if value
    )
    if not endpoints:
        parser.error("model_urls cannot be empty")
    server_metadata = {
        endpoint: dict(_server_metadata(endpoint, args.api_key)) for endpoint in endpoints
    }
    roots = {str(value.get("root") or value.get("id")) for value in server_metadata.values()}
    ids = {str(value.get("id")) for value in server_metadata.values()}
    if ids != {args.model_name}:
        raise ValueError(f"CSA model alias mismatch: {ids}")
    if roots != {args.expected_model_revision}:
        raise ValueError(f"CSA model revision mismatch: {roots}")

    assessment_header = {
        "schema": "csa-assessment-cache-v1",
        "source_commit": observed_commit,
        "prompt_contract": PROMPT_CONTRACT,
        "nli_contract_sha256": _value_sha256(nli_header),
        "model_name": args.model_name,
        "expected_model_revision": args.expected_model_revision,
        "ensemble_sizes": list(sizes),
        "tasks": list(tasks),
        "inputs": input_fingerprint,
    }
    args.assessment_cache = args.assessment_cache.expanduser().resolve()
    assessment_records = _load_jsonl_cache(args.assessment_cache, assessment_header)
    if args.stage in {"assessment", "all"}:
        assessment_records = _run_assessment_stage(
            tasks,
            sizes,
            records_by_task,
            nli_records,
            args.assessment_cache,
            assessment_records,
            endpoints,
            args.model_name,
            args.api_key,
            args.assessment_batch_size,
            min(args.max_workers, len(endpoints)),
            args.request_timeout,
            args.request_retries,
        )
    expected_assessment = expected_nli * len(sizes)
    if len(assessment_records) != expected_assessment:
        raise RuntimeError(
            "CSA assessment cache incomplete: "
            f"{len(assessment_records)}/{expected_assessment}"
        )

    metrics_by_n: Dict[str, Any] = {}
    for size in sizes:
        for task in tasks:
            records = records_by_task[task]
            arrays_by_size[size][task]["CSA"] = np.asarray(
                [
                    assessment_records[(task, str(record["sample_id"]), size)][
                        "csa_risk"
                    ]
                    for record in records
                ],
                dtype=np.float64,
            )
        task_metrics = {
            task: {
                method: _metrics(values["error"], values[method])
                for method in ("PWSC-CI", "CSA")
            }
            for task, values in arrays_by_size[size].items()
        }
        macro_metrics = {
            method: {
                metric: float(
                    np.mean([task_metrics[task][method][metric] for task in tasks])
                )
                for metric in task_metrics[tasks[0]][method]
            }
            for method in ("PWSC-CI", "CSA")
        }
        metrics_by_n[f"N={size}"] = {
            "task_metrics": task_metrics,
            "macro_metrics": macro_metrics,
            "paired_macro_differences": {
                "PWSC-CI - CSA": _paired_macro_bootstrap(
                    arrays_by_size[size], args.seed + size, args.bootstrap
                )
            },
        }

    output = {
        "status": args.evaluation_status,
        "tasks": list(tasks),
        "ensemble_sizes": list(sizes),
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "evaluation_target": "fixed first-path error labels from the PWSC sample CSV",
        "baseline": {
            "name": "Clustered Self-Assessment (matched-cache)",
            "source_repository": "https://github.com/ccqq77/clustered_self_assessment",
            "source_commit": observed_commit,
            "nli_model": CSA_NLI_MODEL_ID,
            "nli_model_revision": _nli_revision(model_path),
            "target_model": server_metadata,
            "formula_fidelity": (
                "released bidirectional-NLI clustering rule; exact full-vocabulary "
                "option-A probability recovered with teacher-forced prompt logprob"
            ),
            "protocol_adaptation": (
                "the frozen stochastic first path replaces CSA's separately decoded "
                "greedy target so labels and candidates are paired across methods"
            ),
        },
        "representation": REPRESENTATION_CONTRACT,
        "inputs": provenance,
        "sidecar_caches": {
            "nli_path": str(args.nli_cache),
            "nli_sha256": _sha256(args.nli_cache),
            "assessment_path": str(args.assessment_cache),
            "assessment_sha256": _sha256(args.assessment_cache),
        },
        "Metrics by N": metrics_by_n,
        "claim_boundary": (
            "This is a named matched-cache adaptation, not an exact reproduction of "
            "CSA's greedy-target generation protocol. It uses only black-box sampled "
            "text and black-box target-model token probabilities."
        ),
    }
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
