"""Publication runner for the canonical PWSC-CI method.

Paper mode evaluates the complete declared split with sixteen sampled paths at
N={1,4,8,16}.  Partial runs require explicit smoke mode.  The command-line
arguments and cache schema remain compatible with existing invocations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from config.settings import ExperimentSettings
from core.adapters import (
    make_adapter,
    math_equivalence_runtime_diagnostics,
    official_math_verifier_runtime_diagnostics,
)
from core.risk_estimator import (
    PWSC_CI_CONTRACT_VERSION,
    PWSCCIRiskEstimator,
    retain_scoring_tokens,
)
from data.loaders import DatasetLoader, TaskSample
from diagnostics.analyzer import ScientificSignalAnalyzer
from llm.generator import APIGenerator
from routing.baselines import SCSVBaselines
from routing.verifier_manager import VerifierManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ise_flow.cli")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore


CORE_TASKS = ("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO")
SECONDARY_TASKS = ("AIME2025", "AIME2026")
PAPER_TASKS = CORE_TASKS + SECONDARY_TASKS
PRIMARY_TASKS = PAPER_TASKS
DEFAULT_ENSEMBLE_SIZES = (1, 4, 8, 16)
CACHE_SCHEMA_VERSION = 2
BOOTSTRAP_SAMPLES = 1000
PWSC_CI_NAME = "PWSC-CI"


def filter_reasoning_logprobs_full(
    logprob_items: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compatibility hook: PWSC-CI scores reasoning and answer tokens alike."""
    return list(logprob_items or [])


def extract_homotopy_outcome(text: str, task_name: str) -> str:
    """Compatibility wrapper around the canonical task adapter."""
    return make_adapter(task_name, [None]).outcome_keys([text])[0]


def parse_ensemble_sizes(value: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        try:
            sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError("ensemble sizes must be comma-separated integers") from exc
    else:
        sizes = tuple(int(item) for item in value)
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("ensemble sizes must be positive")
    if tuple(sorted(set(sizes))) != sizes:
        raise ValueError("ensemble sizes must be unique and increasing")
    return sizes


def parse_stop_token_ids(value: str | Sequence[int]) -> Tuple[int, ...]:
    """Parse the optional vLLM stop-token contract into a canonical tuple."""
    if isinstance(value, str):
        if not value.strip():
            return ()
        try:
            token_ids = tuple(
                int(item.strip()) for item in value.split(",") if item.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "stop token ids must be comma-separated integers"
            ) from exc
    else:
        try:
            token_ids = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError("stop token ids must be integers") from exc
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("stop token ids must be nonnegative")
    return tuple(sorted(set(token_ids)))


def validate_protocol(
    run_mode: str,
    task_name: str,
    sample_limit: int,
    num_samples: int,
    ensemble_sizes: Sequence[int],
) -> None:
    if num_samples < max(ensemble_sizes):
        raise ValueError("num_samples must cover the largest requested ensemble size")
    if run_mode == "paper":
        if task_name not in PAPER_TASKS:
            raise ValueError(f"paper mode is restricted to declared paper tasks: {PAPER_TASKS}")
        if sample_limit != 0:
            raise ValueError(
                "paper mode requires --limit 0 (the complete split); use --mode smoke for subsets"
            )
        if num_samples != 16 or tuple(ensemble_sizes) != DEFAULT_ENSEMBLE_SIZES:
            raise ValueError(
                "paper mode requires --num_samples 16 and --ensemble_sizes 1,4,8,16"
            )
    elif run_mode == "stress":
        if task_name in PAPER_TASKS:
            raise ValueError(
                "stress mode is for optional domains; use paper mode for declared paper tasks"
            )
    elif run_mode == "smoke":
        if sample_limit <= 0:
            raise ValueError("smoke mode requires an explicit positive --limit")
    else:
        raise ValueError(f"unknown run mode: {run_mode}")


def compute_academic_metrics(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    is_correct_list: Optional[Sequence[bool]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    if isinstance(is_correct_list, (int, np.integer)) and seed == 0:
        seed = int(is_correct_list)
        is_correct_list = None
    auroc = ScientificSignalAnalyzer.calculate_auroc(y_true, y_scores)
    auprc = ScientificSignalAnalyzer.calculate_auprc(y_true, y_scores)
    aurc = ScientificSignalAnalyzer.calculate_aurc(y_true, y_scores)
    _, ci_low, ci_high = ScientificSignalAnalyzer.bootstrap_ci(
        y_true,
        y_scores,
        ScientificSignalAnalyzer.calculate_auroc,
        seed=seed,
        n_bootstrap=BOOTSTRAP_SAMPLES,
    )
    result: Dict[str, Any] = {
        "AUROC": auroc,
        "AUPRC": auprc,
        "AURC": aurc,
        "SelectiveAcc@50": ScientificSignalAnalyzer.calculate_selective_accuracy(
            y_true, y_scores, 0.5
        ),
        "AUROC_CI95": [ci_low, ci_high],
        "ECE": None,
        "Brier": None,
    }
    if is_correct_list is not None:
        result["Top-50% ACC"] = ScientificSignalAnalyzer.calculate_top_50_acc(
            is_correct_list, y_scores
        )
    return result


class UniversalDatasetLoader(DatasetLoader):
    def load_any_dataset(
        self, task_name: str, limit: Optional[int] = None
    ) -> List[TaskSample]:
        if task_name == "GSM8K":
            samples = self.load_gsm8k_split()[1]
            return samples[:limit] if limit and limit > 0 else samples
        if task_name == "HumanEvalPlus":
            return self.load_humanevalplus_test(limit)
        if task_name == "MBPPPlus":
            return self.load_mbppplus_test(limit)
        if task_name == "AIME2025":
            return self.load_aime2025_split(limit)
        if task_name == "AIME2026":
            return self.load_aime2026_split(limit)
        if task_name == "FOLIO":
            return self.load_folio_split(limit)
        if task_name == "P-FOLIO":
            return self.load_p_folio_split(limit)
        if task_name == "LiveCodeBench":
            return self.load_livecodebench_split(limit)
        if task_name == "MATH-500":
            return self.load_math500_split(limit)
        raise NotImplementedError(f"Unsupported dataset domain: {task_name}")


def _verifier_id(task_name: str) -> str:
    return {
        "HumanEvalPlus": "humaneval_plus_tester",
        "MBPPPlus": "mbpp_plus_tester",
        "LiveCodeBench": "lcb_tester",
        "AIME2025": "math_official_tester",
        "AIME2026": "math_official_tester",
        "MATH-500": "math_official_tester",
        "FOLIO": "folio_tester",
        "P-FOLIO": "folio_tester",
        "GSM8K": "gsm8k_official_tester",
    }[task_name]


def _sample_metadata(sample: TaskSample, task_name: str) -> Dict[str, Any]:
    metadata = dict(sample.metadata or {})
    metadata.update(sample.prompt_data or {})
    metadata.setdefault("prompt", sample.raw_question)
    if task_name in {"HumanEvalPlus", "MBPPPlus"}:
        metadata.setdefault("entry_point", sample.prompt_data.get("entry_point", ""))
        metadata.setdefault("task_id", sample.sample_id)
    return metadata


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _redact_command(command: Sequence[str]) -> List[str]:
    redacted: List[str] = []
    hide_next = False
    for item in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
        elif item == "--api_key":
            redacted.append(item)
            hide_next = True
        elif item.startswith("--api_key="):
            redacted.append("--api_key=<redacted>")
        else:
            redacted.append(item)
    return redacted


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_sha256(samples: Sequence[TaskSample]) -> str:
    payload = [
        {"sample_id": sample.sample_id, "prompt": sample.raw_question}
        for sample in samples
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _trace_batch_complete(
    traces: Sequence[Dict[str, Any]], num_samples: int
) -> bool:
    return not _trace_batch_failure_reasons(traces, num_samples)


def _trace_batch_failure_reasons(
    traces: Sequence[Dict[str, Any]], num_samples: int
) -> List[str]:
    reasons: List[str] = []
    trace_indices_by_reason: Dict[str, List[int]] = {}
    if len(traces) < num_samples:
        reasons.append(f"missing_choices:{num_samples - len(traces)}")
    for index, trace in enumerate(traces[:num_samples]):
        error = str(trace.get("error", "")).strip()
        if not str(trace.get("text", "")).strip():
            reason = error or "empty_response"
        elif not retain_scoring_tokens(trace.get("logprobs", [])):
            reason = error or "missing_scoring_logprobs"
        else:
            continue
        trace_indices_by_reason.setdefault(reason, []).append(index)
    for reason, indices in trace_indices_by_reason.items():
        if indices == list(range(indices[0], indices[-1] + 1)):
            index_label = (
                str(indices[0])
                if len(indices) == 1
                else f"{indices[0]}-{indices[-1]}"
            )
        else:
            index_label = ",".join(str(index) for index in indices)
        reasons.append(f"traces[{index_label}]:{reason}")
    return reasons


def _majority_index(outcomes: Sequence[str], confidences: Sequence[float]) -> int:
    counts = Counter(outcomes)
    first_position = {key: list(outcomes).index(key) for key in counts}
    confidence_by_key = {
        key: sum(
            float(confidences[index])
            for index, value in enumerate(outcomes)
            if value == key
        )
        for key in counts
    }
    winning_key = max(
        counts,
        key=lambda key: (counts[key], confidence_by_key[key], -first_position[key]),
    )
    candidates = [index for index, value in enumerate(outcomes) if value == winning_key]
    return max(candidates, key=lambda index: (float(confidences[index]), -index))


def _mbr_index(kernel: np.ndarray, weights: Optional[np.ndarray]) -> int:
    matrix = np.asarray(kernel, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return 0
    if weights is None:
        normalized = np.ones(matrix.shape[0], dtype=np.float64) / matrix.shape[0]
    else:
        normalized = np.asarray(weights, dtype=np.float64)
        normalized = (
            normalized / normalized.sum()
            if normalized.sum() > 0
            else np.ones(matrix.shape[0], dtype=np.float64) / matrix.shape[0]
        )
    return int(np.argmax(matrix @ normalized))


class AcademicExperimentRunner:
    def __init__(
        self,
        settings: Optional[ExperimentSettings] = None,
        model_name: str = "deepseek",
        model_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "empty",
        top_k: int = 10,
        dataset_root: str = "/data/newmodel/graphunagent/datasets",
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        max_tokens: int = 12000,
        max_workers: int = 8,
        generation_batch_size: int = 32,
        request_timeout: float = 3600.0,
        max_retries: int = 0,
        stop_token_ids: Optional[Sequence[int]] = None,
        sampling_top_k: Optional[int] = None,
        expected_model_revision: str = "",
        cache_output_dir: str = "",
    ) -> None:
        self.settings = settings or ExperimentSettings()
        self.model_name = model_name
        self.model_url = model_url
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.top_p = None if top_p is None else float(top_p)
        self.max_tokens = int(max_tokens)
        self.max_workers = int(max_workers)
        self.generation_batch_size = int(generation_batch_size)
        self.request_timeout = float(request_timeout)
        self.max_retries = int(max_retries)
        self.stop_token_ids = parse_stop_token_ids(stop_token_ids or ())
        self.sampling_top_k = (
            None if sampling_top_k is None else int(sampling_top_k)
        )
        if self.sampling_top_k is not None and self.sampling_top_k < 0:
            raise ValueError("sampling_top_k must be nonnegative")
        self.expected_model_revision = expected_model_revision
        self.cache_output_dir = (
            Path(cache_output_dir).expanduser().resolve()
            if cache_output_dir
            else None
        )
        if self.max_tokens < 1 or self.max_workers < 1 or self.generation_batch_size < 1:
            raise ValueError(
                "max_tokens, max_workers, and generation_batch_size must be positive"
            )
        if self.request_timeout <= 0 or self.max_retries < 0:
            raise ValueError(
                "request_timeout must be positive and max_retries nonnegative"
            )
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must lie in (0, 1]")
        self.loader = UniversalDatasetLoader(dataset_root)
        self.verifier_mgr = VerifierManager()
        self.generator = APIGenerator(
            model_name=model_name,
            api_key=api_key,
            base_url=model_url,
            seed=self.settings.random_seed,
            top_k_logprobs=top_k,
            request_timeout=self.request_timeout,
            max_retries=self.max_retries,
            stop_token_ids=self.stop_token_ids,
            sampling_top_k=self.sampling_top_k,
        )
        self.risk_estimator = PWSCCIRiskEstimator(
            weakest_k=self.settings.pwsc_ci.weakest_k
        )

    def _model_metadata(
        self, cache_header: Optional[Dict[str, Any]], run_mode: str
    ) -> Dict[str, Any]:
        if cache_header is not None:
            metadata = dict(cache_header.get("model", {}))
        else:
            try:
                metadata = self.generator.describe_model()
            except Exception:
                if run_mode == "paper":
                    raise
                metadata = {
                    "id": self.model_name,
                    "root": None,
                    "metadata_unavailable": True,
                }
        if metadata.get("id") != self.model_name:
            raise ValueError(
                f"model mismatch: cache/server has {metadata.get('id')!r}, "
                f"CLI requested {self.model_name!r}"
            )
        revision = str(
            metadata.get("root") or metadata.get("revision") or metadata.get("id") or ""
        )
        if self.expected_model_revision and revision != self.expected_model_revision:
            raise ValueError(
                f"model revision mismatch: expected {self.expected_model_revision!r}, "
                f"got {revision!r}"
            )
        metadata["resolved_revision"] = revision
        root = Path(revision)
        if root.is_dir():
            descriptor_names = (
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "tokenizer_config.json",
            )
            descriptors = []
            for name in descriptor_names:
                path = root / name
                if path.exists():
                    descriptors.append(
                        {
                            "name": name,
                            "bytes": path.stat().st_size,
                            "sha256": _sha256_file(path),
                        }
                    )
            shards = [
                {"name": path.name, "bytes": path.stat().st_size}
                for path in sorted(root.glob("*.safetensors"))
            ]
            fingerprint_payload = {
                "descriptors": descriptors,
                "weight_shards": shards,
            }
            metadata["local_fingerprint"] = {
                **fingerprint_payload,
                "descriptor_sha256": hashlib.sha256(
                    json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
        return metadata

    @staticmethod
    def _read_cache_header(path: Path) -> Dict[str, Any]:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
        if not first:
            raise ValueError(f"empty generation cache: {path}")
        header = json.loads(first)
        if (
            header.get("record_type") != "meta"
            or header.get("schema_version") != CACHE_SCHEMA_VERSION
        ):
            raise ValueError("cache is not a PWSC-CI schema-v2 generation cache")
        return header

    def _validate_cache_header(
        self,
        header: Dict[str, Any],
        task_name: str,
        samples: Sequence[TaskSample],
        num_samples: int,
        dataset_fingerprint: Dict[str, Any],
        run_mode: str,
    ) -> None:
        expected = {
            "run_mode": run_mode,
            "task": task_name,
            "seed": self.settings.random_seed,
            "num_samples": num_samples,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "requested_top_logprobs": self.top_k,
            "stop_token_ids": list(self.stop_token_ids),
            "sampling_top_k": self.sampling_top_k,
            "generation_protocol": self.generator.generation_protocol(task_name),
            "system_prompt_sha256": hashlib.sha256(
                self.generator._system_prompt(task_name).encode("utf-8")
            ).hexdigest(),
            "input_sha256": _input_sha256(samples),
        }
        for key, value in expected.items():
            observed = header.get(key, [] if key == "stop_token_ids" else None)
            if observed != value:
                raise ValueError(
                    f"cache {key} mismatch: expected {value!r}, got {observed!r}"
                )
        cached_hash_seed = header.get("python_hash_seed")
        if cached_hash_seed is not None and str(cached_hash_seed) != os.environ.get(
            "PYTHONHASHSEED"
        ):
            raise ValueError(
                "cache Python hash seed does not match the current experiment seed"
            )
        if header.get("dataset") != dataset_fingerprint:
            raise ValueError(
                "cache dataset fingerprint does not match current local data"
            )

    def _prepare_generation_cache(
        self,
        task_name: str,
        samples: Sequence[TaskSample],
        num_samples: int,
        run_mode: str,
        dataset_fingerprint: Dict[str, Any],
        cache_path: Optional[str],
        resume_cache: bool = False,
    ) -> Tuple[Path, str, Dict[str, Any]]:
        if cache_path:
            raw_cache_path = str(cache_path)
            if any(character in raw_cache_path for character in "\r\n\0"):
                raise ValueError(
                    "--cache_path contains a newline or NUL control character: "
                    f"{raw_cache_path!r}"
                )
            path = Path(raw_cache_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"generation cache not found: {path}")
            header = self._read_cache_header(path)
            self._validate_cache_header(
                header,
                task_name,
                samples,
                num_samples,
                dataset_fingerprint,
                run_mode,
            )
            model_metadata = self._model_metadata(header, run_mode)
            completed = self._cache_record_count(path, samples, num_samples)
            if completed < len(samples):
                if not resume_cache:
                    raise ValueError(
                        f"generation cache has {completed}/{len(samples)} complete "
                        "records; pass --resume_cache to append the missing records"
                    )
                with path.open("a", encoding="utf-8") as handle:
                    self._generate_cache_records(
                        handle,
                        task_name,
                        samples[completed:],
                        num_samples,
                    )
            return path, _sha256_file(path), model_metadata

        model_metadata = self._model_metadata(None, run_mode)
        cache_dir = self.cache_output_dir or (
            Path(__file__).resolve().parent / "diagnostics" / f"{run_mode}_cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / (
            f"generations_{task_name}_{time.time_ns()}_{_input_sha256(samples)[:10]}.jsonl"
        )
        header = {
            "record_type": "meta",
            "schema_version": CACHE_SCHEMA_VERSION,
            "method": self.settings.pwsc_ci.method_version,
            "run_mode": run_mode,
            "task": task_name,
            "seed": self.settings.random_seed,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "num_samples": num_samples,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "requested_top_logprobs": self.top_k,
            "stop_token_ids": list(self.stop_token_ids),
            "sampling_top_k": self.sampling_top_k,
            "generation_protocol": self.generator.generation_protocol(task_name),
            "system_prompt": self.generator._system_prompt(task_name),
            "system_prompt_sha256": hashlib.sha256(
                self.generator._system_prompt(task_name).encode("utf-8")
            ).hexdigest(),
            "input_sha256": _input_sha256(samples),
            "dataset": dataset_fingerprint,
            "model": model_metadata,
        }
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_json_safe(header), sort_keys=True, ensure_ascii=False)
                + "\n"
            )
            self._generate_cache_records(
                handle, task_name, samples, num_samples
            )
        return path, _sha256_file(path), model_metadata

    def _generate_cache_records(
        self,
        handle: Any,
        task_name: str,
        samples: Sequence[TaskSample],
        num_samples: int,
    ) -> None:
        for start in range(0, len(samples), self.generation_batch_size):
            batch = samples[start : start + self.generation_batch_size]
            generated = self.generator.generate_consistency_samples(
                [sample.raw_question for sample in batch],
                num_samples=num_samples,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                max_workers=self.max_workers,
                task_type=task_name,
            )
            if len(generated) != len(batch):
                raise RuntimeError(
                    f"generator returned {len(generated)} batches for "
                    f"{len(batch)} prompts"
                )
            for sample, traces in zip(batch, generated):
                failure_reasons = _trace_batch_failure_reasons(
                    traces, num_samples
                )
                if failure_reasons:
                    cache_name = getattr(handle, "name", "")
                    resume_hint = ""
                    if cache_name:
                        cache_path = Path(cache_name).resolve()
                        resume_hint = (
                            f" Complete cache prefix retained at {cache_path};"
                            f" resume with --cache_path {cache_path} --resume_cache."
                        )
                    raise RuntimeError(
                        f"generator returned incomplete traces for {sample.sample_id}: "
                        f"{', '.join(failure_reasons)}.{resume_hint}"
                    )
                record = {
                    "record_type": "sample",
                    "sample_id": sample.sample_id,
                    "prompt_sha256": hashlib.sha256(
                        sample.raw_question.encode("utf-8")
                    ).hexdigest(),
                    "traces": traces,
                }
                handle.write(
                    json.dumps(_json_safe(record), sort_keys=True, ensure_ascii=False)
                    + "\n"
                )
            handle.flush()

    @staticmethod
    def _cache_record_count(
        path: Path,
        samples: Sequence[TaskSample],
        num_samples: int,
    ) -> int:
        """Return the complete aligned prefix length of a schema-v2 cache."""
        with path.open(encoding="utf-8") as handle:
            next(handle)
            count = 0
            for line in handle:
                if count >= len(samples):
                    raise ValueError("generation cache contains extra sample records")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "generation cache ends with a malformed record; do not resume it"
                    ) from exc
                sample = samples[count]
                expected_prompt = hashlib.sha256(
                    sample.raw_question.encode("utf-8")
                ).hexdigest()
                if (
                    record.get("record_type") != "sample"
                    or record.get("sample_id") != sample.sample_id
                    or record.get("prompt_sha256") != expected_prompt
                ):
                    raise ValueError(f"cache sample mismatch at index {count}")
                if not _trace_batch_complete(record.get("traces", []), num_samples):
                    raise ValueError(
                        f"cache record {sample.sample_id} is incomplete; do not resume it"
                    )
                count += 1
        return count

    @staticmethod
    def _iter_generation_cache(
        path: Path, samples: Sequence[TaskSample]
    ) -> Iterator[Tuple[TaskSample, List[Dict[str, Any]]]]:
        with path.open(encoding="utf-8") as handle:
            header = json.loads(handle.readline())
            if header.get("record_type") != "meta":
                raise ValueError("generation cache is missing its metadata header")
            count = 0
            for sample, line in zip(samples, handle):
                record = json.loads(line)
                prompt_hash = hashlib.sha256(
                    sample.raw_question.encode("utf-8")
                ).hexdigest()
                if (
                    record.get("record_type") != "sample"
                    or record.get("sample_id") != sample.sample_id
                ):
                    raise ValueError(f"cache sample order mismatch at index {count}")
                if record.get("prompt_sha256") != prompt_hash:
                    raise ValueError(f"cache prompt mismatch for {sample.sample_id}")
                count += 1
                yield sample, list(record.get("traces", []))
            if count != len(samples):
                raise ValueError(
                    f"cache contains {count} aligned samples; expected {len(samples)}"
                )
            if any(line.strip() for line in handle):
                raise ValueError("cache contains extra sample records")

    def preflight(
        self,
        task_name: str,
        sample_limit: int,
        num_samples: int,
        ensemble_sizes: Sequence[int],
        run_mode: str,
        cache_path: Optional[str] = None,
        resume_cache: bool = False,
    ) -> Dict[str, Any]:
        validate_protocol(
            run_mode, task_name, sample_limit, num_samples, ensemble_sizes
        )
        samples = self.loader.load_any_dataset(task_name, sample_limit)
        dataset = self.loader.dataset_fingerprint(task_name)
        cache_header = (
            self._read_cache_header(Path(cache_path).resolve()) if cache_path else None
        )
        model = self._model_metadata(cache_header, run_mode)
        if cache_header:
            self._validate_cache_header(
                cache_header, task_name, samples, num_samples, dataset, run_mode
            )
        completed = (
            self._cache_record_count(Path(cache_path).resolve(), samples, num_samples)
            if cache_path
            else 0
        )
        if cache_path and completed < len(samples) and not resume_cache:
            raise ValueError(
                f"generation cache has {completed}/{len(samples)} complete records; "
                "pass --resume_cache to continue"
            )
        return {
            "status": "ready",
            "method": self.settings.pwsc_ci.method_version,
            "run_mode": run_mode,
            "task": task_name,
            "sample_count": len(samples),
            "num_samples": num_samples,
            "ensemble_sizes": list(ensemble_sizes),
            "seed": self.settings.random_seed,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "max_workers": self.max_workers,
            "generation_batch_size": self.generation_batch_size,
            "stop_token_ids": list(self.stop_token_ids),
            "sampling_top_k": self.sampling_top_k,
            "generation_protocol": self.generator.generation_protocol(task_name),
            "pwsc_ci_contract": PWSC_CI_CONTRACT_VERSION,
            "system_prompt_sha256": hashlib.sha256(
                self.generator._system_prompt(task_name).encode("utf-8")
            ).hexdigest(),
            "model": model,
            "dataset": dataset,
            "cache_path": str(Path(cache_path).resolve()) if cache_path else None,
            "cache_output_dir": (
                str(self.cache_output_dir) if self.cache_output_dir else None
            ),
            "cache_records_completed": completed if cache_path else None,
            "cache_complete": completed == len(samples) if cache_path else None,
            "resume_cache": bool(resume_cache),
        }

    def run_benchmark_experiment(
        self,
        task_name: str = "GSM8K",
        sample_limit: int = 0,
        num_samples: int = 16,
        ensemble_sizes: Sequence[int] = DEFAULT_ENSEMBLE_SIZES,
        run_mode: str = "paper",
        cache_path: Optional[str] = None,
        resume_cache: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float, List[Dict[str, Any]]]:
        run_started = time.perf_counter()
        sizes = parse_ensemble_sizes(ensemble_sizes)
        validate_protocol(run_mode, task_name, sample_limit, num_samples, sizes)
        samples = self.loader.load_any_dataset(task_name, sample_limit)
        if not samples:
            raise ValueError(f"No samples loaded for {task_name}")
        dataset_fingerprint = self.loader.dataset_fingerprint(task_name)
        cache_started = time.perf_counter()
        generation_cache, cache_sha256, model_metadata = self._prepare_generation_cache(
            task_name,
            samples,
            num_samples,
            run_mode,
            dataset_fingerprint,
            cache_path,
            resume_cache,
        )
        cache_or_generation_seconds = time.perf_counter() - cache_started
        verifier = self.verifier_mgr.verifiers[_verifier_id(task_name)]
        worst_k_nll_name = (
            f"Worst-k NLL (k={self.settings.pwsc_ci.weakest_k})"
        )

        labels_by_n: Dict[int, List[int]] = {size: [] for size in sizes}
        scores_by_n: Dict[int, Dict[str, List[float]]] = {
            size: {PWSC_CI_NAME: [], "Mean NLL": [], worst_k_nll_name: []}
            for size in sizes
        }
        for size in sizes:
            if size > 1:
                scores_by_n[size].update(
                    {
                        "Self-Consistency": [],
                        "Semantic Entropy": [],
                        "MBR": [],
                        "Target Consistency (task-kernel adaptation)": [],
                        "KLE (task-kernel adaptation)": [],
                        "CoCoA-MSP (outcome-kernel adaptation)": [],
                        "CoCoA-PPL (outcome-kernel adaptation)": [],
                    }
                )
        selection_correct_by_n: Dict[int, Dict[str, List[bool]]] = {
            size: {
                "First Path": [],
                PWSC_CI_NAME: [],
                "Self-Consistency": [],
                "MBR": [],
            }
            for size in sizes
        }
        ci_information_diagnostics: Dict[int, List[Dict[str, float]]] = {
            size: [] for size in sizes
        }
        rows: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        records: Iterable[Tuple[TaskSample, List[Dict[str, Any]]]] = (
            self._iter_generation_cache(generation_cache, samples)
        )
        if tqdm and os.environ.get("TQDM_DISABLE", "").lower() not in {
            "1",
            "true",
            "yes",
        }:
            records = tqdm(
                records,
                total=len(samples),
                desc=f"PWSC-CI paper evaluation: {task_name}",
            )

        scoring_started = time.perf_counter()
        for sample, traces in records:
            if not _trace_batch_complete(traces, num_samples):
                reason = "incomplete_text_or_logprobs"
                if run_mode == "paper":
                    raise RuntimeError(
                        f"publication run aborted at {sample.sample_id}: {reason}"
                    )
                skipped.append({"sample_id": sample.sample_id, "reason": reason})
                continue
            traces = traces[:num_samples]
            texts = [str(trace["text"]) for trace in traces]
            logprob_paths = [
                filter_reasoning_logprobs_full(trace["logprobs"])
                for trace in traces
            ]
            metadata = _sample_metadata(sample, task_name)
            correctness_cache: Dict[int, bool] = {}

            def is_correct(index: int) -> bool:
                if index not in correctness_cache:
                    correctness_cache[index] = verifier.execute(
                        texts[index], sample.gold_answer, metadata
                    )
                return correctness_cache[index]

            base_correct = is_correct(0)
            base_wrong = int(not base_correct)
            mean_nll = SCSVBaselines.calculate_nll(logprob_paths[0])
            worst_nll = SCSVBaselines.calculate_worst_k_nll(
                logprob_paths[0], self.settings.pwsc_ci.weakest_k
            )

            for size in sizes:
                prefix_texts = texts[:size]
                prefix_logprobs = logprob_paths[:size]
                risk, diagnostics = self.risk_estimator.score(
                    prefix_logprobs,
                    prefix_texts,
                    task_name,
                    [metadata] * size,
                )
                outcomes = diagnostics["outcomes"]
                confidences = diagnostics["q"]
                ci_index = int(diagnostics["selected_index"])
                sc_index = _majority_index(outcomes, confidences)
                task_kernel = diagnostics["task_kernel"]
                mbr_index = _mbr_index(task_kernel, diagnostics.get("weights"))
                components = dict(diagnostics["ci_components"])

                labels_by_n[size].append(base_wrong)
                scores_by_n[size][PWSC_CI_NAME].append(float(risk))
                scores_by_n[size]["Mean NLL"].append(float(mean_nll))
                scores_by_n[size][worst_k_nll_name].append(float(worst_nll))
                if size > 1:
                    scores_by_n[size]["Self-Consistency"].append(
                        SCSVBaselines.calculate_self_consistency(
                            prefix_texts, outcomes
                        )
                    )
                    scores_by_n[size]["Semantic Entropy"].append(
                        SCSVBaselines.calculate_semantic_entropy(
                            prefix_texts, outcomes
                        )
                    )
                    scores_by_n[size]["MBR"].append(
                        SCSVBaselines.calculate_mbr(
                            task_kernel, diagnostics.get("weights")
                        )
                    )
                    scores_by_n[size][
                        "Target Consistency (task-kernel adaptation)"
                    ].append(
                        SCSVBaselines.calculate_target_consistency(
                            task_kernel
                        )
                    )
                    scores_by_n[size]["KLE (task-kernel adaptation)"].append(
                        SCSVBaselines.calculate_kernel_language_entropy(
                            task_kernel
                        )
                    )
                    scores_by_n[size][
                        "CoCoA-MSP (outcome-kernel adaptation)"
                    ].append(
                        SCSVBaselines.calculate_cocoa_msp(
                            prefix_logprobs[0], task_kernel
                        )
                    )
                    scores_by_n[size][
                        "CoCoA-PPL (outcome-kernel adaptation)"
                    ].append(
                        SCSVBaselines.calculate_cocoa_ppl(
                            prefix_logprobs[0], task_kernel
                        )
                    )

                selection_correct_by_n[size]["First Path"].append(base_correct)
                selection_correct_by_n[size][PWSC_CI_NAME].append(
                    is_correct(ci_index)
                )
                selection_correct_by_n[size]["Self-Consistency"].append(
                    is_correct(sc_index)
                )
                selection_correct_by_n[size]["MBR"].append(is_correct(mbr_index))
                ci_information_diagnostics[size].append(components)

                row: Dict[str, Any] = {
                    "sample_id": sample.sample_id,
                    "task": task_name,
                    "ensemble_size": size,
                    "base_is_correct": base_correct,
                    "fixed_first_error": base_wrong,
                    "pwsc_ci_risk": float(risk),
                    "pwsc_ci_holevo_information": components[
                        "holevo_information"
                    ],
                    "pwsc_ci_partition_entropy": components[
                        "partition_entropy"
                    ],
                    "pwsc_ci_target_mass_ratio": components[
                        "target_mass_ratio"
                    ],
                    "pwsc_ci_peer_mass_ratio": components["peer_mass_ratio"],
                    "pwsc_ci_peer_entropy": components["peer_entropy"],
                    "pwsc_ci_mixture_entropy": components["mixture_entropy"],
                    "pwsc_ci_evidence_deficit": components["evidence_deficit"],
                    "pwsc_ci_degenerate": bool(components["degenerate"]),
                    "mean_nll": float(mean_nll),
                    "worst_5_nll": float(worst_nll),
                    "target_path_confidence": float(diagnostics["q"][0]),
                    "mean_path_confidence": float(np.mean(diagnostics["q"])),
                    "valid_path_count": int(diagnostics["valid_count"]),
                    "pwsc_ci_kernel_source": diagnostics["kernel_source"],
                    "pwsc_ci_probe_count": int(
                        diagnostics["kernel_diagnostics"].get("probe_count", 0)
                    ),
                    "pwsc_ci_active_probe_count": int(
                        diagnostics["kernel_diagnostics"].get(
                            "active_probe_count", 0
                        )
                    ),
                    "pwsc_ci_successful_calls": int(
                        diagnostics["kernel_diagnostics"].get(
                            "successful_calls", 0
                        )
                    ),
                    "pwsc_ci_selected_index": ci_index,
                    "pwsc_ci_selected_is_correct": is_correct(ci_index),
                    "self_consistency_selected_index": sc_index,
                    "self_consistency_selected_is_correct": is_correct(sc_index),
                    "mbr_selected_index": mbr_index,
                    "mbr_selected_is_correct": is_correct(mbr_index),
                }
                if size > 1:
                    for name, values in scores_by_n[size].items():
                        if name != PWSC_CI_NAME and values:
                            key = (
                                name.lower()
                                .replace(" ", "_")
                                .replace("-", "_")
                                .replace("(", "")
                                .replace(")", "")
                            )
                            row[f"baseline_{key}"] = float(values[-1])
                rows.append(row)

        scoring_seconds = time.perf_counter() - scoring_started
        analysis_started = time.perf_counter()
        scored_questions = len(labels_by_n[sizes[-1]])
        methods_by_n: Dict[str, Dict[str, Any]] = {}
        paired_by_n: Dict[str, Dict[str, Any]] = {}
        selection_by_n: Dict[str, Dict[str, float]] = {}
        diagnostics_by_n: Dict[str, Dict[str, Any]] = {}
        for size in sizes:
            key = f"N={size}"
            labels = labels_by_n[size]
            methods_by_n[key] = {
                name: compute_academic_metrics(
                    labels,
                    values,
                    seed=self.settings.random_seed,
                )
                for name, values in scores_by_n[size].items()
            }
            comparisons: Dict[str, Any] = {}
            for name, values in scores_by_n[size].items():
                if name == PWSC_CI_NAME:
                    continue
                comparisons[f"{PWSC_CI_NAME} - {name}"] = {
                    "AUROC": ScientificSignalAnalyzer.paired_bootstrap_difference(
                        labels,
                        scores_by_n[size][PWSC_CI_NAME],
                        values,
                        ScientificSignalAnalyzer.calculate_auroc,
                        seed=self.settings.random_seed,
                        n_bootstrap=BOOTSTRAP_SAMPLES,
                    ),
                    "AURC": ScientificSignalAnalyzer.paired_bootstrap_difference(
                        labels,
                        scores_by_n[size][PWSC_CI_NAME],
                        values,
                        ScientificSignalAnalyzer.calculate_aurc,
                        seed=self.settings.random_seed,
                        n_bootstrap=BOOTSTRAP_SAMPLES,
                    ),
                }
            paired_by_n[key] = comparisons
            selection_by_n[key] = {
                name: float(np.mean(values))
                for name, values in selection_correct_by_n[size].items()
            }
            diagnostics_by_n[key] = self._ci_group_report(
                labels, scores_by_n[size][PWSC_CI_NAME]
            )

        max_size = sizes[-1]
        max_key = f"N={max_size}"
        correlation = ScientificSignalAnalyzer.calculate_pearson_correlation(
            scores_by_n[max_size][PWSC_CI_NAME], labels_by_n[max_size]
        )
        diagnostic_report = dict(diagnostics_by_n[max_key])
        diagnostic_report["pwsc_ci_risk_error_correlation"] = correlation
        diagnostic_report["skipped_samples"] = skipped

        publication_reasons: List[str] = []
        if run_mode != "paper":
            publication_reasons.append(f"run_mode={run_mode}")
        if task_name not in PAPER_TASKS:
            publication_reasons.append("not a declared paper task")
        if sample_limit != 0:
            publication_reasons.append("partial dataset")
        if tuple(sizes) != DEFAULT_ENSEMBLE_SIZES or num_samples != 16:
            publication_reasons.append("non-canonical ensemble protocol")
        if skipped or scored_questions != len(samples):
            publication_reasons.append("incomplete sample coverage")

        class_counts = {
            "correct": int(
                len(labels_by_n[max_size]) - sum(labels_by_n[max_size])
            ),
            "incorrect": int(sum(labels_by_n[max_size])),
        }
        statistical_caveats: List[str] = []
        if min(class_counts.values()) < 5:
            statistical_caveats.append(
                "fewer than five questions in one outcome class; AUROC and "
                "bootstrap intervals are unstable"
            )

        ci_summary = {
            f"N={size}": {
                "mean_normalized_information": float(
                    np.mean(
                        [
                            item["normalized_information"]
                            for item in ci_information_diagnostics[size]
                        ]
                    )
                ),
                "mean_holevo_information": float(
                    np.mean(
                        [
                            item["holevo_information"]
                            for item in ci_information_diagnostics[size]
                        ]
                    )
                ),
                "mean_partition_entropy": float(
                    np.mean(
                        [
                            item["partition_entropy"]
                            for item in ci_information_diagnostics[size]
                        ]
                    )
                ),
                "mean_target_mass_ratio": float(
                    np.mean(
                        [
                            item["target_mass_ratio"]
                            for item in ci_information_diagnostics[size]
                        ]
                    )
                ),
                "mean_evidence_deficit": float(
                    np.mean(
                        [
                            item["evidence_deficit"]
                            for item in ci_information_diagnostics[size]
                        ]
                    )
                ),
                "degenerate_count": int(
                    sum(
                        bool(item["degenerate"])
                        for item in ci_information_diagnostics[size]
                    )
                ),
                "maximum_holevo_bound_excess": float(
                    max(
                        item["holevo_information"] - item["partition_entropy"]
                        for item in ci_information_diagnostics[size]
                    )
                ),
            }
            for size in sizes
        }

        analysis_seconds = time.perf_counter() - analysis_started
        total_seconds = time.perf_counter() - run_started
        report = {
            "method": self.settings.pwsc_ci.method_version,
            "task": task_name,
            "benchmark_tier": "core" if task_name in CORE_TASKS else "secondary",
            "run_mode": run_mode,
            "publication_eligible": not publication_reasons,
            "publication_ineligibility_reasons": publication_reasons,
            "n_requested": len(samples),
            "n_scored_questions": scored_questions,
            "n_skipped": len(skipped),
            "evaluation_target": (
                "correctness of the fixed first sampled path for every UQ method"
            ),
            "class_counts": class_counts,
            "statistical_caveats": statistical_caveats,
            "Methods by N": methods_by_n,
            "Selection Accuracy by N": selection_by_n,
            "PWSC-CI Paired Bootstrap Differences by N": paired_by_n,
            "Paired Bootstrap Differences by N": paired_by_n,
            "PWSC-CI Information Diagnostics by N": ci_summary,
            "Math Equivalence Runtime Diagnostics": {
                "gold_label_verification": official_math_verifier_runtime_diagnostics(),
                "task_answer_equivalence": math_equivalence_runtime_diagnostics(),
            },
            "Diagnostics by N": diagnostics_by_n,
            "Single-Path Methods": methods_by_n["N=1"],
            "Multi-Path Methods": methods_by_n[max_key],
            "run_metadata": {
                "seed": self.settings.random_seed,
                "model_identifier": self.model_name,
                "model_revision": model_metadata.get("resolved_revision"),
                "model_metadata": model_metadata,
                "model_endpoint": self.model_url,
                "requested_top_logprobs": self.top_k,
                "stop_token_ids": list(self.stop_token_ids),
                "sampling_top_k": self.sampling_top_k,
                "generation_protocol": self.generator.generation_protocol(task_name),
                "system_prompt_sha256": hashlib.sha256(
                    self.generator._system_prompt(task_name).encode("utf-8")
                ).hexdigest(),
                "weakest_k": self.settings.pwsc_ci.weakest_k,
                "num_samples": num_samples,
                "ensemble_sizes": list(sizes),
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "max_workers": self.max_workers,
                "generation_batch_size": self.generation_batch_size,
                "request_timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "pwsc_ci_contract": PWSC_CI_CONTRACT_VERSION,
                "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                "dataset_root": str(self.loader.base_dir),
                "dataset_fingerprint": dataset_fingerprint,
                "generation_cache_path": str(generation_cache),
                "generation_cache_sha256": cache_sha256,
                "input_sample_sha256": _input_sha256(samples),
                "timing_seconds": {
                    "cache_or_generation": cache_or_generation_seconds,
                    "scoring_and_verification": scoring_seconds,
                    "metric_analysis": analysis_seconds,
                    "total": total_seconds,
                },
                "generation_cache_reused": bool(cache_path),
                "generation_cache_resumed": bool(resume_cache),
                "cache_output_dir": (
                    str(self.cache_output_dir) if self.cache_output_dir else None
                ),
            },
            "diagnostics": diagnostic_report,
            "notes": [
                "All UQ methods use the same fixed first-path error labels for paired comparison.",
                "Answer-selection accuracy is reported separately and never changes UQ labels.",
                "PWSC-CI is the normalized Holevo information of the natural target-versus-peer confidence-mass ensemble.",
                "PWSC-CI is a raw ranking score, not a calibrated error probability.",
                "No test-set min-max normalization, fitted fusion, or majority-vote relabeling is used.",
                "KLE and CoCoA rows are explicitly named task-kernel adaptations.",
                "The generation protocol tag is retained only for immutable cache compatibility; PWSC-CI scoring does not inspect proof certificates.",
            ],
        }
        return report, diagnostic_report, correlation, rows

    @staticmethod
    def _ci_group_report(
        labels: Sequence[int], risks: Sequence[float]
    ) -> Dict[str, Any]:
        correct = [float(risk) for risk, label in zip(risks, labels) if label == 0]
        incorrect = [float(risk) for risk, label in zip(risks, labels) if label == 1]
        result: Dict[str, Any] = {
            "correct_mean_pwsc_ci_risk": (
                float(np.mean(correct)) if correct else float("nan")
            ),
            "incorrect_mean_pwsc_ci_risk": (
                float(np.mean(incorrect)) if incorrect else float("nan")
            ),
            "correct_n": len(correct),
            "incorrect_n": len(incorrect),
            "welch_t": float("nan"),
            "welch_p": float("nan"),
        }
        if correct and incorrect and (np.std(correct) > 0 or np.std(incorrect) > 0):
            try:
                from scipy.stats import ttest_ind

                statistic, pvalue = ttest_ind(
                    correct, incorrect, equal_var=False, nan_policy="omit"
                )
                result["welch_t"] = float(statistic)
                result["welch_p"] = float(pvalue)
            except Exception:
                pass
        return result


def print_final_academic_table(
    report: Dict[str, Any], task_type: str, num_samples: int
) -> None:
    print("\n" + "=" * 98)
    print(
        f"PWSC-CI v1.0 | mode={report['run_mode']} | task={task_type} | "
        f"publication_eligible={report['publication_eligible']}"
    )
    print(
        "Fixed first-path correctness labels; answer-selection accuracy is reported separately."
    )
    print("=" * 98)
    for size_key, methods in report["Methods by N"].items():
        print(
            f"[{size_key}]  {'Method':<42} {'AUROC':>9} {'AUPRC':>9} "
            f"{'AURC':>9} {'SelAcc@50':>12}"
        )
        for name, metrics in methods.items():
            def fmt(metric: str) -> str:
                value = metrics.get(metric)
                return (
                    "NA"
                    if value is None or not np.isfinite(value)
                    else f"{value:.4f}"
                )

            print(
                f"{'':7}{name:<42} {fmt('AUROC'):>9} {fmt('AUPRC'):>9} "
                f"{fmt('AURC'):>9} {fmt('SelectiveAcc@50'):>12}"
            )
        selection = report["Selection Accuracy by N"][size_key]
        selection_text = ", ".join(
            f"{name}={value:.4f}" for name, value in selection.items()
        )
        print(f"{'':7}Selection accuracy: {selection_text}")
        print("-" * 98)
    print("=" * 98 + "\n")


def print_hypothesis_proof_table(
    diagnostics: Dict[str, Any], correlation: float
) -> None:
    """Compatibility print hook for the fourth-value runner contract."""
    print("PWSC-CI diagnostics (descriptive; no hard-coded significance claims)")
    print(
        "  correct mean risk:   "
        f"{diagnostics.get('correct_mean_pwsc_ci_risk')}"
    )
    print(
        "  incorrect mean risk: "
        f"{diagnostics.get('incorrect_mean_pwsc_ci_risk')}"
    )
    print(f"  Welch p-value:       {diagnostics.get('welch_p')}")
    print(f"  risk/error correlation: {correlation}")


def save_results_to_csv(
    task_name: str,
    intermediate_rows: List[Dict[str, Any]],
    report_dict: Dict[str, Any],
    seed: int = 0,
    command: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    if output_dir:
        root = Path(output_dir).expanduser().resolve()
    else:
        root = (
            Path(__file__).resolve().parent
            / "diagnostics"
            / f"{report_dict['run_mode']}_runs"
        )
    root.mkdir(parents=True, exist_ok=True)
    run_id = time.time_ns()
    csv_path = root / f"sample_results_{task_name}_{run_id}.csv"
    fields = sorted({key for row in intermediate_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(intermediate_rows)

    package_versions = {}
    for package in (
        "numpy",
        "scipy",
        "pandas",
        "openai",
        "httpx",
        "evalplus",
        "math-verify",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    report_path = root / f"report_{task_name}_{run_id}.json"
    report_path.write_text(
        json.dumps(_json_safe(report_dict), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    run_metadata = report_dict["run_metadata"]
    manifest = {
        "method": report_dict["method"],
        "run_mode": report_dict["run_mode"],
        "benchmark_tier": report_dict.get("benchmark_tier"),
        "publication_eligible": report_dict["publication_eligible"],
        "publication_ineligibility_reasons": report_dict[
            "publication_ineligibility_reasons"
        ],
        "class_counts": report_dict.get("class_counts"),
        "statistical_caveats": report_dict.get("statistical_caveats", []),
        "task": task_name,
        "seed": seed,
        "command": _redact_command(command or sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions,
        "model_identifier": run_metadata["model_identifier"],
        "model_revision": run_metadata["model_revision"],
        "model_metadata": run_metadata["model_metadata"],
        "dataset_fingerprint": run_metadata["dataset_fingerprint"],
        "sampling": {
            key: run_metadata[key]
            for key in (
                "num_samples",
                "ensemble_sizes",
                "temperature",
                "top_p",
                "max_tokens",
                "requested_top_logprobs",
                "stop_token_ids",
                "sampling_top_k",
                "generation_protocol",
                "system_prompt_sha256",
                "weakest_k",
                "python_hash_seed",
            )
        },
        "transport": {
            key: run_metadata[key]
            for key in (
                "request_timeout",
                "max_retries",
                "max_workers",
                "generation_batch_size",
            )
        },
        "evaluation": {
            "target": report_dict["evaluation_target"],
            "bootstrap_samples": run_metadata["bootstrap_samples"],
            "pwsc_ci_contract": run_metadata["pwsc_ci_contract"],
        },
        "timing_seconds": run_metadata.get("timing_seconds"),
        "generation_cache_reused": run_metadata.get("generation_cache_reused"),
        "generation_cache_resumed": run_metadata.get("generation_cache_resumed"),
        "input_sample_sha256": run_metadata["input_sample_sha256"],
        "generation_cache_path": run_metadata["generation_cache_path"],
        "generation_cache_sha256": run_metadata["generation_cache_sha256"],
        "sample_rows": len(intermediate_rows),
        "csv_path": str(csv_path),
        "csv_sha256": _sha256_file(csv_path),
        "report_path": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "created_unix": time.time(),
    }
    manifest_path = root / f"manifest_{task_name}_{run_id}.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote sample results: %s", csv_path)
    logger.info("Wrote report: %s", report_path)
    logger.info("Wrote manifest: %s", manifest_path)
    return {
        "csv": str(csv_path),
        "report": str(report_path),
        "manifest": str(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PWSC-CI publication experiment pipeline."
    )
    parser.add_argument(
        "--mode", choices=["paper", "stress", "smoke"], default="paper"
    )
    parser.add_argument(
        "--task",
        default="GSM8K",
        choices=[
            "GSM8K",
            "HumanEvalPlus",
            "MBPPPlus",
            "AIME2025",
            "AIME2026",
            "FOLIO",
            "P-FOLIO",
            "LiveCodeBench",
            "MATH-500",
        ],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means the complete split; positive limits require smoke mode",
    )
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--model_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", default="deepseek")
    parser.add_argument("--expected_model_revision", default="")
    parser.add_argument("--api_key", default="empty")
    parser.add_argument(
        "--dataset_root", default="/data/newmodel/graphunagent/datasets"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="API top-logprobs request; PWSC-CI weakest-k remains frozen at k=5",
    )
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--ensemble_sizes", default="1,4,8,16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="optional explicit nucleus threshold; omitted preserves server defaults",
    )
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=32)
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=3600.0,
        help="per-request OpenAI client timeout in seconds",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=0,
        help="OpenAI client retries; keep zero for expensive local generations",
    )
    parser.add_argument(
        "--stop_token_ids",
        default="",
        help=(
            "optional comma-separated vLLM stop token ids, recorded in the "
            "generation cache contract"
        ),
    )
    parser.add_argument(
        "--sampling_top_k",
        type=int,
        default=None,
        help=(
            "optional explicit vLLM sampling top-k; use 0 for neutral vLLM "
            "sampling independently of model generation_config.json"
        ),
    )
    parser.add_argument(
        "--cache_path",
        default="",
        help="reuse a complete schema-v2 generation cache and skip API generation",
    )
    parser.add_argument(
        "--cache_output_dir",
        default="",
        help="optional directory for newly generated schema-v2 caches",
    )
    parser.add_argument(
        "--resume_cache",
        action="store_true",
        help="append only missing records to a validated partial --cache_path",
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate protocol/model/data/cache without generation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.resume_cache and not args.cache_path:
        raise SystemExit("--resume_cache requires --cache_path")
    hash_seed = str(int(args.seed) % (2**32))
    if os.environ.get("PYTHONHASHSEED") != hash_seed:
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)
    sizes = parse_ensemble_sizes(args.ensemble_sizes)
    validate_protocol(args.mode, args.task, args.limit, args.num_samples, sizes)
    random.seed(args.seed)
    np.random.seed(args.seed)
    settings = ExperimentSettings(random_seed=args.seed, use_calibration=False)
    runner = AcademicExperimentRunner(
        settings=settings,
        model_name=args.model_name,
        model_url=args.model_url,
        api_key=args.api_key,
        top_k=args.top_k,
        dataset_root=args.dataset_root,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        generation_batch_size=args.generation_batch_size,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        stop_token_ids=parse_stop_token_ids(args.stop_token_ids),
        sampling_top_k=args.sampling_top_k,
        expected_model_revision=args.expected_model_revision,
        cache_output_dir=args.cache_output_dir,
    )
    try:
        if args.preflight:
            result = runner.preflight(
                args.task,
                args.limit,
                args.num_samples,
                sizes,
                args.mode,
                args.cache_path or None,
                args.resume_cache,
            )
            print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
            return
        report, diagnostics, correlation, rows = runner.run_benchmark_experiment(
            task_name=args.task,
            sample_limit=args.limit,
            num_samples=args.num_samples,
            ensemble_sizes=sizes,
            run_mode=args.mode,
            cache_path=args.cache_path or None,
            resume_cache=args.resume_cache,
        )
        print_final_academic_table(report, args.task, args.num_samples)
        print_hypothesis_proof_table(diagnostics, correlation)
        save_results_to_csv(
            args.task,
            rows,
            report,
            seed=args.seed,
            command=sys.argv,
            output_dir=args.output_dir or None,
        )
    except Exception as exc:
        logger.critical(
            "Pipeline failed without fabricating results: %s", exc, exc_info=True
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
