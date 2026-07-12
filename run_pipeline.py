"""Publication-grade PWSC experiment runner.

The default mode is the frozen paper protocol: a complete primary-domain test
split, sixteen sampled paths, and offline evaluation at N={1,4,8,16}.  Partial
runs require an explicit ``--mode smoke`` so their artifacts cannot be mistaken
for paper results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from config.settings import ExperimentSettings
from core.adapters import make_adapter
from core.risk_estimator import RiskEstimator, retain_scoring_tokens
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


PRIMARY_TASKS = ("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO")
DEFAULT_ENSEMBLE_SIZES = (1, 4, 8, 16)
CACHE_SCHEMA_VERSION = 2
BOOTSTRAP_SAMPLES = 1000


def filter_reasoning_logprobs_full(logprob_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compatibility name; PWSC retains reasoning and answer tokens alike."""
    return list(logprob_items or [])


def extract_homotopy_outcome(text: str, task_name: str) -> str:
    """Compatibility wrapper around the canonical domain adapter."""
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
        if task_name not in PRIMARY_TASKS:
            raise ValueError(f"paper mode is restricted to primary tasks: {PRIMARY_TASKS}")
        if sample_limit != 0:
            raise ValueError("paper mode requires --limit 0 (the complete split); use --mode smoke for subsets")
        if num_samples != 16 or tuple(ensemble_sizes) != DEFAULT_ENSEMBLE_SIZES:
            raise ValueError("paper mode requires --num_samples 16 and --ensemble_sizes 1,4,8,16")
    elif run_mode == "stress":
        if task_name in PRIMARY_TASKS:
            raise ValueError("stress mode is for optional domains; use paper mode for primary tasks")
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
    # Keep the pre-paper helper signature usable for downstream scripts that
    # passed a correctness list as the third positional argument.
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
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "AURC": aurc,
        "SelectiveAcc@50": ScientificSignalAnalyzer.calculate_selective_accuracy(y_true, y_scores, 0.5),
        "AUROC_CI95": [ci_low, ci_high],
        "ECE": None,
        "Brier": None,
        **(
            {"Top-50% ACC": ScientificSignalAnalyzer.calculate_top_50_acc(is_correct_list, y_scores)}
            if is_correct_list is not None
            else {}
        ),
    }


class UniversalDatasetLoader(DatasetLoader):
    def load_any_dataset(self, task_name: str, limit: Optional[int] = None) -> List[TaskSample]:
        if task_name == "GSM8K":
            samples = self.load_gsm8k_split()[1]
            return samples[:limit] if limit and limit > 0 else samples
        if task_name == "HumanEvalPlus":
            return self.load_humanevalplus_test(limit)
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
    if task_name == "HumanEvalPlus":
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
    payload = [{"sample_id": sample.sample_id, "prompt": sample.raw_question} for sample in samples]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _trace_batch_complete(traces: Sequence[Dict[str, Any]], num_samples: int) -> bool:
    if len(traces) < num_samples:
        return False
    for trace in traces[:num_samples]:
        if not str(trace.get("text", "")).strip():
            return False
        if not retain_scoring_tokens(trace.get("logprobs", [])):
            return False
    return True


def _majority_index(outcomes: Sequence[str], confidences: Sequence[float]) -> int:
    counts = Counter(outcomes)
    first_position = {key: list(outcomes).index(key) for key in counts}
    confidence_by_key = {
        key: sum(float(confidences[idx]) for idx, value in enumerate(outcomes) if value == key)
        for key in counts
    }
    winning_key = max(
        counts,
        key=lambda key: (counts[key], confidence_by_key[key], -first_position[key]),
    )
    candidates = [idx for idx, value in enumerate(outcomes) if value == winning_key]
    return max(candidates, key=lambda idx: (float(confidences[idx]), -idx))


def _mbr_index(kernel: np.ndarray, weights: Optional[np.ndarray]) -> int:
    matrix = np.asarray(kernel, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return 0
    if weights is None:
        normalized = np.ones(matrix.shape[0], dtype=np.float64) / matrix.shape[0]
    else:
        normalized = np.asarray(weights, dtype=np.float64)
        normalized = normalized / normalized.sum() if normalized.sum() > 0 else np.ones(matrix.shape[0]) / matrix.shape[0]
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
        max_tokens: int = 12000,
        max_workers: int = 8,
        generation_batch_size: int = 32,
        expected_model_revision: str = "",
    ) -> None:
        self.settings = settings or ExperimentSettings()
        self.model_name = model_name
        self.model_url = model_url
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_workers = int(max_workers)
        self.generation_batch_size = int(generation_batch_size)
        self.expected_model_revision = expected_model_revision
        if self.max_tokens < 1 or self.max_workers < 1 or self.generation_batch_size < 1:
            raise ValueError("max_tokens, max_workers, and generation_batch_size must be positive")
        self.loader = UniversalDatasetLoader(dataset_root)
        self.verifier_mgr = VerifierManager()
        self.generator = APIGenerator(
            model_name=model_name,
            api_key=api_key,
            base_url=model_url,
            seed=self.settings.random_seed,
            top_k_logprobs=top_k,
        )
        self.risk_estimator = RiskEstimator(weakest_k=self.settings.pwsc.weakest_k)

    def _model_metadata(self, cache_header: Optional[Dict[str, Any]], run_mode: str) -> Dict[str, Any]:
        if cache_header is not None:
            metadata = dict(cache_header.get("model", {}))
        else:
            try:
                metadata = self.generator.describe_model()
            except Exception:
                if run_mode == "paper":
                    raise
                metadata = {"id": self.model_name, "root": None, "metadata_unavailable": True}
        if metadata.get("id") != self.model_name:
            raise ValueError(f"model mismatch: cache/server has {metadata.get('id')!r}, CLI requested {self.model_name!r}")
        revision = str(metadata.get("root") or metadata.get("revision") or metadata.get("id") or "")
        if self.expected_model_revision and revision != self.expected_model_revision:
            raise ValueError(f"model revision mismatch: expected {self.expected_model_revision!r}, got {revision!r}")
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
                    descriptors.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
            shards = [
                {"name": path.name, "bytes": path.stat().st_size}
                for path in sorted(root.glob("*.safetensors"))
            ]
            fingerprint_payload = {"descriptors": descriptors, "weight_shards": shards}
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
        if header.get("record_type") != "meta" or header.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("cache is not a PWSC paper-runner v2 cache")
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
            "max_tokens": self.max_tokens,
            "requested_top_logprobs": self.top_k,
            "system_prompt_sha256": hashlib.sha256(
                self.generator._system_prompt(task_name).encode("utf-8")
            ).hexdigest(),
            "input_sha256": _input_sha256(samples),
        }
        for key, value in expected.items():
            if header.get(key) != value:
                raise ValueError(f"cache {key} mismatch: expected {value!r}, got {header.get(key)!r}")
        if header.get("dataset") != dataset_fingerprint:
            raise ValueError("cache dataset fingerprint does not match current local data")

    def _prepare_generation_cache(
        self,
        task_name: str,
        samples: Sequence[TaskSample],
        num_samples: int,
        run_mode: str,
        dataset_fingerprint: Dict[str, Any],
        cache_path: Optional[str],
    ) -> Tuple[Path, str, Dict[str, Any]]:
        if cache_path:
            path = Path(cache_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"generation cache not found: {path}")
            header = self._read_cache_header(path)
            self._validate_cache_header(header, task_name, samples, num_samples, dataset_fingerprint, run_mode)
            model_metadata = self._model_metadata(header, run_mode)
            return path, _sha256_file(path), model_metadata

        model_metadata = self._model_metadata(None, run_mode)
        cache_dir = Path(__file__).resolve().parent / "diagnostics" / f"{run_mode}_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"generations_{task_name}_{time.time_ns()}_{_input_sha256(samples)[:10]}.jsonl"
        header = {
            "record_type": "meta",
            "schema_version": CACHE_SCHEMA_VERSION,
            "method": self.settings.pwsc.method_version,
            "run_mode": run_mode,
            "task": task_name,
            "seed": self.settings.random_seed,
            "num_samples": num_samples,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "requested_top_logprobs": self.top_k,
            "system_prompt": self.generator._system_prompt(task_name),
            "system_prompt_sha256": hashlib.sha256(
                self.generator._system_prompt(task_name).encode("utf-8")
            ).hexdigest(),
            "input_sha256": _input_sha256(samples),
            "dataset": dataset_fingerprint,
            "model": model_metadata,
        }
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(header), sort_keys=True, ensure_ascii=False) + "\n")
            for start in range(0, len(samples), self.generation_batch_size):
                batch = samples[start : start + self.generation_batch_size]
                generated = self.generator.generate_consistency_samples(
                    [sample.raw_question for sample in batch],
                    num_samples=num_samples,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    max_workers=self.max_workers,
                    task_type=task_name,
                )
                if len(generated) != len(batch):
                    raise RuntimeError(f"generator returned {len(generated)} batches for {len(batch)} prompts")
                for sample, traces in zip(batch, generated):
                    record = {
                        "record_type": "sample",
                        "sample_id": sample.sample_id,
                        "prompt_sha256": hashlib.sha256(sample.raw_question.encode("utf-8")).hexdigest(),
                        "traces": traces,
                    }
                    handle.write(json.dumps(_json_safe(record), sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
        return path, _sha256_file(path), model_metadata

    @staticmethod
    def _iter_generation_cache(path: Path, samples: Sequence[TaskSample]) -> Iterator[Tuple[TaskSample, List[Dict[str, Any]]]]:
        with path.open(encoding="utf-8") as handle:
            header = json.loads(handle.readline())
            if header.get("record_type") != "meta":
                raise ValueError("generation cache is missing its metadata header")
            count = 0
            for sample, line in zip(samples, handle):
                record = json.loads(line)
                prompt_hash = hashlib.sha256(sample.raw_question.encode("utf-8")).hexdigest()
                if record.get("record_type") != "sample" or record.get("sample_id") != sample.sample_id:
                    raise ValueError(f"cache sample order mismatch at index {count}")
                if record.get("prompt_sha256") != prompt_hash:
                    raise ValueError(f"cache prompt mismatch for {sample.sample_id}")
                count += 1
                yield sample, list(record.get("traces", []))
            if count != len(samples):
                raise ValueError(f"cache contains {count} aligned samples; expected {len(samples)}")
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
    ) -> Dict[str, Any]:
        validate_protocol(run_mode, task_name, sample_limit, num_samples, ensemble_sizes)
        samples = self.loader.load_any_dataset(task_name, sample_limit)
        dataset = self.loader.dataset_fingerprint(task_name)
        cache_header = self._read_cache_header(Path(cache_path).resolve()) if cache_path else None
        model = self._model_metadata(cache_header, run_mode)
        if cache_header:
            self._validate_cache_header(cache_header, task_name, samples, num_samples, dataset, run_mode)
        return {
            "status": "ready",
            "run_mode": run_mode,
            "task": task_name,
            "sample_count": len(samples),
            "num_samples": num_samples,
            "ensemble_sizes": list(ensemble_sizes),
            "seed": self.settings.random_seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt_sha256": hashlib.sha256(
                self.generator._system_prompt(task_name).encode("utf-8")
            ).hexdigest(),
            "model": model,
            "dataset": dataset,
            "cache_path": str(Path(cache_path).resolve()) if cache_path else None,
        }

    def run_benchmark_experiment(
        self,
        task_name: str = "GSM8K",
        sample_limit: int = 0,
        num_samples: int = 16,
        ensemble_sizes: Sequence[int] = DEFAULT_ENSEMBLE_SIZES,
        run_mode: str = "paper",
        cache_path: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float, List[Dict[str, Any]]]:
        sizes = parse_ensemble_sizes(ensemble_sizes)
        validate_protocol(run_mode, task_name, sample_limit, num_samples, sizes)
        samples = self.loader.load_any_dataset(task_name, sample_limit)
        if not samples:
            raise ValueError(f"No samples loaded for {task_name}")
        dataset_fingerprint = self.loader.dataset_fingerprint(task_name)
        generation_cache, cache_sha256, model_metadata = self._prepare_generation_cache(
            task_name, samples, num_samples, run_mode, dataset_fingerprint, cache_path
        )
        verifier = self.verifier_mgr.verifiers[_verifier_id(task_name)]

        labels_by_n: Dict[int, List[int]] = {size: [] for size in sizes}
        scores_by_n: Dict[int, Dict[str, List[float]]] = {
            size: {"PWSC": [], "Mean NLL": [], "Worst-5 NLL": []} for size in sizes
        }
        for size in sizes:
            if size > 1:
                scores_by_n[size].update({"Self-Consistency": [], "Semantic Entropy": [], "MBR": []})
        selection_correct_by_n: Dict[int, Dict[str, List[bool]]] = {
            size: {"First Path": [], "PWSC": [], "Self-Consistency": [], "MBR": []} for size in sizes
        }
        diagnostic_groups: Dict[int, Dict[str, List[float]]] = {
            size: {"correct": [], "incorrect": []} for size in sizes
        }
        rows: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        records: Iterable[Tuple[TaskSample, List[Dict[str, Any]]]] = self._iter_generation_cache(generation_cache, samples)
        if tqdm:
            records = tqdm(records, total=len(samples), desc=f"PWSC paper evaluation: {task_name}")
        for sample, traces in records:
            if not _trace_batch_complete(traces, num_samples):
                reason = "incomplete_text_or_logprobs"
                if run_mode == "paper":
                    raise RuntimeError(f"publication run aborted at {sample.sample_id}: {reason}")
                skipped.append({"sample_id": sample.sample_id, "reason": reason})
                continue
            traces = traces[:num_samples]
            texts = [str(trace["text"]) for trace in traces]
            logprob_paths = [filter_reasoning_logprobs_full(trace["logprobs"]) for trace in traces]
            metadata = _sample_metadata(sample, task_name)
            correctness_cache: Dict[int, bool] = {}

            def is_correct(index: int) -> bool:
                if index not in correctness_cache:
                    correctness_cache[index] = verifier.execute(texts[index], sample.gold_answer, metadata)
                return correctness_cache[index]

            base_correct = is_correct(0)
            base_wrong = int(not base_correct)
            mean_nll = SCSVBaselines.calculate_nll(logprob_paths[0])
            worst_nll = SCSVBaselines.calculate_worst_k_nll(logprob_paths[0], self.settings.pwsc.weakest_k)

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
                pwsc_index = int(diagnostics["selected_index"])
                sc_index = _majority_index(outcomes, confidences)
                mbr_index = _mbr_index(diagnostics["kernel"], diagnostics.get("weights"))
                sc_risk = SCSVBaselines.calculate_self_consistency(prefix_texts, outcomes)
                se_risk = SCSVBaselines.calculate_semantic_entropy(prefix_texts, outcomes)
                mbr_risk = SCSVBaselines.calculate_mbr(diagnostics["kernel"], diagnostics.get("weights"))

                labels_by_n[size].append(base_wrong)
                scores_by_n[size]["PWSC"].append(float(risk))
                scores_by_n[size]["Mean NLL"].append(mean_nll)
                scores_by_n[size]["Worst-5 NLL"].append(worst_nll)
                if size > 1:
                    scores_by_n[size]["Self-Consistency"].append(sc_risk)
                    scores_by_n[size]["Semantic Entropy"].append(se_risk)
                    scores_by_n[size]["MBR"].append(mbr_risk)
                selection_correct_by_n[size]["First Path"].append(base_correct)
                selection_correct_by_n[size]["PWSC"].append(is_correct(pwsc_index))
                selection_correct_by_n[size]["Self-Consistency"].append(is_correct(sc_index))
                selection_correct_by_n[size]["MBR"].append(is_correct(mbr_index))
                group = "correct" if base_correct else "incorrect"
                diagnostic_groups[size][group].append(float(diagnostics["normalized_spectral_uncertainty"]))
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "task": task_name,
                        "ensemble_size": size,
                        "base_is_correct": base_correct,
                        "pwsc_risk": float(risk),
                        "mean_nll": mean_nll,
                        "worst_5_nll": worst_nll,
                        "self_consistency_risk": sc_risk if size > 1 else None,
                        "semantic_entropy_risk": se_risk if size > 1 else None,
                        "mbr_risk": mbr_risk if size > 1 else None,
                        "pwsc_selected_index": pwsc_index,
                        "pwsc_selected_is_correct": is_correct(pwsc_index),
                        "sc_selected_index": sc_index,
                        "sc_selected_is_correct": is_correct(sc_index),
                        "mbr_selected_index": mbr_index,
                        "mbr_selected_is_correct": is_correct(mbr_index),
                        "valid_path_count": int(diagnostics["valid_count"]),
                        "spectral_uncertainty": float(diagnostics["normalized_spectral_uncertainty"]),
                        "mean_path_confidence": float(diagnostics["mean_path_confidence"]),
                    }
                )

        scored_questions = len(labels_by_n[sizes[0]])
        if scored_questions == 0:
            raise RuntimeError(f"All {len(samples)} samples failed before scoring")
        if run_mode == "paper" and (skipped or scored_questions != len(samples)):
            raise RuntimeError("paper mode requires complete coverage with zero skipped questions")

        methods_by_n: Dict[str, Dict[str, Any]] = {}
        paired_by_n: Dict[str, Dict[str, Any]] = {}
        selection_by_n: Dict[str, Dict[str, float]] = {}
        diagnostics_by_n: Dict[str, Dict[str, Any]] = {}
        for size in sizes:
            key = f"N={size}"
            labels = labels_by_n[size]
            methods_by_n[key] = {
                name: compute_academic_metrics(labels, values, seed=self.settings.random_seed)
                for name, values in scores_by_n[size].items()
            }
            paired_by_n[key] = {}
            for name, values in scores_by_n[size].items():
                if name == "PWSC":
                    continue
                paired_by_n[key][f"PWSC - {name}"] = {
                    "AUROC": ScientificSignalAnalyzer.paired_bootstrap_difference(
                        labels,
                        scores_by_n[size]["PWSC"],
                        values,
                        ScientificSignalAnalyzer.calculate_auroc,
                        seed=self.settings.random_seed,
                        n_bootstrap=BOOTSTRAP_SAMPLES,
                    ),
                    "AURC": ScientificSignalAnalyzer.paired_bootstrap_difference(
                        labels,
                        scores_by_n[size]["PWSC"],
                        values,
                        ScientificSignalAnalyzer.calculate_aurc,
                        seed=self.settings.random_seed,
                        n_bootstrap=BOOTSTRAP_SAMPLES,
                    ),
                }
            selection_by_n[key] = {
                name: float(np.mean(values)) for name, values in selection_correct_by_n[size].items()
            }
            diagnostics_by_n[key] = self._spectral_report(diagnostic_groups[size])

        max_size = sizes[-1]
        max_key = f"N={max_size}"
        correlation = ScientificSignalAnalyzer.calculate_pearson_correlation(
            diagnostic_groups[max_size]["correct"] + diagnostic_groups[max_size]["incorrect"],
            [0] * len(diagnostic_groups[max_size]["correct"]) + [1] * len(diagnostic_groups[max_size]["incorrect"]),
        )
        proof_report = dict(diagnostics_by_n[max_key])
        proof_report["spectral_uncertainty_error_correlation"] = correlation
        proof_report["skipped_samples"] = skipped
        publication_reasons: List[str] = []
        if run_mode != "paper":
            publication_reasons.append(f"run_mode={run_mode}")
        if task_name not in PRIMARY_TASKS:
            publication_reasons.append("not a primary task")
        if sample_limit != 0:
            publication_reasons.append("partial dataset")
        if tuple(sizes) != DEFAULT_ENSEMBLE_SIZES or num_samples != 16:
            publication_reasons.append("non-canonical ensemble protocol")
        if skipped or scored_questions != len(samples):
            publication_reasons.append("incomplete sample coverage")

        report = {
            "method": self.settings.pwsc.method_version,
            "task": task_name,
            "run_mode": run_mode,
            "publication_eligible": not publication_reasons,
            "publication_ineligibility_reasons": publication_reasons,
            "n_requested": len(samples),
            "n_scored_questions": scored_questions,
            "n_skipped": len(skipped),
            "evaluation_target": "correctness of the fixed first sampled path for every UQ method",
            "Methods by N": methods_by_n,
            "Selection Accuracy by N": selection_by_n,
            "Paired Bootstrap Differences by N": paired_by_n,
            "Diagnostics by N": diagnostics_by_n,
            # Compatibility views for downstream table scripts.
            "Single-Path Methods": methods_by_n["N=1"],
            "Multi-Path Methods": methods_by_n[max_key],
            "run_metadata": {
                "seed": self.settings.random_seed,
                "model_identifier": self.model_name,
                "model_revision": model_metadata.get("resolved_revision"),
                "model_metadata": model_metadata,
                "model_endpoint": self.model_url,
                "requested_top_logprobs": self.top_k,
                "system_prompt_sha256": hashlib.sha256(
                    self.generator._system_prompt(task_name).encode("utf-8")
                ).hexdigest(),
                "weakest_k": self.settings.pwsc.weakest_k,
                "num_samples": num_samples,
                "ensemble_sizes": list(sizes),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "max_workers": self.max_workers,
                "generation_batch_size": self.generation_batch_size,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "dataset_root": str(self.loader.base_dir),
                "dataset_fingerprint": dataset_fingerprint,
                "generation_cache_path": str(generation_cache),
                "generation_cache_sha256": cache_sha256,
                "input_sample_sha256": _input_sha256(samples),
            },
            "diagnostics": proof_report,
            "notes": [
                "All UQ methods use the same fixed first-path error labels for paired comparison.",
                "Answer-selection accuracy is reported separately and does not change UQ labels.",
                "Scores are raw risks; ECE/Brier require a disjoint labeled dev calibrator.",
                "No test-set min-max normalization or majority-vote relabeling is used.",
            ],
        }
        return report, proof_report, correlation, rows

    @staticmethod
    def _spectral_report(groups: Dict[str, List[float]]) -> Dict[str, Any]:
        correct, incorrect = groups["correct"], groups["incorrect"]
        result: Dict[str, Any] = {
            "correct_mean": float(np.mean(correct)) if correct else float("nan"),
            "incorrect_mean": float(np.mean(incorrect)) if incorrect else float("nan"),
            "correct_n": len(correct),
            "incorrect_n": len(incorrect),
            "welch_t": float("nan"),
            "welch_p": float("nan"),
        }
        if correct and incorrect and (np.std(correct) > 0 or np.std(incorrect) > 0):
            try:
                from scipy.stats import ttest_ind

                stat, pvalue = ttest_ind(correct, incorrect, equal_var=False, nan_policy="omit")
                result["welch_t"], result["welch_p"] = float(stat), float(pvalue)
            except Exception:
                pass
        return result


def print_final_academic_table(report: Dict[str, Any], task_type: str, num_samples: int) -> None:
    print("\n" + "=" * 98)
    print(
        f"PWSC v1.0 | mode={report['run_mode']} | task={task_type} | "
        f"publication_eligible={report['publication_eligible']}"
    )
    print("Fixed first-path correctness labels; answer-selection accuracy is reported separately.")
    print("=" * 98)
    for size_key, methods in report["Methods by N"].items():
        print(f"[{size_key}]  {'Method':<25} {'AUROC':>10} {'AUPRC':>10} {'AURC':>10} {'SelAcc@50':>12}")
        for name, metrics in methods.items():
            def fmt(metric: str) -> str:
                value = metrics.get(metric)
                return "NA" if value is None or not np.isfinite(value) else f"{value:.4f}"

            print(
                f"{'':7}{name:<25} {fmt('AUROC'):>10} {fmt('AUPRC'):>10} "
                f"{fmt('AURC'):>10} {fmt('SelectiveAcc@50'):>12}"
            )
        selection = report["Selection Accuracy by N"][size_key]
        selection_text = ", ".join(f"{name}={value:.4f}" for name, value in selection.items())
        print(f"{'':7}Selection accuracy: {selection_text}")
        print("-" * 98)
    print("=" * 98 + "\n")


def print_hypothesis_proof_table(proof: Dict[str, Any], correlation: float) -> None:
    print("PWSC diagnostics (descriptive; no hard-coded significance claims)")
    print(f"  spectral uncertainty, correct mean:   {proof.get('correct_mean')}")
    print(f"  spectral uncertainty, incorrect mean: {proof.get('incorrect_mean')}")
    print(f"  Welch p-value:                         {proof.get('welch_p')}")
    print(f"  uncertainty/error correlation:         {correlation}")


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
        root = Path(__file__).resolve().parent / "diagnostics" / f"{report_dict['run_mode']}_runs"
    root.mkdir(parents=True, exist_ok=True)
    run_id = time.time_ns()
    csv_path = root / f"sample_results_{task_name}_{run_id}.csv"
    fields = sorted({key for row in intermediate_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(intermediate_rows)
    package_versions = {}
    for package in ("numpy", "scipy", "pandas", "openai", "httpx", "evalplus", "math-verify"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    report_path = root / f"report_{task_name}_{run_id}.json"
    report_path.write_text(json.dumps(_json_safe(report_dict), indent=2, ensure_ascii=False), encoding="utf-8")
    run_metadata = report_dict["run_metadata"]
    manifest = {
        "method": report_dict["method"],
        "run_mode": report_dict["run_mode"],
        "publication_eligible": report_dict["publication_eligible"],
        "publication_ineligibility_reasons": report_dict["publication_ineligibility_reasons"],
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
                "max_tokens",
                "requested_top_logprobs",
                "system_prompt_sha256",
                "weakest_k",
            )
        },
        "evaluation": {
            "target": report_dict["evaluation_target"],
            "bootstrap_samples": run_metadata["bootstrap_samples"],
        },
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
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote sample results: %s", csv_path)
    logger.info("Wrote report: %s", report_path)
    logger.info("Wrote manifest: %s", manifest_path)
    return {"csv": str(csv_path), "report": str(report_path), "manifest": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PWSC publication experiment pipeline.")
    parser.add_argument("--mode", choices=["paper", "stress", "smoke"], default="paper")
    parser.add_argument(
        "--task",
        default="GSM8K",
        choices=["GSM8K", "HumanEvalPlus", "AIME2025", "AIME2026", "FOLIO", "P-FOLIO", "LiveCodeBench", "MATH-500"],
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means the complete split; positive limits require smoke mode")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--model_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", default="deepseek")
    parser.add_argument("--expected_model_revision", default="")
    parser.add_argument("--api_key", default="empty")
    parser.add_argument("--dataset_root", default="/data/newmodel/graphunagent/datasets")
    parser.add_argument("--top_k", type=int, default=10, help="API top-logprobs request; PWSC weakest-K remains frozen at 5")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--ensemble_sizes", default="1,4,8,16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=32)
    parser.add_argument("--cache_path", default="", help="reuse a complete schema-v2 generation cache and skip API generation")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--preflight", action="store_true", help="validate protocol/model/data/cache without generation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        generation_batch_size=args.generation_batch_size,
        expected_model_revision=args.expected_model_revision,
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
            )
            print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
            return
        report, proof, correlation, rows = runner.run_benchmark_experiment(
            task_name=args.task,
            sample_limit=args.limit,
            num_samples=args.num_samples,
            ensemble_sizes=sizes,
            run_mode=args.mode,
            cache_path=args.cache_path or None,
        )
        print_final_academic_table(report, args.task, args.num_samples)
        print_hypothesis_proof_table(proof, correlation)
        save_results_to_csv(
            args.task,
            rows,
            report,
            seed=args.seed,
            command=sys.argv,
            output_dir=args.output_dir or None,
        )
    except Exception as exc:
        logger.critical("Pipeline failed without fabricating results: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
