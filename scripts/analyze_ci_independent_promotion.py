#!/usr/bin/env python3
"""Execute the preregistered one-shot PWSC-CI independent promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from diagnostics.analyzer import ScientificSignalAnalyzer  # noqa: E402
from run_pipeline import UniversalDatasetLoader  # noqa: E402
from scripts.evaluate_blackbox_2026 import (  # noqa: E402
    RDS_METHODS,
    _load_task_inputs,
    _metrics,
)
from scripts.evaluate_sese_2026 import (  # noqa: E402
    SESE_CACHE_FIELDS,
    SESE_METHODS,
)
from scripts.seal_ci_promotion_comparator import _selection as _baseline_selection  # noqa: E402


TASKS = ("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO")
EXPECTED_COUNTS = {
    "GSM8K": 1319,
    "MATH-500": 500,
    "HumanEvalPlus": 164,
    "FOLIO": 204,
}
EXPECTED_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
EXPECTED_MODEL_ROOT = "/data/newmodel/IE/LLM/Llama-3.1-8B-Instruct"
EXPECTED_PREREG_SHA256 = (
    "444f46ee4c1d1c8f7a6f0ce61aad5548792695d98699859e8130abd6eb543f62"
)
EXPECTED_CI_CONTRACT = "pwsc-ci-normalized-holevo-v1"
EXPECTED_GENERATION_PROTOCOL = "proof-carrying-pc-v1"
ELIGIBLE_METHODS = (*RDS_METHODS, "CSA", *SESE_METHODS)
METRIC_FUNCTIONS = {
    "AUROC": ScientificSignalAnalyzer.calculate_auroc,
    "AURC": ScientificSignalAnalyzer.calculate_aurc,
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


def _load_jsonl(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        try:
            header = json.loads(next(handle))
        except StopIteration as exc:
            raise ValueError(f"empty sidecar: {path}") from exc
        records = [json.loads(line) for line in handle if line.strip()]
    return header, records


def _resolve(path: Path) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def _validate_preregistration(path: Path) -> Dict[str, Any]:
    observed_sha = _sha256(path)
    if observed_sha != EXPECTED_PREREG_SHA256:
        raise RuntimeError(
            f"preregistration drift: expected {EXPECTED_PREREG_SHA256}, got {observed_sha}"
        )
    value = _load_json(path)
    if value.get("status") != "frozen_before_independent_generation":
        raise ValueError("promotion preregistration is not frozen")
    if value.get("score_contract") != EXPECTED_CI_CONTRACT:
        raise ValueError("PWSC-CI score contract drift")
    for relative, expected in value["implementation_sha256"].items():
        observed = _sha256(ROOT / relative)
        if observed != expected:
            raise RuntimeError(
                f"frozen implementation drift for {relative}: {observed} != {expected}"
            )
    return value


def _validate_selection(path: Path) -> Dict[str, Any]:
    value = _load_json(path)
    if value.get("schema") != "pwsc-ci-development-comparator-selection-v1":
        raise ValueError("invalid comparator selection schema")
    if value.get("selection_uses_pwsc_ci_values") is not False:
        raise ValueError("comparator selection is not baseline-only")
    if value.get("preregistration", {}).get("sha256") != EXPECTED_PREREG_SHA256:
        raise ValueError("comparator selection does not reference the frozen preregistration")
    if tuple(value.get("tasks", ())) != TASKS or value.get("ensemble_size") != 16:
        raise ValueError("comparator selection matrix mismatch")
    selected = value["selection"]
    single = selected["strongest_single"]["method"]
    mapping = selected["task_envelope"]["method_by_task"]
    if single not in ELIGIBLE_METHODS:
        raise ValueError(f"ineligible strongest comparator: {single}")
    if set(mapping) != set(TASKS) or any(
        method not in ELIGIBLE_METHODS for method in mapping.values()
    ):
        raise ValueError("ineligible task-envelope comparator mapping")
    expected_selection = _baseline_selection(value["eligible_task_metrics"])
    if selected != expected_selection:
        raise ValueError("comparator selection does not match the frozen baseline-only rule")
    script_path = ROOT / "scripts/seal_ci_promotion_comparator.py"
    if value.get("selection_script", {}).get("sha256") != _sha256(script_path):
        raise ValueError("comparator selection script hash mismatch")
    for name, metadata in value.get("inputs", {}).items():
        input_path = Path(metadata["path"]).resolve()
        if _sha256(input_path) != metadata.get("sha256"):
            raise ValueError(f"development comparator input drift: {name}")
    return value


def _validate_service_manifest(path: Path) -> Dict[str, Any]:
    value = _load_json(path)
    if value.get("schema") != "pwsc-ci-isolated-vllm-service-v1":
        raise ValueError("invalid isolated-service manifest schema")
    if value.get("endpoint") != "http://127.0.0.1:8004/v1":
        raise ValueError("independent service endpoint mismatch")
    if value.get("model", {}).get("served_model_name") != EXPECTED_MODEL_ID:
        raise ValueError("independent service model ID mismatch")
    if value.get("model", {}).get("root") != EXPECTED_MODEL_ROOT:
        raise ValueError("independent service model root mismatch")
    runtime = value.get("runtime", {})
    if runtime.get("generation_config") != "auto" or runtime.get(
        "resolved_default_chat_sampling", {}
    ).get("top_p") != 0.9:
        raise ValueError("independent service sampling default mismatch")
    return value


def _comparison_contract(value: Mapping[str, Any], name: str) -> None:
    if value.get("status") != "independent_promotion_comparison":
        raise ValueError(f"{name} is not an independent promotion comparison")
    if tuple(value.get("tasks", ())) != TASKS:
        raise ValueError(f"{name} task matrix mismatch")
    if int(value.get("seed", -1)) != 20260716:
        raise ValueError(f"{name} seed mismatch")


def _validate_manifest(path: Path, task: str) -> Dict[str, Any]:
    value = _load_json(path)
    if value.get("publication_eligible") is not True:
        raise ValueError(f"{task} manifest is not publication eligible")
    if value.get("publication_ineligibility_reasons"):
        raise ValueError(f"{task} manifest has ineligibility reasons")
    expected = {
        "task": task,
        "seed": 20260716,
        "model_identifier": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_ROOT,
        "sample_rows": EXPECTED_COUNTS[task] * 4,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"{task} manifest {key}: {value.get(key)!r} != {expected_value!r}"
            )
    sampling = value["sampling"]
    sampling_expected = {
        "num_samples": 16,
        "ensemble_sizes": [1, 4, 8, 16],
        "temperature": 0.7,
        "max_tokens": 12000,
        "requested_top_logprobs": 10,
        "generation_protocol": EXPECTED_GENERATION_PROTOCOL,
        "python_hash_seed": "20260716",
    }
    for key, expected_value in sampling_expected.items():
        if sampling.get(key) != expected_value:
            raise ValueError(f"{task} sampling contract mismatch for {key}")
    if value["evaluation"].get("pwsc_ci_contract") != EXPECTED_CI_CONTRACT:
        raise ValueError(f"{task} PWSC-CI contract mismatch")
    if value.get("class_counts", {}).get("correct", 0) < 1 or value.get(
        "class_counts", {}
    ).get("incorrect", 0) < 1:
        raise ValueError(f"{task} has a degenerate fixed-first label distribution")
    return value


def _score_maps(
    rds_score_path: Path, csa_cache_path: Path, sese_cache_path: Path
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, float]],
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], Dict[str, float]],
    Dict[str, Any],
]:
    rds = _load_json(rds_score_path)
    if rds.get("schema") != "rds-paired-score-cache-v1":
        raise ValueError("invalid RDS score-cache schema")
    if tuple(rds.get("tasks", ())) != TASKS or 16 not in rds.get(
        "ensemble_sizes", ()
    ):
        raise ValueError("RDS score-cache matrix mismatch")
    rds_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    for record in rds["records"]:
        key = (str(record["task"]), str(record["sample_id"]))
        if key in rds_map:
            raise ValueError(f"duplicate RDS score: {key}")
        values = record["scores_by_n"]["16"]
        rds_map[key] = {method: float(values[method]) for method in RDS_METHODS}

    csa_header, csa_records = _load_jsonl(csa_cache_path)
    if csa_header.get("schema") != "csa-assessment-cache-v1":
        raise ValueError("invalid CSA assessment-cache schema")
    csa_map: Dict[Tuple[str, str], float] = {}
    for record in csa_records:
        if int(record.get("ensemble_size", -1)) != 16:
            continue
        key = (str(record["task"]), str(record["sample_id"]))
        if key in csa_map:
            raise ValueError(f"duplicate CSA score: {key}")
        csa_map[key] = float(record["csa_risk"])

    sese_header, sese_records = _load_jsonl(sese_cache_path)
    if sese_header.get("schema") != "sese-source-replay-cache-v2":
        raise ValueError("invalid SeSE score-cache schema")
    sese_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    for record in sese_records:
        key = (str(record["task"]), str(record["sample_id"]))
        if key in sese_map:
            raise ValueError(f"duplicate SeSE score: {key}")
        sese_map[key] = {
            method: float(record[SESE_CACHE_FIELDS[method]])
            for method in SESE_METHODS
        }
    return rds_map, csa_map, sese_map, {
        "RDS": {key: value for key, value in rds.items() if key != "records"},
        "CSA_header": csa_header,
        "SeSE_header": sese_header,
    }


def _comparison_task_metrics(
    method: str,
    task: str,
    rds: Mapping[str, Any],
    csa: Mapping[str, Any],
    sese: Mapping[str, Any],
) -> Mapping[str, Any]:
    if method in RDS_METHODS:
        return rds["Metrics by N"]["N=16"]["task_metrics"][task][method]
    if method == "CSA":
        return csa["Metrics by N"]["N=16"]["task_metrics"][task][method]
    return sese["task_metrics"][task][method]


def _assert_metrics_match(
    observed: Mapping[str, float], expected: Mapping[str, Any], method: str, task: str
) -> None:
    for metric in ("AUROC", "AUPRC", "AURC", "SelectiveAcc@50"):
        if not math.isclose(
            float(observed[metric]), float(expected[metric]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{method} {task} {metric} does not match its comparison JSON"
            )


def _difference_point(
    arrays: Mapping[str, Mapping[str, np.ndarray]], method_by_task: Mapping[str, str]
) -> Dict[str, float]:
    return {
        metric: float(
            np.mean(
                [
                    function(values["error"], values["PWSC-CI"])
                    - function(values["error"], values[method_by_task[task]])
                    for task, values in arrays.items()
                ]
            )
        )
        for metric, function in METRIC_FUNCTIONS.items()
    }


def _paired_gate(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    comparators: Mapping[str, Mapping[str, str]],
    seed: int,
    draws: int,
) -> Dict[str, Any]:
    points = {
        name: _difference_point(arrays, method_by_task)
        for name, method_by_task in comparators.items()
    }
    samples = {
        name: {metric: [] for metric in METRIC_FUNCTIONS} for name in comparators
    }
    domain_samples = {
        name: {
            task: {metric: [] for metric in METRIC_FUNCTIONS} for task in TASKS
        }
        for name in comparators
    }
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        indices_by_task = {
            task: rng.integers(0, len(arrays[task]["error"]), len(arrays[task]["error"]))
            for task in TASKS
        }
        for name, method_by_task in comparators.items():
            task_differences = {metric: [] for metric in METRIC_FUNCTIONS}
            for task in TASKS:
                values = arrays[task]
                indices = indices_by_task[task]
                method = method_by_task[task]
                for metric, function in METRIC_FUNCTIONS.items():
                    difference = float(
                        function(values["error"][indices], values["PWSC-CI"][indices])
                        - function(values["error"][indices], values[method][indices])
                    )
                    if math.isfinite(difference):
                        domain_samples[name][task][metric].append(difference)
                        task_differences[metric].append(difference)
            for metric, values in task_differences.items():
                if len(values) == len(TASKS):
                    samples[name][metric].append(float(np.mean(values)))

    output: Dict[str, Any] = {}
    for name, method_by_task in comparators.items():
        intervals = {}
        for metric, point in points[name].items():
            values = np.asarray(samples[name][metric], dtype=np.float64)
            if len(values) != draws:
                raise RuntimeError(f"non-finite bootstrap draws for {name} {metric}")
            intervals[metric] = [
                point,
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
        domain = {}
        for task in TASKS:
            method = method_by_task[task]
            domain[task] = {"comparator": method, "differences": {}}
            for metric, function in METRIC_FUNCTIONS.items():
                values = arrays[task]
                point = float(
                    function(values["error"], values["PWSC-CI"])
                    - function(values["error"], values[method])
                )
                sampled = np.asarray(domain_samples[name][task][metric], dtype=np.float64)
                domain[task]["differences"][metric] = [
                    point,
                    float(np.quantile(sampled, 0.025)),
                    float(np.quantile(sampled, 0.975)),
                ]
        auroc = intervals["AUROC"]
        aurc = intervals["AURC"]
        passes = (
            auroc[0] > 0.0
            and auroc[1] > 0.0
            and aurc[0] < 0.0
            and aurc[2] < 0.0
        )
        output[name] = {
            "method_by_task": dict(method_by_task),
            "paired_macro_differences": intervals,
            "per_domain_differences": domain,
            "passes_joint_auroc_aurc_gate": passes,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--service_manifest",
        type=Path,
        default=ROOT / ".pipeline/docs/pwsc_ci_llama_service_20260716.json",
    )
    parser.add_argument("--rds", type=Path, required=True)
    parser.add_argument("--rds_score_cache", type=Path, required=True)
    parser.add_argument("--csa", type=Path, required=True)
    parser.add_argument("--csa_assessment_cache", type=Path, required=True)
    parser.add_argument("--sese", type=Path, required=True)
    parser.add_argument("--sese_cache", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT
        / ".pipeline/docs/pwsc_ci_independent_promotion_protocol_20260716.json",
    )
    parser.add_argument(
        "--dataset_root", default="/data/newmodel/graphunagent/datasets"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260732)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap != 2000 or args.bootstrap_seed != 20260732:
        parser.error("promotion gate requires 2000 draws and bootstrap seed 20260732")

    paths = {
        "preregistration": _resolve(args.preregistration),
        "selection": _resolve(args.selection),
        "service_manifest": _resolve(args.service_manifest),
        "RDS_comparison": _resolve(args.rds),
        "RDS_score_cache": _resolve(args.rds_score_cache),
        "CSA_comparison": _resolve(args.csa),
        "CSA_assessment_cache": _resolve(args.csa_assessment_cache),
        "SeSE_comparison": _resolve(args.sese),
        "SeSE_cache": _resolve(args.sese_cache),
    }
    prereg = _validate_preregistration(paths["preregistration"])
    selection = _validate_selection(paths["selection"])
    service_manifest = _validate_service_manifest(paths["service_manifest"])
    rds_comparison = _load_json(paths["RDS_comparison"])
    csa_comparison = _load_json(paths["CSA_comparison"])
    sese_comparison = _load_json(paths["SeSE_comparison"])
    for name, value in (
        ("RDS", rds_comparison),
        ("CSA", csa_comparison),
        ("SeSE", sese_comparison),
    ):
        _comparison_contract(value, name)

    if rds_comparison.get("paired_score_cache", {}).get("sha256") != _sha256(
        paths["RDS_score_cache"]
    ):
        raise ValueError("RDS comparison does not bind the paired score cache")
    if csa_comparison.get("sidecar_caches", {}).get("assessment_sha256") != _sha256(
        paths["CSA_assessment_cache"]
    ):
        raise ValueError("CSA comparison does not bind the assessment cache")
    if sese_comparison.get("sidecar_cache", {}).get("sha256") != _sha256(
        paths["SeSE_cache"]
    ):
        raise ValueError("SeSE comparison does not bind the source cache")

    rds_map, csa_map, sese_map, sidecar_headers = _score_maps(
        paths["RDS_score_cache"],
        paths["CSA_assessment_cache"],
        paths["SeSE_cache"],
    )
    loader = UniversalDatasetLoader(args.dataset_root)
    run_dir = args.run_dir.expanduser().resolve()
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    provenance: Dict[str, Any] = {}
    manifests: Dict[str, Any] = {}
    for task in TASKS:
        task_provenance, records, arrays_by_n = _load_task_inputs(
            run_dir, task, (16,), loader
        )
        manifest_path = Path(task_provenance["manifest_path"])
        manifests[task] = _validate_manifest(manifest_path, task)
        values = arrays_by_n[16]
        ids = [str(record["sample_id"]) for record in records]
        expected_keys = {(task, sample_id) for sample_id in ids}
        for name, observed in (
            ("RDS", {key for key in rds_map if key[0] == task}),
            ("CSA", {key for key in csa_map if key[0] == task}),
            ("SeSE", {key for key in sese_map if key[0] == task}),
        ):
            if observed != expected_keys:
                raise ValueError(f"{task} {name} sample IDs do not match the runner")
        for method in RDS_METHODS:
            values[method] = np.asarray(
                [rds_map[(task, sample_id)][method] for sample_id in ids],
                dtype=np.float64,
            )
        values["CSA"] = np.asarray(
            [csa_map[(task, sample_id)] for sample_id in ids], dtype=np.float64
        )
        for method in SESE_METHODS:
            values[method] = np.asarray(
                [sese_map[(task, sample_id)][method] for sample_id in ids],
                dtype=np.float64,
            )
        for method in ELIGIBLE_METHODS:
            observed_metrics = _metrics(values["error"], values[method])
            expected_metrics = _comparison_task_metrics(
                method, task, rds_comparison, csa_comparison, sese_comparison
            )
            _assert_metrics_match(observed_metrics, expected_metrics, method, task)
        arrays[task] = values
        provenance[task] = task_provenance

    if sum(len(values["error"]) for values in arrays.values()) != 2187:
        raise RuntimeError("independent matrix does not contain exactly 2187 questions")

    task_metrics = {
        task: {
            method: _metrics(values["error"], values[method])
            for method in ("PWSC-CI", *ELIGIBLE_METHODS)
        }
        for task, values in arrays.items()
    }
    macro_metrics = {
        method: {
            metric: float(np.mean([task_metrics[task][method][metric] for task in TASKS]))
            for metric in task_metrics[TASKS[0]][method]
        }
        for method in ("PWSC-CI", *ELIGIBLE_METHODS)
    }
    single_method = selection["selection"]["strongest_single"]["method"]
    comparators = {
        "strongest_single": {task: single_method for task in TASKS},
        "task_envelope": dict(
            selection["selection"]["task_envelope"]["method_by_task"]
        ),
    }
    gates = _paired_gate(
        arrays, comparators, args.bootstrap_seed, args.bootstrap
    )
    integrity_passes = all(
        manifest.get("publication_eligible") is True for manifest in manifests.values()
    )
    promotion_go = integrity_passes and all(
        gate["passes_joint_auroc_aurc_gate"] for gate in gates.values()
    )

    output = {
        "schema": "pwsc-ci-independent-promotion-decision-v1",
        "decision": "Go" if promotion_go else "No-Go",
        "canonical_after_decision": "PWSC-CI",
        "promotion_conditions_all_pass": promotion_go,
        "claim": (
            "PWSC-CI passed the preregistered independent-model promotion gate."
            if promotion_go
            else "PWSC-CI did not pass the preregistered independent-model promotion gate."
        ),
        "model": {
            "id": EXPECTED_MODEL_ID,
            "root": EXPECTED_MODEL_ROOT,
            "upstream_revision": prereg["independent_model"]["upstream_revision"],
        },
        "service": service_manifest,
        "seed": 20260716,
        "tasks": list(TASKS),
        "question_count": 2187,
        "ensemble_size": 16,
        "bootstrap": {
            "draws": args.bootstrap,
            "seed": args.bootstrap_seed,
            "contract": "paired within-task resampling followed by four-task macro mean",
        },
        "integrity_passes": integrity_passes,
        "task_metrics": task_metrics,
        "macro_metrics": macro_metrics,
        "gates": gates,
        "selection": selection["selection"],
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "runner_inputs": provenance,
        "sidecar_headers": sidecar_headers,
        "caveats": [
            "SeSE-source-raw preserves the released sign bug; the sign-corrected row is a named one-operation adaptation.",
            "The SeSE source replay is not the paper's directed question-conditioned graph contract.",
            "CSA replaces its separately decoded greedy target with the shared fixed stochastic first path.",
            "The Llama service used the model-native unrequested top-p default 0.9 under vLLM generation-config auto.",
            "AIME, SMT, and LiveCodeBench are excluded from this four-domain gate.",
        ],
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
