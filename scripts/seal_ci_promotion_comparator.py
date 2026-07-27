#!/usr/bin/env python3
"""Seal development-only comparator choices for the PWSC-CI promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO")
RDS_METHODS = ("RDS-L1", "RDS-L2", "RDS-PW-L1", "RDS-PW-L2")
SESE_METHODS = ("SeSE-source-raw", "SeSE-source-sign-corrected")
ELIGIBLE_METHODS = (*RDS_METHODS, "CSA", *SESE_METHODS)
METRICS = ("AUROC", "AUPRC", "AURC", "SelectiveAcc@50")
EXPECTED_PREREG_SHA256 = (
    "444f46ee4c1d1c8f7a6f0ce61aad5548792695d98699859e8130abd6eb543f62"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_contract(data: Mapping[str, Any], path: Path) -> None:
    if data.get("status") != "development_blackbox_baseline_comparison":
        raise ValueError(f"not a development comparison: {path}")
    if int(data.get("seed", -1)) != 20260615:
        raise ValueError(f"unexpected development seed: {path}")
    if tuple(data.get("tasks", ())) != TASKS:
        raise ValueError(f"unexpected task order: {path}")


def _metric_row(value: Mapping[str, Any], method: str, task: str) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for metric in METRICS:
        number = float(value[metric])
        if not math.isfinite(number):
            raise ValueError(f"non-finite {method} {task} {metric}: {number}")
        row[metric] = number
    return row


def _input_fingerprint(data: Mapping[str, Any], task: str) -> Tuple[str, str, str]:
    value = data["inputs"][task]
    return (
        str(value["manifest_sha256"]),
        str(value["sample_csv_sha256"]),
        str(value["cache_sha256"]),
    )


def _extract_methods(
    rds: Mapping[str, Any], csa: Mapping[str, Any], sese: Mapping[str, Any]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    sources = (rds, csa, sese)
    for task in TASKS:
        fingerprints = {_input_fingerprint(source, task) for source in sources}
        if len(fingerprints) != 1:
            raise ValueError(f"baseline inputs differ for {task}: {fingerprints}")

    rds_block = rds["Metrics by N"]["N=16"]["task_metrics"]
    csa_block = csa["Metrics by N"]["N=16"]["task_metrics"]
    sese_block = sese["task_metrics"]
    output: Dict[str, Dict[str, Dict[str, float]]] = {
        method: {} for method in ELIGIBLE_METHODS
    }
    for task in TASKS:
        for method in RDS_METHODS:
            output[method][task] = _metric_row(rds_block[task][method], method, task)
        output["CSA"][task] = _metric_row(csa_block[task]["CSA"], "CSA", task)
        for method in SESE_METHODS:
            output[method][task] = _metric_row(
                sese_block[task][method], method, task
            )
    return output


def _macro(task_rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    return {
        metric: float(sum(row[metric] for row in task_rows) / len(task_rows))
        for metric in METRICS
    }


def _selection(
    methods: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> Dict[str, Any]:
    macro = {
        method: _macro([task_rows[task] for task in TASKS])
        for method, task_rows in methods.items()
    }
    strongest = min(
        methods,
        key=lambda method: (
            -macro[method]["AUROC"],
            macro[method]["AURC"],
            method,
        ),
    )
    task_mapping = {
        task: min(
            methods,
            key=lambda method: (
                -methods[method][task]["AUROC"],
                methods[method][task]["AURC"],
                method,
            ),
        )
        for task in TASKS
    }
    envelope_rows = [methods[task_mapping[task]][task] for task in TASKS]
    return {
        "strongest_single": {
            "method": strongest,
            "macro_metrics": macro[strongest],
        },
        "task_envelope": {
            "method_by_task": task_mapping,
            "macro_metrics": _macro(envelope_rows),
        },
        "all_method_macro_metrics": macro,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT
        / ".pipeline/docs/pwsc_ci_independent_promotion_protocol_20260716.json",
    )
    parser.add_argument("--rds", type=Path, required=True)
    parser.add_argument("--csa", type=Path, required=True)
    parser.add_argument("--sese", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "preregistration": args.preregistration.expanduser().resolve(),
        "RDS": args.rds.expanduser().resolve(),
        "CSA": args.csa.expanduser().resolve(),
        "SeSE": args.sese.expanduser().resolve(),
    }
    prereg_sha = _sha256(paths["preregistration"])
    if prereg_sha != EXPECTED_PREREG_SHA256:
        raise RuntimeError(
            f"promotion preregistration drift: expected {EXPECTED_PREREG_SHA256}, "
            f"got {prereg_sha}"
        )
    inputs = {name: _load(path) for name, path in paths.items() if name != "preregistration"}
    for name, data in inputs.items():
        _validate_contract(data, paths[name])
    methods = _extract_methods(inputs["RDS"], inputs["CSA"], inputs["SeSE"])

    output = {
        "schema": "pwsc-ci-development-comparator-selection-v1",
        "selection_uses_pwsc_ci_values": False,
        "tasks": list(TASKS),
        "ensemble_size": 16,
        "development_seed": 20260615,
        "preregistration": {
            "path": str(paths["preregistration"]),
            "sha256": prereg_sha,
        },
        "selection_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "inputs": {
            name: {"path": str(paths[name]), "sha256": _sha256(paths[name])}
            for name in ("RDS", "CSA", "SeSE")
        },
        "eligible_task_metrics": methods,
        "selection": _selection(methods),
        "rule": {
            "strongest_single": [
                "maximum development macro AUROC",
                "minimum development macro AURC",
                "lexicographic method name",
            ],
            "task_envelope": [
                "maximum development task AUROC",
                "minimum development task AURC",
                "lexicographic method name",
            ],
        },
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

