#!/usr/bin/env python3
"""Replay canonical PWSC-CI scores and compare them with frozen sample rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config.settings import PWSCCIConfig  # noqa: E402
from core.risk_estimator import PWSCCIRiskEstimator  # noqa: E402
from run_pipeline import UniversalDatasetLoader, _sample_metadata  # noqa: E402


FREEZE_PATH = (
    ROOT / "code/diagnostics/pwsc_ci_frozen/pwsc_ci_results_20260717.json"
)
COMPONENT_COLUMNS = {
    "holevo_information": "pwsc_ci_holevo_information",
    "partition_entropy": "pwsc_ci_partition_entropy",
    "target_mass_ratio": "pwsc_ci_target_mass_ratio",
    "evidence_deficit": "pwsc_ci_evidence_deficit",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_task_replay(
    task: str,
    limit: Optional[int] = None,
    start: int = 0,
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    freeze = _load_json(FREEZE_PATH)
    provenance = freeze["provenance"]["manifests"][task]
    cache_path = ROOT / provenance["cache_path"]
    csv_path = ROOT / provenance["sample_csv_path"]

    frozen_rows: Dict[str, Dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["ensemble_size"]) == 16:
                frozen_rows[row["sample_id"]] = row

    loader = UniversalDatasetLoader("/data/newmodel/graphunagent/datasets")
    samples = loader.load_any_dataset(task, 0)
    estimator = PWSCCIRiskEstimator(weakest_k=PWSCCIConfig().weakest_k)
    source_counts: Counter[str] = Counter()
    maximum_difference = 0.0
    maximum_component_difference = 0.0
    checked = 0

    with cache_path.open(encoding="utf-8") as handle:
        next(handle)
        for index, (sample, line) in enumerate(zip(samples, handle)):
            if index < start:
                continue
            if limit is not None and checked >= limit:
                break
            record = json.loads(line)
            if record["sample_id"] != sample.sample_id:
                raise RuntimeError(f"cache/sample mismatch at {sample.sample_id}")
            traces = record["traces"][:16]
            texts = [str(trace["text"]) for trace in traces]
            logprobs = [list(trace["logprobs"]) for trace in traces]
            metadata = _sample_metadata(sample, task)
            observed, diagnostics = estimator.score(
                logprobs, texts, task, [metadata] * 16
            )
            frozen = frozen_rows[sample.sample_id]
            difference = abs(observed - float(frozen["pwsc_ci_risk"]))
            maximum_difference = max(maximum_difference, difference)
            if difference > tolerance:
                raise RuntimeError(
                    f"{task}/{sample.sample_id} risk drift: {difference}; "
                    f"observed={observed}, frozen={frozen['pwsc_ci_risk']}, "
                    f"source={diagnostics['kernel_source']}, "
                    f"kernel_diagnostics={diagnostics['kernel_diagnostics']}"
                )
            for component, column in COMPONENT_COLUMNS.items():
                component_difference = abs(
                    float(diagnostics["ci_components"][component])
                    - float(frozen[column])
                )
                maximum_component_difference = max(
                    maximum_component_difference, component_difference
                )
                if component_difference > tolerance:
                    raise RuntimeError(
                        f"{task}/{sample.sample_id} {component} drift: "
                        f"{component_difference}"
                    )
            source_counts[str(diagnostics["kernel_source"])] += 1
            checked += 1

    available = max(0, len(samples) - start)
    expected = min(available, limit) if limit is not None else available
    if checked != expected:
        raise RuntimeError(f"checked {checked} rows; expected {expected}")
    return {
        "status": "pass",
        "task": task,
        "checked_rows": checked,
        "maximum_risk_difference": maximum_difference,
        "maximum_component_difference": maximum_component_difference,
        "kernel_source_counts": dict(source_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO"),
        default="HumanEvalPlus",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 checks the full split")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    limit = None if args.limit == 0 else args.limit
    print(
        json.dumps(
            validate_task_replay(args.task, limit=limit, start=args.start), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
