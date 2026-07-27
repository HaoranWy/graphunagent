#!/usr/bin/env python3
"""Validate the active PWSC-CI result freeze against immutable provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = (
    ROOT
    / "code/diagnostics/pwsc_ci_frozen/pwsc_ci_results_20260717.json"
)
ALLOWED_METRIC_METHODS = {
    "PWSC-CI",
    "RDS-PW-L1",
    "CSA",
    "SeSE-source-raw",
}


def _load(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _assert_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} drift: {observed!r} != {expected!r}")


def validate_freeze(path: Path = DEFAULT_FREEZE) -> Dict[str, Any]:
    freeze = _load(path)
    _assert_equal(freeze.get("schema"), "pwsc-ci-result-freeze-v1", "schema")
    _assert_equal(freeze.get("method"), "PWSC-CI-v1.0", "method")
    _assert_equal(
        freeze.get("contract"),
        "pwsc-ci-normalized-holevo-v1",
        "score contract",
    )
    _assert_equal(freeze.get("score_column"), "pwsc_ci_risk", "score column")
    _assert_equal(freeze.get("decision"), "Go", "decision")

    provenance = freeze["provenance"]
    decision_path = _resolve(provenance["decision_path"])
    prereg_path = _resolve(provenance["preregistration_path"])
    _assert_equal(
        _sha256(decision_path), provenance["decision_sha256"], "decision hash"
    )
    _assert_equal(
        _sha256(prereg_path),
        provenance["preregistration_sha256"],
        "preregistration hash",
    )
    decision = _load(decision_path)

    scope = freeze["evidence_scope"]
    for key in ("seed", "ensemble_size", "question_count", "tasks", "bootstrap"):
        _assert_equal(scope[key], decision[key], f"evidence scope {key}")
    _assert_equal(scope["model"], decision["model"]["id"], "model id")
    _assert_equal(
        scope["model_root"], decision["model"]["root"], "model revision"
    )
    _assert_equal(
        scope["model_upstream_revision"],
        decision["model"]["upstream_revision"],
        "upstream model revision",
    )

    macro = freeze["macro_metrics"]
    if not ALLOWED_METRIC_METHODS.issubset(macro):
        raise RuntimeError("freeze is missing an admitted metric row")
    pwsc_rows = {name for name in macro if str(name).startswith("PWSC")}
    _assert_equal(pwsc_rows, {"PWSC-CI"}, "active PWSC method rows")
    for method in ALLOWED_METRIC_METHODS:
        _assert_equal(
            macro[method], decision["macro_metrics"][method], f"macro {method}"
        )
    for task, methods in freeze["task_metrics"].items():
        _assert_equal(set(methods), ALLOWED_METRIC_METHODS, f"{task} method set")
        for method, metrics in methods.items():
            _assert_equal(
                metrics,
                decision["task_metrics"][task][method],
                f"{task} {method}",
            )

    gates = freeze["promotion_gates"]
    source_gates = decision["gates"]
    for name in ("strongest_single", "task_envelope"):
        _assert_equal(
            gates[name]["delta_AUROC"],
            source_gates[name]["paired_macro_differences"]["AUROC"],
            f"{name} AUROC gate",
        )
        _assert_equal(
            gates[name]["delta_AURC"],
            source_gates[name]["paired_macro_differences"]["AURC"],
            f"{name} AURC gate",
        )
        _assert_equal(
            gates[name]["passed"],
            source_gates[name]["passes_joint_auroc_aurc_gate"],
            f"{name} decision",
        )

    for task, record in provenance["manifests"].items():
        manifest_path = _resolve(record["manifest_path"])
        cache_path = _resolve(record["cache_path"])
        csv_path = _resolve(record["sample_csv_path"])
        _assert_equal(
            _sha256(manifest_path), record["manifest_sha256"], f"{task} manifest"
        )
        _assert_equal(_sha256(cache_path), record["cache_sha256"], f"{task} cache")
        _assert_equal(
            _sha256(csv_path), record["sample_csv_sha256"], f"{task} sample CSV"
        )
        manifest = _load(manifest_path)
        dataset = freeze["datasets"][task]
        _assert_equal(manifest["seed"], scope["seed"], f"{task} seed")
        _assert_equal(
            manifest["model_revision"], scope["model_root"], f"{task} model"
        )
        _assert_equal(
            manifest["input_sample_sha256"], dataset["input_sha256"], f"{task} input"
        )
        _assert_equal(
            manifest["generation_cache_sha256"],
            record["cache_sha256"],
            f"{task} manifest cache",
        )
        _assert_equal(
            manifest["csv_sha256"],
            record["sample_csv_sha256"],
            f"{task} manifest CSV",
        )
        fingerprint = manifest["dataset_fingerprint"]
        _assert_equal(fingerprint["dataset_id"], dataset["dataset_id"], f"{task} dataset")
        _assert_equal(fingerprint["revision"], dataset["revision"], f"{task} revision")
        _assert_equal(
            fingerprint["files"][0]["sha256"],
            dataset["file_sha256"],
            f"{task} dataset file",
        )

    return {
        "status": "pass",
        "method": freeze["method"],
        "decision_sha256": provenance["decision_sha256"],
        "tasks": scope["tasks"],
        "question_count": scope["question_count"],
    }


def main() -> int:
    print(json.dumps(validate_freeze(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
