from pathlib import Path
import hashlib
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_pwsc_ci_controls import (  # noqa: E402
    _score_controls,
    _uniform_peer_mass,
)
from scripts.summarize_pwsc_ci_supplement import (  # noqa: E402
    _method_column,
    _model_sampling_signature,
    _paired_bootstrap,
    _resolve_bound_file,
    _risk_coverage,
    _single_manifest,
    _validate_cache_sampling_contract,
)


def _path(value):
    return [{"token": "answer", "logprob": value}]


def test_uniform_peer_control_preserves_target_and_peer_mass():
    q = np.asarray([0.8, 0.6, 0.2, 0.0])
    observed = _uniform_peer_mass(q, 0)
    assert observed[0] == q[0]
    assert np.isclose(observed[1:].sum(), q[1:].sum())
    assert np.isclose(observed[1], observed[2])
    assert observed[3] == 0.0


def test_control_scores_include_predeclared_single_factor_sweeps():
    controls, diagnostics = _score_controls(
        [_path(-0.1), _path(-0.2), _path(-0.3)],
        [r"\boxed{1}", r"\boxed{1}", r"\boxed{2}"],
        "GSM8K",
        {"prompt": "question"},
        0,
        (("1", 1), ("5", 5), ("all", 2**31 - 1)),
        (4,),
        (False, True),
    )
    assert 0.0 <= controls["pwsc_ci_risk"] <= 1.0
    assert np.isclose(controls["control_ci_weakest_k_5"], controls["pwsc_ci_risk"])
    assert np.isclose(controls["control_ci_task_kernel"], controls["pwsc_ci_risk"])
    assert "control_ci_uniform_peer_mass" in controls
    assert diagnostics["probe_diagnostics"] == {}


def test_supplemental_summary_keeps_models_separate_and_is_reproducible():
    rows = []
    labels = [0] * 10 + [1] * 10
    for model_index, model in enumerate(("model-a", "model-b")):
        for task in ("GSM8K", "FOLIO"):
            for index, label in enumerate(labels):
                rows.append(
                    {
                        "model_key": model,
                        "seed": 7,
                        "task": task,
                        "sample_id": f"{task}-{index}",
                        "error": label,
                        "PWSC-CI": index / 20 + model_index * 1e-4,
                        "Mean NLL": ((index * 7) % 20) / 20,
                    }
                )
    first = _paired_bootstrap(rows, draws=30, seed=11)
    second = _paired_bootstrap(rows, draws=30, seed=11)
    assert first == second
    assert set(first) == {"model-a", "model-b"}
    curves, metadata = _risk_coverage(rows, draws=10, seed=13)
    assert set(metadata) == {"model-a", "model-b"}
    assert {row["method"] for row in curves} == {"PWSC-CI", "Mean NLL"}


def test_method_column_never_accepts_retired_pwsc_variants():
    assert _method_column(["pwsc_ci_risk", "pwsc_js_risk"], ("pwsc_ci_risk",)) == (
        "pwsc_ci_risk"
    )
    with np.testing.assert_raises(ValueError):
        _method_column(["pwsc_js_risk"], ("pwsc_ci_risk",))


def test_summary_missing_run_dir_reports_stale_root(tmp_path):
    missing = tmp_path / "missing-seed"
    with np.testing.assert_raises_regex(
        FileNotFoundError, "check --run-dir and SUP_ROOT"
    ):
        _single_manifest(missing, "GSM8K")


def test_summary_prefers_relocated_bundle_cache(tmp_path):
    run_dir = tmp_path / "bundle" / "core" / "model" / "seed_7"
    cache_dir = (
        tmp_path / "bundle" / "generation_cache" / "model" / "seed_7"
    )
    run_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    payload = b"bound generation cache\n"
    cache_path = cache_dir / "generations_GSM8K.jsonl"
    cache_path.write_bytes(payload)

    resolved = _resolve_bound_file(
        run_dir,
        str(tmp_path / "retired" / cache_path.name),
        hashlib.sha256(payload).hexdigest(),
        "generation cache",
    )

    assert resolved == cache_path.resolve()


def test_summary_binds_model_specific_stop_and_top_k_contract():
    sampling = {
        "num_samples": 16,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 12000,
        "requested_top_logprobs": 10,
        "stop_token_ids": [106],
        "sampling_top_k": 0,
        "generation_protocol": "proof-carrying-pc-v1",
        "python_hash_seed": "20260716",
    }
    _validate_cache_sampling_contract(sampling, {"sampling": sampling}, "GSM8K")
    mismatched = dict(sampling, sampling_top_k=64)
    with np.testing.assert_raises_regex(
        ValueError, "cache sampling contract mismatch for GSM8K: sampling_top_k"
    ):
        _validate_cache_sampling_contract(
            mismatched, {"sampling": sampling}, "GSM8K"
        )


def test_summary_normalizes_absent_and_empty_stop_contracts():
    absent = {
        "num_samples": 16,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 12000,
        "requested_top_logprobs": 10,
        "generation_protocol": "proof-carrying-pc-v1",
        "python_hash_seed": "20260723",
    }
    empty = dict(absent, stop_token_ids=[])
    _validate_cache_sampling_contract(absent, {"sampling": empty}, "FOLIO")
    assert _model_sampling_signature(absent) == _model_sampling_signature(empty)
    assert _model_sampling_signature(
        dict(absent, stop_token_ids=[106])
    ) != _model_sampling_signature(empty)
