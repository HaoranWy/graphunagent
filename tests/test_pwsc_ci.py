import math
import time

import numpy as np

from core.risk_estimator import (
    PWSCCIRiskEstimator,
    retain_scoring_tokens,
    target_conditional_information_components,
    target_conditional_information_risk,
)
from routing.baselines import SCSVBaselines


def path(values):
    return [
        {"token": f"t{index}", "logprob": value}
        for index, value in enumerate(values)
    ]


def test_weakest_k_path_action_and_control_filtering():
    estimator = PWSCCIRiskEstimator(weakest_k=3)
    values = path([-0.1, -0.2, -1.0, -2.0])
    values.extend(
        [
            {"token": " ", "logprob": -99.0},
            {"token": "<|eos|>", "logprob": -99.0},
            {"token": "bad", "logprob": float("nan")},
        ]
    )
    action = estimator.path_action(values)
    assert action.valid
    assert action.token_count == 4
    assert action.action == np.mean([0.2, 1.0, 2.0])
    assert action.confidence == math.exp(-action.action)


def test_single_path_limit_is_one_minus_confidence():
    estimator = PWSCCIRiskEstimator(weakest_k=5)
    risk, diagnostics = estimator.score(
        [path([-0.2, -0.4])], [r"work \boxed{2}"], "GSM8K"
    )
    assert risk == 1.0 - math.exp(-0.3)
    assert diagnostics["ci_components"]["degenerate"] == 1.0


def test_invalid_or_missing_path_fails_closed():
    estimator = PWSCCIRiskEstimator()
    risk, diagnostics = estimator.score(
        [path([-0.1])],
        ["def target(:"],
        "HumanEvalPlus",
        [{"entry_point": "target"}],
    )
    assert risk == 1.0
    assert diagnostics["valid_count"] == 0
    assert retain_scoring_tokens([]) == []


def test_task_kernel_is_psd_and_diagnostics_are_ci_only():
    risk, diagnostics = PWSCCIRiskEstimator().score(
        [path([-0.1])] * 3,
        [r"\boxed{1}", r"\boxed{1}", r"\boxed{2}"],
        "MATH-500",
    )
    assert np.linalg.eigvalsh(diagnostics["kernel"]).min() >= -1e-10
    assert 0.0 <= risk <= 1.0
    assert diagnostics["method"] == "PWSC-CI-v1.0"
    assert diagnostics["contract_version"] == "pwsc-ci-normalized-holevo-v1"
    assert not {
        "rho",
        "eigenvalues",
        "spectral_entropy",
        "normalized_spectral_uncertainty",
    }.intersection(diagnostics)


def test_identity_and_orthogonal_behavioral_limits():
    assert np.isclose(
        target_conditional_information_risk([0.8, 0.7], np.ones((2, 2))),
        0.0,
    )
    assert np.isclose(
        target_conditional_information_risk([0.8, 0.7], np.eye(2)),
        1.0,
    )


def test_disagreement_increases_risk_at_fixed_confidence():
    estimator = PWSCCIRiskEstimator()
    logprobs = [path([-0.2]), path([-0.2])]
    same, _ = estimator.score(
        logprobs, [r"\boxed{1}", r"\boxed{1}"], "GSM8K"
    )
    different, _ = estimator.score(
        logprobs, [r"\boxed{1}", r"\boxed{2}"], "GSM8K"
    )
    assert different > same


def test_positive_logprob_is_invalid_and_never_imputed():
    estimator = PWSCCIRiskEstimator()
    assert estimator.path_action([-0.2]).valid
    assert not estimator.path_action([0.2]).valid
    risk, diagnostics = estimator.score([[]], [r"\boxed{1}"], "GSM8K")
    assert risk == 1.0
    assert diagnostics["path_confidences"].tolist() == [0.0]


def test_math_semantic_equivalence_is_shared_with_task_kernel():
    _, diagnostics = PWSCCIRiskEstimator().score(
        [path([-0.1]), path([-0.1])],
        [r"\boxed{1/2}", r"\boxed{0.5}"],
        "MATH-500",
    )
    assert diagnostics["outcomes"][0] == diagnostics["outcomes"][1]
    assert diagnostics["raw_outcomes"][0] != diagnostics["raw_outcomes"][1]


def test_math_answer_canonicalization_ignores_cache_protocol_suffix():
    from core.adapters import canonical_math_answer

    text = (
        "The final answer is 6.\n"
        '<proof_certificate>{"final_answer":"999"}</proof_certificate>'
    )
    assert canonical_math_answer(text) == "6"


def test_math_verify_timeout_fails_closed(monkeypatch):
    import math_verify
    from core import adapters

    adapters._math_equivalent_cached.cache_clear()
    monkeypatch.setattr(math_verify, "parse", lambda value, **_kwargs: value)

    def slow_verify(_left, _right, **_kwargs):
        time.sleep(5.0)
        return True

    monkeypatch.setattr(math_verify, "verify", slow_verify)
    started = time.monotonic()
    assert not adapters._math_equivalent("987654321", "987654322")
    assert time.monotonic() - started < 1.5
    assert adapters.math_equivalence_runtime_diagnostics()["timeout_count"] >= 1
    adapters._math_equivalent_cached.cache_clear()


def test_ci_uses_natural_confidence_mass():
    kernel = np.asarray(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    partition_entropy = -(1.0 / 3.0) * math.log(1.0 / 3.0) - (
        2.0 / 3.0
    ) * math.log(2.0 / 3.0)
    expected = (
        partition_entropy - (2.0 / 3.0) * math.log(2.0)
    ) / partition_entropy
    assert np.isclose(
        target_conditional_information_risk([1.0, 1.0, 1.0], kernel),
        expected,
    )


def test_ci_exposes_holevo_bound_and_invariances():
    confidences = np.asarray([0.8, 0.6, 0.2], dtype=np.float64)
    kernel = np.asarray(
        [[1.0, 0.7, 0.1], [0.7, 1.0, 0.2], [0.1, 0.2, 1.0]],
        dtype=np.float64,
    )
    components = target_conditional_information_components(confidences, kernel)
    assert components["holevo_information"] <= components["partition_entropy"] + 1e-10
    assert np.isclose(components["target_mass_ratio"], 0.5)
    assert np.isclose(components["evidence_deficit"], 1.0 - np.mean(confidences))

    scaled = target_conditional_information_components(0.25 * confidences, kernel)
    assert np.isclose(
        scaled["normalized_information"], components["normalized_information"]
    )
    permutation = np.asarray([0, 2, 1])
    permuted = target_conditional_information_components(
        confidences[permutation], kernel[np.ix_(permutation, permutation)]
    )
    assert np.isclose(
        permuted["normalized_information"], components["normalized_information"]
    )


def test_ci_rejects_non_gram_similarity_tables():
    asymmetric = np.asarray([[1.0, 0.2], [0.3, 1.0]], dtype=np.float64)
    with np.testing.assert_raises_regex(ValueError, "symmetric kernel"):
        target_conditional_information_components([0.8, 0.7], asymmetric)

    unnormalized = np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    with np.testing.assert_raises_regex(ValueError, "unit-norm active"):
        target_conditional_information_components([0.8, 0.7], unnormalized)

    non_psd = np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    with np.testing.assert_raises_regex(ValueError, "PSD kernel"):
        target_conditional_information_components([0.8, 0.7], non_psd)


def test_generic_baseline_formulas_remain_explicit_adaptations():
    kernel = np.asarray([[1.0, 0.5], [0.5, 1.0]])
    values = path([-0.2, -0.4])
    assert np.isclose(SCSVBaselines.calculate_cocoa_ppl(values, kernel), 0.15)
    assert np.isclose(SCSVBaselines.calculate_cocoa_msp(values, kernel), 0.3)
    assert SCSVBaselines.calculate_kernel_language_entropy(np.ones((3, 3))) == 0.0
    assert np.isclose(
        SCSVBaselines.calculate_kernel_language_entropy(np.eye(3)), 1.0
    )


def test_code_ci_uses_public_behavior_and_remains_psd():
    prompt = "def increment(x: int) -> int:\n    pass\n"
    texts = [
        "def increment(x: int) -> int:\n    return x + 1",
        "def increment(x: int) -> int:\n    return 1 + x",
        "def increment(x: int) -> int:\n    return x - 1",
    ]
    estimator = PWSCCIRiskEstimator()
    risk, diagnostics = estimator.score(
        [path([-0.1])] * 3,
        texts,
        "HumanEvalPlus",
        [{"prompt": prompt, "entry_point": "increment"}] * 3,
    )
    kernel = diagnostics["kernel"]
    assert diagnostics["kernel_source"] == "type_driven_behavior"
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.isclose(kernel[0, 1], 1.0)
    assert kernel[0, 2] < 1.0
    assert np.linalg.eigvalsh(kernel).min() >= -1e-10
    assert 0.0 <= risk <= 1.0


def test_behavior_kernel_exposes_probe_and_recovery_controls():
    from core.behavioral_kernel import build_behavior_kernel

    prompt = "def increment(x: int) -> int:\n    pass\n"
    texts = [
        "def increment(x: int) -> int:\n    return x + 1",
        "def increment(x: int) -> int:\n    return 1 + x",
    ]
    kernel, diagnostics = build_behavior_kernel(
        texts,
        np.ones(2, dtype=bool),
        prompt,
        "increment",
        max_probes=4,
        recover_single_unsupported=False,
    )
    assert kernel is not None
    assert diagnostics["max_probes"] == 4
    assert diagnostics["probe_count"] <= 4
    assert diagnostics["recover_single_unsupported"] is False
    assert diagnostics["recovered_path_count"] == 0
    with np.testing.assert_raises_regex(ValueError, "max_probes"):
        build_behavior_kernel(texts, np.ones(2, dtype=bool), prompt, "increment", max_probes=0)


def test_universally_unsupported_probe_recovers_timed_out_candidates():
    prompt = "def terminates_only_for_positive(n: int) -> int:\n    pass\n"
    texts = [
        (
            "def terminates_only_for_positive(n: int) -> int:\n"
            "    while n == 0:\n"
            "        pass\n"
            "    return n"
        ),
        (
            "def terminates_only_for_positive(n: int) -> int:\n"
            "    while n == 0:\n"
            "        pass\n"
            "    return n + 1"
        ),
        (
            "def terminates_only_for_positive(n: int) -> int:\n"
            "    if n == 0:\n"
            "        raise ValueError\n"
            "    return n"
        ),
    ]
    _, diagnostics = PWSCCIRiskEstimator().score(
        [path([-0.1])] * 3,
        texts,
        "HumanEvalPlus",
        [
            {"prompt": prompt, "entry_point": "terminates_only_for_positive"}
        ]
        * 3,
    )
    behavior = diagnostics["kernel_diagnostics"]
    assert diagnostics["kernel_source"] == "type_driven_behavior"
    assert behavior["active_probe_count"] == behavior["probe_count"] - 1
    assert behavior["successful_calls_by_path"] == [
        behavior["probe_count"] - 1,
        behavior["probe_count"] - 1,
        behavior["probe_count"] - 1,
    ]
