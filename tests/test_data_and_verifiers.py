import json
import time

from data.loaders import DatasetLoader
from routing.verifier_manager import (
    FOLIOVerifier,
    HumanEvalPlusVerifier,
    LiveCodeBenchVerifier,
    MBPPPlusVerifier,
)


def test_pfolio_expands_real_question_labels():
    samples = DatasetLoader().load_p_folio_split()
    labels = {sample.gold_answer for sample in samples}
    assert labels.issubset({"T", "F", "U"})
    assert len(labels) >= 2
    assert all("Conclusion:" in sample.raw_question for sample in samples)


def test_livecodebench_loader_decodes_private_tests():
    sample = DatasetLoader().load_livecodebench_split(1)[0]
    input_output = sample.metadata["input_output"]
    assert input_output["inputs"]
    assert len(input_output["inputs"]) == len(input_output["outputs"])
    assert sample.gold_answer != "[]"


def test_mbppplus_loader_uses_public_tests_without_reference_code():
    sample = DatasetLoader().load_mbppplus_test(1)[0]
    assert sample.sample_id == "mbppplus_2"
    assert sample.metadata["entry_point"] == "similar_elements"
    assert "def similar_elements(arg0, arg1):" in sample.raw_question
    assert "return tuple(set" not in sample.raw_question
    assert "# >>> similar_elements(" in sample.raw_question


def test_mbppplus_verifier_executes_plus_tests():
    sample = DatasetLoader().load_mbppplus_test(1)[0]
    verifier = MBPPPlusVerifier()
    correct = "def similar_elements(left, right):\n    return tuple(set(left) & set(right))"
    wrong = "def similar_elements(left, right):\n    return tuple()"
    assert verifier.execute(correct, sample.gold_answer, sample.metadata)
    assert not verifier.execute(wrong, sample.gold_answer, sample.metadata)


def test_logic_verifier_uses_last_label_and_normalizes_short_gold():
    verifier = FOLIOVerifier()
    assert verifier.execute("True was considered, therefore False", "F")
    assert not verifier.execute("False was considered, therefore True", "F")


def test_humaneval_verifier_executes_tests_instead_of_returning_true():
    verifier = HumanEvalPlusVerifier()
    tests = "def check(candidate):\n    assert candidate(2) == 4\n"
    metadata = {"entry_point": "square"}
    assert verifier.execute("def square(x):\n    return x * x", tests, metadata)
    assert not verifier.execute("def square(x):\n    return x + 1", tests, metadata)


def test_math_verifier_preserves_commas_in_symbolic_answers():
    from routing.verifier_manager import MATHOfficialVerifier

    verifier = MATHOfficialVerifier()
    candidate = r"\boxed{\left(3,\frac{\pi}{2}\right)}"
    gold = r"\left(3,\frac{\pi}{2}\right)"
    assert verifier.execute(candidate, gold)


def test_math_official_verifier_ignores_proof_certificate_suffix():
    from routing.verifier_manager import MATHOfficialVerifier

    candidate = (
        "The final answer is 6.\n"
        '<proof_certificate>{"final_answer":"999"}</proof_certificate>'
    )
    assert MATHOfficialVerifier().execute(candidate, "6")


def test_math_official_verifier_timeout_fails_closed(monkeypatch):
    import math_verify
    from core import adapters
    from routing.verifier_manager import MATHOfficialVerifier

    adapters._official_math_verify_cached.cache_clear()
    monkeypatch.setattr(math_verify, "parse", lambda value, **_kwargs: value)

    def slow_verify(_left, _right, **_kwargs):
        time.sleep(5.0)
        return True

    monkeypatch.setattr(math_verify, "verify", slow_verify)
    started = time.monotonic()
    assert not MATHOfficialVerifier().execute(
        r"\boxed{987654321}", "987654322"
    )
    assert time.monotonic() - started < 1.5
    diagnostics = adapters.official_math_verifier_runtime_diagnostics()
    assert diagnostics["timeout_count"] >= 1
    assert diagnostics["orientation"] == "gold_to_target"
    adapters._official_math_verify_cached.cache_clear()


def test_math_official_verifier_preserves_gold_target_order(monkeypatch):
    import math_verify
    from core import adapters
    from routing.verifier_manager import MATHOfficialVerifier

    calls = []
    adapters._official_math_verify_cached.cache_clear()
    monkeypatch.setattr(math_verify, "parse", lambda value, **_kwargs: value)

    def record_verify(gold, target, **_kwargs):
        calls.append((gold, target))
        return False

    monkeypatch.setattr(math_verify, "verify", record_verify)
    assert not MATHOfficialVerifier().execute(r"\boxed{1}", "9")
    assert calls == [("9", "1")]
    adapters._official_math_verify_cached.cache_clear()


def test_lcb_official_runner_on_first_problem():
    sample = DatasetLoader().load_livecodebench_split(1)[0]
    code = """import sys
data = sys.stdin.read().split()
out = []
for value in data[1:]:
    out.append('YES' if value in ('abc', 'acb', 'bac', 'cba') else 'NO')
print('\\n'.join(out))
"""
    assert LiveCodeBenchVerifier().execute(code, sample.gold_answer, sample.metadata)
