from scripts.validate_pwsc_ci_freeze import validate_freeze


def test_active_pwsc_ci_freeze_matches_immutable_provenance():
    result = validate_freeze()
    assert result["status"] == "pass"
    assert result["method"] == "PWSC-CI-v1.0"
    assert result["question_count"] == 2187
