from scripts.validate_pwsc_ci_code_replay import validate_task_replay


def test_current_code_replays_a_frozen_code_domain_row():
    result = validate_task_replay("HumanEvalPlus", limit=1)
    assert result["status"] == "pass"
    assert result["checked_rows"] == 1
    assert result["maximum_risk_difference"] <= 1e-10
