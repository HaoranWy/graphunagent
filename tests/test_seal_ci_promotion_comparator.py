import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.seal_ci_promotion_comparator import METRICS, TASKS, _selection


def _row(auroc, aurc):
    return {
        "AUROC": auroc,
        "AUPRC": auroc / 2,
        "AURC": aurc,
        "SelectiveAcc@50": 0.5,
    }


def test_selection_uses_baseline_only_macro_and_fixed_task_envelope():
    methods = {
        "A": {task: _row(0.8, 0.2) for task in TASKS},
        "B": {
            task: _row(0.9 if task == "FOLIO" else 0.7, 0.1)
            for task in TASKS
        },
    }
    selected = _selection(methods)
    assert selected["strongest_single"]["method"] == "A"
    assert selected["task_envelope"]["method_by_task"] == {
        "GSM8K": "A",
        "MATH-500": "A",
        "HumanEvalPlus": "A",
        "FOLIO": "B",
    }
    assert tuple(selected["strongest_single"]["macro_metrics"]) == METRICS


def test_selection_breaks_exact_auroc_ties_with_lower_aurc_then_name():
    methods = {
        "B": {task: _row(0.8, 0.2) for task in TASKS},
        "A": {task: _row(0.8, 0.2) for task in TASKS},
        "C": {task: _row(0.8, 0.1) for task in TASKS},
    }
    selected = _selection(methods)
    assert selected["strongest_single"]["method"] == "C"
    assert set(selected["task_envelope"]["method_by_task"].values()) == {"C"}
