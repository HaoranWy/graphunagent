import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_ci_independent_promotion import TASKS, _paired_gate  # noqa: E402


def test_paired_gate_is_reproducible_and_checks_both_metrics():
    arrays = {}
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
    for offset, task in enumerate(TASKS):
        comparator = np.asarray(
            [0.1, 0.5, 0.7, 0.2, 0.4, 0.8, 0.3, 0.6, 0.9, 0.2] * 2
        ) + offset * 1e-4
        candidate = np.linspace(0.0, 1.0, len(labels)) + offset * 1e-4
        arrays[task] = {
            "error": labels,
            "PWSC-CI": candidate,
            "baseline": comparator,
        }
    comparators = {"single": {task: "baseline" for task in TASKS}}
    first = _paired_gate(arrays, comparators, seed=7, draws=100)
    second = _paired_gate(arrays, comparators, seed=7, draws=100)
    assert first == second
    assert first["single"]["paired_macro_differences"]["AUROC"][0] > 0
    assert first["single"]["paired_macro_differences"]["AURC"][0] < 0
    assert first["single"]["passes_joint_auroc_aurc_gate"] is True
