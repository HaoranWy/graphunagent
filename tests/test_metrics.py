import math

from diagnostics.analyzer import ScientificSignalAnalyzer
from routing.calibration import DevPlattCalibrator


def test_auroc_and_auprc_perfect_ordering():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert ScientificSignalAnalyzer.calculate_auroc(labels, scores) == 1.0
    assert ScientificSignalAnalyzer.calculate_auprc(labels, scores) == 1.0


def test_auroc_uses_average_ranks_for_ties():
    assert ScientificSignalAnalyzer.calculate_auroc([0, 1], [0.5, 0.5]) == 0.5


def test_single_class_metrics_are_undefined_not_fabricated_half():
    assert math.isnan(ScientificSignalAnalyzer.calculate_auroc([0, 0], [0.1, 0.2]))
    assert math.isnan(ScientificSignalAnalyzer.calculate_auprc([0, 0], [0.1, 0.2]))
    assert math.isnan(ScientificSignalAnalyzer.calculate_auprc([1, 1], [0.1, 0.2]))


def test_selective_metrics_keep_low_risk_first():
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8]
    assert ScientificSignalAnalyzer.calculate_selective_accuracy(labels, scores, 0.5) == 1.0
    assert ScientificSignalAnalyzer.calculate_aurc(labels, scores) < 0.5


def test_calibration_is_labeled_dev_only_and_monotone():
    calibrator = DevPlattCalibrator()
    try:
        calibrator.fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], split_name="test")
        assert False, "test-set calibration must be rejected"
    except ValueError:
        pass
    calibrator.fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], split_name="validation")
    assert calibrator.calibrate(0.9) > calibrator.calibrate(0.1)


def test_paired_bootstrap_returns_a_finite_difference_for_mixed_labels():
    labels = [0, 0, 1, 1]
    a = [0.1, 0.2, 0.8, 0.9]
    b = [0.2, 0.3, 0.7, 0.8]
    estimate, low, high = ScientificSignalAnalyzer.paired_bootstrap_difference(
        labels, a, b, ScientificSignalAnalyzer.calculate_auroc, seed=3, n_bootstrap=100
    )
    assert estimate == 0.0
    assert low <= estimate <= high
