"""Statistical metrics with explicit undefined-case handling."""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class ScientificSignalAnalyzer:
    def __init__(self, tracker: Any = None) -> None:
        self.tracker = tracker

    @staticmethod
    def _arrays(y_true: Sequence[int], y_scores: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_true, dtype=np.int64)
        s = np.asarray(y_scores, dtype=np.float64)
        if y.size != s.size:
            raise ValueError("y_true and y_scores must have equal length")
        if s.size:
            s = np.where(np.isfinite(s), s, np.nan)
            finite = s[np.isfinite(s)]
            fill = float(np.median(finite)) if finite.size else 0.0
            s = np.where(np.isfinite(s), s, fill)
        return y, s

    @staticmethod
    def calculate_auroc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
        y, scores = ScientificSignalAnalyzer._arrays(y_true, y_scores)
        pos = y == 1
        neg = y == 0
        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        # Mann-Whitney probability with average ranks for ties. Constructing
        # tie groups in sorted order avoids an O(n^2) equality scan inside
        # every bootstrap replicate.
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        ends = np.r_[starts[1:], scores.size]
        average_ranks = (starts + 1.0 + ends) / 2.0
        sorted_ranks = np.repeat(average_ranks, ends - starts)
        ranks = np.empty(scores.size, dtype=np.float64)
        ranks[order] = sorted_ranks
        u = float(ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0)
        return u / (n_pos * n_neg)

    @staticmethod
    def calculate_auprc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
        y, scores = ScientificSignalAnalyzer._arrays(y_true, y_scores)
        n_pos = int(np.sum(y == 1))
        if n_pos == 0 or n_pos == y.size or y.size == 0:
            return float("nan")
        order = np.argsort(-scores, kind="mergesort")
        y_sorted = y[order]
        score_sorted = scores[order]
        tp = 0
        fp = 0
        previous_recall = 0.0
        area = 0.0
        start = 0
        while start < y_sorted.size:
            end = start + 1
            while end < y_sorted.size and score_sorted[end] == score_sorted[start]:
                end += 1
            tp += int(np.sum(y_sorted[start:end] == 1))
            fp += int(np.sum(y_sorted[start:end] == 0))
            recall = tp / n_pos
            precision = tp / max(tp + fp, 1)
            area += precision * (recall - previous_recall)
            previous_recall = recall
            start = end
        return float(area)

    @staticmethod
    def risk_coverage(y_true: Sequence[int], risk_scores: Sequence[float]) -> Dict[str, List[float]]:
        y, scores = ScientificSignalAnalyzer._arrays(y_true, risk_scores)
        if y.size == 0:
            return {"coverage": [], "selective_error": []}
        order = np.argsort(scores, kind="mergesort")
        sorted_errors = y[order].astype(float)
        cumulative = np.cumsum(sorted_errors)
        coverage = np.arange(1, y.size + 1, dtype=np.float64) / y.size
        selective_error = cumulative / np.arange(1, y.size + 1, dtype=np.float64)
        return {"coverage": coverage.tolist(), "selective_error": selective_error.tolist()}

    @staticmethod
    def calculate_aurc(y_true: Sequence[int], risk_scores: Sequence[float]) -> float:
        curve = ScientificSignalAnalyzer.risk_coverage(y_true, risk_scores)
        if not curve["coverage"]:
            return float("nan")
        return float(np.mean(curve["selective_error"]))

    @staticmethod
    def calculate_selective_accuracy(y_true: Sequence[int], risk_scores: Sequence[float], coverage: float = 0.5) -> float:
        y, scores = ScientificSignalAnalyzer._arrays(y_true, risk_scores)
        if y.size == 0:
            return float("nan")
        keep = max(1, min(y.size, int(math.ceil(float(coverage) * y.size))))
        order = np.argsort(scores, kind="mergesort")[:keep]
        return float(1.0 - np.mean(y[order]))

    @staticmethod
    def calculate_top_50_acc(is_correct_list: Sequence[bool], risk_scores: Sequence[float]) -> float:
        if not is_correct_list:
            return float("nan")
        order = np.argsort(np.asarray(risk_scores, dtype=float), kind="mergesort")
        keep = max(1, len(order) // 2)
        return float(np.mean(np.asarray(is_correct_list, dtype=bool)[order[:keep]]))

    @staticmethod
    def calculate_ece(y_true: Sequence[int], y_probs: Sequence[float], num_bins: int = 10) -> float:
        y, probs = ScientificSignalAnalyzer._arrays(y_true, y_probs)
        if y.size == 0:
            return float("nan")
        probs = np.clip(probs, 0.0, 1.0)
        total = 0.0
        for idx in range(num_bins):
            lower, upper = idx / num_bins, (idx + 1) / num_bins
            mask = (probs >= lower) & ((probs <= upper) if idx == num_bins - 1 else (probs < upper))
            if np.any(mask):
                total += float(mask.mean()) * abs(float(probs[mask].mean()) - float(y[mask].mean()))
        return total

    @staticmethod
    def calculate_brier_score(y_true: Sequence[int], y_probs: Sequence[float]) -> float:
        y, probs = ScientificSignalAnalyzer._arrays(y_true, y_probs)
        if y.size == 0:
            return float("nan")
        return float(np.mean((np.clip(probs, 0.0, 1.0) - y) ** 2))

    @staticmethod
    def calculate_pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
        if len(x) < 2 or len(y) < 2:
            return float("nan")
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if np.std(xa) == 0 or np.std(ya) == 0:
            return float("nan")
        return float(np.corrcoef(xa, ya)[0, 1])

    @staticmethod
    def compute_ast_entropy(code: str) -> float:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, TypeError):
            return float("nan")
        degrees = np.asarray([len(list(ast.iter_child_nodes(node))) + 1e-12 for node in ast.walk(tree)], dtype=float)
        probs = degrees / degrees.sum()
        return float(-np.sum(probs * np.log(probs)))

    @staticmethod
    def bootstrap_ci(
        y_true: Sequence[int],
        scores: Sequence[float],
        metric: Callable[[Sequence[int], Sequence[float]], float],
        seed: int = 0,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        y, s = ScientificSignalAnalyzer._arrays(y_true, scores)
        estimate = float(metric(y, s))
        if y.size < 2:
            return estimate, float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        values: List[float] = []
        for _ in range(int(n_bootstrap)):
            indices = rng.integers(0, y.size, y.size)
            value = float(metric(y[indices], s[indices]))
            if np.isfinite(value):
                values.append(value)
        if not values:
            return estimate, float("nan"), float("nan")
        alpha = (1.0 - confidence) / 2.0
        return estimate, float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))

    @staticmethod
    def paired_bootstrap_difference(
        y_true: Sequence[int],
        scores_a: Sequence[float],
        scores_b: Sequence[float],
        metric: Callable[[Sequence[int], Sequence[float]], float],
        seed: int = 0,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Question-level paired bootstrap for method A minus method B."""
        y, a = ScientificSignalAnalyzer._arrays(y_true, scores_a)
        _, b = ScientificSignalAnalyzer._arrays(y_true, scores_b)
        estimate = float(metric(y, a) - metric(y, b))
        if y.size < 2:
            return estimate, float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        values: List[float] = []
        for _ in range(int(n_bootstrap)):
            indices = rng.integers(0, y.size, y.size)
            value = float(metric(y[indices], a[indices]) - metric(y[indices], b[indices]))
            if np.isfinite(value):
                values.append(value)
        if not values:
            return estimate, float("nan"), float("nan")
        alpha = (1.0 - confidence) / 2.0
        return estimate, float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))

    def audit_m2_scientific_signal(self) -> Dict[str, Any]:
        if self.tracker is None:
            return {"status": "SKIPPED", "reason": "no tracker"}
        logs = [item for item in self.tracker.get_session_logs() if item.get("is_correct") is not None]
        if not logs:
            return {"status": "SKIPPED", "reason": "no verified samples"}
        y = [0 if item["is_correct"] else 1 for item in logs]
        scores = [float(item.get("r_ise_score", item.get("risk", 0.0))) for item in logs]
        return {"size": len(logs), "auroc": self.calculate_auroc(y, scores), "auprc": self.calculate_auprc(y, scores), "aurc": self.calculate_aurc(y, scores)}


__all__ = ["ScientificSignalAnalyzer"]
