"""Non-PWSC uncertainty baselines used by the PWSC-CI experiment."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, List, Optional, Sequence

import numpy as np


class SCSVBaselines:
    @staticmethod
    def _nll_values(token_logprobs: Sequence[Any]) -> np.ndarray:
        values = []
        for item in token_logprobs or []:
            value = item.get("logprob") if isinstance(item, dict) else item
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(max(0.0, -value))
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def calculate_nll(token_logprobs: Sequence[Any], apply_ln: bool = True) -> float:
        values = SCSVBaselines._nll_values(token_logprobs)
        return float(np.mean(values)) if values.size else float("inf")

    @staticmethod
    def calculate_worst_k_nll(token_logprobs: Sequence[Any], k: int = 5) -> float:
        values = SCSVBaselines._nll_values(token_logprobs)
        if values.size == 0:
            return float("inf")
        k = max(1, min(int(k), values.size))
        return float(np.mean(np.partition(values, values.size - k)[-k:]))

    @staticmethod
    def calculate_self_consistency(parallel_trajectories: Sequence[str], final_answers: Optional[Sequence[str]] = None) -> float:
        keys = list(final_answers) if final_answers is not None else list(parallel_trajectories)
        if not keys:
            return 1.0
        counts = Counter(keys)
        return float(1.0 - max(counts.values()) / len(keys))

    @staticmethod
    def calculate_semantic_entropy(parallel_trajectories: Sequence[str], final_answers: Optional[Sequence[str]] = None) -> float:
        keys = list(final_answers) if final_answers is not None else list(parallel_trajectories)
        if len(keys) <= 1:
            return 0.0
        counts = np.asarray(list(Counter(keys).values()), dtype=np.float64)
        probs = counts / counts.sum()
        entropy = -float(np.sum(probs * np.log(probs)))
        return float(np.clip(entropy / math.log(len(keys)), 0.0, 1.0))

    @staticmethod
    def calculate_mbr(kernel: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Quality-weighted task-kernel MBR adaptation: one minus best expected agreement."""
        matrix = np.asarray(kernel, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return 1.0
        if weights is None:
            weights = np.ones(matrix.shape[0], dtype=np.float64) / matrix.shape[0]
        weights = np.asarray(weights, dtype=np.float64)
        if weights.size != matrix.shape[0] or weights.sum() <= 0:
            weights = np.ones(matrix.shape[0], dtype=np.float64) / matrix.shape[0]
        else:
            weights = weights / weights.sum()
        valid = np.any(matrix > 0, axis=1)
        scores = matrix @ weights
        return float(np.clip(1.0 - np.max(scores[valid]) if np.any(valid) else 1.0, 0.0, 1.0))

    @staticmethod
    def calculate_target_consistency(
        kernel: np.ndarray,
        weights: Optional[np.ndarray] = None,
        target_index: int = 0,
    ) -> float:
        """Leave-one-out expected dissimilarity of the evaluated response."""
        matrix = np.asarray(kernel, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] <= 1:
            return 0.0
        n = matrix.shape[0]
        if target_index < 0 or target_index >= n:
            raise IndexError("target_index is outside the kernel")
        mask = np.ones(n, dtype=bool)
        mask[target_index] = False
        if weights is None:
            evidence_weights = np.ones(int(mask.sum()), dtype=np.float64)
        else:
            values = np.asarray(weights, dtype=np.float64)
            if values.size != n:
                raise ValueError("weights must align with the kernel")
            evidence_weights = np.clip(values[mask], 0.0, None)
        total = float(evidence_weights.sum())
        if total <= 0.0:
            return 1.0
        evidence_weights /= total
        dissimilarity = 1.0 - np.clip(matrix[target_index, mask], 0.0, 1.0)
        return float(np.clip(dissimilarity @ evidence_weights, 0.0, 1.0))

    @staticmethod
    def calculate_kernel_language_entropy(kernel: np.ndarray) -> float:
        """VNE of a unit-trace task kernel, explicitly a KLE adaptation."""
        matrix = np.asarray(kernel, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] <= 1:
            return 0.0
        matrix = (matrix + matrix.T) / 2.0
        trace = float(np.trace(matrix))
        if trace <= 0.0:
            return 1.0
        density = matrix / trace
        eigenvalues = np.linalg.eigvalsh(density)
        if float(np.min(eigenvalues)) < -1e-8:
            raise ValueError("KLE adaptation requires a PSD kernel")
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        eigenvalues[eigenvalues < 1e-12] = 0.0
        eigenvalues /= float(eigenvalues.sum())
        entropy = -float(np.sum(eigenvalues[eigenvalues > 0] * np.log(eigenvalues[eigenvalues > 0])))
        return float(np.clip(entropy / math.log(matrix.shape[0]), 0.0, 1.0))

    @staticmethod
    def _cocoa_diversity(kernel: np.ndarray, target_index: int = 0) -> float:
        """Target-conditioned dissimilarity used by the CoCoA formula.

        The official CoCoA implementation obtains sentence similarity from a
        cross-encoder. The PWSC-CI cache has no cross-encoder outputs, so callers
        must pass an explicitly documented task kernel (the current runner
        uses its deterministic outcome/AST kernel).  Keeping this primitive in
        the baseline module makes the substitution auditable and prevents it
        from being mistaken for the official cross-encoder score.
        """
        return SCSVBaselines.calculate_target_consistency(kernel, target_index=target_index)

    @staticmethod
    def calculate_cocoa_msp(
        token_logprobs: Sequence[Any], kernel: np.ndarray, target_index: int = 0
    ) -> float:
        """CoCoA-MSP formula with a cache-only task-kernel adaptation."""
        values = SCSVBaselines._nll_values(token_logprobs)
        if values.size == 0:
            return 1.0
        return float(np.sum(values) * SCSVBaselines._cocoa_diversity(kernel, target_index))

    @staticmethod
    def calculate_cocoa_ppl(
        token_logprobs: Sequence[Any], kernel: np.ndarray, target_index: int = 0
    ) -> float:
        """CoCoA-PPL formula with a cache-only task-kernel adaptation."""
        values = SCSVBaselines._nll_values(token_logprobs)
        if values.size == 0:
            return 1.0
        return float(np.mean(values) * SCSVBaselines._cocoa_diversity(kernel, target_index))

    @staticmethod
    def calculate_self_certainty(token_logprobs: Sequence[Any], apply_ln: bool = True) -> float:
        # Kept as an API-compatible name; it is the mean chosen-token NLL.
        return SCSVBaselines.calculate_nll(token_logprobs, apply_ln=apply_ln)

    @staticmethod
    def calculate_deep_conf(token_logprobs: Sequence[Any], window_size: int = 5, apply_ln: bool = True) -> float:
        return SCSVBaselines.calculate_worst_k_nll(token_logprobs, k=window_size)

    @staticmethod
    def calculate_ll(token_logprobs: Sequence[Any]) -> float:
        values = SCSVBaselines._nll_values(token_logprobs)
        return float(np.mean(values)) if values.size else float("inf")

    @staticmethod
    def calculate_p_true(logprobs: Sequence[Any]) -> float:
        # Legacy callers receive a bounded confidence complement, but the
        # metric is excluded from the paper tables because it is not a
        # separately validated probability model.
        nll = SCSVBaselines.calculate_nll(logprobs)
        return float(np.clip(1.0 - math.exp(-nll), 0.0, 1.0)) if np.isfinite(nll) else 1.0

    @staticmethod
    def calculate_inside_eigen_score(hidden_states_list: Sequence[np.ndarray]) -> float:
        # Hidden-state methods are outside this PWSC-CI runner.
        return float("nan")

    @staticmethod
    def calculate_sar(token_logprobs: Sequence[Any], weights: Optional[Sequence[float]] = None) -> float:
        return float("nan")

    @staticmethod
    def calculate_u_ecc(parallel_trajectories: Sequence[str], sim_matrix: np.ndarray) -> float:
        matrix = np.asarray(sim_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            return 1.0
        distances = 1.0 - np.clip(matrix, -1.0, 1.0)
        return float(np.clip(np.max(np.mean(distances, axis=1)), 0.0, 1.0))

    @staticmethod
    def calculate_u_deg(parallel_trajectories: Sequence[str], sim_matrix: np.ndarray) -> float:
        matrix = np.asarray(sim_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            return 1.0
        agreement = np.mean(matrix, axis=1)
        return float(np.clip(1.0 - np.max(agreement), 0.0, 1.0))


__all__ = ["SCSVBaselines"]
