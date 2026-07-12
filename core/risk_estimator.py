"""Prior-Weighted Spectral Consensus (PWSC) risk estimator.

This module implements the equations in ``PWSC_ALGORITHM_FLOW.md`` and the
compact contract in ``.pipeline/docs/algorithm_spec_v1.md``,
without hidden-state features, random projections, gold-label access, or
post-hoc test-set calibration.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.adapters import TaskAdapter, make_adapter


_SPECIAL_TOKEN_RE = re.compile(r"^<[^>]+>$")


def retain_scoring_tokens(items: Sequence[Any]) -> List[Dict[str, Any]]:
    """Keep finite chosen-token logprobs, excluding control/whitespace tokens."""
    retained: List[Dict[str, Any]] = []
    for raw in items or []:
        if isinstance(raw, dict):
            token = str(raw.get("token", ""))
            value = raw.get("logprob")
            item = dict(raw)
        else:
            # Numeric-only inputs are accepted by the backward-compatible API
            # and receive a synthetic non-whitespace token name.
            token = "__value__"
            value = raw
            item = {"logprob": value}
        try:
            logprob = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(logprob):
            continue
        if not token or token.isspace() or _SPECIAL_TOKEN_RE.match(token):
            continue
        item["token"] = token
        item["logprob"] = logprob
        retained.append(item)
    return retained


@dataclass(frozen=True)
class PathAction:
    action: float
    confidence: float
    valid: bool
    token_count: int


class FisherRaoCovariantPathAction:
    """Compatibility wrapper for the retired path-action class.

    The old implementation exposed a random trajectory embedding.  This
    wrapper exposes only the deterministic PWSC action and a two-value summary
    for callers that still import the legacy symbol; PWSC itself never uses
    the summary as a feature map.
    """

    def __init__(self, logprobs: Sequence[Any], weakest_k: int = 5) -> None:
        self.logprobs = logprobs
        self.weakest_k = weakest_k

    def compute_classical_action(self) -> float:
        return RiskEstimator(weakest_k=self.weakest_k).path_action(self.logprobs).action

    def compute_trajectory_embedding(self) -> np.ndarray:
        action = self.compute_classical_action()
        count = len(retain_scoring_tokens(self.logprobs))
        return np.asarray([0.0 if not np.isfinite(action) else action, float(count)], dtype=np.float64)


class RiskEstimator:
    """Training-free PWSC scorer.

    ``weakest_k`` is selected on development data and defaults to the frozen
    value specified by the project protocol.
    """

    def __init__(
        self,
        verifier_mgr: Optional[Any] = None,
        generator: Optional[Any] = None,
        weakest_k: int = 5,
    ) -> None:
        if weakest_k < 1:
            raise ValueError("weakest_k must be positive")
        self.verifier_mgr = verifier_mgr
        self.generator = generator
        self.weakest_k = int(weakest_k)

    def path_action(self, logprobs: Sequence[Any]) -> PathAction:
        tokens = retain_scoring_tokens(logprobs)
        if not tokens:
            return PathAction(math.inf, 0.0, False, 0)
        nll = np.asarray([-float(item["logprob"]) for item in tokens], dtype=np.float64)
        if np.any(nll < -1e-6):
            return PathAction(math.inf, 0.0, False, len(nll))
        # API log probabilities should be <= 0.  Clamping prevents an invalid
        # provider response from producing q > 1 and violating the score range.
        nll = np.maximum(nll, 0.0)
        k = min(self.weakest_k, len(nll))
        weakest = np.partition(nll, len(nll) - k)[-k:]
        action = float(np.mean(weakest))
        confidence = float(np.clip(math.exp(-action), 0.0, 1.0))
        return PathAction(action, confidence, True, len(nll))

    @staticmethod
    def _metadata_list(
        metadata_list: Optional[Sequence[Optional[Dict[str, Any]]]], n: int
    ) -> List[Optional[Dict[str, Any]]]:
        values = list(metadata_list or [])
        if len(values) < n:
            values.extend([None] * (n - len(values)))
        return values[:n]

    def score(
        self,
        logprobs_list: Sequence[Sequence[Any]],
        texts: Sequence[str],
        schema_name: str = "GSM8K",
        metadata_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Score one or more sampled paths and return risk plus diagnostics."""
        n = len(logprobs_list)
        if n == 0 or len(texts) != n:
            raise ValueError("logprobs_list and texts must be non-empty and aligned")
        metadata = self._metadata_list(metadata_list, n)
        adapter: TaskAdapter = make_adapter(schema_name, metadata)
        actions = [self.path_action(path) for path in logprobs_list]
        action_values = np.asarray([item.action for item in actions], dtype=np.float64)
        confidence = np.asarray([item.confidence for item in actions], dtype=np.float64)
        logprob_valid = np.asarray([item.valid for item in actions], dtype=bool)
        domain_valid = adapter.valid_mask(texts)
        valid = logprob_valid & domain_valid
        q = confidence * valid.astype(np.float64)

        # A missing logprob stream or an invalid response is a zero-confidence
        # path, never a fabricated random observation.
        if not np.any(q > 0.0):
            diagnostics = self._base_diagnostics(
                action_values, confidence, valid, q, adapter, texts
            )
            diagnostics.update(
                {
                    "kernel": np.zeros((n, n), dtype=np.float64),
                    "rho": np.zeros((n, n), dtype=np.float64),
                    "eigenvalues": np.zeros(n, dtype=np.float64),
                    "spectral_entropy": 0.0,
                    "normalized_spectral_uncertainty": 0.0,
                    "mean_path_confidence": 0.0,
                    "selected_index": 0,
                    "selected_outcome": adapter.outcome_keys(texts)[0],
                    "risk": 1.0,
                }
            )
            return 1.0, diagnostics

        # Stable log-sum-exp normalization of q_i = exp(-s_i) for valid paths.
        log_q = np.full(n, -np.inf, dtype=np.float64)
        for idx, item in enumerate(actions):
            if valid[idx]:
                log_q[idx] = -item.action
        shift = float(np.max(log_q[np.isfinite(log_q)]))
        weights = np.zeros(n, dtype=np.float64)
        weights[valid] = np.exp(log_q[valid] - shift)
        weights /= float(np.sum(weights))

        features = adapter.feature_matrix(texts, valid)
        kernel = np.asarray(features @ features.T, dtype=np.float64)
        # Numerical cleanup is explicit; the feature map itself guarantees PSD.
        kernel = (kernel + kernel.T) / 2.0
        if np.any(valid):
            if not np.allclose(np.diag(kernel)[valid], 1.0, atol=1e-8):
                raise ValueError("valid feature vectors must have unit norm")
            valid_kernel = kernel[np.ix_(valid, valid)]
            min_kernel_eig = float(np.min(np.linalg.eigvalsh(valid_kernel)))
            if min_kernel_eig < -1e-8:
                raise ValueError(f"feature kernel is not PSD (min eigenvalue {min_kernel_eig})")

        sqrt_w = np.sqrt(weights)
        rho = np.outer(sqrt_w, sqrt_w) * kernel
        rho = (rho + rho.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(rho)
        if float(np.min(eigenvalues)) < -1e-8:
            raise ValueError("density matrix is not PSD")
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        # Treat machine-level null eigenvalues as exact zeros.  Otherwise a
        # rank-one ensemble can acquire a spurious 1e-15 entropy signal that
        # later statistical tests may mistake for evidence.
        eigenvalues[eigenvalues < 1e-12] = 0.0
        trace = float(np.sum(eigenvalues))
        if trace <= 0.0:
            raise ValueError("density matrix has zero trace")
        eigenvalues /= trace
        valid_count = int(np.sum(valid))
        spectral_entropy = float(
            -np.sum(eigenvalues[eigenvalues > 0] * np.log(eigenvalues[eigenvalues > 0]))
        )
        uncertainty = 0.0 if valid_count <= 1 else spectral_entropy / math.log(valid_count)
        uncertainty = float(max(0.0, min(1.0, uncertainty)))
        mean_confidence = float(np.mean(q))
        risk = float(np.clip(1.0 - (1.0 - uncertainty) * mean_confidence, 0.0, 1.0))

        centrality = kernel @ weights
        candidate_scores = q * centrality
        selected_index = int(np.argmax(candidate_scores))
        outcomes = adapter.outcome_keys(texts)
        diagnostics = self._base_diagnostics(action_values, confidence, valid, q, adapter, texts)
        diagnostics.update(
            {
                "kernel": kernel,
                "rho": rho,
                "eigenvalues": eigenvalues,
                "spectral_entropy": spectral_entropy,
                "normalized_spectral_uncertainty": uncertainty,
                "mean_path_confidence": mean_confidence,
                "weights": weights,
                "centrality": centrality,
                "candidate_scores": candidate_scores,
                "selected_index": selected_index,
                "selected_outcome": outcomes[selected_index],
                "risk": risk,
            }
        )
        return risk, diagnostics

    def _base_diagnostics(
        self,
        actions: np.ndarray,
        confidence: np.ndarray,
        valid: np.ndarray,
        q: np.ndarray,
        adapter: TaskAdapter,
        texts: Sequence[str],
    ) -> Dict[str, Any]:
        return {
            "method": "PWSC-v1.0",
            "weakest_k": self.weakest_k,
            "path_actions": actions,
            "path_confidences": confidence,
            "valid_mask": valid,
            "q": q,
            "valid_count": int(np.sum(valid)),
            "domain_family": adapter.family,
            "outcomes": adapter.outcome_keys(texts),
        }

    # Backward-compatible single-path API used by older callers.
    def estimate_structural_risk(
        self,
        G_sequence: Any = None,
        schema_name: str = "GSM8K",
        trajectory_text: str = "",
        gold_answer: str = "",
        entry_point: str = "",
        logprobs: Optional[Sequence[Any]] = None,
        attentions: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, List[float], Dict[str, Any]]:
        metadata = [{"entry_point": entry_point}]
        risk, diagnostics = self.score([list(logprobs or [])], [trajectory_text], schema_name, metadata)
        curve_length = max(1, len(retain_scoring_tokens(logprobs or [])))
        return risk, 0.0, [risk] * curve_length, diagnostics

    # Backward-compatible ensemble API used by older callers.
    def estimate_ensemble_risk(
        self,
        logprobs_list: List[List[Dict[str, Any]]],
        answers_list: List[str],
        schema_name: str = "GSM8K",
        metadata_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> Tuple[float, Dict[str, Any]]:
        return self.score(logprobs_list, answers_list, schema_name, metadata_list)


__all__ = ["FisherRaoCovariantPathAction", "PathAction", "RiskEstimator", "retain_scoring_tokens"]
