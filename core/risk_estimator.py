"""Canonical PWSC-CI uncertainty estimator.

The implementation follows ``PWSC_ALGORITHM_FLOW.md`` exactly: weakest-token
confidence masses, a task-structural Gram kernel, and normalized target-versus-
peer Holevo information.  It has no learned fusion, hidden-state features,
gold-label access, or alternative PWSC scoring branch.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.adapters import TaskAdapter, make_adapter
from core.behavioral_kernel import build_behavior_kernel


PWSC_CI_CONTRACT_VERSION = "pwsc-ci-normalized-holevo-v1"
_SPECIAL_TOKEN_RE = re.compile(r"^<[^>]+>$")


def _operator_entropy(weights: np.ndarray, gram: np.ndarray) -> float:
    """Von Neumann entropy computed from a weighted Gram representation."""
    sqrt_weights = np.sqrt(weights)
    density = np.outer(sqrt_weights, sqrt_weights) * gram
    density = (density + density.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(density)
    if float(np.min(eigenvalues, initial=0.0)) < -1e-8:
        raise ValueError("PWSC-CI requires a PSD kernel")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    eigenvalues[eigenvalues < 1e-12] = 0.0
    trace = float(eigenvalues.sum())
    if trace <= 0.0:
        return 0.0
    eigenvalues /= trace
    positive = eigenvalues[eigenvalues > 0.0]
    return -float(np.sum(positive * np.log(positive)))


def target_conditional_information_components(
    path_confidences: Sequence[float],
    kernel: np.ndarray,
    target_index: int = 0,
) -> Dict[str, float]:
    """Compute PWSC-CI and every scalar term needed to audit the score.

    Let the fixed target path have confidence mass ``q_t`` and the peers have
    total mass ``Q_-t``.  The target is a pure behavioral state, while the
    confidence-weighted peers form a density operator.  PWSC-CI is their
    Holevo information normalized by the entropy of the natural binary mass
    partition.  The score is bounded in ``[0, 1]`` and is a ranking score, not
    a calibrated probability.
    """
    values = np.asarray(path_confidences, dtype=np.float64)
    matrix = np.asarray(kernel, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("path_confidences must be a non-empty vector")
    if matrix.shape != (values.size, values.size):
        raise ValueError("kernel must be square and align with path_confidences")
    if target_index < 0 or target_index >= values.size:
        raise IndexError("target_index is outside the ensemble")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("path confidences must be finite and lie in [0, 1]")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("kernel must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError("PWSC-CI requires a symmetric kernel")

    active = values > 0.0
    if np.any(active):
        if not np.allclose(np.diag(matrix)[active], 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError("PWSC-CI requires unit-norm active kernel states")
        active_kernel = matrix[np.ix_(active, active)]
        if float(np.min(np.linalg.eigvalsh(active_kernel))) < -1e-8:
            raise ValueError("PWSC-CI requires a PSD kernel")

    target_confidence = float(values[target_index])
    if values.size == 1:
        score = float(1.0 - target_confidence)
        return {
            "normalized_information": score,
            "holevo_information": 0.0,
            "partition_entropy": 0.0,
            "target_mass_ratio": 1.0 if target_confidence > 0.0 else 0.0,
            "peer_mass_ratio": 0.0,
            "peer_entropy": 0.0,
            "mixture_entropy": 0.0,
            "evidence_deficit": score,
            "degenerate": 1.0,
        }

    peer_mask = np.ones(values.size, dtype=bool)
    peer_mask[target_index] = False
    peer_confidences = values[peer_mask]
    peer_mass = float(peer_confidences.sum())
    total_mass = target_confidence + peer_mass
    if target_confidence <= 0.0 or peer_mass <= 0.0 or total_mass <= 0.0:
        return {
            "normalized_information": 1.0,
            "holevo_information": 0.0,
            "partition_entropy": 0.0,
            "target_mass_ratio": target_confidence / total_mass if total_mass > 0.0 else 0.0,
            "peer_mass_ratio": peer_mass / total_mass if total_mass > 0.0 else 0.0,
            "peer_entropy": 0.0,
            "mixture_entropy": 0.0,
            "evidence_deficit": float(1.0 - np.mean(values)),
            "degenerate": 1.0,
        }

    target_mass_ratio = target_confidence / total_mass
    peer_mass_ratio = peer_mass / total_mass
    peer_weights = peer_confidences / peer_mass
    peer_kernel = matrix[np.ix_(peer_mask, peer_mask)]
    peer_entropy = _operator_entropy(peer_weights, peer_kernel)

    mixture_weights = values / total_mass
    mixture_entropy = _operator_entropy(mixture_weights, matrix)
    partition_entropy = -(
        target_mass_ratio * math.log(target_mass_ratio)
        + peer_mass_ratio * math.log(peer_mass_ratio)
    )
    holevo_information = max(0.0, mixture_entropy - peer_mass_ratio * peer_entropy)
    normalized_information = float(
        np.clip(holevo_information / partition_entropy, 0.0, 1.0)
    )
    return {
        "normalized_information": normalized_information,
        "holevo_information": float(holevo_information),
        "partition_entropy": float(partition_entropy),
        "target_mass_ratio": float(target_mass_ratio),
        "peer_mass_ratio": float(peer_mass_ratio),
        "peer_entropy": float(peer_entropy),
        "mixture_entropy": float(mixture_entropy),
        "evidence_deficit": float(1.0 - np.mean(values)),
        "degenerate": 0.0,
    }


def target_conditional_information_risk(
    path_confidences: Sequence[float],
    kernel: np.ndarray,
    target_index: int = 0,
) -> float:
    """Return the canonical PWSC-CI uncertainty score."""
    return target_conditional_information_components(
        path_confidences, kernel, target_index=target_index
    )["normalized_information"]


def retain_scoring_tokens(items: Sequence[Any]) -> List[Dict[str, Any]]:
    """Keep finite chosen-token logprobs, excluding control/whitespace tokens."""
    retained: List[Dict[str, Any]] = []
    for raw in items or []:
        if isinstance(raw, dict):
            token = str(raw.get("token", ""))
            value = raw.get("logprob")
            item = dict(raw)
        else:
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


class PWSCCIRiskEstimator:
    """Training-free implementation of the sole canonical PWSC-CI method."""

    def __init__(self, weakest_k: int = 5) -> None:
        if weakest_k < 1:
            raise ValueError("weakest_k must be positive")
        self.weakest_k = int(weakest_k)

    def path_action(self, logprobs: Sequence[Any]) -> PathAction:
        tokens = retain_scoring_tokens(logprobs)
        if not tokens:
            return PathAction(math.inf, 0.0, False, 0)
        nll = np.asarray([-float(item["logprob"]) for item in tokens], dtype=np.float64)
        if np.any(nll < -1e-6):
            return PathAction(math.inf, 0.0, False, len(nll))
        nll = np.maximum(nll, 0.0)
        k = min(self.weakest_k, len(nll))
        weakest = np.partition(nll, len(nll) - k)[-k:]
        action = float(np.mean(weakest))
        return PathAction(
            action=action,
            confidence=float(np.clip(math.exp(-action), 0.0, 1.0)),
            valid=True,
            token_count=len(nll),
        )

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
        """Score an ensemble with a fixed target at index zero."""
        n = len(logprobs_list)
        if n == 0 or len(texts) != n:
            raise ValueError("logprobs_list and texts must be non-empty and aligned")
        metadata = self._metadata_list(metadata_list, n)
        adapter: TaskAdapter = make_adapter(schema_name, metadata)
        actions = [self.path_action(path) for path in logprobs_list]
        action_values = np.asarray([item.action for item in actions], dtype=np.float64)
        path_confidences = np.asarray([item.confidence for item in actions], dtype=np.float64)
        logprob_valid = np.asarray([item.valid for item in actions], dtype=bool)
        domain_valid = adapter.valid_mask(texts)
        valid = logprob_valid & domain_valid
        q = path_confidences * valid.astype(np.float64)

        features = adapter.feature_matrix(texts, valid)
        task_kernel = np.asarray(features @ features.T, dtype=np.float64)
        task_kernel = (task_kernel + task_kernel.T) / 2.0
        if np.any(valid):
            if not np.allclose(np.diag(task_kernel)[valid], 1.0, atol=1e-8):
                raise ValueError("valid feature vectors must have unit norm")
            valid_kernel = task_kernel[np.ix_(valid, valid)]
            minimum = float(np.min(np.linalg.eigvalsh(valid_kernel)))
            if minimum < -1e-8:
                raise ValueError(f"task kernel is not PSD (min eigenvalue {minimum})")

        kernel = task_kernel
        kernel_info: Dict[str, Any] = {
            "kernel_source": "task_structural",
            "probe_count": 0,
            "active_probe_count": 0,
            "successful_calls": 0,
        }
        if adapter.family == "code" and n > 1:
            public = metadata[0] or {}
            behavior_kernel, kernel_info = build_behavior_kernel(
                texts,
                valid,
                str(public.get("prompt", "")),
                str(public.get("entry_point", "")),
            )
            if behavior_kernel is not None:
                kernel = np.asarray(behavior_kernel, dtype=np.float64)
        kernel = (kernel + kernel.T) / 2.0
        if np.any(valid):
            if not np.allclose(np.diag(kernel)[valid], 1.0, atol=1e-8):
                raise ValueError("PWSC-CI kernel must have unit active diagonal")
            minimum = float(np.min(np.linalg.eigvalsh(kernel[np.ix_(valid, valid)])))
            if minimum < -1e-8:
                raise ValueError(f"PWSC-CI kernel is not PSD (min eigenvalue {minimum})")

        components = target_conditional_information_components(q, kernel, target_index=0)
        risk = float(components["normalized_information"])
        mass = float(q.sum())
        weights = q / mass if mass > 0.0 else np.zeros(n, dtype=np.float64)
        centrality = kernel @ weights
        candidate_scores = q * centrality
        selected_index = int(np.argmax(candidate_scores))
        outcomes = adapter.semantic_class_keys(texts, valid)
        raw_outcomes = adapter.outcome_keys(texts)
        diagnostics: Dict[str, Any] = {
            "method": "PWSC-CI-v1.0",
            "contract_version": PWSC_CI_CONTRACT_VERSION,
            "weakest_k": self.weakest_k,
            "path_actions": action_values,
            "path_confidences": path_confidences,
            "valid_mask": valid,
            "q": q,
            "valid_count": int(np.sum(valid)),
            "domain_family": adapter.family,
            "outcomes": outcomes,
            "raw_outcomes": raw_outcomes,
            "task_kernel": task_kernel,
            "kernel": kernel,
            "kernel_source": kernel_info["kernel_source"],
            "kernel_diagnostics": kernel_info,
            "weights": weights,
            "centrality": centrality,
            "candidate_scores": candidate_scores,
            "selected_index": selected_index,
            "selected_outcome": outcomes[selected_index],
            "selected_raw_outcome": raw_outcomes[selected_index],
            "ci_components": components,
            "risk": risk,
        }
        return risk, diagnostics


__all__ = [
    "PWSC_CI_CONTRACT_VERSION",
    "PWSCCIRiskEstimator",
    "PathAction",
    "retain_scoring_tokens",
    "target_conditional_information_components",
    "target_conditional_information_risk",
]
