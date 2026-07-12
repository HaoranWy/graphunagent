"""Development-only monotone Platt calibration for PWSC risk scores."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


class DevPlattCalibrator:
    """Fit ``P(error|score)`` using labeled development data only."""

    def __init__(self, l2: float = 1e-4, max_iter: int = 500) -> None:
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.center = 0.0
        self.scale = 1.0
        self.weight = 1.0
        self.bias = 0.0
        self.is_fitted = False

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        split_name: str = "dev",
    ) -> Tuple[float, float]:
        split = split_name.strip().lower()
        if "test" in split or not any(name in split for name in ("dev", "val", "calib")):
            raise ValueError("Calibration may only be fitted on a declared dev/validation split")
        x = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 4:
            raise ValueError("scores and labels must be aligned one-dimensional arrays with at least four samples")
        if not np.all(np.isfinite(x)) or not set(np.unique(y)).issubset({0.0, 1.0}) or np.unique(y).size < 2:
            raise ValueError("Calibration requires finite scores and both binary error labels")
        self.center = float(np.mean(x))
        self.scale = float(np.std(x)) or 1.0
        z = (x - self.center) / self.scale

        from scipy.optimize import minimize

        prevalence = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
        initial = np.asarray([0.0, math.log(prevalence / (1.0 - prevalence))], dtype=np.float64)

        def objective(params: np.ndarray) -> Tuple[float, np.ndarray]:
            # exp(theta) makes the mapping monotonically increasing in risk.
            theta, bias = float(params[0]), float(params[1])
            weight = math.exp(float(np.clip(theta, -10.0, 10.0)))
            logits = np.clip(weight * z + bias, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            loss = -float(np.mean(y * np.log(probs + 1e-12) + (1.0 - y) * np.log(1.0 - probs + 1e-12)))
            loss += 0.5 * self.l2 * theta * theta
            residual = probs - y
            gradient = np.asarray([
                float(np.mean(residual * z) * weight + self.l2 * theta),
                float(np.mean(residual)),
            ])
            return loss, gradient

        result = minimize(
            lambda params: objective(params)[0],
            initial,
            jac=lambda params: objective(params)[1],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        if not result.success:
            raise RuntimeError(f"Platt calibration failed: {result.message}")
        self.weight = math.exp(float(np.clip(result.x[0], -10.0, 10.0)))
        self.bias = float(result.x[1])
        self.is_fitted = True
        return self.weight, self.bias

    def calibrate(self, score: float) -> float:
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted on labeled development data first")
        normalized = (float(score) - self.center) / self.scale
        logit = float(np.clip(self.weight * normalized + self.bias, -30.0, 30.0))
        return float(1.0 / (1.0 + math.exp(-logit)))

    def calibrate_many(self, scores: Sequence[float]) -> List[float]:
        return [self.calibrate(score) for score in scores]


# Backward-compatible import name.  Its semantics are now explicitly dev-only.
PriorConditionedCalibrator = DevPlattCalibrator


__all__ = ["DevPlattCalibrator", "PriorConditionedCalibrator"]
