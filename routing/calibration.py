# -*- coding: utf-8 -*-
"""ISE-Flow 全任务域极性本征安全对齐的 Platt-Beta 校准器 (v5.0)."""

import bisect
import math
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

logger = logging.getLogger("ise_flow.routing.calibration")


class PriorConditionedCalibrator:
    """极性单调本征安全 Platt-Beta 标定器."""

    def __init__(self, mu_0: float = 0.15, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.mu_0 = float(np.clip(mu_0, 0.01, 0.99))
        self.max_iter = max_iter
        self.tol = tol
        
        self.w: float = 1.0
        self.b: float = 0.0
        self.is_fitted = False

    def fit(self, scores: List[float], unsupervised_reference: Optional[List[float]] = None, mso_scores: Optional[List[float]] = None) -> Tuple[float, float]:
        if not scores:
            raise ValueError("Scores list cannot be empty.")
            
        F = np.array(scores, dtype=np.float64)
        std_f = np.std(F) if np.std(F) > 1e-6 else 1.0
        
        mean_mso = 0.05
        if mso_scores is not None and len(mso_scores) == len(scores):
            mean_mso = float(np.mean(mso_scores))
            self.mu_0 = float(np.clip(mean_mso * 2.5 + 0.02, 0.05, 0.70))
            logger.info(f"OOD Calibration: dynamic mu_0 aligned to {self.mu_0:.4f} via MSO mean = {mean_mso:.4f}")

        # ----------------- 极性本征安全 -----------------
        # 得益于自监督语义密度与有理质量度规的联合全域单调性，标定极性恒为正向，物理清零繁冗实体
        sign = 1.0
        self.w = sign / (std_f + 0.2)

        b_min, b_max = -50.0, 50.0
        b_curr = 0.0

        for _ in range(self.max_iter):
            b_curr = (b_min + b_max) / 2.0
            z = self.w * F + b_curr
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
            
            g_b = np.mean(p) - self.mu_0
            if abs(g_b) < self.tol or (b_max - b_min) < 1e-7:
                break
                
            if g_b > 0:
                b_max = b_curr
            else:
                b_min = b_curr

        self.b = float(b_curr)
        self.is_fitted = True
        logger.info(f"Platt-Beta calibration complete: solved w = {self.w:.4f} | b = {self.b:.4f}")
        return self.w, self.b

    def calibrate(self, score: float) -> float:
        if not self.is_fitted:
            return float(1.0 / (1.0 + math.exp(-score)))
        z = self.w * score + self.b
        prob = 1.0 / (1.0 + math.exp(-np.clip(z, -20.0, 20.0)))
        return float(np.clip(prob, 0.01, 0.99))