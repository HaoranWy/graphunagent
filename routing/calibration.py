# -*- coding: utf-8 -*-
"""ISE-Flow 非参数分桶校准与无监督分位数校正模块.

学术级升级版：
1. 包含标准的 `SupervisedHybridCalibrator` IPW 分类加权。
2. 新增 `UnsupervisedConsensusAligner`，使用弱监督协方差共识（Covariance Consensus）自适应纠偏特征极性。
"""

import bisect
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

logger = logging.getLogger("ise_flow.routing.calibration")


class UnsupervisedConsensusAligner:
    """基于通用异常锚点协方差共识的自适应极性对齐器."""

    @classmethod
    def align_polarities(
        cls, 
        features: List[List[float]], 
        anchor_index: int = 2
    ) -> Tuple[List[List[float]], List[int]]:
        """在无监督测试集上，利用协方差矩阵自适应辨识并对齐各特征维度的极性。

        Args:
            features: 原始特征矩阵 $X \in \mathbb{R}^{N \times d}$。
                      默认列序为: [r_struct, r_robust, r_parse, log_v]。
            anchor_index: 通用异常锚点 $A$ 的列索引。默认为 2 (r_parse 解析风险)。

        Returns:
            Tuple: (对齐极性后的特征矩阵, 各维度的极性系数 [-1, 1])
        """
        X = np.array(features, dtype=np.float64)
        n_samples, n_features = X.shape
        if n_samples < 2:
            return features, [1] * n_features

        # 提取通用异常锚点 A (解析崩溃风险在任意领域均无条件正向对应真实错误率)
        A = X[:, anchor_index]
        A_mean = np.mean(A)

        polarities = []
        X_aligned = X.copy()

        for j in range(n_features):
            if j == anchor_index:
                polarities.append(1)
                continue

            # 计算特征 X_j 与锚点 A 的经验协方差
            mean_j = np.mean(X[:, j])
            covariance = np.sum((X[:, j] - mean_j) * (A - A_mean)) / (n_samples - 1)

            # 【核心辨识】如果协方差为负，证明该弱指标与异常负相关，需要反转极性
            sign = 1 if covariance >= 0.0 else -1
            polarities.append(sign)

            # 执行极性修正对齐
            X_aligned[:, j] = sign * X[:, j]

        logger.info(
            f"Unsupervised Covariance Polarity Consensus Complete. "
            f"Estimated Dimension Signs (against Anchor R_parse): {polarities}"
        )
        return X_aligned.tolist(), polarities


class NativeGaussianKDE:
    """纯 NumPy 实现的高效一维高斯核密度估计器 (KDE)."""

    def __init__(self, data: np.ndarray, bandwidth: float = 0.4) -> None:
        self.data = np.array(data, dtype=np.float64).flatten()
        self.bandwidth = bandwidth
        self.n = len(self.data)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """评估给定点集处的概率密度."""
        points = np.array(points, dtype=np.float64).flatten()
        # 矢量化并行高斯核矩阵运算
        diff = (points[:, None] - self.data[None, :]) / self.bandwidth
        kernels = np.exp(-0.5 * (diff ** 2)) / np.sqrt(2.0 * np.pi)
        return np.sum(kernels, axis=1) / (self.n * self.bandwidth + 1e-12)


class RobustWeaklySupervisedCalibrator:
    """自抗扰弱监督概率标定器 (RWSC) [PDF Page 8].

    融合 IPW 分类加权、高斯核密度估计（KDE）自适应混淆估计，
    并应用基于闭式解析梯度的自抗扰逻辑回归。
    """

    def __init__(self, l2_penalty: float = 0.1, lr: float = 0.05, epochs: int = 300) -> None:
        self.l2 = l2_penalty
        self.lr = lr
        self.epochs = epochs
        
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted = False

        self.prior_base_rate: float = 0.0
        self.theta_01: float = 0.005  # 假阳性率噪声上限
        self.theta_10: float = 0.020  # 假阴性率噪声下限
        self.training_history: List[float] = []

    def _estimate_noise_with_kde(self, X_scaled: np.ndarray, y_arr: np.ndarray) -> None:
        """利用无监督一维高斯核密度估计（KDE）自适应确定混淆分布极值点."""
        # 选取主轴特征（如第一维度 R_struct）作为 KDE 的估计锚点
        feature_axis = X_scaled[:, 0]
        
        wrong_samples = feature_axis[y_arr == 1.0]
        correct_samples = feature_axis[y_arr == 0.0]

        if len(wrong_samples) < 5 or len(correct_samples) < 5:
            # 样本过少时，采用基于 MCI 审计的稳健先验默认值
            self.theta_01 = 0.01
            self.theta_10 = 0.04
            return

        kde_wrong = NativeGaussianKDE(wrong_samples, bandwidth=0.3)
        kde_correct = NativeGaussianKDE(correct_samples, bandwidth=0.3)

        # 测算两个分布交叉的极值点
        eval_points = np.linspace(-3.0, 3.0, 100)
        dens_wrong = kde_wrong.evaluate(eval_points)
        dens_correct = kde_correct.evaluate(eval_points)

        # 通过极端特征区间（Anchor regions）的密度重叠估算标签噪声 [Ratner et al.]
        overlap = np.minimum(dens_wrong, dens_correct) / (np.maximum(dens_wrong, dens_correct) + 1e-12)
        
        # 假阳性率对应高分值（高异常）区间的误报率
        self.theta_01 = float(np.clip(np.mean(overlap[eval_points > 1.5]), 0.005, 0.05))
        # 假阴性率对应低分值（低异常）区间的漏报率（由 Latex 错配导致）
        self.theta_10 = float(np.clip(np.mean(overlap[eval_points < -1.5]), 0.010, 0.15))

        logger.info(
            f"Gaussian KDE Label Noise Transition Estimated. "
            f"FPR (theta_01): {self.theta_01:.4f} | FNR (theta_10): {self.theta_10:.4f}"
        )

    def fit(self, X: List[List[float]], y: List[bool]) -> None:
        """在含有标签噪声的开发集上运行带闭式解析梯度的自抗扰参数最优化."""
        if not X or not y or len(X) != len(y):
            raise ValueError("Input feature matrix X and labels y must be non-empty.")

        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape

        self.prior_base_rate = float(np.mean(y_arr))

        # 1. 特征标准化
        self.mean = np.mean(X_arr, axis=0)
        self.std = np.std(X_arr, axis=0) + 1e-6
        X_scaled = (X_arr - self.mean) / self.std

        # 2. 调用高斯 KDE 在线评估判定器混淆概率矩阵 T
        self._estimate_noise_with_kde(X_scaled, y_arr)

        # 3. 类别均衡加权 (IPW)
        n_wrong = max(1.0, np.sum(y_arr))
        n_correct = max(1.0, n_samples - n_wrong)
        w_wrong = n_samples / (2.0 * n_wrong)
        w_correct = n_samples / (2.0 * n_correct)
        sample_weights = np.where(y_arr == 1.0, w_wrong, w_correct)

        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.training_history = []

        # 4. 梯度下降学习 (Robust Loss-corrected Logistic Regression)
        # 优化目标：最小化带噪声信道 T 映射后的交叉熵
        for epoch in range(self.epochs):
            # 4.1 估计真实潜在概率 p
            linear_pred = np.dot(X_scaled, self.weights) + self.bias
            p = 1.0 / (1.0 + np.exp(-np.clip(linear_pred, -20.0, 20.0)))

            # 4.2 通过噪声信道 T 映射出观测概率 p_tilde
            # p_tilde = (1 - theta_01 - theta_10) * p + theta_01
            p_tilde = (1.0 - self.theta_01 - self.theta_10) * p + self.theta_01

            # 4.3 计算闭式解析梯度 (Analytical Gradients for Robust Loss)
            # dp_tilde_dp = 1 - theta_01 - theta_10
            # dLoss_dp_tilde = (p_tilde - y) / (p_tilde * (1 - p_tilde))
            # dLoss_dz = dLoss_dp_tilde * dp_tilde_dp * p * (1 - p)
            dp_tilde_dp = 1.0 - self.theta_01 - self.theta_10
            error_vector = (p_tilde - y_arr) / (p_tilde * (1.0 - p_tilde) + 1e-12)
            grad_z = error_vector * dp_tilde_dp * p * (1.0 - p) * sample_weights

            # 4.4 参数更新
            dw = (1.0 / n_samples) * np.dot(X_scaled.T, grad_z) + self.l2 * self.weights
            db = (1.0 / n_samples) * np.sum(grad_z)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # 计算鲁棒修正损失
            loss = -np.mean(sample_weights * (y_arr * np.log(p_tilde + 1e-12) + (1.0 - y_arr) * np.log(1.0 - p_tilde + 1e-12)))
            self.training_history.append(float(loss))

        self.is_fitted = True
        logger.info(
            f"Robust weakly-supervised calibrator (RWSC) fitted. "
            f"Weights: {self.weights.tolist()} | Bias: {self.bias:.4f}"
        )

    def calibrate(self, x: List[float]) -> float:
        if not self.is_fitted or self.weights is None or self.mean is None or self.std is None:
            raise RuntimeError("Calibrator has not been fitted yet.")

        x_arr = np.array(x, dtype=np.float64)
        x_scaled = (x_arr - self.mean) / self.std

        linear_val = np.dot(x_scaled, self.weights) + self.bias
        prob = 1.0 / (1.0 + np.exp(-np.clip(linear_val, -20.0, 20.0)))
        
        return float(np.clip(prob, 0.01, 0.99))


class BinningCalibrator:
    """非参数等频分桶概率校准器."""

    def __init__(
        self,
        num_bins: int = 10,
        alpha: float = 1.0,
        beta: float = 1.0,
        p_min: float = 0.01,
        p_max: float = 0.99
    ) -> None:
        self.num_bins = num_bins
        self.alpha = alpha
        self.beta = beta
        self.p_min = p_min
        self.p_max = p_max

        self.boundaries: np.ndarray = np.array([], dtype=np.float64)
        self.bin_probs: np.ndarray = np.array([], dtype=np.float64)
        self.dev_scores_sorted: List[float] = []
        self.is_fitted = False
        self.is_rank_fitted = False

    def fit(self, r_ise_scores: List[float], labels: List[bool], adaptive: bool = True) -> None:
        if not r_ise_scores or not labels or len(r_ise_scores) != len(labels):
            raise ValueError("Input scores and validation labels must be non-empty.")

        n_samples = len(r_ise_scores)
        scores_arr = np.array(r_ise_scores, dtype=np.float64)
        labels_arr = np.array(labels, dtype=np.bool_)

        B = self.num_bins
        if adaptive and n_samples < B * 20:
            B = max(3, n_samples // 20)

        quantiles = np.linspace(0, 100, B + 1)
        raw_boundaries = np.percentile(scores_arr, quantiles)
        
        self.boundaries = np.unique(raw_boundaries)
        actual_B = len(self.boundaries) - 1

        if actual_B < 1:
            self.boundaries = np.array([scores_arr.min() - 1e-5, scores_arr.max() + 1e-5], dtype=np.float64)
            actual_B = 1

        bin_probs_list = []

        for i in range(actual_B):
            lower = self.boundaries[i]
            upper = self.boundaries[i + 1]

            if i == actual_B - 1:
                in_bin = (scores_arr >= lower) & (scores_arr <= upper)
            else:
                in_bin = (scores_arr >= lower) & (scores_arr < upper)

            bin_labels = labels_arr[in_bin]
            total_in_bin = len(bin_labels)
            wrongs_in_bin = np.sum(bin_labels)

            smoothed_prob = (wrongs_in_bin + self.alpha) / (total_in_bin + self.alpha + self.beta)
            bin_probs_list.append(smoothed_prob)

        self.bin_probs = np.array(bin_probs_list, dtype=np.float64)
        self.is_fitted = True
        
        self.fit_rank_fallback(r_ise_scores)

    def calibrate(self, score: float, fallback_to_rank: bool = False) -> float:
        if not self.is_fitted:
            if fallback_to_rank and self.is_rank_fitted:
                return self.calibrate_rank(score)
            else:
                raise RuntimeError("Calibrator has not been fitted yet.")

        bin_idx = np.digitize(score, self.boundaries) - 1
        bin_idx = max(0, min(bin_idx, len(self.bin_probs) - 1))
        
        raw_prob = self.bin_probs[bin_idx]
        return float(np.clip(raw_prob, self.p_min, self.p_max))

    def fit_rank_fallback(self, r_ise_scores: List[float]) -> None:
        if not r_ise_scores:
            raise ValueError("Empty scores provided for rank calibration fitting.")
        self.dev_scores_sorted = sorted(r_ise_scores)
        self.is_rank_fitted = True

    def calibrate_rank(self, score: float) -> float:
        if not self.is_rank_fitted:
            raise RuntimeError("Rank Calibrator has not been initialized.")

        pos = bisect.bisect_right(self.dev_scores_sorted, score)
        raw_percentile = pos / (len(self.dev_scores_sorted) + 1e-12)
        
        return float(np.clip(raw_percentile, self.p_min, self.p_max))