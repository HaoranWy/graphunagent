# -*- coding: utf-8 -*-
"""ISE-Flow 商空间对齐与本征共形路径作用量 (CSCF-EPA v2.0) 引擎.

重构说明：
1. 兼顾 Single-Path 与 Multi-Path 两种调用模式。
2. 实现商空间投影与同伦路径对齐：仅对落入最大语义等价丛 C_maj 的路径计算几何积分，
   并以系综共识比 (CCD) 进行共形非线性耦合，消融决策层面的错配。
3. 随动回传 MSG, MSO, SBP, EAD_in, EAD_all 五大假设证明指标。
"""

import math
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np

logger = logging.getLogger("ise_flow.core.risk_estimator")


class IntrinsicConformalObserver:
    """无参子词形态学投影器 (ICM).
    
    不引入任何外部随机投影矩阵与超参数，通过对 Token 字符进行无参形态学分解，
    在词表空间中建立本征共形球面的黎曼测地距离。
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.d_arr = np.arange(1, dim + 1, dtype=np.float64)

    def _to_char_ngrams(self, word: str, n_min: int = 2, n_max: int = 4) -> List[str]:
        if not word:
            return []
        ngrams = []
        word_padded = f"<{word}>"
        for n in range(n_min, n_max + 1):
            for i in range(len(word_padded) - n + 1):
                ngrams.append(word_padded[i:i+n])
        return ngrams

    def get_embeddings(self, tokens: List[str]) -> np.ndarray:
        T = len(tokens)
        if T == 0:
            return np.zeros((0, self.dim))
        
        embeds = []
        for tok in tokens:
            ngrams = self._to_char_ngrams(tok)
            if not ngrams:
                h = hash(tok)
                tok_vector = np.sin(h * self.d_arr)
            else:
                hashes = np.array([hash(ngram) for ngram in ngrams], dtype=np.float64)
                tok_vector = np.sum(np.sin(np.outer(hashes, self.d_arr)), axis=0)
            embeds.append(tok_vector)

        embeds_np = np.array(embeds, dtype=np.float64)
        norms = np.linalg.norm(embeds_np, axis=1, keepdims=True) + 1e-8
        return embeds_np / norms


class FisherRaoCovariantPathAction:
    """局域扩散谱阻碍与本征雅可比极化路径作用量变分计算器."""

    def __init__(self, logprobs: List[Dict[str, Any]], observer: IntrinsicConformalObserver, floor_logprob: float = -20.0, eta: float = 1.0) -> None:
        self.logprobs = logprobs
        self.T = len(logprobs)
        self.floor = floor_logprob
        self.eta = eta
        self.observer = observer  # 共享全局观测器

    def _compute_local_spectral_dynamics(self, words: List[str], probs_t: np.ndarray) -> Tuple[float, float, float]:
        """【假设 2 证明核】利用 Zelnik-Manor 局部自适应度规闭式求解局域谱隙与谱阻碍."""
        K = len(words)
        shannon_h = -np.sum(probs_t * np.log(probs_t + 1e-12))
        
        if K <= 2:
            sbp = 1.0 / (shannon_h + 1e-12)
            return 1.0, 0.0, float(sbp)

        embeds = self.observer.get_embeddings(words)
        S = np.clip(np.dot(embeds, embeds.T), -1.0, 1.0)
        D = np.sqrt(np.clip(2.0 * (1.0 - S), 0.0, None))  # 本征共形球面距离

        M = min(3, K - 1)
        sigmas = np.zeros(K)
        for i in range(K):
            sorted_dists = np.sort(D[i])
            sigmas[i] = sorted_dists[M] if sorted_dists[M] > 1e-6 else 1e-3

        W = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                denom = sigmas[i] * sigmas[j]
                W[i, j] = np.exp(- (D[i, j] ** 2) / denom)

        P_diff = np.zeros((K, K))
        for i in range(K):
            row_weights = W[i] * probs_t
            sum_row = np.sum(row_weights) + 1e-12
            P_diff[i] = row_weights / sum_row

        try:
            eigvals = np.real(np.linalg.eigvals(P_diff))
            eigvals = np.sort(eigvals)
            lambda_2 = np.clip(eigvals[-2] if len(eigvals) >= 2 else 0.0, 0.0, 1.0)
            
            spectral_gap = 1.0 - lambda_2
            spectral_obstruction = lambda_2
        except Exception:
            spectral_gap = 1.0
            spectral_obstruction = 0.0

        sbp = spectral_gap / (shannon_h + 1e-12)
        return float(spectral_gap), float(spectral_obstruction), float(sbp)

    def compute_local_kinetic_energies(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """解析求解每一自回归生成步的多维几何动力学指标."""
        if self.T <= 1:
            return np.zeros(self.T), np.zeros(self.T), np.zeros(self.T), np.zeros(self.T), np.zeros(self.T)

        K_raw = np.zeros(self.T)
        H_polar = np.zeros(self.T)
        Spec_Gap = np.zeros(self.T)
        Spec_Obs = np.zeros(self.T)
        SBP_list = np.zeros(self.T)

        for t in range(1, self.T):
            dist_t = self.logprobs[t].get("top_logprobs", {})
            dist_prev = self.logprobs[t-1].get("top_logprobs", {})

            words = list(dist_t.keys())
            if not words:
                continue

            raw_logprobs_t = np.array([dist_t[w] for w in words])
            probs_t = np.exp(raw_logprobs_t)
            probs_t /= (np.sum(probs_t) + 1e-12)

            s_gap, s_obs, sbp = self._compute_local_spectral_dynamics(words, probs_t)
            Spec_Gap[t] = s_gap
            Spec_Obs[t] = s_obs
            SBP_list[t] = sbp

            K_size = len(probs_t)
            if K_size > 1:
                J = np.diag(probs_t) - np.outer(probs_t, probs_t)
                eigvals = np.linalg.eigvalsh(J)
                eigvals = np.clip(eigvals, 0.0, None)
                eigvals = np.sort(eigvals)
                
                non_zero_eigs = eigvals[1:]
                trace_J = np.sum(non_zero_eigs)
                
                if trace_J > 1e-6:
                    norm_eigs = non_zero_eigs / trace_J
                    H_polar[t] = -np.sum(norm_eigs * np.log(norm_eigs + 1e-12))
                else:
                    H_polar[t] = 0.0
            else:
                H_polar[t] = 0.0

            log_p_prev = np.array([dist_prev.get(w, self.floor) for w in words])
            velocities = raw_logprobs_t - log_p_prev
            mean_v = np.sum(probs_t * velocities)
            var_v = np.sum(probs_t * (velocities - mean_v) ** 2)
            K_raw[t] = 0.5 * max(0.0, var_v)

        return K_raw, H_polar, Spec_Gap, Spec_Obs, SBP_list

    def compute_polarized_path_action(self) -> Tuple[float, float, float, float]:
        """求解集成局域谱阻碍压制后的单路径作用量."""
        K_raw, H_polar, Spec_Gap, Spec_Obs, SBP_list = self.compute_local_kinetic_energies()
        if len(K_raw) <= 1:
            return 0.0, 1.0, 0.0, 1.0
        
        K_polarized = K_raw * Spec_Obs * np.exp(self.eta * H_polar)
        
        avg_action = float(np.mean(K_polarized[1:]))
        avg_gap = float(np.mean(Spec_Gap[1:]))
        avg_obs = float(np.mean(Spec_Obs[1:]))
        avg_sbp = float(np.mean(SBP_list[1:]))

        return avg_action, avg_gap, avg_obs, avg_sbp


class RiskEstimator:
    """基于本征谱阻碍与雅可比极化路径作用量的信息几何风险总线."""

    def __init__(self, null_cache_mgr: Any = None, weights_config: Any = None) -> None:
        self.weights = weights_config
        self.observer = IntrinsicConformalObserver()

    def estimate_structural_risk(
        self, 
        G_sequence: Any = None, 
        schema_name: str = "GSM8K",
        trajectory_text: str = "",
        gold_answer: str = "",
        entry_point: str = "",
        logprobs: Optional[List[Dict[str, Any]]] = None,
        attentions: Optional[np.ndarray] = None
    ) -> Tuple[float, float, List[float], Dict[str, Any]]:
        """【单路径模式接口】."""
        if not logprobs:
            return 0.5, 0.0, [0.5], {}

        fr_pa = FisherRaoCovariantPathAction(logprobs, observer=self.observer)
        action_val, avg_gap, avg_obs, avg_sbp = fr_pa.compute_polarized_path_action()
        K_raw, H_polar, Spec_Gap, Spec_Obs, SBP_list = fr_pa.compute_local_kinetic_energies()
        K_polarized = K_raw * Spec_Obs * np.exp(fr_pa.eta * H_polar)

        return action_val, 0.0, K_polarized.tolist(), {
            "raw_r_struct_unbiased": action_val,
            "avg_edge_weight": avg_gap,
            "fiedler_connectivity": avg_obs,
            "attention_entropy": avg_sbp,
            "action_variance": 0.0,
            "consensus_ratio": 1.0,
            "ead_in": 0.0
        }

    def estimate_ensemble_risk(
        self,
        logprobs_list: List[List[Dict[str, Any]]],
        answers_list: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """【假设 1 证明核：商空间投影与同伦路径对齐】.
        
        仅对落入最大语义丛 C_maj 的路径计算几何积分，彻底剥离不相干错误路径的线性污染。
        同时输出全景离散度（EAD_out）与丛内离散度（EAD_in）用于严格理论实证。
        """
        if not logprobs_list:
            return 0.5, {}

        # 1. 商空间等价类划分，识别最大语义共识丛 C_maj
        counter = Counter(answers_list)
        voted_ans, maj_size = counter.most_common(1)[0]
        N = len(logprobs_list)
        consensus_ratio = float(maj_size / N)
        
        single_actions = []
        gaps = []
        obstractions = []
        sbps = []

        # 2. 共享观测器并随动解算
        for path_logprobs in logprobs_list:
            fr_pa = FisherRaoCovariantPathAction(path_logprobs, observer=self.observer)
            action_val, avg_gap, avg_obs, avg_sbp = fr_pa.compute_polarized_path_action()
            single_actions.append(action_val)
            gaps.append(avg_gap)
            obstractions.append(avg_obs)
            sbps.append(avg_sbp)

        # 3. 同伦路径对齐 (HPA)
        # 仅对处于最大共识丛中的测地线进行局域作用量积分
        maj_indices = [i for i, ans in enumerate(answers_list) if ans == voted_ans]
        maj_actions = [single_actions[idx] for idx in maj_indices]
        aligned_mean_action = float(np.mean(maj_actions))

        # 4. 【假设 1 证明核】系综作用量离散度求解
        # ead_out: 全景测地线离散度 (EAD_out)
        ead_out = float(np.std(single_actions)) if len(single_actions) > 1 else 0.0
        # ead_in: 丛内测地线离散度 (EAD_in) -> 证明正确路径收敛于轨道管
        ead_in = float(np.std(maj_actions)) if len(maj_actions) > 1 else 0.0

        # 5. EPA 极化耦合
        epa_val = aligned_mean_action * np.exp(1.0 - consensus_ratio) + ead_out

        extra_diagnostics = {
            "raw_r_struct_unbiased": epa_val,
            "avg_edge_weight": float(np.mean(gaps)),          # Mean MSG
            "fiedler_connectivity": float(np.mean(obstractions)), # Mean MSO
            "attention_entropy": float(np.mean(sbps)),        # Mean SBP
            "action_variance": ead_out,                        # EAD_out (全景)
            "ead_in": ead_in,                                  # EAD_in (丛内)
            "consensus_ratio": consensus_ratio                 # CCD (共识丛密度)
        }

        return epa_val, extra_diagnostics