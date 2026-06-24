# -*- coding: utf-8 -*-
"""ISE-Flow 拓扑风险评估模块 (重正化群尺度无关 REV 引擎版)."""

import math
import logging
import re
from typing import List, Dict, Any, Tuple, Callable
import numpy as np
import networkx as nx

from config.schemas import EdgeType
from config.settings import RISEWeightsConfig
from core.null_cache import NullCacheManager

logger = logging.getLogger("ise_flow.core.risk_estimator")


class RiskEstimator:
    """基于重正化信息流速（REV）波动率的尺度无关拓扑风险评估与编译引擎."""

    def __init__(
        self,
        null_cache_mgr: NullCacheManager,
        weights_config: RISEWeightsConfig,
        omega_weights: Tuple[float, ...] = (0.3, 0.2, 0.3, 0.1, 0.1)
    ) -> None:
        self.cache_mgr = null_cache_mgr
        self.weights = weights_config
        self.omega = omega_weights

    @staticmethod
    def calculate_effective_dimension(G: nx.DiGraph) -> float:
        if G.number_of_nodes() <= 2:
            return 1.0
        U = G.to_undirected()
        try:
            if not nx.is_connected(U):
                components = list(nx.connected_components(U))
                if not components:
                    return 1.0
                largest_cc = U.subgraph(max(components, key=len))
                v_size = largest_cc.number_of_nodes()
                diameter = nx.diameter(largest_cc)
            else:
                v_size = G.number_of_nodes()
                diameter = nx.diameter(U)
            if diameter <= 1:
                return 1.0
            return float(math.log(v_size) / math.log(diameter + 1e-5))
        except Exception:
            return 1.0

    def estimate_structural_risk(
        self, 
        G_sequence: List[nx.DiGraph], 
        schema_name: str,
        trajectory_text: str = ""
    ) -> Tuple[float, float, List[float], Dict[str, Any]]:
        if not G_sequence:
            return 1.0, 1.0, [], {}

        G_0 = G_sequence[0]
        v_0_count = G_0.number_of_nodes()
        d_eff = self.calculate_effective_dimension(G_0)

        # -----------------------------------------------------------------
        # 核心：重正化熵流速（REV）计算
        # -----------------------------------------------------------------
        rev_rates = []
        for k in range(len(G_sequence) - 1):
            G_curr = G_sequence[k]
            G_next = G_sequence[k+1]
            
            v_curr = G_curr.number_of_nodes()
            v_next = G_next.number_of_nodes()
            
            if v_curr <= 1 or v_next <= 1 or v_curr == v_next:
                continue
                
            h_curr = self.cache_mgr.calculate_shannon_entropy(G_curr)
            h_next = self.cache_mgr.calculate_shannon_entropy(G_next)
            
            log_scale_ratio = math.log(v_next) - math.log(v_curr)
            delta_entropy = h_next - h_curr
            
            v_rev = delta_entropy / log_scale_ratio
            rev_rates.append(v_rev)
            
        if rev_rates:
            rev_volatility = float(np.var(rev_rates))
            purified_r_struct = 1.0 - math.exp(-rev_volatility * 15.0)
            purified_r_struct = float(np.clip(purified_r_struct, 0.01, 0.99))
        else:
            purified_r_struct = 0.5

        # -----------------------------------------------------------------
        # 诊断流兼容计算：提取标准 Z-score 数据
        # -----------------------------------------------------------------
        normalized_entropies: List[float] = []
        raw_entropies: List[float] = []
        null_means: List[float] = []
        null_stds: List[float] = []
        lookup_distances: List[float] = []
        mcmc_fallback_count = 0
        cache_statuses: List[str] = []
        miss_count = 0

        for G_k in G_sequence:
            h_raw = self.cache_mgr.calculate_shannon_entropy(G_k)
            mu_N, sigma_N, status, dist = self.cache_mgr.query_baseline(G_k, schema_name)
            
            if status in ["fallback_miss", "fallback_raw"]:
                miss_count += 1
            if status == "mcmc_simulation":
                mcmc_fallback_count += 1

            sigma_N_safe = max(sigma_N, 0.1)
            mu_N_safe = max(mu_N, 0.1)

            normalized_h = (h_raw - mu_N_safe) / (sigma_N_safe + 1e-5)
            normalized_h_clipped = max(0.0, min(3.0, float(normalized_h)))
            
            normalized_entropies.append(normalized_h_clipped)
            raw_entropies.append(float(h_raw))
            null_means.append(float(mu_N))
            null_stds.append(float(sigma_N))
            lookup_distances.append(float(dist))
            cache_statuses.append(status)

        cache_miss_rate = miss_count / len(G_sequence)

        extra_diagnostics = {
            "raw_entropies": raw_entropies,
            "null_means": null_means,
            "null_stds": null_stds,
            "lookup_distances": lookup_distances,
            "mcmc_fallback_count": mcmc_fallback_count,
            "cache_statuses": cache_statuses,
            "effective_dimension": d_eff,
            "rev_rates": [round(r, 4) for r in rev_rates] if rev_rates else [],
            "raw_r_struct_unbiased": purified_r_struct
        }

        return purified_r_struct, cache_miss_rate, normalized_entropies, extra_diagnostics

    @classmethod
    def apply_rule_perturbations(cls, text: str) -> List[str]:
        if not text:
            return []
        perturbed_versions: List[str] = []
        p_space = re.sub(r"(\s+)", r"\1  ", text)
        perturbed_versions.append(p_space)
        p_newline = text.replace("\n", "\n\n")
        perturbed_versions.append(p_newline)
        p_tab = text.replace("    ", "\t") if "    " in text else text.replace("\t", "    ")
        perturbed_versions.append(p_tab)
        p_bullet = text.replace("\n- ", "\n* ").replace("\n* ", "\n- ")
        perturbed_versions.append(p_bullet)
        return perturbed_versions

    def estimate_robustness_risk(
        self,
        raw_text: str,
        task_type: str,
        graph_build_fn: Callable[[str, str], Tuple[nx.DiGraph, float, float, int]],
        coarsen_fn: Callable[[nx.DiGraph], List[nx.DiGraph]],
        base_h_curve: List[float]
    ) -> float:
        if not raw_text or not base_h_curve:
            return 1.0
        perturbed_texts = self.apply_rule_perturbations(raw_text)
        if not perturbed_texts:
            return 1.0
        phi_base = np.array(base_h_curve, dtype=np.float64)
        phi_base_norm = np.linalg.norm(phi_base) + 1e-6
        drift_sums = 0.0

        for p_text in perturbed_texts:
            try:
                g_p, _, _, _ = graph_build_fn(task_type, p_text)
                g_p_seq = coarsen_fn(g_p)
                
                phi_p_list: List[float] = []
                for G_k in g_p_seq:
                    h_p = self.cache_mgr.calculate_shannon_entropy(G_k)
                    mu_N, _, _, _ = self.cache_mgr.query_baseline(G_k, task_type)
                    phi_p_list.append(h_p / (mu_N + 1e-6))
                
                phi_p = np.array(phi_p_list, dtype=np.float64)
                if len(phi_p) < len(phi_base):
                    phi_p = np.pad(phi_p, (0, len(phi_base) - len(phi_p)), "edge")
                elif len(phi_p) > len(phi_base):
                    phi_p = phi_p[:len(phi_base)]
                l2_distance = np.linalg.norm(phi_base - phi_p)
                drift_sums += (l2_distance / phi_base_norm)
            except Exception as e:
                drift_sums += 1.0

        robust_coef = 1.0 - (drift_sums / len(perturbed_texts))
        r_robust = 1.0 - max(0.0, min(1.0, robust_coef))
        return float(r_robust)

    def compile_r_ise(self, r_struct: float, r_parse: float, r_robust: float, r_cache: float) -> float:
        r_ise = (
            self.weights.w_struct * r_struct +
            self.weights.w_parse * r_parse +
            self.weights.w_robust * r_robust +
            self.weights.w_cache * r_cache
        )
        return float(max(0.0, r_ise))