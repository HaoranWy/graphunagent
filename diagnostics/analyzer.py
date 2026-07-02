# -*- coding: utf-8 -*-
"""ISE-Flow 实验诊断与 EHS 指标审计模块."""

import os
import ast
import json
import logging
import math
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger("ise_flow.diagnostics.analyzer")


class ScientificSignalAnalyzer:
    """图拓扑风险诊断、失效案例审计与 EHS 算法性能指标分析器."""

    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker

    # =====================================================================
    # 📌 修复：防标签泄露、平局鲁棒的学术级标准 AUROC 计算
    # =====================================================================
    @staticmethod
    def calculate_auroc(y_true: List[int], y_scores: List[float]) -> float:
        """计算真实的 ROC 曲线下面积 (AUROC) [2].
        
        采用学术界标准的 fractional ranking 处理并列分数 (ties) [2]。
        当全部预测值完全等同（如发生平局）时，AUROC 严格收敛于 0.50 随机概率，阻断标签泄露 [2]。
        """
        if not y_true or not y_scores or len(y_true) != len(y_scores):
            return 0.5
            
        y_true_arr = np.array(y_true, dtype=np.int32)
        y_scores_arr = np.array(y_scores, dtype=np.float64)
        
        n_pos = int(np.sum(y_true_arr == 1))
        n_neg = int(np.sum(y_true_arr == 0))
        if n_pos == 0 or n_neg == 0:
            return 0.5

        # 使用 scipy.stats 提供的中立排序，不存在依赖时使用纯 numpy 实现的 fractional ranking
        try:
            from scipy.stats import rankdata
            ranks = rankdata(y_scores_arr)
        except ImportError:
            # 纯 NumPy 闭式同分排名等价实现
            obs, counts = np.unique(y_scores_arr, return_counts=True)
            unique_ranks = np.zeros_like(obs, dtype=np.float64)
            start_idx = 0
            for i, c in enumerate(counts):
                unique_ranks[i] = start_idx + (c + 1) / 2.0
                start_idx += c
            val_to_rank = dict(zip(obs, unique_ranks))
            ranks = np.array([val_to_rank[x] for x in y_scores_arr])

        rank_sum_pos = np.sum(ranks[y_true_arr == 1])
        u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
        return float(u_stat / (n_pos * n_neg))

    @staticmethod
    def calculate_ece(y_true: List[int], y_probs: List[float], num_bins: int = 10) -> float:
        if not y_true or not y_probs:
            return 1.0
        n_samples = len(y_true)
        bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
        ece = 0.0
        for i in range(num_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = [j for j in range(n_samples) if lower <= y_probs[j] <= upper] if i == num_bins - 1 else \
                     [j for j in range(n_samples) if lower <= y_probs[j] < upper]
            bin_size = len(in_bin)
            if bin_size == 0:
                continue
            bin_probs_mean = sum(y_probs[j] for j in in_bin) / bin_size
            bin_actual_error = sum(y_true[j] for j in in_bin) / bin_size
            ece += (bin_size / n_samples) * abs(bin_actual_error - bin_probs_mean)
        return float(ece)

    @staticmethod
    def calculate_brier_score(y_true: List[int], y_probs: List[float]) -> float:
        if not y_true or not y_probs:
            return 1.0
        return sum((y_probs[i] - y_true[i]) ** 2 for i in range(len(y_true))) / len(y_true)

    @staticmethod
    def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
        if len(x) < 2 or len(y) < 2:
            return 0.0
        x_arr, y_arr = np.array(x), np.array(y)
        num = np.sum((x_arr - np.mean(x_arr)) * (y_arr - np.mean(y_arr)))
        den = np.sqrt(np.sum((x_arr - np.mean(x_arr))**2) * np.sum((y_arr - np.mean(y_arr))**2))
        return float(num / den) if den != 0.0 else 0.0

    @staticmethod
    def compute_ast_entropy(code: str) -> float:
        try:
            tree = ast.parse(code)
            degrees = [len(list(ast.iter_child_nodes(node))) + 1e-5 for node in ast.walk(tree)]
            degrees = np.array(degrees)
            p_v = degrees / np.sum(degrees)
            return float(-np.sum(p_v * np.log(p_v + 1e-12)))
        except Exception:
            return 0.0

    def evaluate_ehs_normalizer(self, logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """在线审计 EHS 归一化器的三大核心物理指标."""
        from data.parsers import ehs_normalize_code
        
        edr_list = []
        raw_nodes_list = []
        ehs_nodes_list = []
        
        for log in logs:
            text = log.get("trajectory_text", "")
            if not text:
                continue
            code_clean = text.split("```python")[-1].split("```")[0].strip() if "```python" in text else text
            
            try:
                raw_tree = ast.parse(code_clean)
                h_raw = self.compute_ast_entropy(code_clean)
                n_raw = len(list(ast.walk(raw_tree)))
                
                ehs_code = ehs_normalize_code(code_clean)
                ehs_tree = ast.parse(ehs_code)
                h_ehs = self.compute_ast_entropy(ehs_code)
                n_ehs = len(list(ast.walk(ehs_tree)))
                
                if h_ehs > 0:
                    edr_list.append(h_raw / h_ehs)
                raw_nodes_list.append(n_raw)
                ehs_nodes_list.append(n_ehs)
            except Exception:
                pass
                
        avg_edr = float(np.mean(edr_list)) if edr_list else 1.0
        sso = float(sum(ehs_nodes_list) / (sum(raw_nodes_list) + 1e-12)) if raw_nodes_list else 1.0
        
        lens = [float(log["trajectory_len"]) for log in logs]
        r_structs = [log["r_ise_score"] for log in logs]
        log_lens = [math.log(l + 1) for l in lens]
        corr = self.calculate_pearson_correlation(log_lens, r_structs)
        ldi = 1.0 - abs(corr)
        
        return {
            "entropy_decimation_ratio_edr": round(avg_edr, 4),
            "length_decoupling_index_ldi": round(ldi, 4),
            "semantic_state_overlap_sso": round(sso, 4)
        }

    def audit_m2_scientific_signal(self) -> Dict[str, Any]:
        logs = self.tracker.get_session_logs()
        valid_logs = [log for log in logs if log.get("is_correct") is not None]
        if len(valid_logs) < 2:
            return {"status": "FAILED", "reason": "Insufficient verified samples."}

        results: Dict[str, Any] = {}
        y_true_all = [0 if log["is_correct"] else 1 for log in valid_logs]
        y_scores_rise_all = [log["r_ise_score"] for log in valid_logs]
        y_scores_len_all = [float(log["trajectory_len"]) for log in valid_logs]

        lens = [float(log["trajectory_len"]) for log in valid_logs]
        log_lens = [math.log(l + 1) for l in lens]
        pearson_corr = self.calculate_pearson_correlation(log_lens, y_scores_rise_all)

        results["all_samples"] = {
            "size": len(valid_logs),
            "auroc_r_ise": round(self.calculate_auroc(y_true_all, y_scores_rise_all), 4),
            "auroc_length_baseline": round(self.calculate_auroc(y_true_all, y_scores_len_all), 4),
            "correlation_length_structural_risk": round(pearson_corr, 4),
            "matcher_label_noise_mci": 0.0
        }

        syntax_valid_logs = [log for log in valid_logs if log.get("is_syntax_valid", 1) == 1]
        if len(syntax_valid_logs) >= 2:
            y_true_sv = [0 if log["is_correct"] else 1 for log in syntax_valid_logs]
            y_scores_rise_sv = [log["r_ise_score"] for log in syntax_valid_logs]
            y_scores_len_sv = [float(log["trajectory_len"]) for log in syntax_valid_logs]
            auroc_rise_sv = self.calculate_auroc(y_true_sv, y_scores_rise_sv)
            auroc_len_sv = self.calculate_auroc(y_true_sv, y_scores_len_sv)
            results["syntax_valid_only"] = {
                "size": len(syntax_valid_logs),
                "auroc_r_ise": round(auroc_rise_sv, 4),
                "auroc_length_baseline": round(auroc_len_sv, 4),
                "auroc_gain_vs_len": round(auroc_rise_sv - auroc_len_sv, 4),
                "go_condition_1_pass": (auroc_rise_sv - auroc_len_sv) >= 0.03
            }
        else:
            results["syntax_valid_only"] = {"status": "SKIPPED"}

        corrects = [log for log in valid_logs if log.get("is_correct") is True]
        wrongs = [log for log in valid_logs if log.get("is_correct") is False]
        matched_logs = []
        if corrects and wrongs:
            used_correct_ids = set()
            for w_log in wrongs:
                w_len = w_log.get("trajectory_len", 0)
                best_match = min((c_log for c_log in corrects if c_log["sample_id"] not in used_correct_ids), 
                                  key=lambda c_log: abs(w_len - c_log.get("trajectory_len", 0)), default=None)
                if best_match:
                    matched_logs.extend([w_log, best_match])
                    used_correct_ids.add(best_match["sample_id"])

        if len(matched_logs) >= 2:
            y_true_matched = [0 if log["is_correct"] else 1 for log in matched_logs]
            y_scores_rise_matched = [log["r_ise_score"] for log in matched_logs]
            auroc_rise_matched = self.calculate_auroc(y_true_matched, y_scores_rise_matched)
            results["length_matched_subset"] = {
                "size": len(matched_logs),
                "auroc_r_ise": round(auroc_rise_matched, 4),
                "go_condition_2_pass": auroc_rise_matched > 0.50
            }
        else:
            results["length_matched_subset"] = {"status": "SKIPPED"}

        results["calibration"] = {
            "ece_score": round(self.calculate_ece(y_true_all, y_scores_rise_all), 4),
            "brier_score": round(self.calculate_brier_score(y_true_all, y_scores_rise_all), 4)
        }

        scale_5_stds, lookup_dists, d_eff_list, spectral_gaps, pr_entropies = [], [], [], [], []
        for log in valid_logs:
            stds = log.get("scale_wise_null_stds", [])
            if len(stds) > 5:
                scale_5_stds.append(stds[5])
            dists = log.get("scale_wise_lookup_distances", [])
            if dists:
                lookup_dists.append(np.mean(dists))
            d_eff_list.append(log.get("effective_dimension", 1.0))
            spectral_gaps.append(log.get("spectral_gap", 0.0))
            pr_entropies.append(log.get("pagerank_entropy", 0.0))

        results["topology_audit"] = {
            "avg_scale_5_null_std": round(float(np.mean(scale_5_stds)), 4) if scale_5_stds else 0.0,
            "avg_lookup_distance": round(float(np.mean(lookup_dists)), 4) if lookup_dists else 0.0,
            "total_mcmc_fallbacks": sum(log.get("mcmc_fallback_count", 0) for log in valid_logs),
            "avg_effective_dimension": round(float(np.mean(d_eff_list)), 4) if d_eff_list else 1.0,
            "avg_spectral_gap": round(float(np.mean(spectral_gaps)), 4) if spectral_gaps else 0.0,
            "avg_pagerank_entropy": round(float(np.mean(pr_entropies)), 4) if pr_entropies else 0.0
        }

        results["ehs_metrics"] = self.evaluate_ehs_normalizer(valid_logs)
        return results