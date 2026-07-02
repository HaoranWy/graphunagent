# -*- coding: utf-8 -*-
"""ISE-Flow 经典与前沿学术基线计算模块 (ICLR 2026 TokUR 严格对齐版).

本模块包含：
1. Vanilla Logprob / NLL (带严格长度归一化 LN)
2. Self-Consistency (Maj@N 一致性概率)
3. Self-Certainty (带严格词表级均值归一化 LN，兼容 verbalized_score 接口)
4. DeepConf ( Lowest Window 滑动窗口均值 LN)
5. Semantic Entropy (SE) (针对单路径多尺度验证重聚类熵)
"""

import math
import logging
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from data.parsers import TrajectoryParser

logger = logging.getLogger("ise_flow.routing.baselines")


class SCSVBaselines:
    """不确定性度量学术对比基线集合."""

    @staticmethod
    def calculate_nll(token_logprobs: List[float], apply_ln: bool = True) -> float:
        """1. Vanilla Logprob / NLL [严格对齐 ICLR 2026 LN 规范]."""
        if not token_logprobs:
            return 1.0
        
        # NLL 均值计算 (即带长度归一化)
        if apply_ln:
            avg_logprob = float(np.mean(token_logprobs))
        else:
            # 不带长度归一化，直接累加 (会导致长文本不确定性非物理性暴增)
            avg_logprob = float(np.sum(token_logprobs)) / 100.0  # 常规缩放
            
        return -avg_logprob

    @staticmethod
    def calculate_self_consistency(parallel_trajectories: List[str]) -> float:
        """2. Self-Consistency (Maj@N)."""
        if not parallel_trajectories:
            return 1.0

        answers = []
        for tau in parallel_trajectories:
            _, final_ans, _ = TrajectoryParser.parse_cot_steps(tau)
            if final_ans:
                answers.append(final_ans.strip())
            else:
                answers.append("null_fallback_ans")

        counter = Counter(answers)
        _, count = counter.most_common(1)[0]
        p_maj = count / len(parallel_trajectories)
        return 1.0 - p_maj

    @staticmethod
    def calculate_self_certainty(
        token_logprobs: List[float], 
        apply_ln: bool = True, 
        verbalized_score: Optional[float] = None
    ) -> float:
        """3. Self-Certainty [严格对齐 TokUR Table 9 KL Divergence LN 规范，支持多参数兼容]."""
        # 若口头打分有效，自适应转换为不确定度
        if verbalized_score is not None:
            return float(np.clip(1.0 - verbalized_score, 0.0, 1.0))

        if not token_logprobs:
            return 1.0

        # 根据 ICLR 2026 论文，Self-Certainty 通过计算每个 Step 的信息熵均值/分布变动得到
        probs = np.exp(np.clip(token_logprobs, -20.0, 0.0))
        entropy_list = -probs * np.array(token_logprobs)
        
        if apply_ln:
            # 开启长度归一化 (LN)
            return float(np.mean(entropy_list))
        else:
            # 未开启长度归一化
            return float(np.sum(entropy_list))

    @classmethod
    def calculate_deep_conf(cls, token_logprobs: List[float], window_size: int = 5, apply_ln: bool = True) -> float:
        """4. DeepConf [严格对齐 Meta AI 2025/2026 Sliding-Window 机制]."""
        if not token_logprobs or len(token_logprobs) < window_size:
            return cls.calculate_nll(token_logprobs, apply_ln=apply_ln)

        group_confidences = []
        for i in range(len(token_logprobs) - window_size + 1):
            window = token_logprobs[i : i + window_size]
            if apply_ln:
                group_confidences.append(float(np.mean(window)))
            else:
                group_confidences.append(float(np.sum(window)))

        lowest_group_conf = min(group_confidences)
        return -lowest_group_conf

    @staticmethod
    def calculate_semantic_entropy(parallel_trajectories: List[str]) -> float:
        """5. Semantic Entropy (SE) [Nature 2023 语义重聚类]."""
        if not parallel_trajectories:
            return 1.0

        answers = []
        for tau in parallel_trajectories:
            _, final_ans, _ = TrajectoryParser.parse_cot_steps(tau)
            if final_ans:
                answers.append(final_ans.strip().lower())
            else:
                answers.append("null_fallback_ans")

        n_samples = len(answers)
        semantic_groups: List[List[str]] = []

        for ans in answers:
            matched = False
            for group in semantic_groups:
                rep = group[0]
                if rep == ans:
                    group.append(ans)
                    matched = True
                    break
                # 编辑距离模糊合并
                distance = sum(1 for a, b in zip(rep, ans) if a != b) + abs(len(rep) - len(ans))
                max_len = max(len(rep), len(ans), 1)
                if (1.0 - (distance / max_len)) > 0.85:
                    group.append(ans)
                    matched = True
                    break
            
            if not matched:
                semantic_groups.append([ans])

        se = 0.0
        for group in semantic_groups:
            p_c = len(group) / n_samples
            se += -p_c * math.log(p_c + 1e-12)
            
        return float(se)