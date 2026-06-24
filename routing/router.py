# -*- coding: utf-8 -*-
"""ISE-Flow 批处理多维预算分配与成本敏感路由器.

本模块基于不确定性概率标定与多维向量预算限制，实现全局性价比（VCR）排序分发，
并在预算告急时自适应执行期望损失最小化兜底决策 [PDF Page 4, 5, 8]。
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from config.schemas import NodeType
from routing.verifier_manager import VerifierManager

logger = logging.getLogger("ise_flow.routing.router")


class BatchRouter:
    """基于性价比（VCR）全局排序的多维成本敏感路由器."""

    def __init__(
        self,
        verifier_mgr: VerifierManager,
        budget_limit: Tuple[float, ...],
        loss_err: float = 10.0,
        loss_abs: float = 2.0,
        coverage_threshold_zeta: float = 0.75,
        safety_priority_constant_M: float = 100000.0
    ) -> None:
        """初始化路由器.

        Args:
            verifier_mgr: 已绑定的验证器管理器实例。
            budget_limit: 多维向量预算上限 B = ($, 延迟, 算力, 风险, 隐私...) [PDF Page 4, 8]。
            loss_err: 信任错误轨迹造成的期望损失 L_err [PDF Page 5]。
            loss_abs: 弃权不作答（Abstain）产生的固有代价 L_abs [PDF Page 5]。
            coverage_threshold_zeta: Parser 覆盖度的安全阈值 ζ。
            safety_priority_constant_M: 强制安全优先队列的常数 M [PDF Page 11]。
        """
        self.verifier_mgr = verifier_mgr
        self.budget_limit = np.array(budget_limit, dtype=np.float64)
        self.l_err = loss_err
        self.l_abs = loss_abs
        self.zeta = coverage_threshold_zeta
        self.M = safety_priority_constant_M

    def route_batch(
        self, 
        batch_data: List[Dict[str, Any]], 
        active_verifiers: Optional[List[str]] = None  # 【新增】支持传入过滤后的可用 verifier 列表
    ) -> List[Dict[str, Any]]:
        """执行批处理全局 VCR 排序预算分配路由 [PDF Page 4, 8].

        Args:
            batch_data: 批输入列表，每个元素字典格式必须包含：
                - "sample_id": 样本唯一标识。
                - "p_wrong": 开发集分桶校准/Rank兜底转换出的错误概率。
                - "syntax_valid": 语法有效性标签（1: 合法, 0: 退化失效）。
                - "coverage": 解析覆盖率。
                - "trajectory_text": 原始生成轨迹。

        Returns:
            List[Dict]: 每一个输入样本的最终决策，包含:
                - "sample_id": 样本 ID。
                - "action": 决策动作 ("trust" / "verify" / "abstain")。
                - "verifier_allocated": 选中的验证器 ID（若未进行verify则为 None）。
                - "actual_cost_vector": 本次动作实际消耗的多维物理成本。
        """
        if not batch_data:
            return []

        # 1. 深度拷贝并初始化多维向量剩余预算 B_remain
        remaining_budget = self.budget_limit.copy()
        
        # 预先登记注册的验证器 ID 列表
        verifier_ids = active_verifiers if active_verifiers else list(self.verifier_mgr.verifiers.keys())

        # 2. 对批次内的每一个样本，计算其最佳 Verifier 候选项、EVV、以及 VCR 性价比
        candidate_actions: List[Dict[str, Any]] = []

        for sample in batch_data:
            p_w = sample["p_wrong"]
            syntax_valid = sample.get("syntax_valid", 1)
            coverage = sample.get("coverage", 1.0)
            
            # 2.1 寻找使得当前样本期望验证价值（EVV）最大化的验证器 v*
            best_v_id = None
            best_evv = -999999.0
            best_scalar_cost = 0.0

            for v_id in verifier_ids:
                evv, c_v = self.verifier_mgr.compute_evv(p_w, v_id)
                if evv > best_evv:
                    best_evv = evv
                    best_v_id = v_id
                    best_scalar_cost = c_v

            # 2.2 判断当前样本是否触及 Parser 安全门控或语法失效 [PDF Page 3, 8]
            is_gated = (coverage < self.zeta) or (syntax_valid == 0)

            # 2.3 计算验证性价比 VCR [PDF Page 4]
            # 若触发安全门控，其验证动作 VCR 强制加上大常数 M 以插队优先强占通道
            evv_positive = max(0.0, best_evv)
            if is_gated:
                vcr = (self.M + evv_positive) / (best_scalar_cost + 1e-6)
            else:
                vcr = evv_positive / (best_scalar_cost + 1e-6)

            candidate_actions.append({
                "sample": sample,
                "best_verifier": best_v_id,
                "best_evv": best_evv,
                "best_scalar_cost": best_scalar_cost,
                "is_gated": is_gated,
                "vcr": vcr
            })

        # 3. 按 VCR/VCR_safe 降序对批动作进行全局排序 [PDF Page 4]
        candidate_actions.sort(key=lambda x: x["vcr"], reverse=True)

        final_decisions: List[Dict[str, Any]] = []

        # 4. 逐个检查多维向量预算，并输出最终决策路径 [PDF Page 4]
        for item in candidate_actions:
            sample = item["sample"]
            v_id = item["best_verifier"]
            evv = item["best_evv"]
            is_gated = item["is_gated"]
            p_w = sample["p_wrong"]

            allocated_verifier = None
            actual_cost = np.zeros_like(remaining_budget)

            # 4.1 只有在最佳 EVV > 0，或样本触及安全门控时，才尝试进行 verify 动作 [PDF Page 4, 8]
            if (evv > 0.0 or is_gated) and v_id:
                # 获取待消费验证器的多维原始成本向量
                spec = self.verifier_mgr.verifiers[v_id].spec
                c_vector = np.array(spec.base_cost_vector, dtype=np.float64)[:len(remaining_budget)]

                # 多维向量预算边界检查： np.all(c_vector <= B_remain) [PDF Page 4]
                if np.all(c_vector <= remaining_budget):
                    action = "verify"
                    allocated_verifier = v_id
                    actual_cost = c_vector
                    # 动态扣减多维预算
                    remaining_budget -= c_vector
                else:
                    # 预算超限，若有便宜的备用验证器可在后续版本拓展，此处直接 fallback 至无验证分支
                    action = self._fallback_unverified_decision(p_w, is_gated)
            else:
                # 期望价值为负，无验证价值，执行无验证分支
                action = self._fallback_unverified_decision(p_w, is_gated)

            final_decisions.append({
                "sample_id": sample["sample_id"],
                "action": action,
                "verifier_allocated": allocated_verifier,
                "actual_cost_vector": actual_cost.tolist()
            })

        # 回归样本原始输入序，便于上层分析器对齐
        id_to_decision = {d["sample_id"]: d for d in final_decisions}
        ordered_decisions = [id_to_decision[s["sample_id"]] for s in batch_data]
        
        logger.info(
            f"Batch routing complete for {len(batch_data)} samples. "
            f"Remaining Vector Budget: {remaining_budget.tolist()}"
        )
        return ordered_decisions

    def _fallback_unverified_decision(self, p_w: float, is_gated: bool) -> str:
        """预算超限或验证价值为负时的安全兜底策略 [PDF Page 4, 5].

        规则：
            1. 若为 Safety Gated 样本，由于绝对不可直接信任，强制输出 `abstain` [PDF Page 8, 11]。
            2. 若为正常样本，比较 E[L|trust] = p_wrong * L_err 与 弃权惩罚 L_abs。
               若损失更小则 `trust`（信任），否则 `abstain`（弃权） [PDF Page 5]。
        """
        if is_gated:
            return "abstain"

        # 比较期望损失 [PDF Page 5]
        expected_trust_loss = p_w * self.l_err
        if expected_trust_loss <= self.l_abs:
            return "trust"
        else:
            return "abstain"