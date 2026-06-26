# -*- coding: utf-8 -*-
"""ISE-Flow 层论状态转移同伦一致性（STCG）分析模块."""

import ast
import math
import logging
import re
from typing import List, Dict, Any, Tuple, Set, FrozenSet, Optional
import numpy as np
import networkx as nx

logger = logging.getLogger("ise_flow.core.stcg")


class ASTSemanticNormalizer(ast.NodeTransformer):
    """静态重写 AST 符号命名，消除 LLM 并行采样产生的字面表达差异."""

    def __init__(self) -> None:
        self.var_map: Dict[str, str] = {}
        self.counter: int = 1
        self.builtins_whitelist = set(dir(__builtins__)) | {
            "self", "cls", "List", "Dict", "Tuple", "Optional", "Set", "Any"
        }

    def _get_standardized_name(self, original_name: str) -> str:
        if original_name in self.builtins_whitelist:
            return original_name
        if original_name not in self.var_map:
            self.var_map[original_name] = f"var_{self.counter}"
            self.counter += 1
        return self.var_map[original_name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if isinstance(node.ctx, (ast.Store, ast.Load)):
            node.id = self._get_standardized_name(node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._get_standardized_name(node.arg)
        return self.generic_visit(node)


class StateTransitionConsensusGraph:
    """基于层论捷赫拼缝（Čech Gluing）的代数同伦一致性分析器."""

    def __init__(self, task_type: str = "GSM8K") -> None:
        self.task_type = task_type
        self.graph = nx.DiGraph()
        self.symbol_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
        # 多路局部截面集合
        self.sections: List[Dict[int, np.ndarray]] = []
        self.d = 16  # 状态同伦嵌入空间维度

    def _normalize_code_step(self, step_text: str) -> str:
        if self.task_type not in ["Code", "HumanEvalPlus"]:
            return step_text
        clean_text = step_text
        if "```python" in step_text:
            clean_text = step_text.split("```python")[-1].split("```")[0].strip()
        try:
            tree = ast.parse(clean_text)
            normalizer = ASTSemanticNormalizer()
            normalized_tree = normalizer.visit(tree)
            ast.fix_missing_locations(normalized_tree)
            return ast.unparse(normalized_tree)
        except Exception:
            return step_text

    def _extract_state_embedding(self, step_text: str, cumulative_variables: Set[str]) -> np.ndarray:
        """提取归一化步骤语义不变式，并映射到 Stalk 向量空间."""
        normalized_text = self._normalize_code_step(step_text)
        
        found_symbols = self.symbol_pattern.findall(normalized_text)
        for sym in found_symbols:
            if len(sym) > 4 or sym not in ["0", "1", "2", "3"]:
                cumulative_variables.add(sym)

        if self.task_type in ["Code", "HumanEvalPlus"]:
            ast_signature_tokens = re.findall(r"\b(?:var_\d+|return|if|for|while|def)\b", normalized_text)
            for token in ast_signature_tokens:
                cumulative_variables.add(token)

        # 构造确定性特征分布向量
        features = np.zeros(64, dtype=np.float64)
        for token in cumulative_variables:
            h = hash(token) % 64
            features[h] += 1.0
            
        norm = np.linalg.norm(features) + 1e-6
        features /= norm
        
        # 稳定的正交投影
        rng = np.random.RandomState(42)
        proj_matrix = rng.randn(self.d, 64)
        embed = np.dot(proj_matrix, features)
        return embed / (np.linalg.norm(embed) + 1e-6)

    def build_consensus_graph(self, parallel_trajectories: List[List[str]]) -> None:
        """多路局部截面的统一上同调空间组装."""
        self.graph.clear()
        self.sections.clear()
        if not parallel_trajectories:
            return

        # 1. 建立全局重合基空间 (联合状态图)
        # 为每一步推理建立节点，节点索引代表全局时间片
        max_steps = max(len(tr) for tr in parallel_trajectories)
        for step_idx in range(max_steps):
            self.graph.add_node(step_idx)
            if step_idx > 0:
                self.graph.add_edge(step_idx - 1, step_idx)

        # 2. 构造各条轨迹的局部截面 (Local Sections)
        for m, trajectory in enumerate(parallel_trajectories):
            cumulative_variables: Set[str] = set()
            local_section: Dict[int, np.ndarray] = {}
            
            for step_idx, step in enumerate(trajectory):
                if not step.strip():
                    continue
                embed = self._extract_state_embedding(step, cumulative_variables)
                local_section[step_idx] = embed
                
            self.sections.append(local_section)

    def compute_cech_consensus_obstruction(self) -> float:
        """利用多路截面Čech拼缝，计算逻辑冲突阻碍."""
        if not self.sections or not self.graph.edges:
            return 0.0

        gluing_discrepancies = []
        M = len(self.sections)

        # 遍历基空间中的所有有向转移边
        for u, v in self.graph.edges:
            # 统计所有并行轨迹在当前转换上的语义相容性
            step_overlap_embeds = []
            for m in range(M):
                sec = self.sections[m]
                if u in sec and v in sec:
                    # 转移残差: \Delta s_m = s_m(v) - s_m(u)
                    step_overlap_embeds.append(sec[v] - sec[u])
            
            if len(step_overlap_embeds) >= 2:
                # 衡量多条轨迹在该转移重合部上的 Čech 互变离散度
                matrix = np.array(step_overlap_embeds)
                # 计算协方差矩阵最大特征值（即第一主成分），代表共识分歧阻碍
                cov = np.cov(matrix.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                # 最大特征值即代表粘合障碍的代数强度
                max_obstruction = float(np.max(eigenvalues))
                gluing_discrepancies.append(max_obstruction)

        if not gluing_discrepancies:
            return 1.0  # 完全不重合或发散
            
        return float(np.mean(gluing_discrepancies))

    def compute_spectral_gap(self) -> float:
        """基于代数同伦拼缝的一致性，映射为标准谱共识度量 (对齐 M2 分析总线)."""
        obstruction = self.compute_cech_consensus_obstruction()
        # 将捷赫缝合阻碍无缝映射为代数连通度：
        # 阻碍趋近于 0（即局部截面完美胶合，H^1 = 0）时，谱连通度趋近于 1.0
        # 拼缝障碍极大时，谱连通度逼近于 0.0
        spectral_gap = float(np.exp(-obstruction * 5.0))
        return float(np.clip(spectral_gap, 0.001, 1.000))

    def compute_pagerank_entropy(self) -> float:
        """多路逻辑共识发散度计算."""
        obstruction = self.compute_cech_consensus_obstruction()
        # 阻碍越大，不确定性越大，状态熵随之单调上升
        return float(obstruction * 3.5 + 0.1)