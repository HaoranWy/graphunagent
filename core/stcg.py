# -*- coding: utf-8 -*-
"""ISE-Flow 状态转移一致性图（STCG）分析模块."""

import ast
import math
import logging
import re
from typing import List, Dict, Any, Tuple, Set, FrozenSet, Optional
import numpy as np
import networkx as nx

logger = logging.getLogger("ise_flow.core.stcg")


class ASTSemanticNormalizer(ast.NodeTransformer):
    """静态重写 AST 符号命名，消除由于 LLM 并行采样产生的多样化命名干扰."""

    def __init__(self) -> None:
        self.var_map: Dict[str, str] = {}
        self.counter: int = 1
        # 保留不进行混淆重命名的内置函数、公共类型以及自引用符号
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
    """基于语义等价物对齐的状态转移一致性有向图模型."""

    def __init__(self, task_type: str = "GSM8K") -> None:
        self.task_type = task_type
        self.graph = nx.DiGraph()
        self.symbol_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")

    def _normalize_code_step(self, step_text: str) -> str:
        """对代码步骤进行静态 AST 语义归一化，解决不同轨迹逻辑相同但命名相异的问题."""
        # 仅对代码任务执行语法级别对齐
        if self.task_type not in ["Code", "HumanEvalPlus"]:
            return step_text

        # 尝试剥离 markdown 符号
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
            # 针对不完整、截断的代码段进行鲁棒性 fallback 降级，避免阻塞全局主管道
            return step_text

    def _extract_state(self, step_text: str, cumulative_variables: Set[str]) -> FrozenSet[str]:
        """将单个步骤的文本内容映射为不可变状态二元组 S = (V, P) 的哈希集合 [PDF Page 1]."""
        # 执行前置语义等价物标准化
        normalized_text = self._normalize_code_step(step_text)

        # 扫描出当前步骤的所有物理数值/实体
        found_symbols = self.symbol_pattern.findall(normalized_text)
        for sym in found_symbols:
            if len(sym) > 4 or sym not in ["0", "1", "2", "3"]:
                cumulative_variables.add(sym)

        # 针对语义等价对齐后的 AST，抽取标准化控制流特征词
        if self.task_type in ["Code", "HumanEvalPlus"]:
            # 捕获局部归一化变量名与其语法层面的关键字
            ast_signature_tokens = re.findall(r"\b(?:var_\d+|return|if|for|while|def)\b", normalized_text)
            for token in ast_signature_tokens:
                cumulative_variables.add(token)

        return frozenset(cumulative_variables)

    def build_consensus_graph(self, parallel_trajectories: List[List[str]]) -> None:
        """融合 $M$ 条并行采样的轨迹，组装一致性状态转移图."""
        self.graph.clear()
        if not parallel_trajectories:
            return

        for m, trajectory in enumerate(parallel_trajectories):
            cumulative_variables: Set[str] = set()
            previous_state: Optional[FrozenSet[str]] = None

            for step in trajectory:
                if not step.strip():
                    continue

                current_state = self._extract_state(step, cumulative_variables)

                if not self.graph.has_node(current_state):
                    self.graph.add_node(current_state, visits=1, state_size=len(current_state))
                else:
                    self.graph.nodes[current_state]["visits"] += 1

                if previous_state is not None:
                    if self.graph.has_edge(previous_state, current_state):
                        self.graph[previous_state][current_state]["weight"] += 1.0
                    else:
                        self.graph.add_edge(previous_state, current_state, weight=1.0)

                previous_state = current_state

    def compute_spectral_gap(self) -> float:
        """计算拉普拉斯谱间隙 $\lambda_2$."""
        if self.graph.number_of_nodes() <= 2:
            return 1.0

        U = self.graph.to_undirected()
        try:
            spectrum = nx.normalized_laplacian_spectrum(U)
            sorted_spectrum = np.sort(spectrum)
            if len(sorted_spectrum) > 1:
                return float(np.clip(sorted_spectrum[1], 0.0, 2.0))
            return 0.0
        except Exception as e:
            logger.debug(f"Laplacian spectral computation failed: {str(e)}")
            try:
                diameter = nx.diameter(U)
                if diameter <= 1:
                    return 1.0
                return float(math.log(self.graph.number_of_nodes()) / math.log(diameter + 1e-5))
            except Exception:
                return 0.0

    def compute_pagerank_entropy(self) -> float:
        """量化逻辑共识凝聚度."""
        if self.graph.number_of_nodes() == 0:
            return 0.0

        try:
            pagerank_dict = nx.pagerank(self.graph, alpha=0.85, weight="weight")
            p_v = np.array(list(pagerank_dict.values()), dtype=np.float64)
            p_v = p_v / np.sum(p_v)
            entropy = -np.sum(p_v * np.log(p_v + 1e-12))
            return float(entropy)
        except Exception as e:
            logger.debug(f"PageRank entropy computation failed: {str(e)}")
            return 0.0