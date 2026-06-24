# -*- coding: utf-8 -*-
"""ISE-Flow 轨迹拓扑组装与自适应篇章连贯性建图（DCG）模块.

升级版：
1. 引入无监督符号二分图投影机制。对步骤内的数值/符号进行 $O(L)$ 正则扫描，
   在不引入外部依赖与专家先验的前提下打破链式同构陷阱。
"""

import re
import os
import logging
from typing import List, Dict, Any, Tuple, Optional, Set

import networkx as nx

from config.schemas import NodeType, EdgeType, HeterogeneousGraphSchema
from data.parsers import TrajectoryParser, ParsingError

SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger("ise_flow.data.graph_builder")


class GraphBuilder:
    """轨迹拓扑图构建器."""

    _encoder: Optional[Any] = None

    @classmethod
    def _get_encoder(cls) -> Optional[Any]:
        if cls._encoder is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                cls._encoder = SentenceTransformer("all-MiniLM-L6-v2")
                if torch.cuda.is_available():
                    cls._encoder = cls._encoder.to("cpu")
                logger.info("Successfully loaded sentence-transformer (all-MiniLM-L6-v2) for DCG.")
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformer: {str(e)}. Fallback enabled.")
        return cls._encoder

    @classmethod
    def build_graph(
        cls,
        task_type: str,
        trajectory_text: str,
        coverage_threshold_zeta: float = 0.75
    ) -> Tuple[nx.DiGraph, float, float, int]:
        """轨迹异构图统一组装入口（带无监督二分图及 Fallback 降级）."""
        steps, final_ans, coverage = TrajectoryParser.parse_cot_steps(trajectory_text)
        ambiguity = 0.0
        
        syntax_valid = 1
        g = nx.DiGraph()

        is_truncated = (
            ("<think>" in trajectory_text and "</think>" not in trajectory_text) or
            (final_ans is None and task_type in ["GSM8K", "MATH"])
        )

        if is_truncated:
            logger.warning("Trajectory truncation detected. Forcing safety fallback to G_chain.")
            syntax_valid = 0
            coverage = 0.0

        try:
            if syntax_valid == 0:
                g = cls._build_chain_graph(steps, final_ans)
            elif task_type in ["GSM8K", "MATH"]:
                g = cls._build_bipartite_discourse_graph(steps, final_ans)
            elif task_type == "Code":
                g = cls._build_python_graph(trajectory_text, steps)
            elif task_type == "SMT":
                g = cls._build_smt_graph(trajectory_text, steps)
            elif task_type == "Agent":
                g = cls._build_agent_graph(trajectory_text, steps)
            else:
                g = cls._build_bipartite_discourse_graph(steps, final_ans)
                
        except ParsingError as pe:
            logger.warning(f"Formal graph assembly failed: {str(pe)}. Executing automated fallback to G_chain...")
            syntax_valid = 0
            coverage = 0.0
            g = cls._build_chain_graph(steps, final_ans)
            
        parser_risk = min(1.0, 1.0 - coverage + ambiguity)
        if syntax_valid == 0:
            parser_risk = 1.0

        g.graph["canonical_nodes"] = cls.get_canonical_nodes(g)
        g.graph["canonical_edges"] = cls.get_canonical_edges(g)
        g.graph["task_type"] = task_type
        g.graph["syntax_valid"] = syntax_valid
        
        return g, coverage, parser_risk, syntax_valid

    @classmethod
    def _build_chain_graph(cls, steps: List[Dict[str, Any]], final_answer: Optional[str]) -> nx.DiGraph:
        """纯链式骨架图生成（当做 Fallback 路径或不连贯降级时调用）."""
        g = nx.DiGraph()
        if not steps:
            return g
        
        for i, step in enumerate(steps):
            g.add_node(
                step["node_id"],
                node_type=NodeType.STEP,
                confidence=1.0,
                payload=step["payload"]
            )
            if i > 0:
                g.add_edge(
                    steps[i-1]["node_id"],
                    step["node_id"],
                    edge_type=EdgeType.SEQUENCE,
                    weight=1.0
                )

        if final_answer:
            ans_id = "ans_node"
            g.add_node(
                ans_id,
                node_type=NodeType.ANSWER,
                confidence=1.0,
                payload={"answer_text": final_answer}
            )
            g.add_edge(
                steps[-1]["node_id"],
                ans_id,
                edge_type=EdgeType.DERIVE,
                weight=1.0
            )
            
        return g

    @classmethod
    def _build_bipartite_discourse_graph(cls, steps: List[Dict[str, Any]], final_answer: Optional[str]) -> nx.DiGraph:
        """【新增核心】无监督符号二分图投影篇章连贯图.

        在 SEQUENCE 推理链的基础上，扫描提取步骤中包含的数学/状态物理数值，
        建立 Step -> Claim -> Step 的双向二分流映射结构，彻底解除同构长度偏置。
        """
        # 1. 初始化标准时序链式骨架
        g = cls._build_chain_graph(steps, final_answer)
        if not steps:
            return g

        # 用于提取纯数值的正则表达式
        symbol_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")

        # 记录已创建的符号节点，避免同名冲突
        created_symbols: Set[str] = set()

        for i, step in enumerate(steps):
            content = str(step["payload"].get("content", ""))
            # 扫描出当前步骤的所有唯一实体数值（去除无用噪声）
            symbols = set(symbol_pattern.findall(content))

            for sym in symbols:
                # 剔除过于平庸的一位数或年月日年份，提高特征信噪比
                if len(sym) > 4 or (sym not in ["0", "1", "2", "3"]):
                    claim_node_id = f"claim_sym_{sym}"
                    
                    # 1. 将匿名实体数值作为逻辑命题（NodeType.CLAIM）注入
                    if claim_node_id not in created_symbols:
                        g.add_node(
                            claim_node_id,
                            node_type=NodeType.CLAIM,
                            confidence=1.0,
                            payload={"symbol_value": sym}
                        )
                        created_symbols.add(claim_node_id)

                    # 2. 建立 Step_i -> Claim_sym 的支撑边 (STEP -> CLAIM 支持 EdgeType.SUPPORT)
                    g.add_edge(
                        step["node_id"],
                        claim_node_id,
                        edge_type=EdgeType.SUPPORT,
                        weight=1.0
                    )

                    # 3. 如果后续步骤再次引用了该符号，建立 Claim_sym -> Step_k 的关联时序边
                    # (CLAIM -> STEP 支持 EdgeType.SEQUENCE)
                    for k in range(i + 1, len(steps)):
                        next_content = str(steps[k]["payload"].get("content", ""))
                        if sym in symbol_pattern.findall(next_content):
                            g.add_edge(
                                claim_node_id,
                                steps[k]["node_id"],
                                edge_type=EdgeType.SEQUENCE,
                                weight=1.0
                            )
                            # 仅针对第一次重现建立最直接的拓扑数据流桥接，防止边饱和过度稠密化
                            break

        return g

    @classmethod
    def _build_python_graph(cls, text: str, steps: List[Dict[str, Any]]) -> nx.DiGraph:
        g = cls._build_chain_graph(steps, None)
        ast_nodes, ast_edges = TrajectoryParser.parse_python_ast(text)
        
        for node in ast_nodes:
            g.add_node(
                node["node_id"],
                node_type=node["node_type"],
                confidence=1.0,
                payload=node["payload"]
            )
            
        for u, v in ast_edges:
            src_t = g.nodes[u]["node_type"]
            dst_t = g.nodes[v]["node_type"]
            e_type = EdgeType.DEPENDENCY if dst_t == NodeType.AST_SYMBOL else EdgeType.AST_CHILD
            
            if HeterogeneousGraphSchema.validate_connection(src_t, dst_t, e_type):
                g.add_edge(u, v, edge_type=e_type, weight=1.0)

        if steps and ast_nodes:
            last_step_id = steps[-1]["node_id"]
            first_ast_stmt = [n["node_id"] for n in ast_nodes if n["node_type"] == NodeType.AST_STMT]
            if first_ast_stmt:
                g.add_edge(
                    last_step_id,
                    first_ast_stmt[0],
                    edge_type=EdgeType.DERIVE,
                    weight=1.0
                )
        return g

    @classmethod
    def _build_smt_graph(cls, text: str, steps: List[Dict[str, Any]]) -> nx.DiGraph:
        g = cls._build_chain_graph(steps, None)
        smt_nodes, smt_edges = TrajectoryParser.parse_smt_lib(text)
        
        for node in smt_nodes:
            g.add_node(
                node["node_id"],
                node_type=node["node_type"],
                confidence=1.0,
                payload=node["payload"]
            )
            
        for u, v in smt_edges:
            src_t = g.nodes[u]["node_type"]
            dst_t = g.nodes[v]["node_type"]
            e_type = EdgeType.SOLVER_CHECK if dst_t == NodeType.SOLVER_OUTCOME else EdgeType.CONSTRAINT_DEP
            
            if HeterogeneousGraphSchema.validate_connection(src_t, dst_t, e_type):
                g.add_edge(u, v, edge_type=e_type, weight=1.0)

        if steps and smt_nodes:
            last_step_id = steps[-1]["node_id"]
            first_smt_assert = [n["node_id"] for n in smt_nodes if n["node_type"] == NodeType.SMT_ASSERT]
            if first_smt_assert:
                g.add_edge(
                    last_step_id,
                    first_smt_assert[0],
                    edge_type=EdgeType.DERIVE,
                    weight=1.0
                )
        return g

    @classmethod
    def _build_agent_graph(cls, text: str, steps: List[Dict[str, Any]]) -> nx.DiGraph:
        g = cls._build_chain_graph(steps, None)
        action_matches = list(re.finditer(r"Action:\s*([^\s\(\)]+)\((.*?)\)", text, re.IGNORECASE))
        obs_matches = list(re.finditer(r"Observation:\s*(.*?)\n", text, re.IGNORECASE))
        
        for i, match in enumerate(action_matches):
            tool_name = match.group(1)
            args = match.group(2)
            act_id = f"action_{i}"
            tool_id = f"tool_{tool_name}"
            
            g.add_node(act_id, node_type=NodeType.ACTION, confidence=1.0, payload={"args": args})
            g.add_node(tool_id, node_type=NodeType.TOOL, confidence=1.0, payload={"name": tool_name})
            
            if steps:
                g.add_edge(steps[min(i, len(steps)-1)]["node_id"], act_id, edge_type=EdgeType.CALL, weight=1.0)
            g.add_edge(act_id, tool_id, edge_type=EdgeType.CALL, weight=1.0)
            
            if i < len(obs_matches):
                obs_content = obs_matches[i].group(1)
                obs_id = f"obs_{i}"
                g.add_node(obs_id, node_type=NodeType.OBSERVATION, confidence=1.0, payload={"content": obs_content})
                g.add_edge(tool_id, obs_id, edge_type=EdgeType.RETURN, weight=1.0)
                
        return g

    @staticmethod
    def get_canonical_nodes(g: nx.DiGraph) -> List[str]:
        return sorted(list(g.nodes()))

    @classmethod
    def get_canonical_edges(cls, g: nx.DiGraph) -> List[Tuple[str, str]]:
        raw_edges = []
        for u, v, data in g.edges(data=True):
            e_type_val = data.get("edge_type", EdgeType.SEQUENCE).value
            weight = data.get("weight", 1.0)
            raw_edges.append((u, v, e_type_val, weight))
        sorted_raw = sorted(raw_edges, key=lambda x: (x[0], x[1], x[2], x[3]))
        return [(u, v) for u, v, _, _ in sorted_raw]