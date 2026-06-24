# -*- coding: utf-8 -*-
"""ISE-Flow 确定性多尺度图粗粒化模块.

本模块通过 Heavy-Edge Matching (HEM) 启发式局部聚合算法，将推理轨迹的异构细粒度图 (G_0)
凝聚为多层级宏观粗粒度图序列 [G_0, G_1, ..., G_K] [PDF Page 3, 4]。
"""

import logging
from typing import List, Dict, Any, Tuple, Set

import networkx as nx

from config.schemas import NodeType, EdgeType

logger = logging.getLogger("ise_flow.core.coarsening")


class HEMCoarsener:
    """基于 Heavy-Edge Matching (HEM) 的确定性多尺度图粗粒化器."""

    @classmethod
    def coarsen(
        self,
        G: nx.DiGraph,
        max_scale_K: int = 5,
        min_node_threshold: int = 3
    ) -> List[nx.DiGraph]:
        """对输入的异构有向图进行多尺度 HEM 粗粒化.

        生成一系列图级压缩序列 G_0, G_1, ..., G_K [PDF Page 3, 4]，
        节点数降至 min_node_threshold 或深度达到 max_scale_K 时终止。

        Args:
            G: 原始细粒度异构图 G_0（已经过 Canonical 排序预处理） [PDF Page 4]。
            max_scale_K: 最大重正化/粗粒化深度级数 K [PDF Page 3]。
            min_node_threshold: 终止聚合的最小图阶数阈值。

        Returns:
            G_sequence: 包含 [G_0, G_1, ..., G_k] (k <= K) 的 networkx.DiGraph 列表 [PDF Page 3, 4]。
        """
        # 深拷贝 G_0 避免破坏输入缓存
        g_current = G.copy()
        g_sequence: List[nx.DiGraph] = [g_current]

        logger.debug(f"Starting HEM coarsening. Initial G_0 order: {g_current.number_of_nodes()}")

        for scale in range(max_scale_K):
            num_nodes = g_current.number_of_nodes()
            if num_nodes <= min_node_threshold:
                logger.debug(f"Scale {scale}: Node count {num_nodes} <= threshold {min_node_threshold}. Stopping.")
                break

            # 1. 寻找当前尺度图的确定性最大匹配
            matching = self._find_heavy_edge_matching(g_current)
            if not matching:
                logger.debug(f"Scale {scale}: No more matchings can be found. Convergence reached.")
                break

            # 2. 合并匹配节点对，生成 G_{scale+1}
            g_coarser = self._contract_graph(g_current, matching, scale + 1)
            g_sequence.append(g_coarser)
            g_current = g_coarser

        return g_sequence

    @classmethod
    def _find_heavy_edge_matching(cls, G: nx.DiGraph) -> List[Tuple[str, str]]:
        """执行带严格 Tie-breaking 规则的 Heavy-Edge 匹配查找 [PDF Page 4].

        通过在词典序层面对起点、终点和边类别进行绑定，彻底消除由于图结构哈希或
        多线程顺序带来的随机性，保证粗粒化在学术实验上的完全一致与可复现。
        """
        # 1. 抽取所有候选边并序列化为包含权重和词典顺序的元组
        # 排序 Key：(-weight, min_id, max_id, edge_type_val) [PDF Page 4]
        candidate_edges: List[Tuple[float, str, str, str]] = []
        for u, v, data in G.edges(data=True):
            if u == v:
                continue  # 忽略自环
            weight = data.get("weight", 1.0)
            edge_type_val = data.get("edge_type", EdgeType.SEQUENCE).value
            
            # 使用词典序稳定端点
            min_id = min(u, v)
            max_id = max(u, v)
            
            candidate_edges.append((weight, min_id, max_id, edge_type_val))

        # 2. 严格按 Canonical 排序：权重最大优先，若相同则依端点词典序和边类型稳定打破对称性 [PDF Page 4]
        candidate_edges.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))

        matched_nodes: Set[str] = set()
        matching: List[Tuple[str, str]] = []

        # 3. 贪心扫描边，生成最大重合度独立集匹配
        for _, min_id, max_id, _ in candidate_edges:
            if min_id not in matched_nodes and max_id not in matched_nodes:
                matching.append((min_id, max_id))
                matched_nodes.add(min_id)
                matched_nodes.add(max_id)

        return matching

    @classmethod
    def _contract_graph(
        cls,
        G: nx.DiGraph,
        matching: List[Tuple[str, str]],
        next_scale: int
    ) -> nx.DiGraph:
        """根据匹配表对图中的顶点和边进行合并压缩 [PDF Page 4]."""
        g_next = nx.DiGraph()
        
        # 1. 建立节点合并映射字典：old_node_id -> new_node_id
        # 为了保证确定性，超级节点的新 ID 采用稳定的 A+B 词典升序拼接
        node_map: Dict[str, str] = {}
        for u, v in matching:
            super_id = f"{min(u, v)}+{max(u, v)}"
            node_map[u] = super_id
            node_map[v] = super_id

        # 孤立点或未匹配点保持原样继承
        for node in G.nodes():
            if node not in node_map:
                node_map[node] = node

        # 2. 合并超级节点的属性与质量分布 [PDF Page 4]
        # Supernode 继承其子节点的 type 分布、度数、质量并累加 confidence
        for super_id in set(node_map.values()):
            # 找到映射至该 super_id 的所有源节点
            constituents = [node for node, mapped in node_map.items() if mapped == super_id]
            
            # 计算融合属性
            sum_conf = sum(G.nodes[c].get("confidence", 1.0) for c in constituents)
            
            # 收集类型分布
            type_distribution: Dict[str, int] = {}
            for c in constituents:
                ntype_val = G.nodes[c].get("node_type", NodeType.STEP).value
                type_distribution[ntype_val] = type_distribution.get(ntype_val, 0) + 1
            
            # 确定超级节点的主类型（取频数最高，若平手则词典序前置稳定打破）
            primary_type_str = sorted(
                list(type_distribution.keys()),
                key=lambda x: (-type_distribution[x], x)
            )[0]
            primary_type = NodeType(primary_type_str)

            # 聚合 payload 数据结构 [PDF Page 4]
            aggregated_payload = {
                "constituents": constituents,
                "type_distribution": type_distribution,
                "scale_level": next_scale,
                "merged_history": [G.nodes[c].get("payload", {}) for c in constituents]
            }

            g_next.add_node(
                super_id,
                node_type=primary_type,
                confidence=min(1.0, sum_conf),  # 截断置信区间
                payload=aggregated_payload
            )

        # 3. 映射并合并边及其权重 [PDF Page 4]
        # 平行边合并时，边权重相加；收缩后的自环作为超级节点的内部质量沉淀或边属性
        for u, v, data in G.edges(data=True):
            new_u = node_map[u]
            new_v = node_map[v]
            weight = data.get("weight", 1.0)
            edge_type = data.get("edge_type", EdgeType.SEQUENCE)

            if new_u == new_v:
                # 形成自环（自合并），将内部边沉淀为 Supernode payload
                g_next.nodes[new_u]["payload"]["internal_edges_count"] = (
                    g_next.nodes[new_u]["payload"].get("internal_edges_count", 0) + 1
                )
                g_next.nodes[new_u]["payload"]["internal_mass"] = (
                    g_next.nodes[new_u]["payload"].get("internal_mass", 0.0) + weight
                )
            else:
                # 建立外部指向边
                if g_next.has_edge(new_u, new_v):
                    # 已有平行边，累加边权 w_uv
                    g_next[new_u][new_v]["weight"] += weight
                else:
                    g_next.add_edge(
                        new_u,
                        new_v,
                        edge_type=edge_type,  # 继承原始核心边属性
                        weight=weight
                    )

        # 4. 在图级属性中重新注册排好序的 Canonical Node & Edge
        # 这确保了任意后续层级 coarsening 的输出也能自始至终稳定复现
        g_next.graph["canonical_nodes"] = sorted(list(g_next.nodes()))
        
        raw_edges = []
        for u, v, data in g_next.edges(data=True):
            e_type_val = data.get("edge_type", EdgeType.SEQUENCE).value
            weight = data.get("weight", 1.0)
            raw_edges.append((u, v, e_type_val, weight))
        sorted_raw = sorted(raw_edges, key=lambda x: (x[0], x[1], x[2], x[3]))
        g_next.graph["canonical_edges"] = [(u, v) for u, v, _, _ in sorted_raw]
        
        return g_next