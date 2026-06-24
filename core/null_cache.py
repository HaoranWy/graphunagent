# -*- coding: utf-8 -*-
"""ISE-Flow 基于 PCFG 产生式树形模拟的零模型缓存网格模块."""

import math
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import networkx as nx

from config.schemas import NodeType

logger = logging.getLogger("ise_flow.core.null_cache")


class NullCacheManager:
    """支持无监督 PCFG-AST 产生式模拟的真实零模型计算与缓存检索引擎."""

    def __init__(
        self,
        cache_data: Optional[Dict[str, Any]] = None,
        num_quantile_bins: int = 10,
        knn_k: int = 5,
        distance_threshold: float = 2.5
    ) -> None:
        self.cache: Dict[str, Dict[str, Dict[str, Any]]] = cache_data or {}
        self.num_quantile_bins = num_quantile_bins
        self.knn_k = knn_k
        self.distance_threshold = distance_threshold
        self.prior_sigmas = np.array([1.2, 1.5, 0.4, 0.15], dtype=np.float64)

        if not self.cache:
            self._prepopulate_cache_grid()

    def _generate_pcfg_synthetic_ast(self, target_nodes_size: int) -> nx.DiGraph:
        """基于 PCFG 树结构产生式产生模拟 AST.
        
        模拟控制流语句、基本块声明与表达式终端的分支结构树形拓扑。
        """
        G = nx.DiGraph()
        G.add_node(0, node_type=NodeType.AST_STMT)
        
        node_counter = 1
        nodes_queue = [0]
        
        while nodes_queue and node_counter < target_nodes_size:
            parent = nodes_queue.pop(0)
            
            # 模拟标准 AST 的分支因子分布
            prob = random.random()
            if prob < 0.3:
                # 语句块分支 (如 Block, Function Body): 分支度 3-4
                branching_factor = random.randint(3, 4)
            elif prob < 0.7:
                # 简单表达式分支 (如 Assign, Binary Op): 分支度 1-2
                branching_factor = random.randint(1, 2)
            else:
                # 终端/变量节点: 叶子节点
                branching_factor = 0
                
            for _ in range(branching_factor):
                if node_counter >= target_nodes_size:
                    break
                child_id = node_counter
                node_counter += 1
                
                # 交叉混合节点特征
                ntype = NodeType.AST_STMT if random.random() > 0.5 else NodeType.AST_SYMBOL
                G.add_node(child_id, node_type=ntype)
                G.add_edge(parent, child_id)
                nodes_queue.append(child_id)
                
        # 鲁棒防错：保证生成的是完整有向树
        if G.number_of_nodes() < target_nodes_size:
            diff = target_nodes_size - G.number_of_nodes()
            for _ in range(diff):
                existing_nodes = list(G.nodes())
                if existing_nodes:
                    chosen = random.choice(existing_nodes)
                    G.add_node(node_counter, node_type=NodeType.AST_SYMBOL)
                    G.add_edge(chosen, node_counter)
                    node_counter += 1
                    
        return G

    def _prepopulate_cache_grid(self) -> None:
        """预填充基础缓存，针对代码与推理分别构建特定分布基线."""
        logger.info("Initializing Null Cache. Compiling multi-domain baselines...")
        schema_list = ["GSM8K", "MATH", "Code", "SMT", "Agent"]
        
        v_sizes = [3, 5, 8, 12, 18, 25, 35, 50, 75, 100]
        densities = [0.05, 0.1, 0.2, 0.3, 0.5]

        for schema in schema_list:
            schema_cache: Dict[str, Dict[str, Any]] = {}
            for v in v_sizes:
                for d in densities:
                    max_edges = v * (v - 1)
                    e = max(v - 1, int(max_edges * d))
                    if e < 1:
                        e = 1

                    sim_entropies = []
                    for _ in range(15):
                        # 【核心修正】若属于代码域，采用 PCFG AST 生成器产生树拓扑作为零模型参照
                        if schema in ["Code", "HumanEvalPlus"]:
                            G_sim = self._generate_pcfg_synthetic_ast(v)
                        else:
                            G_sim = nx.gnm_random_graph(v, e, directed=True)
                            
                        sim_entropies.append(self.calculate_shannon_entropy(G_sim))
                    
                    mu_sim = float(np.mean(sim_entropies))
                    sigma_sim = float(np.std(sim_entropies))
                    if sigma_sim == 0.0:
                        sigma_sim = 0.1

                    h_type_sim = 0.2
                    density_sim = e / (max_edges + 1e-5)
                    z_sim = [math.log(v + 1), math.log(e + 1), h_type_sim, density_sim]

                    v_bin = min(self.num_quantile_bins, int(math.log(v + 1) * 2))
                    e_bin = min(self.num_quantile_bins, int(math.log(e + 1) * 2))
                    h_bin = min(self.num_quantile_bins, int(h_type_sim * 3))
                    bucket_key = f"{v_bin}_{e_bin}_{h_bin}"

                    schema_cache[bucket_key] = {
                        "mean": mu_sim,
                        "std": sigma_sim,
                        "summary_mean": z_sim
                    }
            self.cache[schema] = schema_cache
        logger.info("Null Cache populated with structural CFG-AST and random-graph grids.")

    @staticmethod
    def calculate_shannon_entropy(G: nx.DiGraph, smoothing: float = 1e-5) -> float:
        node_ids = list(G.nodes())
        if not node_ids:
            return 0.0
        degrees = np.array([G.degree(n, weight="weight") for n in node_ids], dtype=np.float64)
        degrees_smoothed = degrees + smoothing
        p_v = degrees_smoothed / np.sum(degrees_smoothed)
        return float(-np.sum(p_v * np.log(p_v + 1e-12)))

    @staticmethod
    def calculate_type_entropy(G: nx.DiGraph) -> float:
        node_count = G.number_of_nodes()
        if node_count == 0:
            return 0.0
        type_counts: Dict[str, int] = {}
        for _, data in G.nodes(data=True):
            ntype = data.get("node_type", NodeType.STEP).value
            type_counts[ntype] = type_counts.get(ntype, 0) + 1
        p_t = np.array([count / node_count for count in type_counts.values()], dtype=np.float64)
        return float(-np.sum(p_t * np.log(p_t + 1e-12)))

    @staticmethod
    def calculate_density(G: nx.DiGraph) -> float:
        v_count = G.number_of_nodes()
        e_count = G.number_of_edges()
        if v_count <= 1:
            return 0.0
        return e_count / (v_count * (v_count - 1) + 1e-5)

    def extract_continuous_summary(self, G: nx.DiGraph) -> np.ndarray:
        v_count = G.number_of_nodes()
        e_count = G.number_of_edges()
        h_type = self.calculate_type_entropy(G)
        density = self.calculate_density(G)
        return np.array([math.log(v_count + 1), math.log(e_count + 1), h_type, density], dtype=np.float64)

    def query_baseline(self, G: nx.DiGraph, schema_name: str) -> Tuple[float, float, str, float]:
        v_count = G.number_of_nodes()
        if v_count == 0:
            return 0.0, 1.0, "fallback_miss", 999.0

        lookup_schema = "Code" if schema_name == "HumanEvalPlus" else schema_name
        schema_cache = self.cache.get(lookup_schema, {})
        
        v_bin = min(self.num_quantile_bins, int(math.log(v_count + 1) * 2))
        e_bin = min(self.num_quantile_bins, int(math.log(G.number_of_edges() + 1) * 2))
        h_type = self.calculate_type_entropy(G)
        h_bin = min(self.num_quantile_bins, int(h_type * 3))
        bucket_key = f"{v_bin}_{e_bin}_{h_bin}"

        if bucket_key in schema_cache:
            item = schema_cache[bucket_key]
            return item["mean"], item["std"], "hit", 0.0

        z_g = self.extract_continuous_summary(G)
        near_hits = self._compute_knn_near_hit(z_g, schema_cache)
        if near_hits:
            return near_hits[0], near_hits[1], "near_hit", near_hits[2]

        return 0.1, 1.0, "fallback_raw", 999.0

    def _compute_knn_near_hit(self, z_g: np.ndarray, schema_cache: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        distances: List[Tuple[float, Dict[str, Any]]] = []
        for bucket_item in schema_cache.values():
            if "summary_mean" not in bucket_item:
                continue
            mu_c = np.array(bucket_item["summary_mean"], dtype=np.float64)
            diff_normalized = (z_g - mu_c) / (self.prior_sigmas + 1e-6)
            dist = np.sqrt(np.sum(diff_normalized ** 2))
            distances.append((dist, bucket_item))

        if not distances:
            return None

        distances.sort(key=lambda x: x[0])
        knn_neighbors = distances[:self.knn_k]

        total_inv_dist = 0.0
        weighted_mu = 0.0
        weighted_sigma = 0.0

        for dist, item in knn_neighbors:
            inv_d = 1.0 / (dist + 1e-6)
            total_inv_dist += inv_d
            weighted_mu += item["mean"] * inv_d
            weighted_sigma += item["std"] * inv_d

        if total_inv_dist == 0.0:
            return None

        return (weighted_mu / total_inv_dist), (weighted_sigma / total_inv_dist), distances[0][0]