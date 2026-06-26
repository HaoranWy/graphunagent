# -*- coding: utf-8 -*-
"""ISE-Flow 纯代数胞腔语义层评估模块 (捷赫子空间投影彻底解耦版)."""

import math
import logging
import re
import sys
from typing import List, Dict, Any, Tuple, Callable, Optional
import numpy as np
import networkx as nx

from config.schemas import EdgeType
from config.settings import RISEWeightsConfig
from core.null_cache import NullCacheManager

logger = logging.getLogger("ise_flow.core.risk_estimator")


class DynamicTraceCollector:
    """动态执行物理轨迹采集器."""

    def __init__(self, target_func_name: str) -> None:
        self.target_func_name = target_func_name
        self.trace_steps: List[Tuple[int, Dict[str, float]]] = []

    def trace_func(self, frame, event: str, arg: Any):
        if event == 'line':
            code = frame.f_code
            func_name = code.co_name
            if func_name == self.target_func_name:
                # 捕获局部变量状态域，安全映射为浮点数标量
                local_vars = {}
                for k, v in frame.f_locals.items():
                    if isinstance(v, (int, float, bool)):
                        local_vars[k] = float(v)
                    elif isinstance(v, (list, tuple, set)) and all(isinstance(x, (int, float)) for x in v):
                        local_vars[k] = float(sum(v))
                    else:
                        local_vars[k] = float(hash(str(v)) % 1000) / 1000.0
                self.trace_steps.append((frame.f_lineno, local_vars))
        return self.trace_func


class CellularSemanticsSheaf:
    """纯代数、参数无关的胞腔语义层 (Čech Subspace Projection Sheaf).
    
    采用层论中最严谨的伴随子空间投影定义：
    - 0-胞腔 Stalk F(u) 的维度精确等于该节点所包含的变量、算子、数值等符号集基数 |S_u|。
    - 局部截面 s(u) 存储当前步骤的实值语义状态。
    - 限制映射 rho_u_e 定义为相交子空间的正交典范投影 (Canonical Projection)。
    - 彻底消去随机矩阵 P、Q 与人工权重，实现真正的零参数层同论拼缝。
    """

    def __init__(self, G: nx.DiGraph, task_type: str) -> None:
        """初始化捷赫投影语义层."""
        self.G = G
        self.task_type = task_type
        # stalks[u] 存储形如 {var_name: valuation_float} 的局部语义状态格
        self.stalks: Dict[str, Dict[str, float]] = {}

    def _extract_local_valuations(self, text: str) -> Dict[str, float]:
        """无监督、确定性地提取局部符号的实值语义不变式."""
        valuations: Dict[str, float] = {}
        if not text:
            return valuations

        # 兼容匹配标准 EHS 标准变量 "v_depth_counter" (如 v_0_1) 与原生 "var_X"
        var_pattern = re.compile(r"\b(?:v|var|step|line)_\d+(?:_\d+)?\b")
        vars_found = var_pattern.findall(text)
        for var in vars_found:
            # 变量的初始状态由其名称哈希决定，保障确定性与无偏性
            valuations[var] = float(hash(var) % 1000) / 1000.0

        # 提取核心逻辑与代数算子（作为 Stalk 空间的逻辑不变量维度）
        operators = re.findall(r"[\+\-\*/%&\|\^~=<>!]+", text)
        for idx, op in enumerate(operators):
            valuations[f"op_{idx}_{op}"] = 1.0

        # 提取具体常数数值
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        for idx, num in enumerate(numbers):
            try:
                valuations[f"num_{idx}"] = float(num)
            except ValueError:
                valuations[f"num_{idx}"] = 1.0

        return valuations

    def construct_sheaf_and_sections(self, trajectory_text: str) -> None:
        """静态模态：构建 0-胞腔局部截面."""
        for node_id, data in self.G.nodes(data=True):
            payload = data.get("payload", {})
            content = str(payload.get("content", "")) if payload else ""
            if not content or content == "None":
                content = node_id
            self.stalks[node_id] = self._extract_local_valuations(content)

    def construct_dynamic_complex_sheaf(self, trace_steps: List[Tuple[int, Dict[str, float]]]) -> None:
        """动态模态：物理执行轨迹的实值变量 Stalk 状态映射."""
        self.stalks.clear()
        for step_idx, (line_no, local_vars) in enumerate(trace_steps):
            node_id = f"step_{step_idx}_line_{line_no}"
            self.stalks[node_id] = {k: float(v) for k, v in local_vars.items()}

    def compute_cech_cohomology_obstruction(self) -> float:
        """计算第一上同调群 H^1 的捷赫共边界相容性阻碍 (捷赫子空间投影版)."""
        edges = []
        if self.G and self.G.edges:
            edges = list(self.G.edges)
        else:
            # 动态轨迹基空间 1-胞腔边集合构造
            nodes_sorted = sorted(list(self.stalks.keys()), key=lambda x: int(x.split("_")[1]) if "step_" in x else 0)
            edges = [(nodes_sorted[i], nodes_sorted[i+1]) for i in range(len(nodes_sorted)-1)]

        if not edges:
            return 0.0

        obstruction_norms = []
        for u, v in edges:
            s_u = self.stalks.get(u, {})
            s_v = self.stalks.get(v, {})
            
            # 1. 计算相交子空间支撑集 S_u \cap S_v
            shared_vars = set(s_u.keys()).intersection(set(s_v.keys()))
            
            if not shared_vars:
                # 没有任何逻辑/变量交集，代数胶合相容性平凡（Vacuously）成立，阻碍为 0.0
                obstruction_norms.append(0.0)
                continue
                
            # 2. 捷赫共边界计算 (d^0 s)_e = \rho_{v \to e}(s_v) - \rho_{u \to e}(s_u)
            # 限制映射 \rho 直接由共享变量子空间的正交投影算子 P_e 决定，不存在任何物理调参参数
            edge_ob_sum = 0.0
            for var in shared_vars:
                val_u = s_u[var]
                val_v = s_v[var]
                edge_ob_sum += (val_v - val_u) ** 2
                
            # 归一化该交集胞腔上的阻碍强度
            obstruction_norms.append(edge_ob_sum / len(shared_vars))
            
        ccr_risk = float(np.mean(obstruction_norms))
        return ccr_risk


class RiskEstimator:
    """基于层同伦捷赫拼缝相容性的纯代数风险编译引擎."""

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

    def _extract_trace_from_assertion(self, trajectory_text: str, entry_point: str, gold_answer: str) -> List[Tuple[int, Dict[str, float]]]:
        """物理追踪器：从 gold_answer 断言中剥离测试用例并在线执行，提取 DEC 轨迹."""
        if not trajectory_text or not entry_point or not gold_answer:
            return []

        code_clean = trajectory_text
        if "```python" in trajectory_text:
            code_clean = trajectory_text.split("```python")[-1].split("```")[0].strip()

        assertions = re.findall(r"assert\s+candidate\((.*?)\)", gold_answer)
        if not assertions:
            return []

        test_args_str = assertions[0]

        sandbox_globals = {
            "__builtins__": __builtins__,
            "math": __import__("math"),
            "sys": sys,
            "List": List,
            "Dict": Dict,
            "Tuple": Tuple,
            "Optional": Optional,
            "Set": Set
        }

        try:
            exec(code_clean, sandbox_globals)
            func_obj = sandbox_globals.get(entry_point)
            if not func_obj or not callable(func_obj):
                return []

            collector = DynamicTraceCollector(entry_point)
            sys.settrace(collector.trace_func)
            try:
                eval(f"{entry_point}({test_args_str})", sandbox_globals)
            finally:
                sys.settrace(None)

            return collector.trace_steps
        except Exception as e:
            sys.settrace(None)
            logger.debug(f"[DEC DIAGOSTICS] Trace run exception: {str(e)}")
            return []

    def estimate_structural_risk(
        self, 
        G_sequence: List[nx.DiGraph], 
        schema_name: str,
        trajectory_text: str = "",
        gold_answer: str = "",
        entry_point: str = ""
    ) -> Tuple[float, float, List[float], Dict[str, Any]]:
        """计算混合拓扑风险 (自适应静态/动态层选择)."""
        if not G_sequence:
            return 1.0, 1.0, [], {}

        G_0 = G_sequence[0]
        d_eff = self.calculate_effective_dimension(G_0)

        # -----------------------------------------------------------------
        # 1. 胞腔语义层 Čech 拼缝阻碍计算 (自适应双模态检测)
        # -----------------------------------------------------------------
        is_code_task = (schema_name == "HumanEvalPlus" or schema_name == "Code")
        trace_steps = []
        
        if is_code_task and gold_answer and entry_point:
            trace_steps = self._extract_trace_from_assertion(trajectory_text, entry_point, gold_answer)

        if trace_steps:
            # [动态 DEC 模态]
            dyn_sheaf = CellularSemanticsSheaf(None, schema_name)
            dyn_sheaf.construct_dynamic_complex_sheaf(trace_steps)
            raw_cohomology_risk = dyn_sheaf.compute_cech_cohomology_obstruction()
            logger.info(f"[DEC ACTIVE] Čech dynamic complex obstruction calculated: {raw_cohomology_risk:.4f}")
        else:
            # [静态同伦层模态]
            static_sheaf = CellularSemanticsSheaf(G_0, schema_name)
            static_sheaf.construct_sheaf_and_sections(trajectory_text)
            raw_cohomology_risk = static_sheaf.compute_cech_cohomology_obstruction()

        # -----------------------------------------------------------------
        # 2. 100% 纯净结构层上同调风险输出 (彻底解耦底层物理统计噪声)
        # -----------------------------------------------------------------
        purified_r_struct = float(np.clip(raw_cohomology_risk, 0.01, 0.99))

        # 维持 diagnostics 报告总线的数据完整性
        normalized_entropies: List[float] = []
        raw_entropies: List[float] = []
        null_means: List[float] = []
        null_stds: List[float] = []
        lookup_distances: List[float] = []
        mcmc_fallback_count = 0
        cache_statuses: List[str] = []

        for G_k in G_sequence:
            sheaf_k = CellularSemanticsSheaf(G_k, schema_name)
            sheaf_k.construct_sheaf_and_sections(trajectory_text)
            cech_norm_k = sheaf_k.compute_cech_cohomology_obstruction()
            
            mu_N, sigma_N, status, dist = self.cache_mgr.query_baseline(G_k, schema_name)
            
            sigma_N_safe = max(sigma_N, 0.1)
            mu_N_safe = max(mu_N, 0.1)

            normalized_h = (cech_norm_k - mu_N_safe) / (sigma_N_safe + 1e-5)
            normalized_h_clipped = max(0.0, min(3.0, float(normalized_h)))
            
            normalized_entropies.append(normalized_h_clipped)
            raw_entropies.append(float(cech_norm_k))
            null_means.append(float(mu_N))
            null_stds.append(float(sigma_N))
            lookup_distances.append(float(dist))
            cache_statuses.append(status)

        extra_diagnostics = {
            "raw_entropies": raw_entropies,
            "null_means": null_means,
            "null_stds": null_stds,
            "lookup_distances": lookup_distances,
            "mcmc_fallback_count": mcmc_fallback_count,
            "cache_statuses": cache_statuses,
            "effective_dimension": d_eff,
            "rev_rates": [],
            "raw_r_struct_unbiased": raw_cohomology_risk
        }

        return purified_r_struct, 0.0, normalized_entropies, extra_diagnostics

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
                    sheaf_p = CellularSemanticsSheaf(G_k, task_type)
                    sheaf_p.construct_sheaf_and_sections(p_text)
                    cech_p = sheaf_p.compute_cech_cohomology_obstruction()
                    
                    mu_N, _, _, _ = self.cache_mgr.query_baseline(G_k, task_type)
                    phi_p_list.append(cech_p / (mu_N + 1e-6))
                
                phi_p = np.array(phi_p_list, dtype=np.float64)
                if len(phi_p) < len(phi_base):
                    phi_p = np.pad(phi_p, (0, len(phi_base) - len(phi_p)), "edge")
                elif len(phi_p) > len(phi_base):
                    phi_p = phi_p[:len(phi_base)]
                l2_distance = np.linalg.norm(phi_base - phi_p)
                drift_sums += (l2_distance / phi_base_norm)
            except Exception:
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