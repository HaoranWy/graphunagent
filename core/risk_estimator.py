
"""
ISE-Flow 量子正则系综自由能 UQ 引擎 (v7.7-ZeroParam-GrandUnified).
基于吉布斯自由能 f = (1-beta)*NLL + H_VN/\ln N 与玻尔兹曼透射几率 P_wrong = 1 - e^{-f}，
绝对零超参数、连续可微地达成单路径与多路径、代码域与数学逻辑域的终极对称大一统。
"""

import ast
import math
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# 🛡️ 严格导入解析器，消除 NameError 崩溃
from data.parsers import TrajectoryParser

logger = logging.getLogger("ise_flow.core.risk_estimator")


class FisherRaoCovariantPathAction:
    def __init__(self, logprobs: List[Dict[str, Any]], stalk_dim: int = 10) -> None:
        self.logprobs = logprobs
        self.T = len(logprobs)
        self.d = stalk_dim

    def compute_classical_action(self) -> float:
        """路径强度经典作用量 (平均 NLL) — 零参数单路径基石。"""
        if not self.logprobs:
            return 2.0
        ll_sum = sum(item["logprob"] for item in self.logprobs)
        return float(-ll_sum / self.T)

    def compute_trajectory_embedding(self) -> np.ndarray:
        rng = np.random.RandomState(42)
        projection_matrix = rng.normal(0.0, 1.0, (1000, self.d))
        embeds = []
        for t in range(min(self.T, 100)):
            dist_t = self.logprobs[t].get("top_logprobs", {})
            words = list(dist_t.keys())
            if not words:
                continue
            token_hash = hash(words[0]) % 1000
            word_vector = projection_matrix[token_hash]
            prob_weight = math.exp(self.logprobs[t]["logprob"])
            embeds.append(word_vector * prob_weight)
        if not embeds:
            dummy = np.zeros(self.d)
            dummy[0] = 1.0
            return dummy
        h = np.mean(embeds, axis=0)
        return h / (np.linalg.norm(h) + 1e-12)


class RiskEstimator:
    """基于量子正则系综自由能的第一性原理大一统 UQ 引擎。"""

    def __init__(self, verifier_mgr: Optional[Any] = None, generator: Optional[Any] = None) -> None:
        self.verifier_mgr = verifier_mgr
        self.generator = generator

    def _extract_ans(self, text: str, schema_name: str) -> str:
        if not text:
            return "null"
        text_clean = text.strip()
        if schema_name in ["HumanEvalPlus", "LiveCodeBench"]:
            if "```python" in text_clean:
                return text_clean.split("```python")[-1].split("```")[0].strip()
            return text_clean
        elif schema_name in ["FOLIO", "P-FOLIO"]:
            text_lower = text_clean.lower()
            matches = list(re.finditer(r"\b(true|false|uncertain)\b", text_lower))
            if matches:
                return matches[-1].group(1)
            return text_clean
        _, final_ans, _ = TrajectoryParser.parse_cot_steps(text_clean)
        return final_ans.strip() if final_ans else text_clean

    def estimate_structural_risk(
        self, 
        G_sequence: Any = None, 
        schema_name: str = "GSM8K",
        trajectory_text: str = "",
        gold_answer: str = "",
        entry_point: str = "",
        logprobs: Optional[List[Dict[str, Any]]] = None,
        attentions: Optional[np.ndarray] = None
    ) -> Tuple[float, float, List[float], Dict[str, Any]]:
        """
        【单路径模式接口 - N=1】单路径下无共识， Wrong 几率等价于自由能透射几率：
        $$ P_wrong = 1.0 - e^{-(1-\beta)*S_class} $$
        """
        if not logprobs:
            return 0.5, 0.0, [0.5], {}

        T = len(logprobs)
        path_action = FisherRaoCovariantPathAction(logprobs)
        S_class = path_action.compute_classical_action()
        
        # 1. 提取单路径自监督 AST 特征
        target_types = (
            ast.Assign, ast.AugAssign, ast.For, ast.While, ast.If,
            ast.FunctionDef, ast.Return, ast.Call, ast.BinOp,
            ast.Compare, ast.List, ast.Dict, ast.Set, ast.Name, ast.Constant
        )
        v_ast = np.zeros(len(target_types), dtype=np.float64)
        code_clean = trajectory_text.split("```python")[-1].split("```")[0].strip() if "```python" in trajectory_text else trajectory_text
        try:
            tree = ast.parse(code_clean)
            node_map = {t: idx for idx, t in enumerate(target_types)}
            for node in ast.walk(tree):
                node_type = type(node)
                if node_type in node_map:
                    v_ast[node_map[node_type]] += 1.0
        except Exception:
            pass
            
        beta = 1.0 if np.any(v_ast > 0.0) else 0.0
        
        # 2. 玻尔兹曼透射几率映射 (自监督消偏)
        p_wrong = float(1.0 - np.exp(-(1.0 - beta) * S_class))

        extra_diagnostics = {
            "raw_r_struct_unbiased": p_wrong,
            "avg_edge_weight": S_class,
            "fiedler_connectivity": 0.0,
            "attention_entropy": 0.0,
            "hodge_exact_ratio": 1.0 - p_wrong,
            "hodge_coexact_ratio": p_wrong,
            "hodge_harmonic_ratio": 0.0,
            "cech_obstruction": 0.0,
            "action_variance": S_class,
            "ead_in": 0.0,
            "consensus_ratio": 1.0
        }
        return p_wrong, 0.0, [p_wrong] * T, extra_diagnostics

    def estimate_ensemble_risk(
        self,
        logprobs_list: List[List[Dict[str, Any]]],
        answers_list: List[str],
        schema_name: str = "GSM8K",
        **kwargs  
    ) -> Tuple[float, Dict[str, Any]]:
        """
        【多路径系综接口 - N>1】解算连续双同伦密度算子的冯诺伊曼熵，并对经典能级迹进行正则系综对合。
        """
        if not logprobs_list:
            return 0.5, {}

        N = len(logprobs_list)
        class_actions = []
        embeddings = []
        
        for path_logprobs in logprobs_list:
            path_action = FisherRaoCovariantPathAction(path_logprobs)
            class_actions.append(path_action.compute_classical_action())
            embeddings.append(path_action.compute_trajectory_embedding())
            
        class_actions = np.array(class_actions)
        embeddings = np.array(embeddings)

        # 1. 求解自回归文本空间原始 Gram 相似度矩阵 $S_{raw}$
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-12
        embeddings_normalized = embeddings / norms
        S_raw = np.dot(embeddings_normalized, embeddings_normalized.T)

        # 2. 无参中位数平移核 (Conformal Median Bandwidth)
        D = 1.0 - S_raw
        off_diag_mask = ~np.eye(N, dtype=bool)
        if np.any(off_diag_mask) and N > 1:
            median_dist = np.median(D[off_diag_mask])
            gamma = 2.0 / (median_dist + 1e-8)
        else:
            gamma = 8.0
            
        S_text = np.exp(gamma * (S_raw - 1.0))

        # 3. 求解语法特征向量
        ast_vectors = []
        ast_valid_count = 0
        for text in answers_list:
            target_types = (
                ast.Assign, ast.AugAssign, ast.For, ast.While, ast.If,
                ast.FunctionDef, ast.Return, ast.Call, ast.BinOp,
                ast.Compare, ast.List, ast.Dict, ast.Set, ast.Name, ast.Constant
            )
            v_ast = np.zeros(len(target_types), dtype=np.float64)
            code_clean = text.split("```python")[-1].split("```")[0].strip() if "```python" in text else text
            try:
                tree = ast.parse(code_clean)
                node_map = {t: idx for idx, t in enumerate(target_types)}
                for node in ast.walk(tree):
                    node_type = type(node)
                    if node_type in node_map:
                        v_ast[node_map[node_type]] += 1.0
            except Exception:
                pass
            ast_vectors.append(v_ast)
            if np.any(v_ast > 0.0):
                ast_valid_count += 1
                
        beta = float(ast_valid_count) / float(N) if N > 0 else 0.0
        
        S_ast = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            S_ast[i, i] = 1.0
            for j in range(i + 1, N):
                v_i, v_j = ast_vectors[i], ast_vectors[j]
                norm_i, norm_j = np.linalg.norm(v_i), np.linalg.norm(v_j)
                sim = np.dot(v_i, v_j) / (norm_i * norm_j + 1e-12) if (norm_i > 1e-12 and norm_j > 1e-12) else 0.0
                S_ast[i, j] = sim
                S_ast[j, i] = sim

        # 4. 求解硬同伦等价投影矩阵 M_hard
        cleaned_answers = [self._extract_ans(ans, schema_name) for ans in answers_list]
        M_hard = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if cleaned_answers[i] == cleaned_answers[j]:
                    M_hard[i, j] = 1.0

        # 5. 【大一统核设计】K = (1-beta) * M_hard + beta * S_ast
        K = (1.0 - beta) * M_hard + beta * S_ast

        # 6. 🛡️ 【第一性原理突破：混态迹等变对合】
        # 计算玻尔兹曼信心权重 w_i = e^{-(1-beta)*S_class_i} / \sum e^{-(1-beta)*S_class_j}
        p_prior = np.exp(-(1.0 - beta) * class_actions)
        w = p_prior / (np.sum(p_prior) + 1e-12)
        sqrt_w = np.sqrt(w)
        
        # 构建密度矩阵 \rho = \sqrt{w_i} K_{ij} \sqrt{w_j}
        rho = np.outer(sqrt_w, sqrt_w) * K
        
        # 求解本征谱
        try:
            evals = np.linalg.eigvalsh(rho)
            evals = np.clip(evals, 0.0, None)
            rho_evals = evals / (np.sum(evals) + 1e-12)
        except Exception:
            rho_evals = np.ones(N) / N

        # 7. 计算本征冯·诺伊曼谱熵 H_VN = -\sum \lambda_k \ln \lambda_k
        h_vn = -np.sum(rho_evals * np.log(rho_evals + 1e-12))
        u_semantic = float(h_vn / np.log(N)) if N > 1 else 0.0
        u_semantic = float(np.clip(u_semantic, 0.0, 1.0))

        # 8. 【吉布斯大一统自由能公式 (Gibbs Free Energy UQ)】
        # f = (1-beta) * \bar{S_class} + U_semantic
        # P_wrong = 1.0 - exp(-f)
        nll_mean = float(np.mean(class_actions))
        free_energy_density = (1.0 - beta) * nll_mean + u_semantic
        
        r_ensemble_pi = float(1.0 - np.exp(-free_energy_density))
        r_ensemble_pi = float(np.clip(r_ensemble_pi, 0.0, 1.0))

        # 精准解耦落盘数据
        extra_diagnostics = {
            "raw_r_struct_unbiased": r_ensemble_pi,
            "avg_edge_weight": nll_mean,                    # 1a 映射: 真实平均经典 NLL
            "fiedler_connectivity": h_vn,                   # 2a 映射: 真实本征冯·诺伊曼谱熵 H_VN
            "attention_entropy": u_semantic,                 # 3a 映射: 归一化不确定度
            "hodge_exact_ratio": 1.0 - r_ensemble_pi,
            "hodge_coexact_ratio": r_ensemble_pi,
            "hodge_harmonic_ratio": 0.0,
            "cech_obstruction": float(np.std(class_actions)),
            "action_variance": float(np.std(class_actions)),
            "ead_in": 0.0,
            "consensus_ratio": 1.0 - r_ensemble_pi
        }

        return r_ensemble_pi, extra_diagnostics