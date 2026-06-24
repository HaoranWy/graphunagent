# -*- coding: utf-8 -*-
"""ISE-Flow 真实实验调度流水线编排模块.

升级版：
1. 完整连通 `estimate_structural_risk` 返回的 `extra_diagnostics` 字典并分流。
2. 将开发集标定器的 weights 和 prior base rate 导出为类属性。
"""

import sys
import logging
import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from config.settings import ExperimentSettings
from data.loaders import DatasetLoader, DatasetSplitter, TaskSample
from llm.generator import APIGenerator
from data.graph_builder import GraphBuilder
from core.coarsening import HEMCoarsener
from core.null_cache import NullCacheManager
from core.risk_estimator import RiskEstimator
from routing.calibration import SupervisedHybridCalibrator
from routing.verifier_manager import VerifierManager
from routing.router import BatchRouter
from diagnostics.tracker import TrajectoryTracker
from diagnostics.analyzer import ScientificSignalAnalyzer

logger = logging.getLogger("ise_flow.experiments.run_exp")


class ExperimentRunner:
    """真实学术实验控制总线."""

    def __init__(
        self, 
        settings: Optional[ExperimentSettings] = None,
        model_name: str = "deepseek",
        model_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "empty"
    ) -> None:
        self.settings = settings or ExperimentSettings()
        
        self.loader = DatasetLoader()
        self.cache_mgr = NullCacheManager(
            cache_data=None, 
            num_quantile_bins=self.settings.cache.num_quantile_bins,
            knn_k=self.settings.cache.knn_k_default,
            distance_threshold=self.settings.cache.distance_threshold_D_max
        )
        self.risk_estimator = RiskEstimator(
            null_cache_mgr=self.cache_mgr,
            weights_config=self.settings.rise_weights
        )
        self.verifier_mgr = VerifierManager()
        self.tracker = TrajectoryTracker()
        
        self.generator = APIGenerator(
            model_name=model_name,
            api_key=api_key,
            base_url=model_url,
            seed=self.settings.random_seed
        )

        # 标定器白盒输出指标
        self.calibrator_prior_base_rate: float = 0.0
        self.calibrator_weights: List[float] = []
        self.calibrator_bias: float = 0.0

    def execute_l1_pipeline(self, task_name: str = "GSM8K", sample_limit: int = 30) -> Dict[str, Any]:
        """运行 L1 推理任务的真实 Train/Dev/Test 流水线 [PDF Page 10]."""
        logger.info(f"============ Executing REAL L1 Pipeline for Task: {task_name} ============")
        
        if task_name == "GSM8K":
            dev_samples, test_samples = self.loader.load_gsm8k_split()
            if sample_limit and sample_limit > 0:
                dev_limit = max(5, int(sample_limit * 0.4))
                test_limit = max(5, int(sample_limit * 0.6))
                dev_samples = dev_samples[:dev_limit]
                test_samples = test_samples[:test_limit]
        else:
            raise NotImplementedError("Only real GSM8K loader is implemented under strict conditions.")

        logger.info(f"Requesting real-time trajectories from vLLM server...")
        dev_prompts = [s.raw_question for s in dev_samples]
        test_prompts = [s.raw_question for s in test_samples]

        dev_sc_traces = self.generator.generate_consistency_samples(dev_prompts, num_samples=3)
        test_sc_traces = self.generator.generate_consistency_samples(test_prompts, num_samples=3)

        # 3. 开发集特征提取 (Dev Set)
        logger.info("Extracting topological & length features on Dev Set...")
        dev_feature_matrix: List[List[float]] = []
        dev_labels: List[bool] = []

        for idx, (sample, traces) in enumerate(zip(dev_samples, dev_sc_traces)):
            tau = traces[0]
            g, coverage, parser_risk, syntax_valid = GraphBuilder.build_graph(task_name, tau)
            g_seq = HEMCoarsener.coarsen(g, max_scale_K=self.settings.coarsening.max_scale_K)
            
            # 【改进】接入 extra_diag
            r_struct, r_cache, h_curve, extra_diag = self.risk_estimator.estimate_structural_risk(g_seq, task_name)
            r_robust = self.risk_estimator.estimate_robustness_risk(
                tau, task_name, GraphBuilder.build_graph, HEMCoarsener.coarsen, h_curve
            )

            log_v = math.log(g.number_of_nodes() + 1)
            dev_feature_matrix.append([r_struct, r_robust, parser_risk, log_v])
            
            is_wrong = str(sample.gold_answer).strip() not in tau
            dev_labels.append(is_wrong)

        # 4. 拟合监督混合标定器并导出白盒数据
        calibrator = SupervisedHybridCalibrator(l2_penalty=0.1, lr=0.05, epochs=300)
        calibrator.fit(dev_feature_matrix, dev_labels)

        # 缓存标定器诊断量
        self.calibrator_prior_base_rate = calibrator.prior_base_rate
        self.calibrator_weights = calibrator.weights.tolist() if calibrator.weights is not None else []
        self.calibrator_bias = calibrator.bias

        # 5. 测试集路由诊断 (Test Set)
        logger.info("Running Batch-level cost-aware routing on Test Set...")
        test_routing_inputs: List[Dict[str, Any]] = []

        for idx, (sample, traces) in enumerate(zip(test_samples, test_sc_traces)):
            tau = traces[0]
            g, coverage, parser_risk, syntax_valid = GraphBuilder.build_graph(task_name, tau)
            g_seq = HEMCoarsener.coarsen(g, max_scale_K=self.settings.coarsening.max_scale_K)
            
            # 【改进】接入 extra_diag 并捕获中间原始物理流
            r_struct, r_cache, h_curve, extra_diag = self.risk_estimator.estimate_structural_risk(g_seq, task_name)
            r_robust = self.risk_estimator.estimate_robustness_risk(
                tau, task_name, GraphBuilder.build_graph, HEMCoarsener.coarsen, h_curve
            )

            log_v = math.log(g.number_of_nodes() + 1)
            x_test = [r_struct, r_robust, parser_risk, log_v]
            p_wrong = calibrator.calibrate(x_test)
            r_ise = p_wrong

            test_routing_inputs.append({
                "sample_id": sample.sample_id,
                "p_wrong": p_wrong,
                "syntax_valid": syntax_valid,
                "coverage": coverage,
                "trajectory_text": tau,
                "raw_question": sample.raw_question,
                "gold_answer": sample.gold_answer,
                "normalized_entropy_curve": h_curve,
                "cache_statuses": extra_diag.get("cache_statuses", []),
                "r_ise_score": r_ise,
                "task_type": task_name,
                # 【核心改进】携带原始底层特征供 Tracker 落盘
                "scale_wise_raw_entropies": extra_diag.get("raw_entropies"),
                "scale_wise_null_means": extra_diag.get("null_means"),
                "scale_wise_null_stds": extra_diag.get("null_stds"),
                "scale_wise_lookup_distances": extra_diag.get("lookup_distances"),
                "mcmc_fallback_count": extra_diag.get("mcmc_fallback_count", 0)
            })

        active_verifiers = ["exact_answer_checker"] if task_name in ["GSM8K", "MATH"] else ["python_syntax_checker"]

        router = BatchRouter(
            verifier_mgr=self.verifier_mgr,
            budget_limit=self.settings.cost_budget.budget_vector_20,
            loss_err=10.0,
            loss_abs=2.0,
            coverage_threshold_zeta=self.settings.safety_gate.coverage_threshold_zeta
        )
        routing_decisions = router.route_batch(test_routing_inputs, active_verifiers=active_verifiers)

        # 6. 执行物理校验与 Tracker 原子记录
        id_to_sample_input = {item["sample_id"]: item for item in test_routing_inputs}
        for idx, decision in enumerate(routing_decisions):
            s_id = decision["sample_id"]
            action = decision["action"]
            v_id = decision["verifier_allocated"]
            item = id_to_sample_input[s_id]

            is_correct = str(item["gold_answer"]).strip() in item["trajectory_text"]

            if action == "verify" and v_id:
                passed = self.verifier_mgr.verifiers[v_id].execute(item["trajectory_text"], item["gold_answer"])
                is_correct = passed 

            self.tracker.log_run(
                sample_id=s_id,
                task_type=item["task_type"],
                raw_question=item["raw_question"],
                trajectory_text=item["trajectory_text"],
                is_syntax_valid=item["syntax_valid"],
                parser_coverage=item["coverage"],
                parser_risk=item["r_ise_score"],
                normalized_entropy_curve=item["normalized_entropy_curve"],
                cache_statuses=item["cache_statuses"],
                r_ise_score=item["r_ise_score"],
                calibrated_p_wrong=item["p_wrong"],
                action_taken=action,
                verifier_allocated=v_id,
                actual_cost_vector=decision["actual_cost_vector"],
                is_correct=is_correct,
                # 【核心改进】传入原子诊断流
                scale_wise_raw_entropies=item.get("scale_wise_raw_entropies"),
                scale_wise_null_means=item.get("scale_wise_null_means"),
                scale_wise_null_stds=item.get("scale_wise_null_stds"),
                scale_wise_lookup_distances=item.get("scale_wise_lookup_distances"),
                mcmc_fallback_count=item.get("mcmc_fallback_count", 0)
            )

        analyzer = ScientificSignalAnalyzer(self.tracker)
        audit_report = analyzer.audit_m2_scientific_signal()
        return audit_report