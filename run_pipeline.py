# -*- coding: utf-8 -*-
"""ISE-Flow 启动与学术调度评测流水线 (CSS-CCR 动态追踪与物理长度独立高阶融合版)."""

import sys
import random
import argparse
import logging
import math
import subprocess
import tempfile
import json
import os
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from config.settings import ExperimentSettings, CostBudgetConfig
from data.loaders import DatasetLoader, TaskSample
from llm.generator import APIGenerator
from data.graph_builder import GraphBuilder
from data.parsers import TrajectoryParser
from core.coarsening import HEMCoarsener
from core.null_cache import NullCacheManager
from core.risk_estimator import RiskEstimator
from core.stcg import StateTransitionConsensusGraph
from routing.calibration import BinningCalibrator, UnsupervisedConsensusAligner, RobustWeaklySupervisedCalibrator
from routing.verifier_manager import VerifierManager
from routing.router import BatchRouter
from diagnostics.tracker import TrajectoryTracker
from diagnostics.analyzer import ScientificSignalAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ise_flow.cli")


def check_required_package(package_name: str) -> bool:
    """检查特定学术评测包是否安装。若未安装，则输出醒目的终端警告看板。"""
    import importlib.util
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        return True
        
    print("\n" + "!" * 80)
    print(f" [ENVIRONMENT WARNING] Required benchmark library '{package_name}' is MISSING.")
    print(f" To run this evaluation under the official engine, please install it manually:")
    print(f" >>> pip install {package_name} --upgrade")
    print(f" Note: If you are running in an offline environment, we will auto-detect and bind your")
    print(f" local datasets. But the executable package '{package_name}' must be installed first.")
    print("!" * 80 + "\n")
    return False


def auto_configure_offline_datasets() -> None:
    """自动探测本地离线数据集，并将其注册进系统的环境变量，强制 evalplus 停止联网下载尝试。"""
    # 依据 loaders.py 的默认地址进行探测
    base_dir_from_loader = "/data/newmodel/graphunagent/datasets/humanevalplus/human-eval/data"
    
    possible_paths = [
        os.path.join(base_dir_from_loader, "HumanEvalPlus.jsonl.gz"),
        os.path.join(base_dir_from_loader, "HumanEvalPlus.jsonl"),
        "/data/newmodel/graphunagent/datasets/humanevalplus/human-eval/data/HumanEvalPlus.jsonl.gz"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            os.environ["HUMANEVAL_OVERRIDE_PATH"] = path
            logger.info(f"Successfully locked offline local HumanEvalPlus dataset: {path}")
            return
            
    logger.warning(
        "Could not detect local offline HumanEvalPlus.jsonl.gz in standard directories. "
        "If you run into HF connection errors, make sure to download HumanEvalPlus.jsonl.gz and "
        "place it inside your humanevalplus/data/ folder."
    )


class ExperimentRunner:

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

        self.calibrator_prior_base_rate = 0.0
        self.calibrator_weights: List[float] = []
        self.calibrator_bias = 0.0
        self.unsupervised_polarities: List[int] = []
        self.theta_01 = 0.0
        self.theta_10 = 0.0

    def _extract_stcg_features(self, task_name: str, traces: List[str]) -> Tuple[float, float]:
        parsed_traces = []
        for trace_text in traces:
            steps, _, _ = TrajectoryParser.parse_cot_steps(trace_text)
            step_texts = [step["payload"].get("content", "") for step in steps]
            parsed_traces.append(step_texts)
        
        stcg = StateTransitionConsensusGraph(task_type=task_name)
        stcg.build_consensus_graph(parsed_traces)
        return stcg.compute_spectral_gap(), stcg.compute_pagerank_entropy()

    def execute_l1_pipeline(self, task_name: str = "GSM8K", sample_limit: int = 30, eval_engine: str = "auto") -> Tuple[Dict[str, Any], str]:
        """运行 L1 流水线并自适应路由至指定学术评测引擎."""
        
        # 1. 确定最终激活的评测后端
        selected_engine = eval_engine
        if selected_engine == "auto":
            if task_name == "HumanEvalPlus":
                selected_engine = "evalplus"
            elif task_name == "GSM8K":
                selected_engine = "gsm8k_exact"
            else:
                selected_engine = "custom_sandbox"
                
        # 2. 检查特定依赖
        if selected_engine == "evalplus":
            if not check_required_package("evalplus"):
                logger.warning("Failing back safely to 'custom_sandbox' evaluation due to environment absence.")
                selected_engine = "custom_sandbox"
            else:
                # 若安装了官方 evalplus 包，执行自动化的本地离线数据锁死机制
                auto_configure_offline_datasets()

        logger.info(f"============ Executing L1 Pipeline | Task: {task_name} | Resolved Engine: {selected_engine.upper()} ============")

        # 3. 物理加载学术数据集
        if task_name == "GSM8K":
            dev_samples, test_samples = self.loader.load_gsm8k_split()
            if sample_limit and sample_limit > 0:
                dev_limit = max(5, int(sample_limit * 0.4))
                test_limit = max(5, int(sample_limit * 0.6))
                dev_samples = dev_samples[:dev_limit]
                test_samples = test_samples[:test_limit]
        elif task_name == "HumanEvalPlus":
            test_samples = self.loader.load_humanevalplus_test()
            dev_samples = []
            if sample_limit and sample_limit > 0:
                test_samples = test_samples[:sample_limit]
        else:
            raise NotImplementedError(f"Task {task_name} is not supported.")

        logger.info(f"Generating real-time trajectories from LLM... Sample size: {len(test_samples)}")
        test_prompts = [s.raw_question for s in test_samples]
        test_sc_traces = self.generator.generate_consistency_samples(test_prompts, num_samples=3, task_type=task_name)

        # 诊断生成文本
        if test_sc_traces and len(test_sc_traces) > 0:
            sample_trace = test_sc_traces[0][0]
            logger.info(f"[DIAGNOSTICS] Sample Generated Trajectory Length: {len(sample_trace)} chars.")
            logger.info(f"[DIAGNOSTICS] Sample Content Preview:\n{sample_trace[:200]}\n...")
        else:
            logger.error("[DIAGNOSTICS] Generated trajectories list is EMPTY.")

        use_unsupervised_rank = (task_name == "HumanEvalPlus")
        rank_calibrator = BinningCalibrator(num_bins=10)

        # 监督开发集拟合
        if not use_unsupervised_rank and dev_samples:
            dev_prompts = [s.raw_question for s in dev_samples]
            dev_sc_traces = self.generator.generate_consistency_samples(dev_prompts, num_samples=3, task_type=task_name)
            
            dev_feature_matrix, dev_labels = [], []
            for sample, traces in zip(dev_samples, dev_sc_traces):
                tau = traces[0]
                g, coverage, parser_risk, syntax_valid = GraphBuilder.build_graph(task_name, tau)
                g_seq = HEMCoarsener.coarsen(g, max_scale_K=self.settings.coarsening.max_scale_K)
                r_struct, _, h_curve, _ = self.risk_estimator.estimate_structural_risk(g_seq, task_name, trajectory_text=tau)
                r_robust = self.risk_estimator.estimate_robustness_risk(tau, task_name, GraphBuilder.build_graph, HEMCoarsener.coarsen, h_curve)
                spec_gap, pr_entropy = self._extract_stcg_features(task_name, traces)
                
                # 【物理长度修改点 1】将第四维度的 log(V) 升级为物理文本长度对数 log(L)
                dev_feature_matrix.append([r_struct, r_robust, parser_risk, math.log(len(tau) + 1), spec_gap, pr_entropy])
                dev_labels.append(str(sample.gold_answer).strip() not in tau)

            dev_feature_matrix, self.unsupervised_polarities = UnsupervisedConsensusAligner.align_polarities(dev_feature_matrix, anchor_index=5)
            calibrator = RobustWeaklySupervisedCalibrator()
            calibrator.fit(dev_feature_matrix, dev_labels)
            
            self.calibrator_prior_base_rate = calibrator.prior_base_rate
            self.calibrator_weights = calibrator.weights.tolist() if calibrator.weights is not None else []
            self.calibrator_bias = calibrator.bias
            self.theta_01 = calibrator.theta_01
            self.theta_10 = calibrator.theta_10

        # 测试集特征处理
        test_routing_inputs = []
        raw_test_features = []
        for sample, traces in zip(test_samples, test_sc_traces):
            tau = traces[0]
            g, coverage, parser_risk, syntax_valid = GraphBuilder.build_graph(task_name, tau)
            g_seq = HEMCoarsener.coarsen(g, max_scale_K=self.settings.coarsening.max_scale_K)
            
            r_struct, r_cache, h_curve, extra_diag = self.risk_estimator.estimate_structural_risk(
                g_seq, 
                task_name, 
                trajectory_text=tau,
                gold_answer=sample.gold_answer if task_name == "HumanEvalPlus" else "",
                entry_point=sample.prompt_data.get("entry_point", "") if task_name == "HumanEvalPlus" else ""
            )
            r_robust = self.risk_estimator.estimate_robustness_risk(tau, task_name, GraphBuilder.build_graph, HEMCoarsener.coarsen, h_curve)
            spec_gap, pr_entropy = self._extract_stcg_features(task_name, traces)
            
            # 【物理长度修改点 2】将第四维度的 log(V) 升级为物理文本长度对数 log(L)
            raw_test_features.append([r_struct, r_robust, parser_risk, math.log(len(tau) + 1), spec_gap, pr_entropy])
            test_routing_inputs.append({
                "sample_id": sample.sample_id,
                "syntax_valid": syntax_valid,
                "coverage": coverage,
                "parser_risk": parser_risk,
                "trajectory_text": tau,
                "raw_question": sample.raw_question,
                "gold_answer": sample.gold_answer,
                "normalized_entropy_curve": h_curve,
                "cache_statuses": extra_diag.get("cache_statuses", []),
                "task_type": task_name,
                "scale_wise_raw_entropies": extra_diag.get("raw_entropies"),
                "scale_wise_null_means": extra_diag.get("null_means"),
                "scale_wise_null_stds": extra_diag.get("null_stds"),
                "scale_wise_lookup_distances": extra_diag.get("lookup_distances"),
                "mcmc_fallback_count": extra_diag.get("mcmc_fallback_count", 0),
                "effective_dimension": extra_diag.get("effective_dimension", 1.0),
                "r_cache": r_cache,
                "entry_point": sample.prompt_data.get("entry_point", ""),
                "spectral_gap": spec_gap,
                "pagerank_entropy": pr_entropy
            })

        # 物理极性强锁定
        self.unsupervised_polarities = [1, 1, 1, 1, -1, 1]
        aligned_test_features = []
        for feat in raw_test_features:
            aligned_feat = [f * p for f, p in zip(feat, self.unsupervised_polarities)]
            aligned_test_features.append(aligned_feat)

        if use_unsupervised_rank:
            all_r_ise_scores = []
            for features_aligned, item in zip(aligned_test_features, test_routing_inputs):
                # 物理极性对齐线性组合 (保留 20% 物理长度不确定性先验)
                r_ise_raw = (
                    0.30 * features_aligned[0] + # 100% 纯上同调代数阻碍
                    0.15 * features_aligned[1] + 
                    0.15 * features_aligned[2] + 
                    0.20 * features_aligned[3] + # 物理文本长度对数 log(L) 的决策面融合
                    0.10 * features_aligned[4] + 
                    0.10 * features_aligned[5]
                )
                all_r_ise_scores.append(r_ise_raw)
                item["r_ise_raw"] = r_ise_raw

            rank_calibrator.fit_rank_fallback(all_r_ise_scores)
            for item in test_routing_inputs:
                p_wrong = rank_calibrator.calibrate_rank(item["r_ise_raw"])
                item["p_wrong"] = p_wrong
                item["r_ise_score"] = p_wrong

            median_score = np.median(all_r_ise_scores)
            y_pseudo = np.array([score > median_score for score in all_r_ise_scores], dtype=np.float64)
            temp_calibrator = RobustWeaklySupervisedCalibrator()
            X_scaled_temp = (np.array(aligned_test_features) - np.mean(aligned_test_features, axis=0)) / (np.std(aligned_test_features, axis=0) + 1e-6)
            temp_calibrator._estimate_noise_with_kde(X_scaled_temp, y_pseudo)
            self.theta_01 = temp_calibrator.theta_01
            self.theta_10 = temp_calibrator.theta_10
        else:
            for features_aligned, item in zip(aligned_test_features, test_routing_inputs):
                p_wrong = calibrator.calibrate(features_aligned)
                item["p_wrong"] = p_wrong
                item["r_ise_score"] = p_wrong

        # 官方 evalplus 子进程触发执行逻辑
        official_eval_map = {}
        if selected_engine == "evalplus":
            logger.info("Executing official HumanEval+ (evalplus.evaluate) benchmark test suite...")
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w", encoding="utf-8") as tmp_file:
                temp_jsonl_name = tmp_file.name
                for item in test_routing_inputs:
                    code_block = item["trajectory_text"].split("```python")[-1].split("```")[0].strip() if "```python" in item["trajectory_text"] else item["trajectory_text"]
                    tmp_file.write(json.dumps({
                        "task_id": item["sample_id"],
                        "completion": code_block
                    }) + "\n")
            
            try:
                cmd = ["python", "-m", "evalplus.evaluate", "--dataset", "humaneval", "--samples", temp_jsonl_name]
                # 捕获子进程标准输出和标准错误输出，进行环境追踪
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(f"evalplus subprocess executed successfully. Raw stdout:\n{result.stdout[:600]}\n...")
                
                result_json_path = temp_jsonl_name.replace(".jsonl", "_eval_results.json")
                if os.path.exists(result_json_path):
                    with open(result_json_path, "r", encoding="utf-8") as rf:
                        eval_data = json.load(rf)
                    
                    eval_dict = eval_data.get("eval", {})
                    logger.info(f"[DIAGNOSTICS] Loaded {len(eval_dict)} records from official evalplus results.")
                    
                    # 强兼容性的自适应多 Key 解析器（同时处理 base_status, plus_status 与 status）
                    for raw_id, test_runs in eval_dict.items():
                        norm_id = str(raw_id).replace("_", "/").strip() # 归一化为 "HumanEval/163" 格式
                        
                        passed = True
                        for run in test_runs:
                            # 提取可用的状态原语
                            st = run.get("status")
                            base_st = run.get("base_status")
                            plus_st = run.get("plus_status")
                            
                            # 只要 base/plus/status 中有任何一个不为 "pass"，则说明存在测试用例失效
                            if base_st is not None and base_st != "pass":
                                passed = False
                            if plus_st is not None and plus_st != "pass":
                                passed = False
                            if st is not None and st != "pass":
                                passed = False
                                
                        official_eval_map[norm_id] = passed
                    
                    total_runs = len(official_eval_map)
                    passed_runs = sum(1 for val in official_eval_map.values() if val is True)
                    logger.info(f"[DIAGNOSTICS] Normalized Official Map Status: Passed {passed_runs} / Total {total_runs}")
                else:
                    logger.warning(f"evalplus results file missing at {result_json_path}. Fallback to custom_sandbox.")
                    selected_engine = "custom_sandbox"
            except Exception as se:
                logger.error(f"evalplus subprocess invocation failed: {str(se)}.")
                if hasattr(se, 'stdout') and se.stdout:
                    logger.error(f"[DIAGNOSTICS] Subprocess stdout:\n{se.stdout}")
                if hasattr(se, 'stderr') and se.stderr:
                    logger.error(f"[DIAGNOSTICS] Subprocess stderr:\n{se.stderr}")
                logger.warning(" [FALLBACK DETECTED] Force-switching active engine to 'custom_sandbox'!")
                selected_engine = "custom_sandbox"
            finally:
                for p in [temp_jsonl_name, temp_jsonl_name.replace(".jsonl", "_eval_results.json")]:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass

        # 路由选择分配
        active_verifiers = ["humaneval_plus_tester"] if task_name == "HumanEvalPlus" else ["exact_answer_checker"]
        router = BatchRouter(
            verifier_mgr=self.verifier_mgr,
            budget_limit=self.settings.cost_budget.budget_vector_20,
            loss_err=10.0,
            loss_abs=2.0,
            coverage_threshold_zeta=self.settings.safety_gate.coverage_threshold_zeta
        )
        routing_decisions = router.route_batch(test_routing_inputs, active_verifiers=active_verifiers)

        # 日志原子追踪写入
        id_to_sample_input = {item["sample_id"]: item for item in test_routing_inputs}
        correct_count = 0
        incorrect_count = 0
        missing_count = 0

        for decision in routing_decisions:
            s_id = decision["sample_id"]
            action = decision["action"]
            v_id = decision["verifier_allocated"]
            item = id_to_sample_input[s_id]

            # 强兼容性的双向 ID 匹配
            normalized_s_id = str(s_id).replace("_", "/").strip()
            alternate_s_id = str(s_id).replace("/", "_").strip()

            if selected_engine == "evalplus" and (normalized_s_id in official_eval_map or alternate_s_id in official_eval_map):
                matched_id = normalized_s_id if normalized_s_id in official_eval_map else alternate_s_id
                is_correct = official_eval_map[matched_id]
            elif task_name == "HumanEvalPlus" and selected_engine == "custom_sandbox":
                is_correct = self.verifier_mgr.verifiers["humaneval_plus_tester"].execute(
                    item["trajectory_text"], 
                    item["gold_answer"],
                    metadata={"entry_point": item.get("entry_point", "")}
                )
            elif task_name == "GSM8K":
                is_correct = self.verifier_mgr.verifiers["exact_answer_checker"].execute(item["trajectory_text"], item["gold_answer"])
            else:
                if selected_engine == "evalplus":
                    missing_count += 1
                is_correct = self.verifier_mgr.verifiers[active_verifiers[0]].execute(
                    item["trajectory_text"], 
                    item["gold_answer"],
                    metadata={"entry_point": item.get("entry_point", "")}
                )

            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1

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
                scale_wise_raw_entropies=item.get("scale_wise_raw_entropies"),
                scale_wise_null_means=item.get("scale_wise_null_means"),
                scale_wise_null_stds=item.get("scale_wise_null_stds"),
                scale_wise_lookup_distances=item.get("scale_wise_lookup_distances"),
                mcmc_fallback_count=item.get("mcmc_fallback_count", 0),
                effective_dimension=item.get("effective_dimension", 1.0),
                spectral_gap=item.get("spectral_gap", 0.0),
                pagerank_entropy=item.get("pagerank_entropy", 0.0),
                theta_01=self.theta_01,
                theta_10=self.theta_10
            )

        logger.info(
            f"[DIAGNOSTICS] Pipeline Evaluation Finished. "
            f"True Correct (Pass): {correct_count} | True Incorrect (Fail): {incorrect_count} | "
            f"Official Map Missing Matches: {missing_count}"
        )

        analyzer = ScientificSignalAnalyzer(self.tracker)
        audit_report = analyzer.audit_m2_scientific_signal()
        return audit_report, selected_engine


def print_academic_report(
    task_type: str, 
    report: Dict[str, Any], 
    budget_ratio: str, 
    session_id: str,
    runner: ExperimentRunner,
    resolved_engine: str
) -> None:
    print("\n" + "=" * 80)
    print(f"               ISE-Flow v6 REAL Academia Evaluation Report")
    print(f"               Task: {task_type} | Verifier Budget Ratio: {budget_ratio}% | Resolved Engine: {resolved_engine.upper()}")
    print("=" * 80)
    
    print("\n### 📖 新增拓扑物理与逻辑共识指标 (State-Transition Spectral Indicators)")
    print("- **Avg Spectral Gap (代数连通度/谱间隙 λ2)**: 衡量多路逻辑状态的连通共识力。")
    print("- **Avg PageRank Entropy (共识谱熵)**: 评估并行状态转移路径的发散不确定性。")
    print("- **Estimated Noise Channel (KDE 弱监督噪声矩阵)**: 在线自适应估算判定器的假阳性率 (theta_01) 和假阴性率 (theta_10)。")

    print("\n### 🪐 8. PCFG-AST EHS-Normalizer 性能诊断指标")
    ehs = report.get("ehs_metrics", {})
    print(f"- **Avg Entropy Decimation Ratio (EDR 熵压缩比)**: {ehs.get('entropy_decimation_ratio_edr', 'N/A')} (值越大代表骨架化效果越强)")
    print(f"- **Length Decoupling Index (LDI 长度解耦指数)**: {ehs.get('length_decoupling_index_ldi', 'N/A')} (越接近 1.0 说明完全剔成了行数偏置)")
    print(f"- **Semantic State Overlap (SSO 语义状态重合度)**: {ehs.get('semantic_state_overlap_sso', 'N/A')} (值越小代表局部重命名去重率越高)")

    print("\n### 1. Stratified Performance Table (分层拓扑质量表)")
    print("| Cohort / Subset | Sample Size | R_ISE AUROC | Length Baseline AUROC | Gain (Δ AUROC) |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    all_s = report.get("all_samples", {})
    if all_s:
        gain_all = all_s.get("auroc_r_ise", 0.5) - all_s.get("auroc_length_baseline", 0.5)
        print(f"| All Samples (全量) | {all_s.get('size')} | {all_s.get('auroc_r_ise'):.4f} | {all_s.get('auroc_length_baseline'):.4f} | {gain_all:+.4f} |")
    
    sv_s = report.get("syntax_valid_only", {})
    if sv_s and sv_s.get("status") != "SKIPPED":
        print(f"| Syntax-Valid Only (语法合法) | {sv_s.get('size')} | {sv_s.get('auroc_r_ise'):.4f} | {sv_s.get('auroc_length_baseline'):.4f} | {sv_s.get('auroc_gain_vs_len'):+.4f} |")
    else:
         print(f"| Syntax-Valid Only | N/A (Skipped) | - | - | - |")
         
    lm_s = report.get("length_matched_subset", {})
    if lm_s and lm_s.get("status") != "SKIPPED":
         print(f"| Length-Matched Subset (长度匹配) | {lm_s.get('size')} | {lm_s.get('auroc_r_ise'):.4f} | N/A (Matched) | - |")
    else:
        print(f"| Length-Matched Subset | N/A (Skipped) | - | - | - |")

    print("\n### 2. Multi-Scale Topology & Calibration Diagnostics (多尺度拓扑物理审计)")
    audit = report.get("topology_audit", {})
    print(f"- **Avg Effective Dimension (流形谱维度 d_eff)**: {audit.get('avg_effective_dimension', 'N/A')}")
    print(f"- **Matcher Label Noise (MCI 标签匹配污染率)**: {all_s.get('matcher_label_noise_mci', 'N/A')}")
    print(f"- **Pearson Correlation Corr(log(L), R_struct)**: {all_s.get('correlation_length_structural_risk', 'N/A')}")
    print(f"- **Average Scale 5 Null Std (临界标准差稳定下限)**: {audit.get('avg_scale_5_null_std', 'N/A')}")
    print(f"- **Average KNN Lookup Distance (平均近邻匹配几何距离)**: {audit.get('avg_lookup_distance', 'N/A')}")

    print("\n### 📊 3. Spectral Consensus & Label Noise Audit (逻辑一致性谱共识审计)")
    print(f"- **Average Spectral Gap (谱间隙/代数连通度 λ2)**: {audit.get('avg_spectral_gap', 'N/A')}")
    print(f"- **Average PageRank Entropy (谱共识状态熵)**: {audit.get('avg_pagerank_entropy', 'N/A')}")
    print(f"- **Estimated Label Noise Channel (核密度自抗扰信道矩阵)**:")
    print(f"  * 假阳性率 FPR (theta_01): {runner.theta_01:.4f}")
    print(f"  * 假阴性率 FNR (theta_10): {runner.theta_10:.4f}")

    print("\n### 4. Probability Calibration Error (概率校准误差)")
    cal = report.get("calibration", {})
    print(f"- **Expected Calibration Error (ECE)**: {cal.get('ece_score', 'N/A')}")
    print(f"- **Brier Score (BS)**: {cal.get('brier_score', 'N/A')}")

    print("\n### 7. Go/No-go Scientific Decision Audit (预实验科学决策审计)")
    decision = report.get("global_go_decision", "NO-GO")
    print(f"- **M2 Go-Condition 1 (Syntax-Valid Gain >= 0.03)**: {'PASS' if sv_s.get('go_condition_1_pass') else 'FAIL'}")
    print(f"- **M2 Go-Condition 2 (Length-Matched AUC > 0.50)**: {'PASS' if lm_s.get('go_condition_2_pass') else 'FAIL'}")
    
    color_code = "\033[92m" if decision == "GO" else "\033[91m"
    print(f"- **Final Academic Recommendation (决策结论)**: {color_code}{decision}\033[0m")
    print("\n" + "=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ISE-Flow CLI Run Harness.")
    parser.add_argument("--task", type=str, default="GSM8K", choices=["GSM8K", "HumanEvalPlus"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budget_ratio", type=str, default="20", choices=["20", "30"])
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--model_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", type=str, default="deepseek")
    parser.add_argument("--api_key", type=str, default="empty")
    
    # 细粒度引擎自动配置参数
    parser.add_argument("--eval_engine", type=str, default="auto", 
                        choices=["auto", "evalplus", "gsm8k_exact", "custom_sandbox"],
                        help="Specify the benchmark engine: 'evalplus' for HumanEval+, 'gsm8k_exact' for GSM8K, 'custom_sandbox' for thread sandbox, or 'auto' for auto-alignment.")

    args = parser.parse_args()
    set_global_seeds(args.seed)

    cb_config = CostBudgetConfig()
    custom_cost_budget = CostBudgetConfig(
        alpha_vector=cb_config.alpha_vector,
        budget_vector_20=cb_config.budget_vector_20,
        budget_vector_30=cb_config.budget_vector_30,
        pre_routing_ratio=cb_config.pre_routing_ratio
    )
    exp_settings = ExperimentSettings(
        cost_budget=custom_cost_budget,
        random_seed=args.seed,
        use_calibration=True
    )

    runner = ExperimentRunner(
        settings=exp_settings,
        model_name=args.model_name,
        model_url=args.model_url,
        api_key=args.api_key
    )

    try:
        report, resolved_engine = runner.execute_l1_pipeline(
            task_name=args.task, 
            sample_limit=args.limit,
            eval_engine=args.eval_engine
        )
        print_academic_report(
            args.task, 
            report, 
            args.budget_ratio, 
            runner.tracker.session_id,
            runner,
            resolved_engine
        )
    except Exception as e:
        logger.critical(f"Global pipeline crash: {str(e)}", exc_info=True)
        sys.exit(1)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"System random seed globally locked to: {seed}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()