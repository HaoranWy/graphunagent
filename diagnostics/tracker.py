# -*- coding: utf-8 -*-
"""ISE-Flow 轨迹运行指标追踪器.

升级版：
1. 新增记录状态转移一致性图谱指标（spectral_gap, pagerank_entropy）。
2. 新增记录自抗扰标定器估计的弱监督观测噪声参数（theta_01, theta_10）。
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ise_flow.diagnostics.tracker")


class TrajectoryTracker:
    """样本级运行时指标与中间状态追踪器."""

    def __init__(self, log_dir: str = "diagnostics/logs", session_id: Optional[str] = None) -> None:
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        if session_id is None:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_id = f"{now_str}_ise_run"
        else:
            self.session_id = session_id

        self.log_file_path = os.path.join(self.log_dir, f"{self.session_id}.jsonl")
        logger.info(f"Diagnostics tracker initialized. Target session path: {self.log_file_path}")

    def log_run(
        self,
        sample_id: str,
        task_type: str,
        raw_question: str,
        trajectory_text: str,
        is_syntax_valid: int,
        parser_coverage: float,
        parser_risk: float,
        normalized_entropy_curve: List[float],
        cache_statuses: List[str],
        r_ise_score: float,
        calibrated_p_wrong: float,
        action_taken: str,
        verifier_allocated: Optional[str],
        actual_cost_vector: List[float],
        is_correct: Optional[bool] = None,
        scale_wise_raw_entropies: Optional[List[float]] = None,
        scale_wise_null_means: Optional[List[float]] = None,
        scale_wise_null_stds: Optional[List[float]] = None,
        scale_wise_lookup_distances: Optional[List[float]] = None,
        mcmc_fallback_count: int = 0,
        effective_dimension: float = 1.0,
        # 【新增改进】记录状态一致性图特征
        spectral_gap: float = 0.0,
        pagerank_entropy: float = 0.0,
        # 【新增改进】记录标定器弱监督噪声评估参数
        theta_01: float = 0.0,
        theta_10: float = 0.0
    ) -> None:
        record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": sample_id,
            "task_type": task_type,
            "raw_question": raw_question,
            "trajectory_text": trajectory_text,
            "raw_question_len": len(raw_question),
            "trajectory_len": len(trajectory_text),
            "is_syntax_valid": is_syntax_valid,
            "parser_coverage": round(parser_coverage, 4),
            "parser_risk": round(parser_risk, 4),
            "normalized_entropy_curve": [round(h, 4) for h in normalized_entropy_curve],
            "cache_statuses": cache_statuses,
            "r_ise_score": round(r_ise_score, 4),
            "calibrated_p_wrong": round(calibrated_p_wrong, 4),
            "action_taken": action_taken,
            "verifier_allocated": verifier_allocated,
            "actual_cost_vector": [round(c, 4) for c in actual_cost_vector],
            "is_correct": is_correct,
            "scale_wise_raw_entropies": [round(h, 4) for h in scale_wise_raw_entropies] if scale_wise_raw_entropies else [],
            "scale_wise_null_means": [round(m, 4) for m in scale_wise_null_means] if scale_wise_null_means else [],
            "scale_wise_null_stds": [round(s, 4) for s in scale_wise_null_stds] if scale_wise_null_stds else [],
            "scale_wise_lookup_distances": [round(d, 4) for d in scale_wise_lookup_distances] if scale_wise_lookup_distances else [],
            "mcmc_fallback_count": mcmc_fallback_count,
            "effective_dimension": round(effective_dimension, 4),
            # 【新增落盘】
            "spectral_gap": round(spectral_gap, 4),
            "pagerank_entropy": round(pagerank_entropy, 4),
            "theta_01": round(theta_01, 4),
            "theta_10": round(theta_10, 4)
        }

        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"Failed to write diagnostic log for sample {sample_id}: {str(e)}")

    def get_session_logs(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_file_path):
            return []
        logs: List[Dict[str, Any]] = []
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line.strip()))
            return logs
        except Exception as e:
            logger.error(f"Error reading session logs: {str(e)}")
            return []