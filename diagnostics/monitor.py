# -*- coding: utf-8 -*-
"""ISE-Flow 全生命周期实验运行诊断看板生成器.

本模块读取 Tracker 生成的 JSONL 诊断文件，将其转化为直观、可交互、
带颜色高亮的全流程决策轨迹审计报表（Diagnostics Dashboard）。
"""

import os
import json
import glob
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger("ise_flow.diagnostics.monitor")


class ExecutionMonitor:
    """全流程运行诊断监控器."""

    @staticmethod
    def generate_html_dashboard(log_file_path: str, output_dir: str = "diagnostics/plots") -> str:
        """读取日志，生成高可视化的 Markdown / HTML 端到端运行状态监控看板."""
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"Target log file not found: {log_file_path}")

        session_id = os.path.basename(log_file_path).replace(".jsonl", "")
        dashboard_path = os.path.join(output_dir, f"execution_monitor_{session_id}.md")
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据
        records: List[Dict[str, Any]] = []
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        # 统计基础状态
        total_eval = len(records)
        action_counts = {"trust": 0, "verify": 0, "abstain": 0}
        success_counts = {"trust": 0, "verify": 0, "abstain": 0}
        syntax_errors = 0

        for r in records:
            act = r["action_taken"]
            action_counts[act] = action_counts.get(act, 0) + 1
            if r.get("is_correct") is True:
                success_counts[act] = success_counts.get(act, 0) + 1
            if r.get("is_syntax_valid") == 0:
                syntax_errors += 1

        # 写入 Markdown 仪表盘
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(f"# 📊 ISE-Flow v6 Full-Process Execution Monitor Dashboard\n")
            f.write(f"- **Audit Session ID**: `{session_id}`\n")
            f.write(f"- **Generation Time**: `{datetime.now().isoformat()}`\n")
            f.write(f"- **Total Evaluated Trajectories**: **{total_eval}**\n\n")

            # 饼图式数据面板
            f.write(f"### 📈 1. 运行状态概览 (Global Status Summary)\n")
            f.write(f"| 决策动作 (Action) | 触发频数 (Count) | 占比 (Ratio) | 校验通过率 (Pass Rate) |\n")
            f.write(f"| :--- | :---: | :---: | :---: |\n")
            for act in ["trust", "verify", "abstain"]:
                count = action_counts[act]
                ratio = count / total_eval if total_eval > 0 else 0.0
                pass_rate = success_counts[act] / count if count > 0 else 0.0
                f.write(f"| `{act.upper()}` | {count} | {ratio:.2%} | {pass_rate:.2%} |\n")
            f.write(f"\n- **语法不完整/截断退化样本数 (Truncated/Syntax Invalid)**: **{syntax_errors}** (退化率: {syntax_errors/total_eval:.2%})\n\n")

            # 端到端决策链大追踪表
            f.write(f"### 🔍 2. 样本端到端决策链追踪面板 (End-to-End Decision Chain Panel)\n")
            f.write(f"> **阅读指南**：追踪每一个轨迹从底层物理指标（文本长、图阶数等），经过 R_struct、R_ISE，校准为 P_wrong，最后输出决策并获得外部验证的完整全生命周期。\n\n")
            
            f.write(
                "| 样本 ID | 文本长度 | 语法状态 | 归一化熵曲线 $\\mathbf{h}(\\tau)$ | 综合风险 $R_{ISE}$ | 标定错误率 $P_{wrong}$ | 决策动作 | 物理结果 |\n"
            )
            f.write(
                "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |\n"
            )

            for r in records:
                syntax_status = "🟢 合法" if r["is_syntax_valid"] == 1 else "🔴 截断退化"
                result_status = "✅ 正确" if r["is_correct"] is True else "❌ 错误"
                
                # 动作颜色高亮
                act = r["action_taken"].upper()
                if act == "TRUST":
                    act_display = f"**🟢 {act}**"
                elif act == "VERIFY":
                    act_display = f"**🔵 {act}**"
                else:
                    act_display = f"**🟡 {act}**"

                # 提取曲线简写
                curve_str = " ➜ ".join([f"{h:.2f}" for h in r["normalized_entropy_curve"]])

                f.write(
                    f"| {r['sample_id']} | {r['trajectory_len']} | {syntax_status} | {curve_str} | "
                    f"{r['r_ise_score']:.4f} | {r['calibrated_p_wrong']:.4f} | {act_display} | {result_status} |\n"
                )

        logger.info(f"Full-process dashboard exported successfully to: {dashboard_path}")
        return dashboard_path


# 让 analyzer 自动联动触发
# 在 ise_flow/diagnostics/analyzer.py 的 audit_m2_scientific_signal 方法末尾补充：
# ...
# self.generate_visualizations()
# self.export_failure_analysis()
# from ise_flow.diagnostics.monitor import ExecutionMonitor
# ExecutionMonitor.generate_html_dashboard(self.tracker.log_file_path)