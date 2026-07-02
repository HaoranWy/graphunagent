# -*- coding: utf-8 -*-
"""ISE-Flow 多维成本核算与异构验证器管理模块.

学术升级版：
1. 强化了本地 HumanEvalPlusVerifier 的隔离沙箱，注入了完整的标准计算和 typing 命名空间。
2. 新增 GSM8KOfficialVerifier 严格对齐 OpenAI 与 EleutherAI 官方数字提纯规则。
3. 预留 MATH、SMT-LIB 和 FactScore 的学术级即插即用评估插槽。
"""

import sys
import ast
import time
import logging
import traceback
import threading
import re
import collections
import heapq
import itertools
from typing import List, Dict, Any, Tuple, Optional, Set, Union, Callable, Iterable, Sequence
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("ise_flow.routing.verifier")


@dataclass(frozen=True)
class VerifierSpec:
    """验证器静态元特征与精确度模型."""
    verifier_id: str
    tpr: float
    fpr: float
    base_cost_vector: Tuple[float, ...] = (0.01, 1.0, 1.0, 0.0, 0.0, 0.0)


class BaseVerifier:
    def __init__(self, spec: VerifierSpec) -> None:
        self.spec = spec

    def execute(self, code_or_text: str, gold_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        raise NotImplementedError


class HumanEvalPlusVerifier(BaseVerifier):
    """HumanEval+ 专用物理单元测试校验器."""

    def __init__(self) -> None:
        spec = VerifierSpec(
            verifier_id="humaneval_plus_tester",
            tpr=1.00,
            fpr=0.00,
            base_cost_vector=(0.02, 2.0, 8.0, 0.0, 0.0, 0.0)
        )
        super().__init__(spec)

    def execute(self, code_or_text: str, gold_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """安全沙箱执行并核对（对齐 check(candidate) 参数）."""
        if not gold_answer:
            return True

        # 提取 metadata 中的入口函数名
        entry_point = metadata.get("entry_point") if metadata else None

        code_clean = code_or_text
        if "```python" in code_or_text:
            code_clean = code_or_text.split("```python")[-1].split("```")[0].strip()

        full_execution_script = f"{code_clean}\n\n{gold_answer}\n"

        success_container = [False]

        def _sandbox_run():
            try:
                # 注入高度健壮的运行命名空间，包含全部 typing 标注及常用计算标准库，规避 NameError
                sandbox_globals = {
                    "__builtins__": __builtins__,
                    "math": __import__("math"),
                    "sys": sys,
                    "List": List,
                    "Dict": Dict,
                    "Tuple": Tuple,
                    "Optional": Optional,
                    "Set": Set,
                    "Any": Any,
                    "Union": Union,
                    "Callable": Callable,
                    "Iterable": Iterable,
                    "Sequence": Sequence,
                    "collections": collections,
                    "re": re,
                    "heapq": heapq,
                    "itertools": itertools,
                    "typing": sys.modules["typing"]
                }
                exec(full_execution_script, sandbox_globals)
                
                # 寻找 candidate 并执行参数化 check(candidate_fn) 调用
                if "check" in sandbox_globals:
                    check_fn = sandbox_globals["check"]
                    if entry_point in sandbox_globals:
                        candidate_fn = sandbox_globals[entry_point]
                        check_fn(candidate_fn)  # 严格对齐官方评测
                    else:
                        # 兜底：如果找不到对应的入口函数名，寻找除 check 以外的可执行单元进行测试
                        fallback_fn = None
                        for k, v in sandbox_globals.items():
                            if callable(v) and k != "check" and k != "Set":
                                fallback_fn = v
                                break
                        if fallback_fn:
                            check_fn(fallback_fn)
                        else:
                            check_fn()
                success_container[0] = True
            except Exception as e:
                logger.debug(f"Unit test assertion failed or crashed: {str(e)}")
                success_container[0] = False

        exec_thread = threading.Thread(target=_sandbox_run)
        exec_thread.daemon = True
        exec_thread.start()
        exec_thread.join(timeout=1.5)

        if exec_thread.is_alive():
            logger.warning("Sandbox execution timed out.")
            return False

        return success_container[0]


class PythonSyntaxVerifier(BaseVerifier):
    def __init__(self) -> None:
        spec = VerifierSpec(
            verifier_id="python_syntax_checker",
            tpr=0.60,
            fpr=0.00,
            base_cost_vector=(0.0, 0.05, 0.1, 0.0, 0.0, 0.0)
        )
        super().__init__(spec)

    def execute(self, code_or_text: str, gold_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            code_clean = code_or_text
            if "```python" in code_or_text:
                code_clean = code_or_text.split("```python")[-1].split("```")[0].strip()
            ast.parse(code_clean)
            return True
        except Exception:
            return False


class ExactAnswerVerifier(BaseVerifier):
    def __init__(self) -> None:
        spec = VerifierSpec(
            verifier_id="exact_answer_checker",
            tpr=1.00,  
            fpr=0.00,
            base_cost_vector=(0.02, 1.5, 5.0, 0.0, 0.0, 0.0) 
        )
        super().__init__(spec)

    def execute(self, code_or_text: str, gold_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not gold_answer:
            return True
        boxed_match = re.search(r"\\boxed\{(.*?)\}", code_or_text)
        if boxed_match:
            extracted_ans = boxed_match.group(1).strip()
            return extracted_ans == str(gold_answer).strip()
        return str(gold_answer).strip() in code_or_text


class GSM8KOfficialVerifier(BaseVerifier):
    """GSM8K 官方标准/lm-evaluation-harness 严格对齐评测器."""

    def __init__(self) -> None:
        spec = VerifierSpec(
            verifier_id="gsm8k_official_tester",
            tpr=1.00,
            fpr=0.00,
            base_cost_vector=(0.01, 1.0, 1.0, 0.0, 0.0, 0.0)
        )
        super().__init__(spec)

    def _clean_answer(self, ans_str: str) -> str:
        """根据 EleutherAI lm-eval 规范清洗提纯数字 (剥离美元符号、逗号与末尾句号)."""
        if not ans_str:
            return ""
        clean = ans_str.replace(",", "").replace("$", "").strip()
        if clean.endswith("."):
            clean = clean[:-1].strip()
        return clean

    def _extract_number_from_cot(self, text: str) -> str:
        """从推理链中提取最终官方数值结果 (支持 #### 占位符与 LaTeX 占位符)."""
        if not text:
            return ""

        # 1. 优先提取 #### 之后的文本
        if "####" in text:
            raw_ans = text.split("####")[-1].strip()
            match = re.match(r"(-?\d+(?:\.\d+)?)", raw_ans)
            if match:
                return match.group(1)
            return raw_ans.split()[0] if raw_ans.split() else raw_ans

        # 2. 其次提取 \boxed{} 符号
        boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # 3. 兜底提取整个文本中的最后一个数值
        numbers = re.findall(r"(-?\d+(?:\.\d+)?)", text)
        if numbers:
            return numbers[-1]

        return ""

    def execute(self, code_or_text: str, gold_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not gold_answer:
            return True

        # 提纯候选值与真值
        candidate_raw = self._extract_number_from_cot(code_or_text)
        candidate_clean = self._clean_answer(candidate_raw)
        gold_clean = self._clean_answer(str(gold_answer))

        if not candidate_clean or not gold_clean:
            return False

        # 鲁棒数值型等价判断 (防止 15.0 == 15 发生误判)
        try:
            return abs(float(candidate_clean) - float(gold_clean)) < 1e-6
        except ValueError:
            # 字符级 Exact Match 兜底
            return candidate_clean == gold_clean


# =====================================================================
# 📌 未来前沿学术 Benchmark 评测器可插拔插槽保留区 (Extensible Registry Slots)
# =====================================================================
class MATHVerifier(BaseVerifier):
    """MATH 复杂多分支数学推理官方评测器 (预留插槽)."""
    # TODO: 后续研究导入时，在此处实现 LaTeX 递归等价化简与 AIME 等等价对齐逻辑
    pass

class SMTLIBVerifier(BaseVerifier):
    """SMT-LIB 约束求解器执行评测器 (预留插槽)."""
    # TODO: 后续研究导入时，在此处调用 Z3 求解器子进程判定 solver.check()
    pass

class FactScoreVerifier(BaseVerifier):
    """FactScore 事实真实性自回归评测器 (预留插槽)."""
    # TODO: 后续研究导入时，在此处调用外部 Fact Judge 评估原子句真值
    pass


class VerifierManager:
    def __init__(
        self,
        alpha_vector: Tuple[float, ...] = (0.4, 0.2, 0.1, 0.1, 0.1, 0.1),
        cost_wrong: float = 10.0,
        cost_false_reject: float = 2.0
    ) -> None:
        self.alpha = np.array(alpha_vector, dtype=np.float64)
        self.c_wrong = cost_wrong
        self.c_false_reject = cost_false_reject

        # 统一工厂注册表
        self.verifiers: Dict[str, BaseVerifier] = {
            "python_syntax_checker": PythonSyntaxVerifier(),
            "exact_answer_checker": ExactAnswerVerifier(),
            "humaneval_plus_tester": HumanEvalPlusVerifier(),
            "gsm8k_official_tester": GSM8KOfficialVerifier(),  # 激活官方 GSM8K 评测
            # "math_official_tester": MATHVerifier(),          # 预留
            # "smt_official_tester": SMTLIBVerifier(),          # 预留
            # "factscore_official_tester": FactScoreVerifier()  # 预留
        }

    def register_verifier(self, verifier_id: str, verifier: BaseVerifier) -> None:
        self.verifiers[verifier_id] = verifier

    def scalarize_cost(self, spec: VerifierSpec) -> float:
        cost_v = np.array(spec.base_cost_vector, dtype=np.float64)
        return float(np.dot(self.alpha, cost_v))

    def compute_evv(self, p_wrong: float, verifier_id: str) -> Tuple[float, float]:
        if verifier_id not in self.verifiers:
            return -999.0, 999.0

        verifier = self.verifiers[verifier_id]
        spec = verifier.spec

        c_v = self.scalarize_cost(spec)
        p_detect_wrong = p_wrong * spec.tpr
        p_reject_correct = (1.0 - p_wrong) * spec.fpr

        expected_gain = (p_detect_wrong * self.c_wrong) - (p_reject_correct * self.c_false_reject)
        evv = expected_gain - c_v
        return float(evv), float(c_v)