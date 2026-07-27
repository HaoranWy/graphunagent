"""Fail-closed verifiers for the supported benchmark domains.

Execution-based verifiers run in a child process with a hard timeout.  They
are evaluation tools only; none of them is consulted by PWSC-CI while scoring a
response.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import multiprocessing as mp
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from core.adapters import (
    bounded_math_verify,
    canonical_logic_label,
    canonical_math_answer,
    extract_last_boxed,
    extract_python_code,
)

logger = logging.getLogger("ise_flow.routing.verifier")


@dataclass(frozen=True)
class VerifierSpec:
    verifier_id: str
    tpr: float = 1.0
    fpr: float = 0.0
    base_cost_vector: Tuple[float, ...] = (0.01, 1.0, 1.0, 0.0, 0.0, 0.0)


def _exec_program_worker(program: str, result_queue: Any) -> None:
    try:
        namespace: Dict[str, Any] = {"__name__": "__main__"}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(program, namespace)
        result_queue.put(True)
    except BaseException:
        result_queue.put(False)


def _lcb_worker(code: str, input_output: Dict[str, Any], result_queue: Any) -> None:
    try:
        # Import the official evaluator by file path.  Importing the package
        # initializer on Python 3.10 executes a Python-3.11-only
        # ``sys.set_int_max_str_digits`` call before testing_util is reached.
        import importlib.util

        path = "/data/newmodel/graphunagent/datasets/livecodebench/LiveCodeBench/lcb_runner/evaluation/testing_util.py"
        spec = importlib.util.spec_from_file_location("pwsc_lcb_testing_util", path)
        if spec is None or spec.loader is None:
            result_queue.put(False)
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run_test = module.run_test

        sample = {"input_output": json.dumps(input_output)}
        results, _metadata = run_test(sample, test=code, timeout=6)
        passed = bool(results) and all(item is True for item in results)
        result_queue.put(passed)
    except BaseException:
        result_queue.put(False)


def _run_child(target: Any, args: Tuple[Any, ...], timeout: float) -> bool:
    """Run an untrusted evaluator child and return false on timeout/error."""
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover - Windows fallback
        ctx = mp.get_context()
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=target, args=(*args, queue))
    process.daemon = True
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join()
        return False
    try:
        return bool(queue.get_nowait())
    except Exception:
        return False


class BaseVerifier:
    def __init__(self, spec: VerifierSpec) -> None:
        self.spec = spec

    def execute(
        self,
        code_or_text: str,
        gold_answer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        raise NotImplementedError


class HumanEvalPlusVerifier(BaseVerifier):
    def __init__(self) -> None:
        super().__init__(VerifierSpec("humaneval_plus_tester", base_cost_vector=(0.02, 2.0, 8.0, 0.0, 0.0, 0.0)))

    def execute(self, code_or_text: str, gold_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        metadata = metadata or {}
        entry = str(metadata.get("entry_point", ""))
        tests = str(gold_answer or "")
        if not entry or not tests.strip() or not extract_python_code(code_or_text).strip():
            return False
        program = (
            extract_python_code(code_or_text)
            + "\n\n"
            + f"candidate = {entry}\n"
            + tests
            + "\n\ncheck(candidate)\n"
        )
        return _run_child(_exec_program_worker, (program,), timeout=8.0)


class MBPPPlusVerifier(BaseVerifier):
    """Execute the released MBPP+ tests after scoring has finished."""

    def __init__(self) -> None:
        super().__init__(
            VerifierSpec(
                "mbpp_plus_tester",
                base_cost_vector=(0.02, 2.0, 8.0, 0.0, 0.0, 0.0),
            )
        )

    def execute(
        self,
        code_or_text: str,
        gold_answer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        code = extract_python_code(code_or_text)
        tests = str(gold_answer or "")
        if not code.strip() or not tests.strip():
            return False
        return _run_child(
            _exec_program_worker,
            (code + "\n\n" + tests,),
            timeout=8.0,
        )


class MATHOfficialVerifier(BaseVerifier):
    def __init__(self) -> None:
        super().__init__(VerifierSpec("math_official_tester", base_cost_vector=(0.01, 1.5, 3.0, 0.0, 0.0, 0.0)))

    @staticmethod
    def _answer(text: str) -> str:
        return canonical_math_answer(text)

    def execute(self, code_or_text: str, gold_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not str(gold_answer or "").strip():
            return False
        candidate, gold = self._answer(code_or_text), self._answer(str(gold_answer))
        if not candidate or not gold:
            return False
        if candidate.lower() == gold.lower():
            return True
        if bounded_math_verify(gold, candidate):
            return True
        # Conservative numeric fallback; symbolic strings require exact
        # normalized equality rather than unsafe eval().
        try:
            return math.isclose(float(candidate), float(gold), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False


class FOLIOVerifier(BaseVerifier):
    def __init__(self) -> None:
        super().__init__(VerifierSpec("folio_tester", base_cost_vector=(0.01, 0.5, 1.0, 0.0, 0.0, 0.0)))

    def execute(self, code_or_text: str, gold_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        gold = canonical_logic_label(str(gold_answer or ""))
        pred = canonical_logic_label(code_or_text)
        return gold in {"true", "false", "uncertain"} and pred == gold


class LiveCodeBenchVerifier(BaseVerifier):
    def __init__(self) -> None:
        super().__init__(VerifierSpec("lcb_tester", base_cost_vector=(0.03, 2.5, 10.0, 0.0, 0.0, 0.0)))

    def execute(self, code_or_text: str, gold_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        metadata = metadata or {}
        input_output = metadata.get("input_output")
        if not isinstance(input_output, dict):
            return False
        if not input_output.get("inputs") or not input_output.get("outputs"):
            return False
        code = extract_python_code(code_or_text)
        if not code.strip():
            return False
        return _run_child(_lcb_worker, (code, input_output), timeout=12.0)


class GSM8KOfficialVerifier(BaseVerifier):
    def __init__(self) -> None:
        super().__init__(VerifierSpec("gsm8k_official_tester", base_cost_vector=(0.01, 1.0, 1.0, 0.0, 0.0, 0.0)))

    @staticmethod
    def _number(text: str) -> str:
        value = str(text or "")
        if "####" in value:
            value = value.rsplit("####", 1)[1]
        boxed = extract_last_boxed(value)
        if boxed is not None:
            value = boxed
        matches = re.findall(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        return matches[-1] if matches else value.strip().rstrip(".")

    def execute(self, code_or_text: str, gold_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        gold = self._number(str(gold_answer or ""))
        pred = self._number(code_or_text)
        if not gold or not pred:
            return False
        try:
            return math.isclose(float(pred), float(gold), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return pred == gold


class VerifierManager:
    def __init__(self, alpha_vector: Tuple[float, ...] = (0.4, 0.2, 0.1, 0.1, 0.1, 0.1)) -> None:
        self.alpha = alpha_vector
        self.verifiers = {
            "humaneval_plus_tester": HumanEvalPlusVerifier(),
            "mbpp_plus_tester": MBPPPlusVerifier(),
            "math_official_tester": MATHOfficialVerifier(),
            "folio_tester": FOLIOVerifier(),
            "lcb_tester": LiveCodeBenchVerifier(),
            "gsm8k_official_tester": GSM8KOfficialVerifier(),
        }

    def execute(self, verifier_id: str, text: str, gold_answer: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        if verifier_id not in self.verifiers:
            raise KeyError(f"Unknown verifier: {verifier_id}")
        return self.verifiers[verifier_id].execute(text, gold_answer, metadata)

    def compute_evv(self, calibrated_p_wrong: float, verifier_id: str) -> Tuple[float, float]:
        """Legacy router utility, valid only for calibrated error probabilities."""
        probability = float(calibrated_p_wrong)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("calibrated_p_wrong must lie in [0, 1]")
        verifier = self.verifiers[verifier_id]
        cost = sum(
            float(weight) * float(component)
            for weight, component in zip(self.alpha, verifier.spec.base_cost_vector)
        )
        post_verify_error = probability * (1.0 - verifier.spec.tpr) + (1.0 - probability) * verifier.spec.fpr
        return float(probability - post_verify_error - cost), float(cost)


__all__ = ["BaseVerifier", "VerifierManager", "VerifierSpec"]
