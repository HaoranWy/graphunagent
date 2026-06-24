# -*- coding: utf-8 -*-
"""ISE-Flow 高可解析性轨迹 Prompt 模板.

本模块定义了 L1 推理、代码和形式化任务的高可解构提示模板。
模板中通过显式注入时序标签、语法隔离符与最终答案包裹块，
确保生成的推理轨迹 (τ) 能无阻碍地映射到相应图 Schema (G_chain/G_formal) [PDF Page 3, 10]。
"""

from typing import Dict, Any


class PromptTemplates:
    """静态 Prompt 模板库与格式约束指令集.

    此类提供了 GSM8K、MATH、Python (HumanEval+/MBPP+) 以及 SMT-LIB
    四大 L1 主线任务的提示生成函数。
    """

    # =====================================================================
    # 1. GSM8K (自然语言多步数学推理)
    # =====================================================================
    GSM8K_SYSTEM: str = (
        "You are an expert mathematical reasoning agent. Your task is to solve the given grade-school "
        "math question step-by-step. To ensure your reasoning chain is highly verifiable, you must "
        "divide your steps into explicit, sequential blocks labeled with '[STEP X]' and '[/STEP X]', "
        "where X is the step number. Write the final numerical answer inside a LaTeX '\\boxed{}' block.\n\n"
        "Strictly follow this output format:\n"
        "[STEP 1]\n"
        "<Your first logical reasoning and calculation step here>\n"
        "[/STEP 1]\n"
        "[STEP 2]\n"
        "<Your second step building on step 1>\n"
        "[/STEP 2]\n"
        "...\n"
        "Therefore, the final answer is \\boxed{<final numerical answer>}"
    )

    GSM8K_USER_TEMPLATE: str = (
        "Question: {question}\n\n"
        "Please provide your step-by-step reasoning chain and final boxed answer below:"
    )

 
    # =====================================================================
    # 2. MATH (复杂多分支数学推理)
    # =====================================================================
    MATH_SYSTEM: str = (
        "You are an elite research mathematician. Solve the following advanced mathematics problem "
        "step-by-step. For verification safety, you must segment your logical reasoning, definitions, "
        "and calculations into numbered step blocks starting with '[STEP X]' and ending with '[/STEP X]'. "
        "Write the final simplified result enclosed strictly within '\\boxed{}' (e.g., \\boxed{\\frac{3}{4}}).\n\n"
        "Strictly follow this output format:\n"
        "[STEP 1]\n"
        "<State your variables, assumptions, or algebraic setups>\n"
        "[/STEP 1]\n"
        "[STEP 2]\n"
        "<Perform derivations or expansions>\n"
        "[/STEP 2]\n"
        "...\n"
        "Therefore, the final answer is \\boxed{<final simplified mathematical result>}"
    )

    MATH_USER_TEMPLATE: str = (
        "Problem: {problem}\n\n"
        "Solve step-by-step, dividing your chain into logical step blocks, and box your final answer:"
    )

    # =====================================================================
    # 3. Python AST Code Generation (HumanEval+ / MBPP+)
    # =====================================================================
    PYTHON_CODE_SYSTEM: str = (
        "You are a master Python programmer. Your task is to implement the requested function "
        "based on its docstring and signature. To ensure your generated code can be parsed into an "
        "Abstract Syntax Tree (AST) without syntax failures, you must only output a single, complete "
        "Python code block enclosed inside standard markdown ```python ... ``` tags.\n\n"
        "Constraints:\n"
        "- Do not output any conversational introduction or post-hoc explanation.\n"
        "- Do not write comments outside the python code block.\n"
        "- The code block must be syntactically valid and import all necessary modules.\n\n"
        "Example Output Format:\n"
        "```python\n"
        "def solve(x: int) -> int:\n"
        "    # implementation\n"
        "    return x + 1\n"
        "```"
    )

    PYTHON_CODE_USER_TEMPLATE: str = (
        "Function Stub and Requirements:\n"
        "{signature}\n"
        "\"\"\"\n"
        "{docstring}\n"
        "\"\"\"\n\n"
        "Please generate only the python code block solving this requirements:"
    )

    # =====================================================================
    # 4. SMT-LIB v2 Formula Generation (形式化约束求解)
    # =====================================================================
    SMT_LIB_SYSTEM: str = (
        "You are a formal verification specialist. Your task is to translate the given logical "
        "constraints or puzzle into a syntactically correct SMT-LIB v2 script compatible with solvers "
        "like Z3. To allow dependency parsing of your assertions, structure your script explicitly.\n\n"
        "Requirements:\n"
        "- Output exactly one code block enclosed in ```smt2 ... ``` tags.\n"
        "- Explicitly declare all logic variables using `declare-fun`.\n"
        "- Separate logical conditions into explicit `assert` statements.\n"
        "- End the script with `check-sat` and optionally `get-model`.\n"
        "- Do not write natural language descriptions outside the code block.\n\n"
        "Example Output Format:\n"
        "```smt2\n"
        "(declare-fun x () Int)\n"
        "(assert (> x 10))\n"
        "(check-sat)\n"
        "```"
    )

    SMT_LIB_USER_TEMPLATE: str = (
        "Logical Requirements to represent in SMT-LIB v2:\n"
        "{specifications}\n\n"
        "Generate the formal smt2 script below:"
    )


def get_gsm8k_prompt(question: str) -> Dict[str, str]:
    """获取格式化后的 GSM8K 双端 Prompt 字典.

    Args:
        question: 原始数学问题。

    Returns:
        包含 "system" 和 "user" 字段的字符串映射表。
    """
    return {
        "system": PromptTemplates.GSM8K_SYSTEM,
        "user": PromptTemplates.GSM8K_USER_TEMPLATE.format(question=question)
    }


def get_math_prompt(problem: str) -> Dict[str, str]:
    """获取格式化后的 MATH 双端 Prompt 字典.

    Args:
        problem: 原始复杂数学难题。

    Returns:
        包含 "system" 和 "user" 字段的字符串映射表。
    """
    return {
        "system": PromptTemplates.MATH_SYSTEM,
        "user": PromptTemplates.MATH_USER_TEMPLATE.format(problem=problem)
    }


def get_python_prompt(signature: str, docstring: str) -> Dict[str, str]:
    """获取格式化后的 Python AST 代码生成双端 Prompt 字典.

    Args:
        signature: 目标函数定义（函数名与入参出参提示）。
        docstring: 详尽的单元测试说明与功能 docstring。

    Returns:
        包含 "system" 和 "user" 字段的字符串映射表。
    """
    return {
        "system": PromptTemplates.PYTHON_CODE_SYSTEM,
        "user": PromptTemplates.PYTHON_CODE_USER_TEMPLATE.format(
            signature=signature,
            docstring=docstring
        )
    }


def get_smt_prompt(specifications: str) -> Dict[str, str]:
    """获取格式化后的 SMT-LIB 逻辑翻译双端 Prompt 字典.

    Args:
        specifications: 形式化规格说明、命题或前置假设。

    Returns:
        包含 "system" 和 "user" 字段的字符串映射表。
    """
    return {
        "system": PromptTemplates.SMT_LIB_SYSTEM,
        "user": PromptTemplates.SMT_LIB_USER_TEMPLATE.format(specifications=specifications)
    }