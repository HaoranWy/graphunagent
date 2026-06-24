# -*- coding: utf-8 -*-
"""ISE-Flow 语法解析与 EHS 归一化器模块 (修复 ParsingError 导入冲突)."""

import ast
import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("ise_flow.data.parsers")


class ParsingError(Exception):
    """自定义图结构解析异常.
    
    当形式化（AST / SMT）代码块存在致命语法错误、格式截断
    且无法强行构建 formal graph 时抛出，用于触发上层 fallback 机制。
    """
    pass


class EHSNormalizer(ast.NodeTransformer):
    """熵同态拓扑骨架归一化器 (Entropic Homomorphic Skeletonizer).

    通过拓扑作用域重命名、交换律子树排序和线性控制流骨架化收缩，
    消去图规模（长度）带来的非物理噪声。
    """

    def __init__(self) -> None:
        self.scope_stack: List[Dict[str, str]] = [{}]
        self.depth: int = 0
        self.counter: int = 0
        self.whitelist = {"self", "cls", "math", "List", "Dict", "Tuple", "Optional", "Set", "Any"}

    def _get_std_id(self, original_id: str) -> str:
        if original_id in dir(__builtins__) or original_id in self.whitelist:
            return original_id
        for scope in reversed(self.scope_stack):
            if original_id in scope:
                return scope[original_id]
        
        self.counter += 1
        std_id = f"v_{self.depth}_{self.counter}"
        self.scope_stack[-1][original_id] = std_id
        return self.counter_node(std_id)

    def counter_node(self, std_id: str) -> str:
        return std_id

    def _enter_scope(self) -> None:
        self.depth += 1
        self.scope_stack.append({})

    def _exit_scope(self) -> None:
        self.depth -= 1
        self.scope_stack.pop()

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if isinstance(node.ctx, (ast.Store, ast.Load)):
            node.id = self._get_std_id(node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._get_std_id(node.arg)
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        node = self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)):
            try:
                left_src = ast.unparse(node.left)
                right_src = ast.unparse(node.right)
                if left_src > right_src:
                    node.left, node.right = node.right, node.left
            except Exception:
                pass
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.BoolOp:
        node = self.generic_visit(node)
        if isinstance(node.op, (ast.And, ast.Or)):
            try:
                node.values.sort(key=lambda x: ast.unparse(x))
            except Exception:
                pass
        return node

    def _skeletonize_body(self, body_nodes: List[ast.AST]) -> List[ast.AST]:
        new_body = []
        linear_buffer: List[ast.AST] = []

        for node in body_nodes:
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Return, ast.Yield, ast.FunctionDef, ast.ClassDef)):
                if linear_buffer:
                    combined_src = "; ".join(ast.unparse(n) for n in linear_buffer)
                    virtual_node = ast.Assign(
                        targets=[ast.Name(id="v_linear_block", ctx=ast.Store())],
                        value=ast.Constant(value=combined_src)
                    )
                    new_body.append(virtual_node)
                    linear_buffer = []
                new_body.append(self.visit(node))
            else:
                linear_buffer.append(node)

        if linear_buffer:
            combined_src = "; ".join(ast.unparse(n) for n in linear_buffer)
            virtual_node = ast.Assign(
                targets=[ast.Name(id="v_linear_block", ctx=ast.Store())],
                value=ast.Constant(value=combined_src)
            )
            new_body.append(virtual_node)

        return new_body

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._skeletonize_body(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = self._get_std_id(node.name)
        self._enter_scope()
        if node.args:
            node.args = self.visit(node.args)
        node.body = self._skeletonize_body(node.body)
        self._exit_scope()
        return node


def ehs_normalize_code(source_code: str) -> str:
    """熵同态拓扑骨架归一化器统一外部调用入口."""
    try:
        tree = ast.parse(source_code)
        transformer = EHSNormalizer()
        normalized_tree = transformer.visit(tree)
        ast.fix_missing_locations(normalized_tree)
        return ast.unparse(normalized_tree)
    except Exception:
        return source_code


class TrajectoryParser:
    """通用语义轨迹解析器."""

    STEP_PATTERN = re.compile(r"\[STEP\s+(\d+)\](.*?)\[/STEP\s+\1\]", re.DOTALL | re.IGNORECASE)
    BOXED_PATTERN = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
    PYTHON_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

    @classmethod
    def parse_cot_steps(cls, text: str) -> Tuple[List[Dict[str, Any]], Optional[str], float]:
        if not text:
            return [], None, 0.0

        steps: List[Dict[str, Any]] = []
        matches = list(cls.STEP_PATTERN.finditer(text))
        total_len = len(text)
        parsed_chars = 0

        if matches:
            for match in matches:
                step_num = int(match.group(1))
                content = match.group(2).strip()
                parsed_chars += len(match.group(0))
                steps.append({
                    "node_id": f"step_{step_num}",
                    "payload": {"content": content, "step_index": step_num}
                })
        else:
            raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
            step_idx = 1
            for line in raw_lines:
                if "\\boxed" in line and len(line) < 100:
                    continue
                steps.append({
                    "node_id": f"step_fallback_{step_idx}",
                    "payload": {"content": line, "step_index": step_idx}
                })
                parsed_chars += len(line)
                step_idx += 1

        ans_match = cls.BOXED_PATTERN.search(text)
        final_answer = ans_match.group(1).strip() if ans_match else None
        
        coverage = parsed_chars / (total_len + 1e-6)
        return steps, final_answer, min(1.0, coverage)

    @classmethod
    def parse_python_ast(cls, code_text: str) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        match = cls.PYTHON_BLOCK_PATTERN.search(code_text)
        source_code = match.group(1) if match else code_text
        
        # 在解析 Python AST 时，前置执行 EHS 归一化对齐拓扑
        normalized_code = ehs_normalize_code(source_code)

        try:
            tree = ast.parse(normalized_code)
        except SyntaxError as se:
            raise ParsingError(f"AST SyntaxError: line {se.lineno}, {se.msg}")

        nodes: List[Dict[str, Any]] = []
        edges: List[Tuple[str, str]] = []

        class ASTVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_stmt_id = None
                self.stmt_idx = 1

            def visit_FunctionDef(self, node: ast.FunctionDef):
                func_id = f"func_{node.name}"
                nodes.append({
                    "node_id": func_id,
                    "node_type": "AST_STMT",
                    "payload": {"name": node.name}
                })
                for arg in node.args.args:
                    arg_id = f"sym_arg_{arg.arg}"
                    nodes.append({
                        "node_id": arg_id,
                        "node_type": "AST_SYMBOL",
                        "payload": {"name": arg.arg}
                    })
                    edges.append((func_id, arg_id))
                
                old_stmt = self.current_stmt_id
                self.current_stmt_id = func_id
                self.generic_visit(node)
                self.current_stmt_id = old_stmt

            def visit_Assign(self, node: ast.Assign):
                stmt_id = f"assign_{self.stmt_idx}"
                self.stmt_idx += 1
                nodes.append({
                    "node_id": stmt_id,
                    "node_type": "AST_STMT",
                    "payload": {}
                })
                if self.current_stmt_id:
                    edges.append((self.current_stmt_id, stmt_id))
                
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        target_id = f"sym_var_{target.id}"
                        nodes.append({
                            "node_id": target_id,
                            "node_type": "AST_SYMBOL",
                            "payload": {"name": target.id}
                        })
                        edges.append((stmt_id, target_id))
                self.generic_visit(node)

        visitor = ASTVisitor()
        visitor.visit(tree)
        return nodes, edges