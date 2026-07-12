"""Task adapters used by PWSC.

The adapter is deliberately small and deterministic.  It owns only answer
canonicalisation, validity checks, and the feature map used by the consensus
kernel; it never reads a gold label.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_CODE_TASKS = {"CODE", "HUMANEVALPLUS", "LIVECODEBENCH", "HUMANEVAL"}
_LOGIC_TASKS = {"FOLIO", "P-FOLIO", "PFOLIO", "SMT"}
_MATH_TASKS = {"GSM8K", "MATH", "MATH-500", "AIME", "AIME2025", "AIME2026"}


def _task_family(schema_name: str) -> str:
    name = str(schema_name or "").upper()
    if name in _CODE_TASKS or "CODE" in name:
        return "code"
    if name in _LOGIC_TASKS or "FOLIO" in name or "LOGIC" in name:
        return "logic"
    if name in _MATH_TASKS or "MATH" in name or "AIME" in name or "GSM" in name:
        return "math"
    return "text"


def extract_python_code(text: str) -> str:
    """Return the first fenced Python block, or the complete response."""
    text = str(text or "")
    fenced = re.findall(r"```(?:python|py)\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        return fenced[-1].strip()
    return text.strip()


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract the last balanced ``\\boxed{...}`` expression."""
    text = str(text or "")
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    brace = text.find("{", start)
    depth = 0
    for idx in range(brace, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : idx].strip()
    return text[brace + 1 :].strip() or None


def canonical_logic_label(text: str) -> str:
    """Use the last explicit three-way logic label in a response."""
    value = str(text or "").lower()
    matches = list(re.finditer(r"\b(true|false|uncertain|t|f|u)\b", value))
    if not matches:
        return "unknown"
    label = matches[-1].group(1)
    return {"t": "true", "f": "false", "u": "uncertain"}.get(label, label)


def canonical_math_answer(text: str) -> str:
    """Create a conservative string key for a mathematical final answer."""
    raw = str(text or "")
    value = extract_last_boxed(raw)
    if value is None:
        answer_matches = re.findall(
            r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\n]+)", raw, flags=re.I
        )
        if answer_matches:
            value = answer_matches[-1]
        elif re.search(r"\\(?:frac|sqrt|left|right|pi|theta|cdot|sum|int|begin)", raw):
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            value = lines[-1] if lines else raw
        else:
            numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", raw.replace(",", ""))
            value = numbers[-1] if numbers else (raw.splitlines()[-1] if raw.splitlines() else raw)
    value = value.replace("$", "")
    if re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?", value):
        value = value.replace(",", "")
    value = re.sub(r"\s+", "", value).strip().rstrip(".")
    return value.lower() or "unknown"


def normalized_code_key(text: str) -> str:
    """A structural key for code baselines, excluding locations and names."""
    source = extract_python_code(text)
    try:
        tree = ast.parse(source)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except (SyntaxError, ValueError, TypeError):
        return "invalid:" + re.sub(r"\s+", " ", source).strip()


_AST_FEATURE_TYPES: Tuple[type, ...] = (
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Return,
    ast.Yield,
    ast.Call,
    ast.BinOp,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Subscript,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
    ast.Assert,
    ast.Match,
)


def code_ast_feature(text: str) -> Tuple[np.ndarray, bool]:
    """Return a fixed operator/control histogram and parse validity."""
    source = extract_python_code(text)
    vector = np.zeros(len(_AST_FEATURE_TYPES), dtype=np.float64)
    if not source:
        return vector, False
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return vector, False
    positions = {node_type: idx for idx, node_type in enumerate(_AST_FEATURE_TYPES)}
    for node in ast.walk(tree):
        idx = positions.get(type(node))
        if idx is not None:
            vector[idx] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector, True


def _entry_point_present(text: str, entry_point: str) -> bool:
    if not entry_point:
        return True
    try:
        tree = ast.parse(extract_python_code(text))
    except (SyntaxError, ValueError, TypeError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in ast.walk(tree)
    )


def _math_equivalent(left: str, right: str) -> bool:
    """Best-effort semantic equality used only to group sampled answers."""
    if left == right:
        return True
    try:
        from math_verify import parse, verify

        return bool(verify(parse(left), parse(right)))
    except Exception:
        return False


def transitive_classes(keys: Sequence[str], semantic: bool = False) -> List[int]:
    """Build transitive equivalence classes (union-find, not pairwise labels)."""
    n = len(keys)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i):
            if keys[i] == keys[j] or (semantic and _math_equivalent(keys[i], keys[j])):
                union(i, j)
    roots: Dict[int, int] = {}
    labels: List[int] = []
    for idx in range(n):
        root = find(idx)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return labels


@dataclass
class TaskAdapter:
    schema_name: str
    metadata: Sequence[Optional[Dict[str, Any]]]

    @property
    def family(self) -> str:
        return _task_family(self.schema_name)

    def outcome_keys(self, texts: Sequence[str]) -> List[str]:
        if self.family == "code":
            return [normalized_code_key(text) for text in texts]
        if self.family == "logic":
            return [canonical_logic_label(text) for text in texts]
        if self.family == "math":
            return [canonical_math_answer(text) for text in texts]
        return [re.sub(r"\s+", " ", str(text or "").strip().lower()) for text in texts]

    def valid_mask(self, texts: Sequence[str]) -> np.ndarray:
        mask = np.ones(len(texts), dtype=bool)
        for idx, text in enumerate(texts):
            if not str(text or "").strip():
                mask[idx] = False
                continue
            if self.family == "code":
                feature, parsed = code_ast_feature(text)
                meta = self.metadata[idx] if idx < len(self.metadata) else None
                entry = str((meta or {}).get("entry_point", ""))
                mask[idx] = parsed and np.linalg.norm(feature) > 0.0 and _entry_point_present(text, entry)
        return mask

    def feature_matrix(self, texts: Sequence[str], valid_mask: np.ndarray) -> np.ndarray:
        n = len(texts)
        if self.family == "code":
            matrix = np.zeros((n, len(_AST_FEATURE_TYPES)), dtype=np.float64)
            for idx, text in enumerate(texts):
                if valid_mask[idx]:
                    matrix[idx], _ = code_ast_feature(text)
            return matrix

        keys = self.outcome_keys(texts)
        classes = transitive_classes(keys, semantic=self.family == "math")
        matrix = np.zeros((n, max(classes, default=-1) + 1), dtype=np.float64)
        for idx, cls in enumerate(classes):
            if valid_mask[idx]:
                matrix[idx, cls] = 1.0
        return matrix


def make_adapter(schema_name: str, metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None) -> TaskAdapter:
    return TaskAdapter(schema_name=schema_name, metadata=list(metadata or []))


__all__ = [
    "TaskAdapter",
    "canonical_logic_label",
    "canonical_math_answer",
    "code_ast_feature",
    "extract_last_boxed",
    "extract_python_code",
    "make_adapter",
    "normalized_code_key",
    "transitive_classes",
]
