"""Gold-free executable code kernel used by canonical PWSC-CI.

Probes are derived only from the public function signature, type annotations,
and inputs shown in docstring examples. Benchmark tests, expected outputs, and
canonical solutions are never read. If public observations are insufficient,
PWSC-CI falls back to the deterministic AST feature kernel.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import multiprocessing as mp
import pickle
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.adapters import extract_python_code


_SAFE_IMPORT_ROOTS = {
    "bisect",
    "collections",
    "copy",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
}
_BLOCKED_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "input",
    "open",
    "quit",
}
def _annotation_text(annotation: Optional[ast.expr]) -> str:
    if annotation is None:
        return ""
    try:
        return (
            ast.unparse(annotation)
            .replace("typing.", "")
            .replace(" ", "")
            .lower()
        )
    except Exception:
        return ""


def _scalar_values(kind: str, name: str) -> List[Any]:
    if kind == "bool":
        return [False, True]
    if kind == "float":
        return [0.0, 0.5, -0.5, 1.0, 2.5, -2.0, 10.0]
    if kind == "str":
        values = ["", "a", "ab", "abc", "a b", "(()())", "Hello, World!", "123"]
        if any(part in name for part in ("paren", "bracket")):
            values = ["", "()", "(())", "()()", "(()", ")(", "(()())"]
        return values
    values = [0, 1, -1, 2, 3, 5, 10, 17]
    if name in {"n", "k", "size", "length", "count", "num"} or name.startswith(
        "num"
    ):
        values = [0, 1, 2, 3, 5, 10, 17]
    return values


def _values_for_parameter(annotation: str, name: str) -> List[Any]:
    value = annotation
    if not value:
        string_names = {
            "s",
            "text",
            "txt",
            "string",
            "word",
            "sentence",
            "message",
            "date",
            "file_name",
            "class_name",
            "operator",
            "operand",
        }
        list_names = {
            "l",
            "lst",
            "lst1",
            "lst2",
            "arr",
            "array",
            "grid",
            "nums",
            "grades",
            "extensions",
            "words",
            "values",
            "items",
        }
        if name.lower() in string_names:
            value = "str"
        elif name.lower() in list_names:
            value = (
                "list[str]"
                if name.lower() in {"extensions", "words"}
                else "list[int]"
            )
        elif name.lower() in {"dict", "mapping"}:
            value = "dict"
    if value.startswith("optional[") and value.endswith("]"):
        return [None, *_values_for_parameter(value[9:-1], name)]
    if value in {"str", "string"}:
        return _scalar_values("str", name)
    if value in {"float", "number"}:
        return _scalar_values("float", name)
    if value == "bool":
        return _scalar_values("bool", name)
    if value == "int" or not value:
        return _scalar_values("int", name)
    if value == "list" or any(
        value.startswith(prefix)
        for prefix in ("list[", "sequence[", "iterable[")
    ):
        inner = value[value.find("[") + 1 : -1] if "[" in value else "any"
        if "str" in inner:
            return [[], [""], ["a"], ["a", "b"], ["ab", "a"], ["x", "x"]]
        if "float" in inner:
            return [
                [],
                [0.0],
                [1.0],
                [-1.0, 0.5],
                [1.0, 2.0],
                [2.0, 1.0],
                [1.0, 1.0, 2.0],
            ]
        if "any" in inner:
            return [[], [0], ["a"], [0, "a"], [1, 1, 2], [True, False]]
        return [
            [],
            [0],
            [1],
            [-1],
            [1, 2],
            [2, 1],
            [1, 1, 2],
            [-2, 0, 3],
            list(range(5)),
        ]
    if value.startswith("tuple["):
        inner = value[6:-1]
        if "str" in inner:
            return [(), ("",), ("a",), ("a", "b"), ("x", "x")]
        return [(), (0,), (1,), (-1,), (1, 2), (2, 1), (-2, 0, 3)]
    if value.startswith("set["):
        return [set(), {0}, {1}, {1, 2}, {-1, 0, 2}]
    if value.startswith("dict[") or value == "dict":
        return [{}, {"a": 1}, {"a": 1, "b": 2}, {0: 0, 1: 1}]
    if value == "any":
        return [0, 1, -1, "", "a", [], [1]]
    return _scalar_values("int", name)


def _expected_function(prompt: str, entry_point: str) -> Optional[ast.FunctionDef]:
    try:
        tree = ast.parse(prompt)
    except (SyntaxError, ValueError, TypeError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return node
    return None


def _example_probes(
    prompt: str, entry_point: str
) -> List[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
    probes: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    for line in str(prompt or "").splitlines():
        if ">>>" not in line:
            continue
        expression = line.split(">>>", 1)[1].strip()
        try:
            parsed = ast.parse(expression, mode="eval").body
            if not isinstance(parsed, ast.Call):
                continue
            if not isinstance(parsed.func, ast.Name) or parsed.func.id != entry_point:
                continue
            args = tuple(ast.literal_eval(item) for item in parsed.args)
            kwargs = {
                item.arg: ast.literal_eval(item.value)
                for item in parsed.keywords
                if item.arg is not None
            }
            probes.append((args, kwargs))
        except (SyntaxError, ValueError, TypeError):
            continue
    return probes


def _observed_pool(values: Sequence[Any], fallback: Sequence[Any]) -> List[Any]:
    observed = list(values)
    if not observed:
        return list(fallback)
    sample = observed[0]
    if isinstance(sample, bool):
        generated = _scalar_values("bool", "")
    elif isinstance(sample, int):
        generated = _scalar_values("int", "")
    elif isinstance(sample, float):
        generated = _scalar_values("float", "")
    elif isinstance(sample, str):
        generated = _scalar_values("str", "")
    elif isinstance(sample, list):
        flattened = [
            item for value in observed if isinstance(value, list) for item in value
        ]
        if flattened and all(isinstance(item, str) for item in flattened):
            generated = _values_for_parameter("list[str]", "")
        elif flattened and all(isinstance(item, (int, float)) for item in flattened):
            generated = _values_for_parameter(
                "list[float]"
                if any(isinstance(item, float) for item in flattened)
                else "list[int]",
                "",
            )
        else:
            generated = _values_for_parameter("list[any]", "")
    elif isinstance(sample, tuple):
        generated = _values_for_parameter("tuple[any]", "")
    elif isinstance(sample, dict):
        generated = _values_for_parameter("dict", "")
    elif isinstance(sample, (set, frozenset)):
        generated = _values_for_parameter("set[any]", "")
    else:
        generated = list(fallback)
    merged: List[Any] = []
    seen = set()
    for item in [*observed, *generated, *fallback]:
        key = repr(item)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


@lru_cache(maxsize=512)
def generate_behavior_probes(
    prompt: str,
    entry_point: str,
    max_probes: int = 16,
) -> Tuple[Tuple[Tuple[Any, ...], Tuple[Tuple[str, Any], ...]], ...]:
    """Build deterministic type-driven calls without expected outputs."""
    function = _expected_function(prompt, entry_point)
    if function is None:
        return tuple()
    probes = _example_probes(prompt, entry_point)
    positional = [*function.args.posonlyargs, *function.args.args]
    pools = []
    for parameter_index, parameter in enumerate(positional):
        fallback = _values_for_parameter(
            _annotation_text(parameter.annotation), parameter.arg.lower()
        )
        observed = [
            args[parameter_index]
            for args, _ in probes
            if parameter_index < len(args)
        ]
        pools.append(_observed_pool(observed, fallback))
    generic_count = 1 if not pools else 12
    for probe_index in range(generic_count):
        args = tuple(
            pool[(probe_index + parameter_index * 3) % len(pool)]
            for parameter_index, pool in enumerate(pools)
        )
        probes.append((args, {}))

    frozen: List[Tuple[Tuple[Any, ...], Tuple[Tuple[str, Any], ...]]] = []
    seen = set()
    for args, kwargs in probes:
        try:
            key = repr((args, sorted(kwargs.items())))
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        frozen.append((tuple(args), tuple(sorted(kwargs.items()))))
        if len(frozen) >= max(1, int(max_probes)):
            break
    return tuple(frozen)


def _program_is_allowed(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.split(".", 1)[0] not in _SAFE_IMPORT_ROOTS
                for alias in node.names
            ):
                return False
        elif isinstance(node, ast.ImportFrom):
            if (
                not node.module
                or node.module.split(".", 1)[0] not in _SAFE_IMPORT_ROOTS
            ):
                return False
        elif isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            return False
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False
    return True


def _normalized_value(value: Any, depth: int = 0) -> Any:
    if depth >= 5:
        return ["depth", type(value).__name__]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return ["float", "nan"]
        if math.isinf(value):
            return ["float", "inf" if value > 0 else "-inf"]
        return ["float", format(value, ".12g")]
    if isinstance(value, list):
        return [
            "list",
            len(value),
            *[_normalized_value(item, depth + 1) for item in value[:64]],
        ]
    if isinstance(value, tuple):
        return [
            "tuple",
            len(value),
            *[_normalized_value(item, depth + 1) for item in value[:64]],
        ]
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: repr(item[0]))[:64]
        return [
            "dict",
            len(value),
            *[
                [
                    _normalized_value(key, depth + 1),
                    _normalized_value(item, depth + 1),
                ]
                for key, item in items
            ],
        ]
    if isinstance(value, (set, frozenset)):
        items = sorted(
            (_normalized_value(item, depth + 1) for item in value), key=repr
        )[:64]
        return [type(value).__name__, len(value), *items]
    return ["object", type(value).__name__, repr(value)[:256]]


def _behavior_worker(
    source: str,
    entry_point: str,
    probes: Sequence[Tuple[Tuple[Any, ...], Tuple[Tuple[str, Any], ...]]],
    result_connection: Any,
) -> None:
    try:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        except Exception:
            pass
        namespace: Dict[str, Any] = {"__name__": "__pwsc_probe__"}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            exec(source, namespace)
            function = namespace.get(entry_point)
            if not callable(function):
                result_connection.send(tuple("error" for _ in probes))
                return
            outcomes = []
            for args, keyword_items in probes:
                try:
                    result = function(
                        *copy.deepcopy(args),
                        **copy.deepcopy(dict(keyword_items)),
                    )
                    payload = json.dumps(
                        _normalized_value(result),
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    outcomes.append("ok:" + hashlib.sha256(payload).hexdigest())
                except BaseException:
                    outcomes.append("error")
        result_connection.send(tuple(outcomes))
    except BaseException:
        try:
            result_connection.send(tuple("error" for _ in probes))
        except BaseException:
            pass
    finally:
        try:
            result_connection.close()
        except BaseException:
            pass


@lru_cache(maxsize=256)
def _program_fingerprint(
    source: str,
    entry_point: str,
    probes_blob: bytes,
) -> Tuple[str, ...]:
    probes = pickle.loads(probes_blob)
    if not source.strip() or not probes or not _program_is_allowed(source):
        return tuple("error" for _ in probes)
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover
        ctx = mp.get_context()
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_behavior_worker,
        args=(source, entry_point, probes, sender),
    )
    process.daemon = True
    process.start()
    sender.close()
    process.join(2.5)
    if process.is_alive():
        process.kill()
        process.join()
        receiver.close()
        return tuple("error" for _ in probes)
    try:
        if not receiver.poll(0.5):
            raise TimeoutError("probe worker produced no result")
        result = tuple(receiver.recv())
    except (EOFError, OSError, TimeoutError):
        receiver.close()
        return tuple("error" for _ in probes)
    receiver.close()
    return result if len(result) == len(probes) else tuple("error" for _ in probes)


def _collect_behavior_observations(
    texts: Sequence[str],
    valid: np.ndarray,
    prompt: str,
    entry_point: str,
    max_probes: int = 16,
    recover_single_unsupported: bool = True,
) -> Tuple[
    Tuple[Tuple[Tuple[Any, ...], Tuple[Tuple[str, Any], ...]], ...],
    List[Tuple[str, ...]],
    List[int],
    List[int],
    int,
]:
    probes = generate_behavior_probes(
        str(prompt or ""), str(entry_point or ""), max_probes=max_probes
    )
    n = len(texts)
    if n == 0 or valid.size != n or not probes or not entry_point:
        return probes, [], [], [0] * n, 0

    probes_blob = pickle.dumps(probes, protocol=pickle.HIGHEST_PROTOCOL)
    fingerprints: List[Tuple[str, ...]] = []
    sources: List[str] = []
    for index, text in enumerate(texts):
        if valid[index]:
            source = extract_python_code(text)
            sources.append(source)
            fingerprints.append(
                _program_fingerprint(source, entry_point, probes_blob)
            )
        else:
            sources.append("")
            fingerprints.append(tuple("error" for _ in probes))

    supported_probes = [
        probe_index
        for probe_index in range(len(probes))
        if any(
            valid[index]
            and fingerprints[index][probe_index].startswith("ok:")
            for index in range(n)
        )
    ]
    recovered_path_count = 0
    if recover_single_unsupported and len(supported_probes) == len(probes) - 1:
        supported_calls = tuple(probes[index] for index in supported_probes)
        supported_blob = pickle.dumps(
            supported_calls, protocol=pickle.HIGHEST_PROTOCOL
        )
        for index, fingerprint in enumerate(fingerprints):
            if not valid[index] or any(
                token.startswith("ok:") for token in fingerprint
            ):
                continue
            recovered = _program_fingerprint(
                sources[index], entry_point, supported_blob
            )
            expanded = ["error"] * len(probes)
            for recovered_index, probe_index in enumerate(supported_probes):
                expanded[probe_index] = recovered[recovered_index]
            fingerprints[index] = tuple(expanded)
            recovered_path_count += 1
    active_probes = [
        probe_index
        for probe_index in range(len(probes))
        if sum(
            valid[index]
            and fingerprints[index][probe_index].startswith("ok:")
            for index in range(n)
        )
        >= 2
    ]
    successful_by_path = [
        int(sum(token.startswith("ok:") for token in fingerprint))
        for fingerprint in fingerprints
    ]
    return (
        probes,
        fingerprints,
        active_probes,
        successful_by_path,
        recovered_path_count,
    )


def build_behavior_kernel(
    texts: Sequence[str],
    valid_mask: np.ndarray,
    prompt: str,
    entry_point: str,
    *,
    max_probes: int = 16,
    recover_single_unsupported: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Return a PSD exact-output agreement kernel over public probes."""
    if max_probes < 1:
        raise ValueError("max_probes must be positive")
    n = len(texts)
    valid = np.asarray(valid_mask, dtype=bool)
    (
        probes,
        fingerprints,
        active_probes,
        successful_by_path,
        recovered_path_count,
    ) = _collect_behavior_observations(
        texts,
        valid,
        prompt,
        entry_point,
        max_probes=max_probes,
        recover_single_unsupported=recover_single_unsupported,
    )
    diagnostics: Dict[str, Any] = {
        "max_probes": int(max_probes),
        "recover_single_unsupported": bool(recover_single_unsupported),
        "recovered_path_count": int(recovered_path_count),
        "probe_count": len(probes),
        "active_probe_count": 0,
        "successful_calls": 0,
        "successful_calls_by_path": [0] * n,
        "checked_obligations_by_path": [0] * n,
        "passed_obligations_by_path": [0] * n,
        "coverage_by_path": [0.0] * n,
        "kernel_source": "ast_fallback",
    }
    if n == 0 or valid.size != n or not probes or not entry_point:
        return None, diagnostics

    diagnostics["active_probe_count"] = len(active_probes)
    diagnostics["successful_calls_by_path"] = successful_by_path
    diagnostics["successful_calls"] = int(sum(successful_by_path))
    if not active_probes:
        return None, diagnostics

    active_successes = [
        int(
            sum(
                fingerprints[index][probe].startswith("ok:")
                for probe in active_probes
            )
        )
        if valid[index]
        else 0
        for index in range(n)
    ]
    diagnostics["checked_obligations_by_path"] = [
        len(active_probes) if valid[index] else 0 for index in range(n)
    ]
    diagnostics["passed_obligations_by_path"] = active_successes
    diagnostics["coverage_by_path"] = [
        len(active_probes) / len(probes) if valid[index] else 0.0
        for index in range(n)
    ]

    feature_ids: Dict[Tuple[int, str], int] = {}
    row_features: List[List[int]] = [[] for _ in range(n)]
    for probe_index in active_probes:
        for candidate_index in range(n):
            token = fingerprints[candidate_index][probe_index]
            category = (
                token
                if valid[candidate_index] and token.startswith("ok:")
                else f"error:{candidate_index}"
            )
            key = (probe_index, category)
            feature_index = feature_ids.setdefault(key, len(feature_ids))
            row_features[candidate_index].append(feature_index)

    features = np.zeros((n, len(feature_ids)), dtype=np.float64)
    scale = math.sqrt(len(active_probes))
    for candidate_index, indices in enumerate(row_features):
        if valid[candidate_index]:
            features[candidate_index, indices] = 1.0 / scale
    kernel = features @ features.T
    diagnostics["kernel_source"] = "type_driven_behavior"
    return kernel, diagnostics


__all__ = ["build_behavior_kernel", "generate_behavior_probes"]
