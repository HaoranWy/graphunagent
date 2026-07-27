#!/usr/bin/env python3
"""Build a traceable real-data motivation figure for Q-RASC.

The figure is intentionally illustrative rather than a second aggregate result:
it selects one local-quality case and one public code-behavior case from the
immutable PWSC-CI promotion freeze.  The selection rules, source hashes, raw
measurements, and rendered-artifact hashes are emitted to a JSON sidecar.

No Qwen development cache, supplemental rerun, or retired PWSC score is read.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/q-rasc-motivation-matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1784304000")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.adapters import extract_python_code, make_adapter  # noqa: E402
from core.behavioral_kernel import _collect_behavior_observations  # noqa: E402
from core.risk_estimator import PWSCCIRiskEstimator  # noqa: E402
from run_pipeline import UniversalDatasetLoader, _input_sha256, _sample_metadata  # noqa: E402


DEFAULT_FREEZE = ROOT / "code/diagnostics/pwsc_ci_frozen/pwsc_ci_results_20260717.json"
DEFAULT_COMPONENT_STATS = ROOT / "figures/pwsc-ci-ablation-stats.json"
ENSEMBLE_SIZE = 16
LOCAL_TASK = "GSM8K"
STRUCTURE_TASK = "HumanEvalPlus"
LOCAL_MIN_GAP = 3.0
LOCAL_MIN_TOKENS = 128
LOCAL_MAX_TOKENS = 2500
STRUCTURE_MAX_CONFIDENCE_GAP = 0.02
SAMPLE_ID_PATTERN = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
WORD_PATTERN = re.compile(r"[A-Za-z]{2,}")
QUALITY_ESTIMATOR = PWSCCIRiskEstimator(weakest_k=5)

INK = "#172129"
MUTED = "#5D6872"
GRID = "#D9E0E4"
CORAL = "#D85B50"
TEAL = "#007E78"
GOLD = "#C38A13"
GOLD_PALE = "#FFF7E6"
TEAL_PALE = "#EAF6F4"
CORAL_PALE = "#FCEDEC"


class FigureSourceError(RuntimeError):
    """Raised when an immutable figure input does not match the active freeze."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise FigureSourceError(f"invalid Boolean value {value!r}")


def _as_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FigureSourceError(f"invalid numeric {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise FigureSourceError(f"non-finite {label}: {value!r}")
    return result


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise FigureSourceError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def _resolve_root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_bundle(
    freeze: Mapping[str, Any], task: str
) -> tuple[list[Dict[str, str]], Dict[str, Any]]:
    scope = freeze["evidence_scope"]
    source = freeze["provenance"]["manifests"][task]
    dataset = freeze["datasets"][task]
    manifest_path = _resolve_root_path(source["manifest_path"])
    csv_path = _resolve_root_path(source["sample_csv_path"])
    cache_path = _resolve_root_path(source["cache_path"])
    for path, expected, label in (
        (manifest_path, source["manifest_sha256"], f"{task} manifest"),
        (csv_path, source["sample_csv_sha256"], f"{task} sample CSV"),
        (cache_path, source["cache_sha256"], f"{task} generation cache"),
    ):
        if not path.exists():
            raise FigureSourceError(f"missing {label}: {path}")
        _require_hash(path, expected, label)

    manifest = _load_json(manifest_path)
    expected_manifest = {
        "task": task,
        "seed": scope["seed"],
        "model_identifier": scope["model"],
        "model_revision": scope["model_root"],
        "input_sample_sha256": dataset["input_sha256"],
        "generation_cache_sha256": source["cache_sha256"],
        "csv_sha256": source["sample_csv_sha256"],
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise FigureSourceError(
                f"{task} manifest {key} mismatch: expected {expected!r}, "
                f"observed {manifest.get(key)!r}"
            )
    sampling = manifest.get("sampling", {})
    if sampling.get("ensemble_sizes") != [1, 4, 8, ENSEMBLE_SIZE]:
        raise FigureSourceError(f"{task} manifest does not bind the canonical prefixes")
    if sampling.get("weakest_k") != 5:
        raise FigureSourceError(f"{task} manifest does not bind weakest-k=5")
    if sampling.get("python_hash_seed") != scope["python_hash_seed"]:
        raise FigureSourceError(f"{task} Python hash seed mismatch")
    fingerprint = manifest.get("dataset_fingerprint", {}).get("files", [{}])[0]
    if fingerprint.get("sha256") != dataset["file_sha256"]:
        raise FigureSourceError(f"{task} dataset fingerprint mismatch")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["ensemble_size"]) == ENSEMBLE_SIZE]
    expected_count = int(dataset["question_count"])
    if len(rows) != expected_count:
        raise FigureSourceError(
            f"{task}/N={ENSEMBLE_SIZE} has {len(rows)} rows; expected {expected_count}"
        )
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise FigureSourceError(f"{task}/N={ENSEMBLE_SIZE} contains duplicate IDs")

    provenance = {
        "task": task,
        "question_count": expected_count,
        "dataset_id": dataset["dataset_id"],
        "dataset_revision": dataset["revision"],
        "dataset_file_sha256": dataset["file_sha256"],
        "input_sha256": dataset["input_sha256"],
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": source["manifest_sha256"],
        "sample_csv_path": _relative(csv_path),
        "sample_csv_sha256": source["sample_csv_sha256"],
        "generation_cache_path": _relative(cache_path),
        "generation_cache_sha256": source["cache_sha256"],
    }
    return rows, provenance


def _iter_records(cache_path: Path, wanted_ids: Iterable[str]) -> Iterator[Dict[str, Any]]:
    wanted = set(wanted_ids)
    found: set[str] = set()
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            match = SAMPLE_ID_PATTERN.search(line[:1024])
            if match is None or match.group(1) not in wanted:
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if sample_id not in wanted:
                continue
            found.add(sample_id)
            yield record
    missing = wanted - found
    if missing:
        raise FigureSourceError(f"cache is missing selected sample IDs: {sorted(missing)[:6]}")


def _trace_metrics(trace: Mapping[str, Any]) -> Dict[str, Any]:
    items = list(trace.get("logprobs") or [])
    values: list[tuple[int, float, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        try:
            logprob = float(item.get("logprob"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(logprob):
            continue
        values.append((index, max(0.0, -logprob), str(item.get("token", ""))))
    if len(values) < 5:
        raise FigureSourceError("trace has fewer than five finite chosen-token log probabilities")
    weakest = sorted(values, key=lambda item: (-item[1], item[0]))[:5]
    mean_nll = float(np.mean([item[1] for item in values]))
    worst_nll = float(np.mean([item[1] for item in weakest]))
    action = QUALITY_ESTIMATOR.path_action(items)
    if not action.valid:
        raise FigureSourceError("trace fails the canonical chosen-token quality filter")
    return {
        "token_count": len(values),
        "mean_nll": mean_nll,
        "worst_5_nll": worst_nll,
        "confidence": float(action.confidence),
        "quality_token_count": int(action.token_count),
        "weakest": weakest,
        "text": str(trace.get("text", "")),
        "tokens": values,
    }


def _assert_close(observed: float, expected: float, label: str, atol: float = 1e-9) -> None:
    if not math.isclose(observed, expected, abs_tol=atol, rel_tol=1e-9):
        raise FigureSourceError(
            f"{label} mismatch: expected {expected:.12g}, observed {observed:.12g}"
        )


def _is_readable_local_trace(metrics: Mapping[str, Any]) -> bool:
    token_count = int(metrics["token_count"])
    text = str(metrics["text"])
    words = [word.lower() for word in WORD_PATTERN.findall(text)]
    if not (LOCAL_MIN_TOKENS <= token_count <= LOCAL_MAX_TOKENS):
        return False
    if len(words) < 30:
        return False
    return len(set(words)) / len(words) >= 0.08


def _visible_token(value: str) -> str:
    text = str(value).replace("\n", "\\n").replace("\t", "\\t").strip()
    if not text:
        return "space"
    if len(text) > 11:
        return text[:10] + "."
    return text


def _token_context(metrics: Mapping[str, Any], index: int, radius: int = 3) -> str:
    values = list(metrics["tokens"])
    start = max(0, index - radius)
    end = min(len(values), index + radius + 1)
    return " ".join(_visible_token(token) for _idx, _nll, token in values[start:end])


def _select_local_example(
    rows: Sequence[Mapping[str, str]], cache_path: Path
) -> tuple[Dict[str, Any], int]:
    candidates = []
    for row in rows:
        if _parse_bool(row["base_is_correct"]):
            continue
        if int(row["valid_path_count"]) != ENSEMBLE_SIZE:
            continue
        mean_nll = _as_float(row["mean_nll"], "GSM8K mean_nll")
        worst_nll = _as_float(row["worst_5_nll"], "GSM8K worst_5_nll")
        gap = worst_nll - mean_nll
        if gap >= LOCAL_MIN_GAP:
            candidates.append((row["sample_id"], row, gap))
    if not candidates:
        raise FigureSourceError("no GSM8K local-quality candidate meets the minimum NLL gap")

    candidate_rows = {sample_id: (row, gap) for sample_id, row, gap in candidates}
    records = {record["sample_id"]: record for record in _iter_records(cache_path, candidate_rows)}
    scored = []
    for sample_id, (row, gap) in candidate_rows.items():
        record = records[sample_id]
        traces = list(record.get("traces") or [])
        if len(traces) < ENSEMBLE_SIZE:
            raise FigureSourceError(f"{sample_id} does not provide {ENSEMBLE_SIZE} traces")
        metrics = _trace_metrics(traces[0])
        _assert_close(metrics["mean_nll"], _as_float(row["mean_nll"], "mean_nll"), f"{sample_id} mean NLL")
        _assert_close(metrics["worst_5_nll"], _as_float(row["worst_5_nll"], "worst_5_nll"), f"{sample_id} worst-5 NLL")
        scored.append((not _is_readable_local_trace(metrics), -gap, sample_id, row, metrics))
    scored.sort(key=lambda item: item[:3])
    unreadable, _negative_gap, sample_id, row, metrics = scored[0]
    if unreadable:
        raise FigureSourceError(
            "no local-quality candidate has a compact, non-repetitive frozen trace"
        )

    weakest = []
    for index, nll, token in metrics["weakest"]:
        weakest.append(
            {
                "token_index": index,
                "token": str(token),
                "display_token": _visible_token(token),
                "nll": nll,
                "logprob": -nll,
                "confidence": math.exp(-nll),
                "context": _token_context(metrics, index),
            }
        )
    return (
        {
            "task": LOCAL_TASK,
            "sample_id": sample_id,
            "anchor_is_correct": False,
            "valid_path_count": int(row["valid_path_count"]),
            "mean_nll": metrics["mean_nll"],
            "worst_5_nll": metrics["worst_5_nll"],
            "nll_gap": metrics["worst_5_nll"] - metrics["mean_nll"],
            "full_trace_confidence": math.exp(-metrics["mean_nll"]),
            "weakest_5_confidence": metrics["confidence"],
            "weakest_k": 5,
            "trace_index": 0,
            "trace_token_count": int(metrics["token_count"]),
            "trace_nll_profile": [
                [int(index), float(nll)] for index, nll, _token in metrics["tokens"]
            ],
            "weakest_tokens": weakest,
        },
        len(candidates),
    )


def _compact_function(text: str, entry_point: str) -> Dict[str, Any] | None:
    source = extract_python_code(text)
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError, ValueError):
        return None
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == entry_point
        ),
        None,
    )
    if function is None:
        return None
    compact = copy.deepcopy(function)
    compact.decorator_list = []
    if (
        compact.body
        and isinstance(compact.body[0], ast.Expr)
        and isinstance(compact.body[0].value, ast.Constant)
        and isinstance(compact.body[0].value.value, str)
    ):
        compact.body = compact.body[1:]
    if not compact.body:
        return None
    compact_source = ast.unparse(ast.fix_missing_locations(compact)).strip()
    first_assignment = next(
        (
            node
            for node in compact.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        None,
    )
    first_loop = next(
        (node for node in compact.body if isinstance(node, (ast.For, ast.While))),
        None,
    )
    if first_assignment is None or first_loop is None:
        return None
    excerpt = [
        ast.unparse(first_assignment).strip(),
        ast.unparse(first_loop).splitlines()[0].strip(),
        "    ...",
    ]
    return {
        "response_sha256": _sha256_text(str(text)),
        "extracted_code_sha256": _sha256_text(source),
        "compact_function": compact_source,
        "compact_line_count": len(compact_source.splitlines()),
        "excerpt": excerpt,
    }


def _empty_list_fingerprint() -> str:
    payload = json.dumps(
        ["list", 0], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return "ok:" + hashlib.sha256(payload).hexdigest()


def _select_structure_example(
    rows: Sequence[Mapping[str, str]],
    cache_path: Path,
    samples: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, int]]:
    candidate_rows = {
        row["sample_id"]: row
        for row in rows
        if not _parse_bool(row["base_is_correct"])
        and int(row["valid_path_count"]) == ENSEMBLE_SIZE
        and int(row["behavior_active_probe_count"]) > 0
    }
    if not candidate_rows:
        raise FigureSourceError("no HumanEvalPlus behavior candidates are available")

    records = {
        record["sample_id"]: record
        for record in _iter_records(cache_path, candidate_rows)
    }
    expected_empty_list = _empty_list_fingerprint()
    ranked: list[tuple[Any, ...]] = []
    examined = 0
    for sample_id, row in candidate_rows.items():
        sample = samples.get(sample_id)
        if sample is None:
            raise FigureSourceError(f"dataset is missing selected candidate {sample_id}")
        metadata = _sample_metadata(sample, STRUCTURE_TASK)
        entry_point = str(metadata.get("entry_point", ""))
        prompt = str(metadata.get("prompt", ""))
        record = records[sample_id]
        traces = list(record.get("traces") or [])[:ENSEMBLE_SIZE]
        if len(traces) != ENSEMBLE_SIZE:
            raise FigureSourceError(f"{sample_id} does not provide {ENSEMBLE_SIZE} traces")
        texts = [str(trace.get("text", "")) for trace in traces]
        adapter = make_adapter(STRUCTURE_TASK, [dict(metadata)] * ENSEMBLE_SIZE)
        valid = np.asarray(adapter.valid_mask(texts), dtype=bool)
        if int(np.sum(valid)) != int(row["valid_path_count"]):
            raise FigureSourceError(f"{sample_id} code-validity count drift")
        (
            probes,
            fingerprints,
            active_probes,
            successful_by_path,
            _recovered_path_count,
        ) = _collect_behavior_observations(
            texts, valid, prompt, entry_point, max_probes=16
        )
        if len(probes) != int(row["behavior_probe_count"]):
            raise FigureSourceError(f"{sample_id} public-probe count drift")
        if len(active_probes) != int(row["behavior_active_probe_count"]):
            raise FigureSourceError(f"{sample_id} active-probe count drift")
        if sum(successful_by_path) != int(row["behavior_successful_calls"]):
            raise FigureSourceError(f"{sample_id} successful-call count drift")
        confidences = [
            float(_trace_metrics(trace)["confidence"]) for trace in traces
        ]
        _assert_close(
            float(np.mean(confidences)),
            _as_float(row["mean_path_confidence"], "mean_path_confidence"),
            f"{sample_id} mean path confidence",
        )
        anchor_program = _compact_function(texts[0], entry_point)
        examined += 1
        if anchor_program is None:
            continue
        for peer_index in range(1, ENSEMBLE_SIZE):
            confidence_gap = abs(confidences[0] - confidences[peer_index])
            if confidence_gap > STRUCTURE_MAX_CONFIDENCE_GAP:
                continue
            peer_program = _compact_function(texts[peer_index], entry_point)
            if peer_program is None:
                continue
            for probe_index in active_probes:
                args, keyword_items = probes[probe_index]
                if args != ([],) or keyword_items:
                    continue
                if fingerprints[0][probe_index] != "error":
                    continue
                if fingerprints[peer_index][probe_index] != expected_empty_list:
                    continue
                agreement = float(
                    np.mean(
                        [
                            fingerprints[0][index]
                            == fingerprints[peer_index][index]
                            for index in active_probes
                        ]
                    )
                )
                payload = {
                    "task": STRUCTURE_TASK,
                    "sample_id": sample_id,
                    "anchor_is_correct": False,
                    "valid_path_count": int(row["valid_path_count"]),
                    "entry_point": entry_point,
                    "prompt_sha256": _sha256_text(prompt),
                    "trace_indices": [0, peer_index],
                    "anchor_confidence": confidences[0],
                    "peer_confidence": confidences[peer_index],
                    "anchor_peer_confidence_gap": confidence_gap,
                    "confidence_definition": "exp(-mean weakest-5 chosen-token NLL)",
                    "public_probe": {
                        "probe_index": int(probe_index),
                        "call": f"{entry_point}([])",
                        "source": "type-driven from the public list input signature",
                        "anchor_observation": "error",
                        "peer_observation": "[]",
                        "anchor_fingerprint": fingerprints[0][probe_index],
                        "peer_fingerprint": fingerprints[peer_index][probe_index],
                    },
                    "active_probe_count": len(active_probes),
                    "pair_behavior_agreement": agreement,
                    "kernel_source": "type_driven_behavior",
                    "anchor_program": anchor_program,
                    "peer_program": peer_program,
                    "q_rasc_risk": _as_float(row["pwsc_ci_risk"], "pwsc_ci_risk"),
                }
                ranked.append(
                    (
                        int(anchor_program["compact_line_count"])
                        + int(peer_program["compact_line_count"]),
                        confidence_gap,
                        sample_id,
                        peer_index,
                        probe_index,
                        payload,
                    )
                )
    if not ranked:
        raise FigureSourceError(
            "no compact HumanEvalPlus empty-input behavior pair meets the rule"
        )
    ranked.sort(key=lambda item: item[:5])
    return ranked[0][-1], {
        "candidate_rows_examined": examined,
        "eligible_response_pairs": len(ranked),
    }


def _load_component_evidence(
    stats_path: Path, freeze_path: Path, freeze: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_hash(freeze_path, _sha256(freeze_path), "active freeze")
    stats = _load_json(stats_path)
    source = stats.get("source_freeze", {})
    expected_freeze = _relative(freeze_path)
    if source.get("path") != expected_freeze or source.get("sha256") != _sha256(freeze_path):
        raise FigureSourceError("component-evidence sidecar is not bound to the active freeze")
    for task in (LOCAL_TASK, STRUCTURE_TASK):
        expected = freeze["provenance"]["manifests"][task]
        observed = stats.get("provenance", {}).get(task, {})
        if observed.get("sample_csv_sha256") != expected["sample_csv_sha256"]:
            raise FigureSourceError(f"component-evidence {task} CSV source drift")
        if observed.get("generation_cache_sha256") != expected["cache_sha256"]:
            raise FigureSourceError(f"component-evidence {task} cache source drift")
    n16 = stats["macro_auroc"]["N=16"]
    ci95 = stats["macro_auroc_ci95"]["N=16"]
    contrasts = stats["n16_paired_macro_contrasts"]
    expected_pwsc = float(freeze["macro_metrics"]["PWSC-CI"]["AUROC"])
    _assert_close(float(n16["pwsc_ci"]), expected_pwsc, "component-evidence Q-RASC AUROC")
    return {
        "path": _relative(stats_path),
        "sha256": _sha256(stats_path),
        "macro_auroc": {
            "mean_nll": float(n16["mean_nll"]),
            "worst_5": float(n16["quality_only"]),
            "target_consistency": float(n16["structure_only"]),
            "q_rasc": float(n16["pwsc_ci"]),
        },
        "macro_auroc_ci95": {
            "mean_nll": [float(value) for value in ci95["mean_nll"]],
            "worst_5": [float(value) for value in ci95["quality_only"]],
            "target_consistency": [float(value) for value in ci95["structure_only"]],
            "q_rasc": [float(value) for value in ci95["pwsc_ci"]],
        },
        "paired_contrasts": {
            "worst_5_minus_mean_nll": {
                key: float(value) for key, value in contrasts["bottleneck"].items()
            },
            "q_rasc_minus_target_consistency": {
                key: float(value) for key, value in contrasts["confidence_fusion"].items()
            },
        },
        "bootstrap_draws": int(stats["analysis"]["bootstrap_draws"]),
        "bootstrap_seed": int(stats["analysis"]["bootstrap_seed"]),
    }


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def _draw_figure(
    local: Mapping[str, Any], structure: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path, Path]:
    _configure_matplotlib()
    fig = plt.figure(figsize=(3.38, 5.35), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.06, 0.94),
        hspace=0.20,
        left=0.205,
        right=0.965,
        top=0.970,
        bottom=0.080,
    )

    # A. A continuous zoom containing all five weakest tokens, followed by a
    # confidence comparison derived monotonically from the frozen NLL values.
    panel_a = outer[0].subgridspec(
        4, 1, height_ratios=(0.44, 0.88, 0.20, 0.55), hspace=0.10
    )
    a_head = fig.add_subplot(panel_a[0])
    a_head.axis("off")
    a_head.text(0.0, 0.96, "A  LOCAL QUALITY", color=CORAL, fontsize=7.7, fontweight="bold", va="top")
    a_head.text(0.0, 0.56, "Full-trace confidence hides five weak tokens", color=INK, fontsize=8.9, fontweight="bold", va="top")
    a_head.text(
        0.0,
        0.04,
        f"Incorrect GSM8K response  |  {local['trace_token_count']:,} generated tokens",
        color=MUTED,
        fontsize=7.0,
        va="bottom",
    )

    trace_ax = fig.add_subplot(panel_a[1])
    profile = np.asarray(local["trace_nll_profile"], dtype=float)
    positions = profile[:, 0]
    nll_values = profile[:, 1]
    token_confidences = np.exp(-nll_values)
    zoom_max = 300
    zoom_mask = positions <= zoom_max
    trace_ax.plot(
        positions[zoom_mask],
        token_confidences[zoom_mask],
        color="#95A3AC",
        linewidth=0.62,
        zorder=2,
    )
    trace_ax.set_xlim(0, zoom_max)
    trace_ax.set_yscale("log")
    trace_ax.set_ylim(0.002, 1.05)
    trace_ax.set_yticks((0.01, 0.1, 1.0), ("0.01", "0.1", "1"))
    trace_ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    trace_ax.set_xticks((0, 50, 100, 150, 200, 250, 300))
    trace_ax.set_ylabel("token confidence  (log scale)", color=MUTED, fontsize=6.8, labelpad=2)
    trace_ax.set_xlabel("token position  (zoom: 0-300)", color=MUTED, fontsize=6.8, labelpad=2)
    trace_ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.75)
    trace_ax.spines["top"].set_visible(False)
    trace_ax.spines["right"].set_visible(False)
    trace_ax.spines["left"].set_visible(False)
    trace_ax.spines["bottom"].set_color(GRID)
    trace_ax.tick_params(axis="both", colors=MUTED, labelsize=6.8, length=2.0, width=0.5)

    weakest = sorted(local["weakest_tokens"], key=lambda item: int(item["token_index"]))
    for item in weakest:
        token_index = int(item["token_index"])
        token_confidence = float(item["confidence"])
        # A short rug tick preserves the exact position while keeping nearby
        # weakest-token markers individually visible in the single-column view.
        trace_ax.vlines(
            token_index,
            0.0021,
            0.0027,
            color=CORAL,
            linewidth=0.85,
            zorder=4,
        )
        trace_ax.scatter(
            [token_index],
            [token_confidence],
            s=11,
            color=CORAL,
            edgecolors="white",
            linewidths=0.50,
            zorder=5,
        )

    bar_ax = fig.add_subplot(panel_a[3])
    bar_values = (
        float(local["full_trace_confidence"]),
        float(local["weakest_5_confidence"]),
    )
    bar_y = (1, 0)
    bar_ax.barh(bar_y, bar_values, height=0.46, color=("#AAB5BC", CORAL), edgecolor="none")
    bar_ax.scatter(bar_values, bar_y, s=15, color=("#8D9AA3", CORAL), edgecolors="white", linewidths=0.45, zorder=4)
    bar_ax.set_yticks(bar_y, ("full-trace confidence", "weakest-5 confidence"))
    bar_ax.set_xlim(0, 1.0)
    bar_ax.set_xticks((0, 0.5, 1.0))
    bar_ax.set_xlabel("confidence  (higher = more confident)", color=MUTED, fontsize=6.8, labelpad=2)
    bar_ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.75)
    bar_ax.set_axisbelow(True)
    bar_ax.tick_params(axis="x", labelsize=6.6, colors=MUTED, length=2.0, width=0.5)
    bar_ax.tick_params(axis="y", length=0, labelsize=7.0, colors=INK, pad=4)
    for y_value, value, color in zip(bar_y, bar_values, (MUTED, CORAL)):
        bar_ax.text(value + 0.025, y_value, f"{value:.3f}", color=color, fontsize=7.4, fontweight="bold", va="center")
    for spine in bar_ax.spines.values():
        spine.set_visible(False)

    # B. Put two real generated programs side by side. Their weakest-token
    # confidence is nearly identical, but a public empty-list call separates
    # their executable behavior because their initial control states differ.
    panel_b = outer[1].subgridspec(2, 1, height_ratios=(0.43, 1.25), hspace=0.04)
    b_head = fig.add_subplot(panel_b[0])
    b_head.axis("off")
    b_head.text(0.0, 0.96, "B  PROGRAM STRUCTURE", color=TEAL, fontsize=7.7, fontweight="bold", va="top")
    b_head.text(0.0, 0.56, "Nearly equal confidence, different behavior", color=INK, fontsize=8.9, fontweight="bold", va="top")
    b_head.text(
        0.0,
        0.04,
        f"HumanEvalPlus  |  public probe: {structure['public_probe']['call']}",
        color=MUTED,
        fontsize=7.0,
        va="bottom",
    )

    path_ax = fig.add_subplot(panel_b[1])
    path_ax.set_xlim(0, 1)
    path_ax.set_ylim(0, 1)
    path_ax.axis("off")
    centers = (0.25, 0.75)
    traces = (int(structure["trace_indices"][0]), int(structure["trace_indices"][1]))
    confidences = (
        float(structure["anchor_confidence"]),
        float(structure["peer_confidence"]),
    )
    programs = (structure["anchor_program"], structure["peer_program"])
    accents = (CORAL, TEAL)
    fills = (CORAL_PALE, TEAL_PALE)
    confidence_gap = float(structure["anchor_peer_confidence_gap"])
    path_ax.text(0.5, 0.975, "weakest-5 confidence", color=MUTED, fontsize=6.6, ha="center", va="top")
    path_ax.text(centers[0], 0.885, f"{confidences[0]:.4f}", color=CORAL, fontsize=7.0, fontweight="bold", ha="center", va="center")
    path_ax.text(0.5, 0.885, f"gap = {confidence_gap:.4f}", color=INK, fontsize=6.4, fontweight="bold", ha="center", va="center")
    path_ax.text(centers[1], 0.885, f"{confidences[1]:.4f}", color=TEAL, fontsize=7.0, fontweight="bold", ha="center", va="center")
    for center, trace_index, confidence, program, accent, fill in zip(
        centers, traces, confidences, programs, accents, fills
    ):
        path_ax.text(
            center,
            0.795,
            f"response {trace_index}",
            color=accent,
            fontsize=6.8,
            fontweight="bold",
            ha="center",
            va="center",
        )
        path_ax.add_patch(
            FancyBboxPatch(
                (center - 0.225, 0.500),
                0.45,
                0.245,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                facecolor=fill,
                edgecolor=accent,
                linewidth=0.75,
            )
        )
        for line_index, line in enumerate(program["excerpt"]):
            path_ax.text(
                center - 0.205,
                0.680 - 0.073 * line_index,
                str(line),
                color=accent if line_index == 0 else INK,
                fontsize=5.6,
                fontfamily="monospace",
                fontweight="bold" if line_index == 0 else "normal",
                ha="left",
                va="center",
            )

    state_labels = ("empty input", "empty input")
    state_details = ("numbers[0] is missing", "loop runs 0 times")
    state_outputs = ("IndexError", "returns []")
    for center, accent in zip(centers, accents):
        path_ax.annotate(
            "",
            xy=(center, 0.405),
            xytext=(center, 0.475),
            arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": accent},
        )
    for center, label, detail, output, accent in zip(
        centers, state_labels, state_details, state_outputs, accents
    ):
        path_ax.text(center, 0.370, label, color=INK, fontsize=6.4, fontfamily="monospace", fontweight="bold", ha="center", va="center")
        path_ax.text(center, 0.300, detail, color=MUTED, fontsize=6.3, ha="center", va="center")
        path_ax.text(center, 0.215, output, color=accent, fontsize=7.4, fontweight="bold", ha="center", va="center")
    path_ax.plot((0.24, 0.76), (0.115, 0.115), color=GRID, linewidth=0.8)
    path_ax.text(
        0.5,
        0.065,
        "public probe confirms different behavior",
        color=INK,
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        va="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "q-rasc-motivation.pdf"
    svg_path = output_dir / "q-rasc-motivation.svg"
    png_path = output_dir / "q-rasc-motivation.png"
    metadata = {
        "Title": "Q-RASC motivation: local quality and task structure",
        "Author": "Anonymous",
        "Creator": "scripts/plot_q_rasc_motivation.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.01, metadata=metadata)
    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return pdf_path, svg_path, png_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the traceable Q-RASC real-data motivation figure."
    )
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--component-stats", type=Path, default=DEFAULT_COMPONENT_STATS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--tex-figure-dir", type=Path, default=ROOT / "AuthorKit27/Figures")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/newmodel/graphunagent/datasets"),
    )
    args = parser.parse_args()

    freeze_path = args.freeze.expanduser().resolve()
    stats_path = args.component_stats.expanduser().resolve()
    if not freeze_path.exists() or not stats_path.exists():
        raise FigureSourceError("freeze or component-evidence sidecar is missing")
    freeze = _load_json(freeze_path)
    if freeze.get("method") != "PWSC-CI-v1.0":
        raise FigureSourceError("active freeze does not name PWSC-CI-v1.0")
    if freeze.get("contract") != "pwsc-ci-normalized-holevo-v1":
        raise FigureSourceError("active freeze contract mismatch")
    scope = freeze.get("evidence_scope", {})
    if scope.get("ensemble_size") != ENSEMBLE_SIZE:
        raise FigureSourceError("active freeze does not bind N=16")
    if tuple(scope.get("tasks", ())) != ("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO"):
        raise FigureSourceError("active freeze task scope drift")

    local_rows, local_provenance = _load_bundle(freeze, LOCAL_TASK)
    structure_rows, structure_provenance = _load_bundle(freeze, STRUCTURE_TASK)
    local_cache = _resolve_root_path(freeze["provenance"]["manifests"][LOCAL_TASK]["cache_path"])
    structure_cache = _resolve_root_path(freeze["provenance"]["manifests"][STRUCTURE_TASK]["cache_path"])
    structure_manifest_path = _resolve_root_path(
        freeze["provenance"]["manifests"][STRUCTURE_TASK]["manifest_path"]
    )
    structure_manifest = _load_json(structure_manifest_path)
    loader = UniversalDatasetLoader(str(args.dataset_root.expanduser().resolve()))
    observed_fingerprint = loader.dataset_fingerprint(STRUCTURE_TASK)
    if observed_fingerprint != structure_manifest.get("dataset_fingerprint"):
        raise FigureSourceError("HumanEvalPlus local dataset fingerprint has drifted")
    structure_samples = loader.load_any_dataset(STRUCTURE_TASK, 0)
    if _input_sha256(structure_samples) != structure_manifest.get("input_sample_sha256"):
        raise FigureSourceError("HumanEvalPlus input sample hash has drifted")
    samples_by_id = {sample.sample_id: sample for sample in structure_samples}
    local, local_candidate_count = _select_local_example(local_rows, local_cache)
    structure, structure_selection = _select_structure_example(
        structure_rows, structure_cache, samples_by_id
    )
    evidence = _load_component_evidence(stats_path, freeze_path, freeze)
    pdf_path, svg_path, png_path = _draw_figure(local, structure, args.output_dir)

    tex_dir = args.tex_figure_dir.expanduser().resolve()
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_pdf = tex_dir / pdf_path.name
    tex_svg = tex_dir / svg_path.name
    ghostscript = shutil.which("gs")
    if ghostscript is None:
        raise FigureSourceError("Ghostscript is required for the font-outlined AuthorKit PDF")
    outlined = subprocess.run(
        [
            ghostscript,
            "-q",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/prepress",
            "-dNoOutputFonts",
            f"-sOutputFile={tex_pdf}",
            str(pdf_path),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if outlined.returncode != 0:
        raise FigureSourceError(
            "failed to outline the AuthorKit motivation PDF:\n" + outlined.stdout[-4000:]
        )
    shutil.copyfile(svg_path, tex_svg)

    sidecar = {
        "schema": "q-rasc-motivation-figure-v8",
        "paper_label": "Q-RASC",
        "canonical_method": freeze["method"],
        "contract": freeze["contract"],
        "evidence_scope": freeze["evidence_scope"],
        "source_freeze": {"path": _relative(freeze_path), "sha256": _sha256(freeze_path)},
        "sources": {LOCAL_TASK: local_provenance, STRUCTURE_TASK: structure_provenance},
        "selection_rule": {
            "local_quality": {
                "universe": "GSM8K N=16 rows with an incorrect fixed first anchor and 16 valid paths",
                "minimum_nll_gap": LOCAL_MIN_GAP,
                "readability_gate": {
                    "min_scored_tokens": LOCAL_MIN_TOKENS,
                    "max_scored_tokens": LOCAL_MAX_TOKENS,
                    "minimum_word_count": 30,
                    "minimum_unique_word_ratio": 0.08,
                },
                "tie_break": "highest worst-5-minus-mean NLL gap, then lexicographic sample_id",
                "candidate_count_before_cache_gate": local_candidate_count,
            },
            "task_structure": {
                "universe": "HumanEvalPlus N=16 rows with an incorrect fixed anchor, 16 valid paths, and an active public behavior kernel",
                "requirements": [
                    "both response functions parse and contain a top-level assignment and loop",
                    "the type-driven public probe is entry_point([])",
                    "the anchor errors while the peer returns the empty list",
                    "absolute anchor-versus-peer weakest-5 confidence gap is at most 0.02",
                ],
                "tie_break": "fewest combined compact-function lines, lower confidence gap, lexicographic sample_id, peer index, then probe index",
                **structure_selection,
            },
        },
        "figure_scope": {
            "layout": "single-column vertical A/B",
            "rendered_evidence": "two mechanically selected sample-level examples",
            "component_evidence_rendered": False,
            "panel_a_encoding": "log-scaled chosen-token confidence over the continuous 0-to-300 zoom containing all five globally weakest tokens, plus full-trace and weakest-5 confidence summaries derived as exp(-mean NLL); no area fill is rendered",
            "panel_b_encoding": "exact weakest-5 confidence values and gap, generated-code excerpts, and explicit IndexError-versus-empty-list behavior on a type-driven public probe; no later-defined algorithm symbol is rendered",
        },
        "examples": {"local_quality": local, "task_structure": structure},
        "component_evidence": evidence,
        "artifacts": {
            "pdf": {"path": _relative(pdf_path), "sha256": _sha256(pdf_path)},
            "svg": {"path": _relative(svg_path), "sha256": _sha256(svg_path)},
            "png": {"path": _relative(png_path), "sha256": _sha256(png_path)},
            "tex_pdf": {"path": _relative(tex_pdf), "sha256": _sha256(tex_pdf)},
            "tex_svg": {"path": _relative(tex_svg), "sha256": _sha256(tex_svg)},
        },
    }
    sidecar_path = args.output_dir / "q-rasc-motivation-sidecar.json"
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "local_sample": local["sample_id"],
                "structure_sample": structure["sample_id"],
                "pdf": _relative(pdf_path),
                "sidecar": _relative(sidecar_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
