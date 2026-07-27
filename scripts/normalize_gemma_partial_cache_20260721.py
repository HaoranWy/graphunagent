#!/usr/bin/env python3
"""Normalize a paused Gemma cache before resuming with an explicit stop id.

The incident cache contains long repeated ``<end_of_turn>`` tails because the
old client did not send vLLM's ``stop_token_ids`` extension.  This script never
mutates that source cache.  It writes a new schema-v2 cache whose traces stop
before the first Gemma turn terminator and whose header records the source
path/hash and the explicit stop contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import orjson
except ImportError:  # pragma: no cover - standard-library fallback
    orjson = None


CACHE_SCHEMA_VERSION = 2
DEFAULT_STOP_TOKEN_ID = 106
DEFAULT_STOP_TOKEN_TEXTS = ("<end_of_turn>", "<eos>")


def _parse_ids(value: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("stop token ids must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values):
        raise ValueError("stop token ids must contain nonnegative integers")
    return tuple(sorted(set(values)))


def _json_bytes(value: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        .encode("utf-8")
    )


def _json_loads(raw: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def _validate_header(header: Dict[str, Any]) -> None:
    if (
        header.get("record_type") != "meta"
        or header.get("schema_version") != CACHE_SCHEMA_VERSION
    ):
        raise ValueError("source is not a PWSC-CI schema-v2 cache")
    if not header.get("task") or not header.get("model", {}).get("id"):
        raise ValueError("source cache header lacks task or model identity")
    if int(header.get("num_samples", 0)) < 1:
        raise ValueError("source cache header has an invalid num_samples")


def _first_stop_index(
    logprobs: Sequence[Dict[str, Any]], stop_texts: Sequence[str]
) -> int:
    stops = set(stop_texts)
    for index, item in enumerate(logprobs):
        if isinstance(item, dict) and str(item.get("token", "")) in stops:
            return index
    return -1


def normalize_cache(
    source_path: Path,
    output_path: Path,
    stop_token_ids: Sequence[int],
    stop_token_texts: Sequence[str],
) -> Dict[str, Any]:
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if source_path == output_path:
        raise ValueError("source and output cache must be different files")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    before = source_path.stat()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")

    source_digest = hashlib.sha256()
    record_count = 0
    trace_count = 0
    removed_items = 0
    length_capped_traces = 0
    try:
        with source_path.open("rb") as source, temporary.open("w+b") as output:
            raw_header = source.readline()
            if not raw_header:
                raise ValueError("source cache is empty")
            source_digest.update(raw_header)
            try:
                header = _json_loads(raw_header)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError("source cache has an invalid metadata line") from exc
            _validate_header(header)

            transformed_header = dict(header)
            transformed_header["stop_token_ids"] = list(stop_token_ids)
            transformed_header["stop_policy"] = "explicit-token-id-v1"
            transformed_header["cache_transform"] = {
                "name": "truncate-before-first-stop-token-v1",
                "source_cache": str(source_path),
                "source_cache_sha256": "0" * 64,
                "stop_token_ids": list(stop_token_ids),
                "stop_token_texts": list(stop_token_texts),
                "no_stop_marker_policy": "preserve-only-at-declared-max-tokens",
            }
            placeholder = _json_bytes(transformed_header) + b"\n"
            output.write(placeholder)

            for raw_line in source:
                source_digest.update(raw_line)
                if not raw_line.strip():
                    raise ValueError(
                        f"source cache contains a blank line after record {record_count}"
                    )
                try:
                    record = _json_loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"source cache has a malformed record at index {record_count}"
                    ) from exc
                if record.get("record_type") != "sample":
                    raise ValueError(
                        f"source cache record {record_count} is not a sample record"
                    )
                traces = record.get("traces")
                expected_traces = int(header["num_samples"])
                if not isinstance(traces, list) or len(traces) != expected_traces:
                    raise ValueError(
                        f"source cache record {record_count} has "
                        f"{len(traces) if isinstance(traces, list) else 'non-list'} "
                        f"traces; expected {expected_traces}"
                    )

                transformed_traces: List[Dict[str, Any]] = []
                for trace_index, trace in enumerate(traces):
                    if not isinstance(trace, dict):
                        raise ValueError(
                            f"source cache trace {record_count}/{trace_index} is not an object"
                        )
                    logprobs = trace.get("logprobs")
                    if not isinstance(logprobs, list):
                        raise ValueError(
                            f"source cache trace {record_count}/{trace_index} lacks logprobs"
                        )
                    stop_index = _first_stop_index(logprobs, stop_token_texts)
                    if stop_index == 0:
                        raise ValueError(
                            f"source cache trace {record_count}/{trace_index} starts "
                            "with a stop marker and has no scoring tokens"
                        )
                    normalized_trace = dict(trace)
                    if stop_index > 0:
                        normalized_trace["logprobs"] = list(logprobs[:stop_index])
                        removed_items += len(logprobs) - stop_index
                    else:
                        declared_max = int(header.get("max_tokens", 0))
                        if declared_max < 1 or len(logprobs) != declared_max:
                            raise ValueError(
                                f"source cache trace {record_count}/{trace_index} has no "
                                "stop marker and {len(logprobs)} tokens; expected exactly "
                                f"the declared max_tokens={declared_max}"
                            )
                        normalized_trace["logprobs"] = list(logprobs)
                        length_capped_traces += 1
                    trace_count += 1
                    transformed_traces.append(normalized_trace)
                transformed_record = dict(record)
                transformed_record["traces"] = transformed_traces
                output.write(_json_bytes(transformed_record) + b"\n")
                record_count += 1
                if record_count % 25 == 0:
                    print(
                        f"normalized {record_count} records; "
                        f"removed {removed_items} repeated logprob items",
                        file=sys.stderr,
                        flush=True,
                    )

            after = source_path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(
                    "source cache changed while being normalized; keep the client paused "
                    "and rerun with a fresh output path"
                )

            final_header = dict(transformed_header)
            final_header["cache_transform"] = dict(
                transformed_header["cache_transform"],
                source_cache_sha256=source_digest.hexdigest(),
            )
            final_bytes = _json_bytes(final_header) + b"\n"
            if len(final_bytes) != len(placeholder):
                raise AssertionError("cache header placeholder length changed")
            output.seek(0)
            output.write(final_bytes)
            output.flush()

        temporary.replace(output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    output_digest = hashlib.sha256()
    with output_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            output_digest.update(chunk)
    return {
        "source": str(source_path),
        "source_sha256": source_digest.hexdigest(),
        "output": str(output_path),
        "output_sha256": output_digest.hexdigest(),
        "records": record_count,
        "traces": trace_count,
        "removed_logprob_items": removed_items,
        "length_capped_traces": length_capped_traces,
        "stop_token_ids": list(stop_token_ids),
        "stop_token_texts": list(stop_token_texts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stop-token-ids", default=str(DEFAULT_STOP_TOKEN_ID))
    parser.add_argument(
        "--stop-token-texts",
        default=",".join(DEFAULT_STOP_TOKEN_TEXTS),
        help="comma-separated serialized token strings searched in each trace",
    )
    args = parser.parse_args()
    stop_texts = tuple(item for item in args.stop_token_texts.split(",") if item)
    if not stop_texts:
        parser.error("--stop-token-texts must contain at least one token string")
    result = normalize_cache(
        args.source,
        args.output,
        _parse_ids(args.stop_token_ids),
        stop_texts,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
