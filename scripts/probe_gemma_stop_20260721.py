#!/usr/bin/env python3
"""Fail unless the Gemma endpoint honors its explicit end-of-turn stop id."""

from __future__ import annotations

import argparse
import json

import httpx
from openai import OpenAI

from llm.generator import APIGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument(
        "--expected-root", default="/data/newmodel/IE/LLM/gemma-3-12b-it"
    )
    parser.add_argument("--stop-token-id", type=int, default=106)
    parser.add_argument("--sampling-top-k", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    timeout = httpx.Timeout(600.0, connect=30.0)
    client = OpenAI(
        api_key="empty",
        base_url=args.base_url,
        timeout=timeout,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=timeout),
    )
    models = [model.model_dump() for model in client.models.list().data]
    selected = next((model for model in models if model.get("id") == args.model), None)
    if selected is None:
        raise RuntimeError(f"model {args.model!r} is not served by {args.base_url}")
    observed_root = selected.get("root") or selected.get("revision")
    if observed_root != args.expected_root:
        raise RuntimeError(
            f"model revision mismatch: expected {args.expected_root!r}, "
            f"got {observed_root!r}"
        )

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": APIGenerator._system_prompt("GSM8K")},
            {"role": "user", "content": "What is 1 + 1? Keep the solution short."},
        ],
        n=1,
        temperature=0.0,
        seed=20260716,
        logprobs=True,
        top_logprobs=10,
        max_tokens=args.max_tokens,
        extra_body={
            "stop_token_ids": [args.stop_token_id],
            "top_k": args.sampling_top_k,
        },
    )
    if not response.choices:
        raise RuntimeError("Gemma stop probe returned no choices")
    choice = response.choices[0]
    token_items = list(choice.logprobs.content or []) if choice.logprobs else []
    if choice.finish_reason != "stop":
        raise RuntimeError(
            f"Gemma stop probe ended by {choice.finish_reason!r}, not 'stop'"
        )
    if not token_items or len(token_items) >= args.max_tokens:
        raise RuntimeError(
            f"Gemma stop probe returned {len(token_items)} tokens for a "
            f"{args.max_tokens}-token cap"
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "model": args.model,
                "revision": observed_root,
                "finish_reason": choice.finish_reason,
                "returned_logprob_tokens": len(token_items),
                "max_tokens": args.max_tokens,
                "stop_token_ids": [args.stop_token_id],
                "sampling_top_k": args.sampling_top_k,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
