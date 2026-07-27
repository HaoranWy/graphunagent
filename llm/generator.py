"""OpenAI-compatible generator used by the unchanged pipeline CLI."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("ise_flow.llm.generator")

# This generation/cache contract predates the PWSC-CI-only scorer.  It is kept
# verbatim so existing immutable caches remain replayable; proof certificates
# are generation metadata and are not consumed by PWSC-CI scoring.
PROOF_CARRYING_GENERATION_PROTOCOL = "proof-carrying-pc-v1"

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore


class APIGenerator:
    def __init__(
        self,
        model_name: str = "deepseek",
        api_key: str = "empty",
        base_url: str = "http://127.0.0.1:8000/v1",
        seed: int = 20260615,
        top_k_logprobs: int = 5,
        request_timeout: float = 3600.0,
        max_retries: int = 0,
        stop_token_ids: Optional[Sequence[int]] = None,
        sampling_top_k: Optional[int] = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError("The openai package is required for the vLLM-compatible endpoint")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self.model_name = model_name
        self.seed = seed
        self.base_url = base_url
        self.api_key = api_key
        self.top_k = int(top_k_logprobs)
        self.request_timeout = float(request_timeout)
        self.max_retries = int(max_retries)
        self.stop_token_ids = tuple(
            sorted({int(token_id) for token_id in (stop_token_ids or ())})
        )
        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError("stop_token_ids must be nonnegative integers")
        self.sampling_top_k = (
            None if sampling_top_k is None else int(sampling_top_k)
        )
        if self.sampling_top_k is not None and self.sampling_top_k < 0:
            raise ValueError("sampling_top_k must be nonnegative")
        # Local vLLM endpoints must not inherit an unrelated corporate proxy;
        # this also keeps localhost behavior identical in CI and on a laptop.
        timeout = (
            httpx.Timeout(
                self.request_timeout,
                connect=min(30.0, self.request_timeout),
            )
            if httpx is not None
            else self.request_timeout
        )
        http_client = (
            httpx.Client(trust_env=False, timeout=timeout)
            if httpx is not None
            else None
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=self.max_retries,
            http_client=http_client,
        )

    def describe_model(self) -> Dict[str, Any]:
        """Return the exact server-side model metadata used by a run."""
        response = self.client.models.list()
        candidates = []
        for model in response.data:
            if hasattr(model, "model_dump"):
                candidates.append(model.model_dump())
            else:  # pragma: no cover - compatibility with older SDK objects
                candidates.append(dict(model))
        selected = next((item for item in candidates if item.get("id") == self.model_name), None)
        if selected is None:
            raise ValueError(f"Model {self.model_name!r} is not exposed by {self.base_url}")
        return selected

    @staticmethod
    def generation_protocol(task_type: str) -> str:
        """Return the generation contract recorded in every cache header."""
        return PROOF_CARRYING_GENERATION_PROTOCOL

    @staticmethod
    def _system_prompt(task_type: str) -> str:
        name = str(task_type).upper()
        if "CODE" in name or "HUMANEVAL" in name or "MBPP" in name:
            return (
                "Implement the requested Python program. You may reason privately, "
                "then return the complete implementation in a single ```python``` block. "
                "The executable program is the proof object; do not append a proof certificate."
            )
        if "FOLIO" in name or "LOGIC" in name or "SMT" in name:
            return (
                "Solve the logic problem by returning exactly one compact JSON proof certificate between "
                "<proof_certificate> and </proof_certificate>. The object must have version "
                "'pc-v1', domain 'logic', a nonempty steps list, final_step, and final_answer. "
                "The final_answer field is the only answer label and must be exactly True, False, or "
                "Uncertain as a JSON string; write no content outside the certificate tags. "
                "Each step needs id, kind, depends_on, and claim. Kinds are premise, derivation, "
                "and conclusion. A premise step must also give the zero-based premise_index and "
                "quote that public premise exactly. A derivation step must give a rule name and "
                "dependency ids. The final_step must be a conclusion whose claim equals "
                "final_answer. Use exactly one outer JSON object with this shape, replacing the "
                "example contents: "
                "{\"version\":\"pc-v1\",\"domain\":\"logic\",\"steps\":["
                "{\"id\":\"p0\",\"kind\":\"premise\",\"depends_on\":[],"
                "\"premise_index\":0,\"claim\":\"exact public premise\"},"
                "{\"id\":\"d1\",\"kind\":\"derivation\",\"depends_on\":[\"p0\"],"
                "\"rule\":\"rule name\",\"claim\":\"derived claim\"},"
                "{\"id\":\"c1\",\"kind\":\"conclusion\",\"depends_on\":[\"d1\"],"
                "\"claim\":\"True\"}],\"final_step\":\"c1\",\"final_answer\":\"True\"}. "
                "Every depends_on value must exactly match an earlier id; include only premises used by "
                "the final proof ancestry. Do not emit JSONL or several adjacent JSON objects. "
                "The certificate must be the entire response, its literal first "
                "line must be <proof_certificate>, and its literal last line must be "
                "</proof_certificate>. These tags are mandatory. Use valid JSON with double quotes; "
                "never use a ```json Markdown fence."
            )
        return (
            r"Solve the problem step by step and give the ordinary final answer in \boxed{...}. "
            "Then append exactly one compact JSON proof certificate between "
            "<proof_certificate> and </proof_certificate>. The object must have version "
            "'pc-v1', domain 'math', a nonempty steps list, final_step, and final_answer. "
            "Every step must have id, kind 'equivalence', depends_on, lhs, and rhs. Write lhs and "
            "rhs as exact ASCII arithmetic or symbolic expressions using +, -, *, /, **, "
            "parentheses, and standard functions such as sqrt. Each equality must be exact, not "
            "decimal approximation. Every id, dependency id, lhs, rhs, final_step, and final_answer "
            "must be a double-quoted JSON string, even when it contains only a number; never place "
            "an unquoted arithmetic expression in JSON. The final step rhs must equal final_answer, which must match "
            r"the boxed answer. The certificate must be the final response content, its literal first "
            "line must be <proof_certificate>, and its literal last line must be "
            "</proof_certificate>. These tags are mandatory. Use valid JSON with double quotes; "
            "never use a ```json Markdown fence."
        )

    @staticmethod
    def _choice_to_sample(choice: Any) -> Dict[str, Any]:
        text = str(getattr(getattr(choice, "message", None), "content", "") or "")
        logprob_items: List[Dict[str, Any]] = []
        logprobs = getattr(choice, "logprobs", None)
        content = getattr(logprobs, "content", None) if logprobs is not None else None
        for token_item in content or []:
            token = str(getattr(token_item, "token", "") or "")
            value = getattr(token_item, "logprob", None)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            top = {}
            for alternative in getattr(token_item, "top_logprobs", None) or []:
                try:
                    top[str(alternative.token)] = float(alternative.logprob)
                except (TypeError, ValueError):
                    continue
            top.setdefault(token, value)
            logprob_items.append({"token": token, "logprob": value, "top_logprobs": top})
        sample: Dict[str, Any] = {"text": text, "logprobs": logprob_items, "attentions": None}
        if not text:
            sample["error"] = "empty_response"
        elif not logprob_items:
            sample["error"] = "missing_logprobs"
        return sample

    def generate_consistency_samples(
        self,
        prompts: List[str],
        num_samples: int = 3,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_workers: int = 32,
        task_type: str = "GSM8K",
    ) -> List[List[Dict[str, Any]]]:
        if not prompts:
            return []
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if top_p is not None and not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p must lie in (0, 1]")
        system_content = self._system_prompt(task_type)
        results: List[Optional[List[Dict[str, Any]]]] = [None] * len(prompts)

        def worker(index: int, prompt: str) -> Tuple[int, List[Dict[str, Any]]]:
            request: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                "n": num_samples,
                "temperature": temperature if num_samples > 1 else 0.0,
                "seed": self.seed,
                "logprobs": True,
                "top_logprobs": self.top_k,
            }
            if top_p is not None:
                request["top_p"] = float(top_p)
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
            vllm_extensions: Dict[str, Any] = {}
            if self.stop_token_ids:
                # vLLM exposes this field as an OpenAI-compatible extension;
                # extra_body keeps it out of the strict SDK request schema.
                vllm_extensions["stop_token_ids"] = list(self.stop_token_ids)
            if self.sampling_top_k is not None:
                vllm_extensions["top_k"] = self.sampling_top_k
            if vllm_extensions:
                request["extra_body"] = vllm_extensions
            try:
                response = self.client.chat.completions.create(**request)
                samples = [self._choice_to_sample(choice) for choice in (response.choices or [])]
                if len(samples) < num_samples:
                    samples.extend(
                        [{"text": "", "logprobs": [], "attentions": None, "error": "missing_choice"}]
                        for _ in range(num_samples - len(samples))
                    )
                return index, samples[:num_samples]
            except Exception as exc:
                logger.error("generation failed for prompt %s: %s", index, exc)
                return index, [
                    {"text": "", "logprobs": [], "attentions": None, "error": f"generation_error:{type(exc).__name__}"}
                    for _ in range(num_samples)
                ]

        workers = max(1, min(int(max_workers), len(prompts)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker, idx, prompt) for idx, prompt in enumerate(prompts)]
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Sampling trajectories") if tqdm else as_completed(futures)
            for future in iterator:
                idx, samples = future.result()
                results[idx] = samples
        return [
            samples if samples is not None else [{"text": "", "logprobs": [], "attentions": None, "error": "missing_result"} for _ in range(num_samples)]
            for samples in results
        ]


__all__ = ["APIGenerator", "PROOF_CARRYING_GENERATION_PROTOCOL"]
