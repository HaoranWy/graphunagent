"""OpenAI-compatible generator used by the unchanged pipeline CLI."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ise_flow.llm.generator")

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


class ConformalSemanticObserver:
    """Compatibility shim for old callers.

    PWSC does not use token embeddings.  This shim returns an exact token
    equality kernel so legacy imports cannot silently reintroduce a random
    projection baseline.
    """

    def compute_conformal_similarity(self, tokens: List[str]):
        import numpy as np

        if not tokens:
            return np.zeros((0, 0), dtype=np.float64)
        return (np.asarray(tokens, dtype=object)[:, None] == np.asarray(tokens, dtype=object)[None, :]).astype(np.float64)


class APIGenerator:
    def __init__(
        self,
        model_name: str = "deepseek",
        api_key: str = "empty",
        base_url: str = "http://127.0.0.1:8000/v1",
        seed: int = 20260615,
        top_k_logprobs: int = 5,
    ) -> None:
        if OpenAI is None:
            raise ImportError("The openai package is required for the vLLM-compatible endpoint")
        self.model_name = model_name
        self.seed = seed
        self.base_url = base_url
        self.api_key = api_key
        self.top_k = int(top_k_logprobs)
        # Local vLLM endpoints must not inherit an unrelated corporate proxy;
        # this also keeps localhost behavior identical in CI and on a laptop.
        http_client = httpx.Client(trust_env=False) if httpx is not None else None
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

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
    def _system_prompt(task_type: str) -> str:
        name = str(task_type).upper()
        if "CODE" in name or "HUMANEVAL" in name:
            return (
                "Implement the requested Python program. You may reason privately, "
                "then return the complete implementation in a single ```python``` block."
            )
        if "FOLIO" in name or "LOGIC" in name:
            return (
                "Solve the logic problem. Explain the reasoning briefly and end with exactly "
                "one of: True, False, Uncertain."
            )
        return (
            r"Solve the problem step by step. End with one final answer in \boxed{...} "
            "when a mathematical answer is appropriate."
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
        max_tokens: Optional[int] = None,
        max_workers: int = 32,
        task_type: str = "GSM8K",
    ) -> List[List[Dict[str, Any]]]:
        if not prompts:
            return []
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
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
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
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


__all__ = ["APIGenerator", "ConformalSemanticObserver"]
