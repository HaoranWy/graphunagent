# -*- coding: utf-8 -*-
"""APIGenerator 客户端与 vLLM 概率流随动提取引擎.

重构说明：
1. 统一采用高吞吐的 vLLM 部署模式。
2. 开启 vLLM 的 top_logprobs 接口，将每个生成步骤的 Top-K 候选分布作为统计流形的本征坐标直接导出。
3. 彻底剥离人工的随机投影与形态学 N-gram，实现零显存额外开销。
"""

import os
import logging
import random
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger("ise_flow.llm.generator")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class APIGenerator:
    """高并发 vLLM 推理客户端 (支持 Top-K 概率流导出)."""

    def __init__(
        self,
        model_name: str = "deepseek",
        api_key: str = "empty",
        base_url: str = "http://127.0.0.1:8000/v1",
        seed: int = 20260615,
        top_k_logprobs: int = 5
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.base_url = base_url
        self.api_key = api_key
        self.top_k = top_k_logprobs
        
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "[Fatal Error] 'openai' library is required to interface with your local vLLM server."
            )
            
        logger.info(f"Connecting to live out-of-process vLLM Server at: {self.base_url}")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_consistency_samples(
        self,
        prompts: List[str],
        num_samples: int = 3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_workers: int = 32,
        task_type: str = "GSM8K"
    ) -> List[List[Dict[str, Any]]]:
        if not prompts:
            return []

        if task_type in ["HumanEvalPlus", "Code"]:
            system_content = (
                "You are a master Python programmer. Your task is to implement the requested function. "
                "Please output your detailed thinking processes (using CoT) inside a <think> ... </think> block, "
                "and then provide ONLY the complete Python code block enclosed in standard markdown ```python ... ``` tags."
            )
        else:
            system_content = (
                "You are an expert mathematical reasoning agent. Solve the given question step-by-step. "
                "Label steps explicitly with '[STEP X]' and '[/STEP X]' blocks, and enclose final answers in \\boxed{}."
            )

        results: List[Optional[List[Dict[str, Any]]]] = [None] * len(prompts)

        def _worker(index: int, prompt_text: str) -> Tuple[int, List[Dict[str, Any]]]:
            try:
                request_kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt_text}
                    ],
                    "n": num_samples,
                    "temperature": temperature if num_samples > 1 else 0.0,
                    "seed": self.seed,
                    "logprobs": True,  
                    "top_logprobs": self.top_k  # 物理提取 Top-K 对数似然，供 Fisher-Rao 路径作用量使用
                }

                if max_tokens is not None:
                    request_kwargs["max_tokens"] = max_tokens

                response = self.client.chat.completions.create(**request_kwargs)
                prompt_samples = []
                for choice in response.choices:
                    text_content = choice.message.content
                    
                    logprob_list = []
                    if choice.logprobs and choice.logprobs.content:
                        for token_logprob in choice.logprobs.content:
                            # 序列化当前 Step 概率单纯形上的 Top-K 坐标
                            top_dist = {}
                            if token_logprob.top_logprobs:
                                for item in token_logprob.top_logprobs:
                                    top_dist[item.token] = item.logprob
                            
                            # 确保当前生成的 Token 强制在 Top 分布中
                            if token_logprob.token not in top_dist:
                                top_dist[token_logprob.token] = token_logprob.logprob

                            logprob_list.append({
                                "token": token_logprob.token,
                                "logprob": token_logprob.logprob,
                                "top_logprobs": top_dist
                            })
                    
                    if not logprob_list:
                        logprob_list = [
                            {"token": "dummy", "logprob": -0.01, "top_logprobs": {"dummy": -0.01}} 
                            for _ in range(max(10, len(text_content)//4))
                        ]

                    prompt_samples.append({
                        "text": text_content,
                        "logprobs": logprob_list,
                        "attentions": None  # FR-PA 摒弃离散注意力矩阵
                    })

                return index, prompt_samples
            except Exception as e:
                logger.error(f"vLLM ThreadWorker error on index {index}: {str(e)}")
                dummy_logprobs = [{"token": "dummy", "logprob": -0.01, "top_logprobs": {"dummy": -0.01}} for _ in range(10)]
                empty_samples = [
                    {"text": "", "logprobs": dummy_logprobs, "attentions": None}
                    for _ in range(num_samples)
                ]
                return index, empty_samples

        logger.info(
            f"Dispatching {len(prompts)} prompts to vLLM Server (task: {task_type}, concurrency: {max_workers}, top_logprobs: {self.top_k})...."
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker, i, prompt): i 
                for i, prompt in enumerate(prompts)
            }

            if TQDM_AVAILABLE:
                progress_bar = tqdm(
                    as_completed(futures),
                    total=len(prompts),
                    desc=f"Sampling Trajectories ({self.model_name})"
                )
            else:
                progress_bar = as_completed(futures)

            for future in progress_bar:
                idx, prompt_samples = future.result()
                results[idx] = prompt_samples

        final_results = [
            res if res is not None else [
                {"text": "", "logprobs": [], "attentions": None} for _ in range(num_samples)
            ] 
            for res in results
        ]
        return final_results