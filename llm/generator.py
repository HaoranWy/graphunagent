# -*- coding: utf-8 -*-
"""APIGenerator 并发生成客户端.

升级版：
1. 引入 task_type 动态分流。针对 Code 任务，自适应注入 PYTHON_CODE 提示词，
   要求思维链写在 <think> 内，最终实现输出在标准 ```python 中，防止格式坍塌。
"""

import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """绑定到本地运行的 vLLM Serve 服务端的高并发推理轨迹生成客户端."""

    def __init__(
        self,
        model_name: str = "deepseek",
        api_key: str = "empty",
        base_url: str = "http://127.0.0.1:8000/v1",
        seed: int = 20260615
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.base_url = base_url
        self.api_key = api_key
        
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
        # 【核心改进】引入任务类型分流
        task_type: str = "GSM8K"
    ) -> List[List[str]]:
        if not prompts:
            return []

        # 动态分流 System Prompt
        if task_type in ["HumanEvalPlus", "Code"]:
            system_content = (
                "You are a master Python programmer. Your task is to implement the requested function. "
                "Please output your detailed thinking processes (using CoT) inside a <think> ... </think> block, "
                "and then provide ONLY the complete Python code block enclosed in standard markdown ```python ... ``` tags."
            )
        else:
            system_content = (
                "You are a precise reasoning agent. Please output your detailed step-by-step thinking processes "
                "(using CoT) before generating the final boxed answer inside a LaTeX '\\boxed{}' block."
            )

        results: List[Optional[List[str]]] = [None] * len(prompts)

        def _worker(index: int, prompt_text: str) -> Tuple[int, List[str]]:
            try:
                request_kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt_text}
                    ],
                    "n": num_samples,
                    "temperature": temperature if num_samples > 1 else 0.0,
                    "seed": self.seed
                }

                if max_tokens is not None:
                    request_kwargs["max_tokens"] = max_tokens

                response = self.client.chat.completions.create(**request_kwargs)
                prompt_samples = [choice.message.content for choice in response.choices]
                return index, prompt_samples
            except Exception as e:
                logger.error(f"vLLM ThreadWorker error on index {index}: {str(e)}")
                return index, ["" for _ in range(num_samples)]

        logger.info(
            f"Dispatching {len(prompts)} prompts to vLLM Server (task: {task_type}, concurrency: {max_workers})..."
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
            res if res is not None else ["" for _ in range(num_samples)] 
            for res in results
        ]
        return final_results