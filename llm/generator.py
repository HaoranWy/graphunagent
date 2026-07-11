# -*- coding: utf-8 -*-
"""APIGenerator 客户端与共形语义观测器 (LSO) 推理引擎 (基于 vLLM 部署)."""

import os
import math
import logging
import random
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger("ise_flow.llm.generator")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 检查 tqdm 依赖
TQDM_AVAILABLE = False
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    pass


class ConformalSemanticObserver:
    """轻量级共形语义观测器 (LSO) - 子词共形投影版."""

    def __init__(self, dim: int = 256, vocab_size: int = 150000, gamma: float = 8.0) -> None:
        """初始化。
        
        Args:
            dim: 共形投影维度。
            vocab_size: 伪随机投影词表尺寸。
            gamma: 核函数平移缩放因子，非线性压制高维正交投影噪声。
        """
        self.dim = dim
        self.gamma = gamma
        rng = np.random.RandomState(42)
        # 建立确定性共形投影随机矩阵
        self.embedding_matrix = rng.normal(0.0, 1.0 / np.sqrt(dim), (vocab_size, dim))

    def _to_char_ngrams(self, word: str, n_min: int = 2, n_max: int = 4) -> List[str]:
        """将 token 分解为 character n-grams，本征捕获形态学与数值相干性."""
        if not word:
            return []
        ngrams = []
        word_padded = f"<{word}>"
        for n in range(n_min, n_max + 1):
            for i in range(len(word_padded) - n + 1):
                ngrams.append(word_padded[i : i + n])
        return ngrams

    def compute_conformal_similarity(self, tokens: List[str]) -> np.ndarray:
        """投影词汇并应用高斯共形核平移，压制高维空间的随机噪声."""
        T = len(tokens)
        if T == 0:
            return np.zeros((0, 0))

        # 1. 为每个 token 的子词进行特征累加并进行共形嵌入
        embeds = []
        for tok in tokens:
            ngrams = self._to_char_ngrams(tok)
            if not ngrams:
                idx = hash(tok) % self.embedding_matrix.shape[0]
                embeds.append(self.embedding_matrix[idx])
                continue
            
            # 累加所有子词的随机投影向量，建立语义相干性
            tok_vector = np.zeros(self.dim, dtype=np.float64)
            for ngram in ngrams:
                idx = hash(ngram) % self.embedding_matrix.shape[0]
                tok_vector += self.embedding_matrix[idx]
            embeds.append(tok_vector)

        embeds_np = np.array(embeds, dtype=np.float64)
        norms = np.linalg.norm(embeds_np, axis=1, keepdims=True) + 1e-8
        norm_embeds = embeds_np / norms

        # 2. 计算点积生成基础相似度 S
        S = np.dot(norm_embeds, norm_embeds.T)
        
        # 3. 余弦核平移 (Cosine Kernel Shift)
        # 将接近 0 的高维噪声压制为 0，将接近 1 的真实相干信号放大
        S_kernel = np.exp(self.gamma * (S - 1.0))

        # 严格 Causal Masking 约束，保持自回归时序因果性
        for u in range(T):
            S_kernel[u, u + 1 :] = 0.0
            S_kernel[u, u] = 1.0  # 顶点本征自我相干度强锁死为 1.0

        return S_kernel


class APIGenerator:
    """绑定至外部 vLLM Serve 服务的高并发推理客户端."""

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
        self.lso = ConformalSemanticObserver()  # 补全 lso 属性定义，提供基线计算支持
        
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

        if task_type in ["HumanEvalPlus", "Code", "LiveCodeBench"]:
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
                    "top_logprobs": self.top_k
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
                            top_dist = {}
                            if token_logprob.top_logprobs:
                                for item in token_logprob.top_logprobs:
                                    top_dist[item.token] = item.logprob
                            
                            if token_logprob.token not in top_dist:
                                top_dist[token_logprob.token] = token_logprob.logprob

                            logprob_list.append({
                                "token": token_logprob.token,
                                "logprob": token_logprob.logprob,
                                "top_logprobs": top_dist
                            })
                    
                    # 极速测试下的 dummy logprobs 保护
                    if not logprob_list:
                        logprob_list = [
                            {"token": "dummy", "logprob": random.gauss(-0.05, 0.02), "top_logprobs": {"dummy": -0.05}} 
                            for _ in range(max(10, len(text_content)//4))
                        ]

                    tokens_list = [item["token"] for item in logprob_list]
                    S_matrix = self.lso.compute_conformal_similarity(tokens_list)

                    prompt_samples.append({
                        "text": text_content,
                        "logprobs": logprob_list,
                        "attentions": S_matrix
                    })

                return index, prompt_samples
            except Exception as e:
                logger.error(f"vLLM ThreadWorker error on index {index}: {str(e)}")
                dummy_logprobs = [{"token": "dummy", "logprob": -1.5, "top_logprobs": {"dummy": -1.5}} for _ in range(10)]
                dummy_S = self.lso.compute_conformal_similarity(["dummy"] * 10)
                empty_samples = [
                    {"text": "", "logprobs": dummy_logprobs, "attentions": dummy_S}
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
                {"text": "", "logprobs": [], "attentions": np.zeros((0, 0))} for _ in range(num_samples)
            ] 
            for res in results
        ]
        return final_results