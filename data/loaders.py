# -*- coding: utf-8 -*-
"""ISE-Flow 真实学术数据集加载与分层划分模块.

升级版：新增对本地 Parquet 格式 HumanEval+ 数据集的物理读取。
"""

import os
import logging
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("ise_flow.data.loaders")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass(frozen=True)
class TaskSample:
    """ISE-Flow 统一格式的学术任务样本实体."""
    sample_id: str
    task_type: str
    prompt_data: Dict[str, str]
    raw_question: str
    gold_answer: str
    difficulty: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)


class DatasetLoader:
    """真实学术数据集加载器."""

    def __init__(self, local_cache_dir: Optional[str] = None) -> None:
        self.local_cache_dir = local_cache_dir or "data/cache"
        if not os.path.exists(self.local_cache_dir):
            os.makedirs(self.local_cache_dir, exist_ok=True)

    def load_gsm8k_split(self) -> Tuple[List[TaskSample], List[TaskSample]]:
        local_parquet_dir = "/data/newmodel/graphunagent/datasets/gsm8k/main"
        train_path = os.path.join(local_parquet_dir, "train-00000-of-00001.parquet")
        test_path = os.path.join(local_parquet_dir, "test-00000-of-00001.parquet")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Local official GSM8K train/test parquet files missing.")

        df_test = pd.read_parquet(test_path)
        test_samples = []
        for i, row in df_test.iterrows():
            gold_val = row["answer"].split("####")[-1].strip()
            test_samples.append(TaskSample(
                sample_id=f"gsm8k_test_{i}",
                task_type="GSM8K",
                prompt_data={"question": row["question"]},
                raw_question=row["question"],
                gold_answer=gold_val
            ))

        df_train = pd.read_parquet(train_path)
        df_train_sampled = df_train.sample(n=1000, random_state=20260615)
        dev_samples = []
        for i, row in df_train_sampled.iterrows():
            gold_val = row["answer"].split("####")[-1].strip()
            dev_samples.append(TaskSample(
                sample_id=f"gsm8k_dev_{i}",
                task_type="GSM8K",
                prompt_data={"question": row["question"]},
                raw_question=row["question"],
                gold_answer=gold_val
            ))

        return dev_samples, test_samples

    # 【新增核心】加载本地 HumanEvalPlus Parquet 数据
    def load_humanevalplus_test(self) -> List[TaskSample]:
        """物理读取本地官方 HumanEval+ 测试分片."""
        local_parquet_path = "/data/newmodel/graphunagent/datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet"
        
        if not os.path.exists(local_parquet_path):
            raise FileNotFoundError(f"HumanEval+ parquet file not found at: {local_parquet_path}")

        logger.info(f"Loading official HumanEval+ TEST set from: {local_parquet_path}")
        df = pd.read_parquet(local_parquet_path)
        
        test_samples = []
        for i, row in df.iterrows():
            # 提取 HumanEval+ 核心字段：
            # prompt (函数定义与docstring), canonical_solution (标准解), test (单元测试断言), entry_point (入口函数名)
            task_id = row.get("task_id", f"HE_{i}")
            prompt = row.get("prompt", "")
            test_code = row.get("test", "")
            entry_point = row.get("entry_point", "")
            
            test_samples.append(TaskSample(
                sample_id=str(task_id),
                task_type="Code",
                prompt_data={
                    "prompt": prompt,
                    "test_code": test_code,
                    "entry_point": entry_point
                },
                raw_question=prompt,
                gold_answer=test_code,  # 单元测试断言作为判定黄金标准
                metadata={"entry_point": entry_point}
            ))
            
        return test_samples