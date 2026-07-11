# -*- coding: utf-8 -*-
"""DatasetLoader 模块.

支持 GSM8K, HumanEvalPlus, AIME2025, AIME2026, FOLIO, P-FOLIO, LiveCodeBench, MATH-500
八大前沿数据集的本地 Parquet/JSONL/CSV 文件无缝加载。
完全消除了键名错配引发的 KeyError，并对 NaN 空值进行了安全归一化。
"""

import os
import json
import logging
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
    sample_id: str
    task_type: str
    prompt_data: Dict[str, str]
    raw_question: str
    gold_answer: str
    difficulty: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)


class DatasetLoader:

    def __init__(self, base_dir: str = "/data/newmodel/graphunagent/datasets") -> None:
        self.base_dir = base_dir

    def load_gsm8k_split(self) -> Tuple[List[TaskSample], List[TaskSample]]:
        local_dir = os.path.join(self.base_dir, "gsm8k/main")
        test_path = os.path.join(local_dir, "test-00000-of-00001.parquet")
        train_path = os.path.join(local_dir, "train-00000-of-00001.parquet")

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
        return [], test_samples

    def load_humanevalplus_test(self) -> List[TaskSample]:
        """物理读取本地官方 HumanEval+ 测试分片 (Parquet 格式)."""
        local_parquet_path = os.path.join(self.base_dir, "humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")
        
        if not os.path.exists(local_parquet_path):
            local_parquet_path = "/data/newmodel/graphunagent/datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet"
            if not os.path.exists(local_parquet_path):
                raise FileNotFoundError(f"HumanEvalPlus Parquet file not found at: {local_parquet_path}")

        logger.info(f"Loading official HumanEval+ TEST set from Parquet: {local_parquet_path}")
        df = pd.read_parquet(local_parquet_path)
        
        test_samples = []
        for i, row in df.iterrows():
            task_id = row.get("task_id", f"HE_{i}")
            prompt = row.get("prompt", "")
            test_code = row.get("test", "")
            entry_point = row.get("entry_point", "")
            
            test_samples.append(TaskSample(
                sample_id=str(task_id).replace("/", "_"),
                task_type="Code",
                prompt_data={
                    "prompt": prompt,
                    "test_code": test_code,
                    "entry_point": entry_point
                },
                raw_question=prompt,
                gold_answer=test_code,
                metadata={"entry_point": entry_point, "task_id": task_id}
            ))
            
        return test_samples

    def load_aime2025_split(self) -> List[TaskSample]:
        aime_dir = os.path.join(self.base_dir, "AIME2025")
        test_samples = []
        for split in ["aime2025-I.jsonl", "aime2025-II.jsonl"]:
            path = os.path.join(aime_dir, split)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    # 宽口径提取键值
                    raw_q = data.get("question") or data.get("problem") or ""
                    gold = data.get("answer") or data.get("solution") or ""
                    test_samples.append(TaskSample(
                        sample_id=f"aime2025_{split.split('.')[0]}_{idx}",
                        task_type="AIME",
                        prompt_data={"question": str(raw_q)},
                        raw_question=str(raw_q),
                        gold_answer=str(gold).strip()
                    ))
        return test_samples

    def load_aime2026_split(self) -> List[TaskSample]:
        path = os.path.join(self.base_dir, "AIME2026/aime2026.jsonl")
        test_samples = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                # 核心修复二：使用 get() 宽口径，防止 AIME2026 的 'problem' 键名引发 KeyError
                raw_q = data.get("question") or data.get("problem") or ""
                gold = data.get("answer") or data.get("solution") or ""
                test_samples.append(TaskSample(
                    sample_id=f"aime2026_{idx}",
                    task_type="AIME",
                    prompt_data={"question": str(raw_q)},
                    raw_question=str(raw_q),
                    gold_answer=str(gold).strip()
                ))
        return test_samples

    def load_folio_split(self) -> List[TaskSample]:
        path = os.path.join(self.base_dir, "folio/validation.jsonl")
        test_samples = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                premises_list = data.get("premises", [])
                premises_str = "\n".join(premises_list) if isinstance(premises_list, list) else str(premises_list)
                conclusion = data.get("conclusion", "")
                label = data.get("label", "")
                
                prompt = f"Premises:\n{premises_str}\n\nConclusion: {conclusion}\nIs the conclusion True, False, or Uncertain?"
                test_samples.append(TaskSample(
                    sample_id=f"folio_{idx}",
                    task_type="FOLIO",
                    prompt_data={"question": prompt},
                    raw_question=prompt,
                    gold_answer=str(label).strip()
                ))
        return test_samples

    def load_p_folio_split(self) -> List[TaskSample]:
        path = os.path.join(self.base_dir, "P-FOLIO/P-FOLIO.csv")
        df = pd.read_csv(path)
        test_samples = []
        for idx, row in df.iterrows():
            premises = str(row.get("premises", row.get("story", ""))).strip()
            conclusion = str(row.get("conclusion", "")).strip()
            label = str(row.get("label", row.get("target", ""))).strip()
            
            # 安全归一化 NaN
            if pd.isna(row.get("label")) and pd.isna(row.get("target")):
                label = "uncertain"
                
            prompt = f"Premises:\n{premises}\n\nConclusion: {conclusion}\nIs the conclusion True, False, or Uncertain?"
            test_samples.append(TaskSample(
                sample_id=f"pfolio_{idx}",
                task_type="FOLIO",
                prompt_data={"question": prompt},
                raw_question=prompt,
                gold_answer=label
            ))
        return test_samples

    def load_livecodebench_split(self) -> List[TaskSample]:
        test_samples = []
        lcb_dir = os.path.join(self.base_dir, "livecodebench")
        for i in range(1, 7):
            path = os.path.join(lcb_dir, f"test{i}.jsonl")
            if not os.path.exists(path):
                path = os.path.join(lcb_dir, "test.jsonl")
                if not os.path.exists(path):
                    continue
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    # 核心修复三：对 LCB 的 test_cases 键名采用多层 fallback 获取，防止返回空列表
                    test_cases = data.get("test_cases") or data.get("test_case") or data.get("tests") or data.get("test") or []
                    
                    test_samples.append(TaskSample(
                        sample_id=f"lcb_{i}_{idx}",
                        task_type="LiveCodeBench",
                        prompt_data={
                            "question": data.get("question", ""),
                            "test_cases": json.dumps(test_cases)
                        },
                        raw_question=data.get("question", ""),
                        gold_answer=json.dumps(test_cases)
                    ))
            if path.endswith("test.jsonl"):
                break
        return test_samples

    def load_math500_split(self) -> List[TaskSample]:
        path = os.path.join(self.base_dir, "MATH-500/test.jsonl")
        test_samples = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                raw_q = data.get("problem") or data.get("question") or ""
                gold = data.get("solution") or data.get("answer") or ""
                test_samples.append(TaskSample(
                    sample_id=f"math500_{idx}",
                    task_type="MATH",
                    prompt_data={"question": str(raw_q)},
                    raw_question=str(raw_q),
                    gold_answer=str(gold).strip()
                ))
        return test_samples