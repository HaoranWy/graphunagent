"""Deterministic local dataset loaders for the PWSC experiments."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import pickle
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - only relevant in a minimal install
    pd = None

logger = logging.getLogger("ise_flow.data.loaders")


@dataclass(frozen=True)
class TaskSample:
    sample_id: str
    task_type: str
    prompt_data: Dict[str, Any]
    raw_question: str
    gold_answer: str
    difficulty: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)


def _first(mapping: Dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None and str(value).lower() not in {"nan", "none"}:
            return value
    return default


class DatasetLoader:
    def __init__(self, base_dir: str = "/data/newmodel/graphunagent/datasets") -> None:
        self.base_dir = Path(base_dir)

    def _require(self, path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def dataset_fingerprint(self, task_name: str) -> Dict[str, Any]:
        """Return immutable source metadata for a run manifest."""
        paths: List[Path]
        if task_name == "GSM8K":
            paths = [self.base_dir / "gsm8k" / "main" / "test-00000-of-00001.parquet"]
        elif task_name == "HumanEvalPlus":
            paths = [self.base_dir / "humanevalplus" / "data" / "test-00000-of-00001-5973903632b82d40.parquet"]
        elif task_name == "MATH-500":
            paths = [self.base_dir / "MATH-500" / "test.jsonl"]
        elif task_name == "FOLIO":
            paths = [self.base_dir / "folio" / "validation.jsonl"]
        elif task_name == "P-FOLIO":
            paths = [self.base_dir / "P-FOLIO" / "FOLIO.csv", self.base_dir / "P-FOLIO" / "P-FOLIO.csv"]
        elif task_name == "AIME2025":
            paths = [self.base_dir / "AIME2025" / "aime2025-I.jsonl", self.base_dir / "AIME2025" / "aime2025-II.jsonl"]
        elif task_name == "AIME2026":
            paths = [self.base_dir / "AIME2026" / "aime2026.jsonl"]
        elif task_name == "LiveCodeBench":
            paths = [self.base_dir / "livecodebench" / ".hfd" / "repo_metadata.json"]
        else:
            raise NotImplementedError(f"No dataset fingerprint mapping for {task_name}")
        files = []
        for path in paths:
            self._require(path)
            files.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": self._sha256(path)})
        revision = None
        dataset_id = task_name
        metadata_roots = {
            "GSM8K": self.base_dir / "gsm8k",
            "MATH-500": self.base_dir / "MATH-500",
            "HumanEvalPlus": self.base_dir / "humanevalplus",
            "FOLIO": self.base_dir / "folio",
            "P-FOLIO": self.base_dir / "P-FOLIO",
            "AIME2025": self.base_dir / "AIME2025",
            "AIME2026": self.base_dir / "AIME2026",
            "LiveCodeBench": self.base_dir / "livecodebench",
        }
        metadata_path = metadata_roots[task_name] / ".hfd" / "repo_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                revision = metadata.get("sha")
                dataset_id = metadata.get("id", dataset_id)
            except (OSError, ValueError):
                pass
        return {"dataset_id": dataset_id, "revision": revision, "files": files}

    @staticmethod
    def _limit(items: List[TaskSample], limit: Optional[int]) -> List[TaskSample]:
        return items if limit is None or limit <= 0 else items[:limit]

    def load_gsm8k_split(self) -> Tuple[List[TaskSample], List[TaskSample]]:
        if pd is None:
            raise ImportError("pandas is required for GSM8K parquet files")
        local_dir = self.base_dir / "gsm8k" / "main"
        test_path = self._require(local_dir / "test-00000-of-00001.parquet")
        train_path = local_dir / "train-00000-of-00001.parquet"
        test_df = pd.read_parquet(test_path)

        def convert(frame: Any, prefix: str) -> List[TaskSample]:
            rows: List[TaskSample] = []
            for idx, row in frame.iterrows():
                answer = str(row.get("answer", ""))
                gold = answer.split("####")[-1].strip() if "####" in answer else answer
                question = str(row.get("question", ""))
                rows.append(TaskSample(f"{prefix}_{idx}", "GSM8K", {"question": question}, question, gold))
            return rows

        return (convert(pd.read_parquet(train_path), "gsm8k_train") if train_path.exists() else [], convert(test_df, "gsm8k_test"))

    def load_humanevalplus_test(self, limit: Optional[int] = None) -> List[TaskSample]:
        if pd is None:
            raise ImportError("pandas is required for HumanEvalPlus parquet files")
        path = self._require(self.base_dir / "humanevalplus" / "data" / "test-00000-of-00001-5973903632b82d40.parquet")
        frame = pd.read_parquet(path)
        samples: List[TaskSample] = []
        for idx, row in frame.iterrows():
            task_id = str(row.get("task_id", f"HumanEval/{idx}"))
            prompt = str(row.get("prompt", ""))
            entry = str(row.get("entry_point", ""))
            test_code = str(row.get("test", ""))
            samples.append(
                TaskSample(
                    sample_id=task_id.replace("/", "_"),
                    task_type="HumanEvalPlus",
                    prompt_data={"prompt": prompt, "test_code": test_code, "entry_point": entry},
                    raw_question=prompt,
                    gold_answer=test_code,
                    metadata={"entry_point": entry, "task_id": task_id},
                )
            )
        return self._limit(samples, limit)

    def load_aime2025_split(self, limit: Optional[int] = None) -> List[TaskSample]:
        samples: List[TaskSample] = []
        root = self.base_dir / "AIME2025"
        for filename in ("aime2025-I.jsonl", "aime2025-II.jsonl"):
            path = root / filename
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    data = json.loads(line)
                    question = str(_first(data, "question", "problem", default=""))
                    answer = str(_first(data, "answer", "solution", default="")).strip()
                    samples.append(TaskSample(f"aime2025_{filename}_{idx}", "AIME2025", {"question": question}, question, answer))
                    if limit and len(samples) >= limit:
                        return samples
        return samples

    def load_aime2026_split(self, limit: Optional[int] = None) -> List[TaskSample]:
        path = self._require(self.base_dir / "AIME2026" / "aime2026.jsonl")
        samples: List[TaskSample] = []
        with path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                data = json.loads(line)
                question = str(_first(data, "question", "problem", default=""))
                answer = str(_first(data, "answer", "solution", default="")).strip()
                samples.append(TaskSample(f"aime2026_{idx}", "AIME2026", {"question": question}, question, answer))
                if limit and len(samples) >= limit:
                    break
        return samples

    @staticmethod
    def _folio_prompt(premises: Any, conclusion: Any) -> str:
        if isinstance(premises, list):
            premise_text = "\n".join(str(item) for item in premises)
        else:
            premise_text = str(premises or "")
        return f"Premises:\n{premise_text}\n\nConclusion: {conclusion}\nIs the conclusion True, False, or Uncertain?"

    def load_folio_split(self, limit: Optional[int] = None) -> List[TaskSample]:
        path = self._require(self.base_dir / "folio" / "validation.jsonl")
        samples: List[TaskSample] = []
        with path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                data = json.loads(line)
                premises = _first(data, "premises", "story", default=[])
                conclusion = str(_first(data, "conclusion", default=""))
                label = str(_first(data, "label", "truth", "truth_value", default=""))
                prompt = self._folio_prompt(premises, conclusion)
                samples.append(TaskSample(f"folio_{idx}", "FOLIO", {"question": prompt}, prompt, label))
                if limit and len(samples) >= limit:
                    break
        return samples

    def load_p_folio_split(self, limit: Optional[int] = None) -> List[TaskSample]:
        """Load P-FOLIO questions from its paired FOLIO table.

        ``P-FOLIO.csv`` is a derivation-chain table and intentionally has no
        question-level premise/conclusion/label columns.  The release ships a
        paired ``FOLIO.csv`` containing the question-level labels; using that
        table avoids silently converting missing labels to ``uncertain``.
        """
        if pd is None:
            raise ImportError("pandas is required for P-FOLIO CSV files")
        root = self.base_dir / "P-FOLIO"
        question_path = self._require(root / "FOLIO.csv")
        frame = pd.read_csv(question_path)
        required = {"Premises - NL", "Conclusions - NL", "Truth Values"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"P-FOLIO question table missing columns: {sorted(missing)}")

        samples: List[TaskSample] = []
        for row_idx, row in frame.iterrows():
            premises = str(row["Premises - NL"] if pd.notna(row["Premises - NL"]) else "")
            conclusions = str(row["Conclusions - NL"] if pd.notna(row["Conclusions - NL"]) else "").splitlines()
            labels = str(row["Truth Values"] if pd.notna(row["Truth Values"]) else "").splitlines()
            if not conclusions or len(conclusions) != len(labels):
                logger.warning("Skipping P-FOLIO row %s with unaligned conclusions/labels", row_idx)
                continue
            for conclusion_idx, (conclusion, label) in enumerate(zip(conclusions, labels)):
                label = label.strip().upper()
                if not conclusion.strip() or label not in {"T", "F", "U"}:
                    logger.warning("Skipping P-FOLIO row %s conclusion %s with invalid label", row_idx, conclusion_idx)
                    continue
                prompt = self._folio_prompt(premises, conclusion)
                samples.append(
                    TaskSample(
                        f"pfolio_{row_idx}_{conclusion_idx}",
                        "P-FOLIO",
                        {"question": prompt},
                        prompt,
                        label,
                        metadata={"source_row": int(row_idx), "conclusion_index": conclusion_idx, "derivation_table": str(root / "P-FOLIO.csv")},
                    )
                )
                if limit and len(samples) >= limit:
                    return samples
        if not samples:
            raise ValueError("P-FOLIO loader produced no labeled samples")
        return samples

    @staticmethod
    def _decode_lcb_private(value: Any) -> List[Dict[str, Any]]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        try:
            return json.loads(str(value))
        except Exception:
            try:
                raw = base64.b64decode(str(value).encode("utf-8"))
                return json.loads(pickle.loads(zlib.decompress(raw)))
            except Exception as exc:
                raise ValueError("Unable to decode LiveCodeBench private_test_cases") from exc

    def load_livecodebench_split(self, limit: Optional[int] = None, release_files: Optional[Iterable[str]] = None) -> List[TaskSample]:
        root = self._require(self.base_dir / "livecodebench")
        names = list(release_files or ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"))
        samples: List[TaskSample] = []
        for filename in names:
            path = root / filename
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    data = json.loads(line)
                    public = json.loads(data.get("public_test_cases") or "[]")
                    private = self._decode_lcb_private(data.get("private_test_cases"))
                    tests = public + private
                    metadata = json.loads(data.get("metadata") or "{}")
                    fn_name = metadata.get("func_name")
                    input_output = {
                        "inputs": [str(item.get("input", "")) for item in tests],
                        "outputs": [str(item.get("output", "")) for item in tests],
                        "fn_name": fn_name,
                    }
                    title = str(data.get("question_title", ""))
                    question = str(data.get("question_content", ""))
                    starter = str(data.get("starter_code", ""))
                    prompt = "\n\n".join(part for part in (title, question, starter) if part)
                    sample_id = f"lcb_{data.get('question_id', filename)}"
                    samples.append(
                        TaskSample(
                            sample_id,
                            "LiveCodeBench",
                            {"question": prompt, "starter_code": starter},
                            prompt,
                            json.dumps(tests, ensure_ascii=False),
                            metadata={
                                "question_id": data.get("question_id"),
                                "platform": data.get("platform"),
                                "func_name": fn_name,
                                "input_output": input_output,
                                "dataset_file": filename,
                            },
                        )
                    )
                    if limit and len(samples) >= limit:
                        return samples
        if not samples:
            raise FileNotFoundError(f"No LiveCodeBench JSONL files found under {root}")
        return samples

    def load_math500_split(self, limit: Optional[int] = None) -> List[TaskSample]:
        path = self._require(self.base_dir / "MATH-500" / "test.jsonl")
        samples: List[TaskSample] = []
        with path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                data = json.loads(line)
                question = str(_first(data, "problem", "question", default=""))
                answer = _first(data, "answer", default=None)
                if answer is None or str(answer).strip() == "":
                    answer = data.get("solution", "")
                samples.append(TaskSample(f"math500_{idx}", "MATH-500", {"question": question}, question, str(answer).strip()))
                if limit and len(samples) >= limit:
                    break
        return samples
