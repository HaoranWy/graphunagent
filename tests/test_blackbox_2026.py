import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_blackbox_2026 import _rds_scores
import scripts.evaluate_csa_2026 as csa
from scripts.evaluate_csa_2026 import (
    NONE_OF_THE_ABOVE,
    _assessment_prompt,
    _official_cluster_choices,
)


def test_rds_is_zero_for_identical_unit_embeddings():
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    scores = _rds_scores(embeddings, np.asarray([0.1, 0.2, 0.3]))
    assert all(np.isclose(value, 0.0) for value in scores.values())


def test_rds_matches_centroid_distance_definition():
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    scores = _rds_scores(embeddings, np.asarray([0.2, 0.2]))
    assert np.isclose(scores["RDS-L1"], 2.0)
    assert np.isclose(scores["RDS-L2"], np.sqrt(2.0))
    assert np.isclose(scores["RDS-PW-L1"], 1.0)
    assert np.isclose(scores["RDS-PW-L2"], 1.0 / np.sqrt(2.0))


def test_probability_weighted_rds_handles_missing_path_likelihood():
    embeddings = np.eye(3, dtype=np.float64)
    scores = _rds_scores(
        embeddings, np.asarray([0.1, float("inf"), 0.5], dtype=np.float64)
    )
    assert all(np.isfinite(value) and value >= 0.0 for value in scores.values())


def test_csa_released_bidirectional_clustering_keeps_target_as_option_a():
    representations = ["target", "equivalent", "different", "equivalent"]
    unique = sorted(set(representations))
    labels = np.zeros((3, 3), dtype=np.int64)
    positions = {value: index for index, value in enumerate(unique)}
    labels[positions["target"], positions["equivalent"]] = 2
    labels[positions["equivalent"], positions["target"]] = 2

    choices = _official_cluster_choices(representations, unique, labels)
    assert choices == ["target", "different", NONE_OF_THE_ABOVE]


def test_csa_assessment_prompt_has_source_compatible_option_boundary():
    prompt = _assessment_prompt("What is correct?", ["yes", "no", NONE_OF_THE_ABOVE])
    assert "(A) yes" in prompt
    assert "(C) None of the above" in prompt
    assert prompt.endswith("The answer is (")


def test_csa_assessment_single_prompt_batches_are_resumable(tmp_path, monkeypatch):
    records = [
        {
            "sample_id": "sample-1",
            "question": "First question?",
            "representations": ["first answer"],
        },
        {
            "sample_id": "sample-2",
            "question": "Second question?",
            "representations": ["second answer"],
        },
    ]
    nli_records = {
        ("GSM8K", record["sample_id"], 0): {"semantic_labels": [[0]]}
        for record in records
    }
    request_sizes = []

    def fake_request(endpoint, model_name, api_key, prompts, timeout, retries):
        request_sizes.append(len(prompts))
        return [-0.25] * len(prompts)

    monkeypatch.setattr(csa, "_request_with_retries", fake_request)
    cache_path = tmp_path / "csa_assessment.jsonl"
    cache_records = csa._run_assessment_stage(
        tasks=["GSM8K"],
        sizes=[1],
        records_by_task={"GSM8K": records},
        nli_records=nli_records,
        cache_path=cache_path,
        cache_records={},
        endpoints=["http://127.0.0.1:8000/v1"],
        model_name="test-model",
        api_key="empty",
        batch_size=1,
        max_workers=1,
        timeout=30.0,
        retries=0,
    )

    assert request_sizes == [1, 1]
    assert len(cache_records) == 2
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 2
