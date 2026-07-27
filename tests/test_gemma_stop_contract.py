import hashlib
import json
from types import SimpleNamespace

import pytest

from config.settings import ExperimentSettings
from run_pipeline import AcademicExperimentRunner, parse_stop_token_ids
from scripts.normalize_gemma_partial_cache_20260721 import normalize_cache


def test_stop_token_ids_are_canonical_and_sent_via_extra_body(monkeypatch):
    assert parse_stop_token_ids("") == ()
    assert parse_stop_token_ids("106,1,106") == (1, 106)
    with pytest.raises(ValueError, match="nonnegative"):
        parse_stop_token_ids("-1")

    runner = AcademicExperimentRunner(
        settings=ExperimentSettings(random_seed=7),
        model_name="google/gemma-3-12b-it",
        stop_token_ids=(106,),
        sampling_top_k=0,
    )
    observed = {}

    def create(**request):
        observed.update(request)
        token = SimpleNamespace(token="answer", logprob=-0.1, top_logprobs=[])
        choice = SimpleNamespace(
            message=SimpleNamespace(content="answer"),
            logprobs=SimpleNamespace(content=[token]),
        )
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(runner.generator.client.chat.completions, "create", create)
    generated = runner.generator.generate_consistency_samples(
        ["question"], num_samples=1, max_tokens=8, max_workers=1
    )
    assert observed["extra_body"] == {"stop_token_ids": [106], "top_k": 0}
    assert generated[0][0]["text"] == "answer"


def test_explicit_stop_contract_rejects_legacy_cache_header():
    runner = AcademicExperimentRunner(
        settings=ExperimentSettings(random_seed=7),
        model_name="google/gemma-3-12b-it",
        stop_token_ids=(106,),
    )
    samples = runner.loader.load_any_dataset("GSM8K", 1)
    dataset = runner.loader.dataset_fingerprint("GSM8K")
    header = {
        "run_mode": "smoke",
        "task": "GSM8K",
        "seed": 7,
        "num_samples": 1,
        "temperature": 0.7,
        "top_p": None,
        "max_tokens": 12000,
        "requested_top_logprobs": 10,
        "generation_protocol": runner.generator.generation_protocol("GSM8K"),
        "system_prompt_sha256": hashlib.sha256(
            runner.generator._system_prompt("GSM8K").encode("utf-8")
        ).hexdigest(),
        "input_sha256": hashlib.sha256(
            "\n".join(sample.sample_id for sample in samples).encode("utf-8")
        ).hexdigest(),
        "dataset": dataset,
    }
    with pytest.raises(ValueError, match="stop_token_ids"):
        runner._validate_cache_header(
            header, "GSM8K", samples, 1, dataset, "smoke"
        )


def test_explicit_sampling_top_k_rejects_implicit_cache_header():
    runner = AcademicExperimentRunner(
        settings=ExperimentSettings(random_seed=7),
        model_name="google/gemma-3-12b-it",
        stop_token_ids=(106,),
        sampling_top_k=0,
    )
    samples = runner.loader.load_any_dataset("GSM8K", 1)
    dataset = runner.loader.dataset_fingerprint("GSM8K")
    header = {
        "run_mode": "smoke",
        "task": "GSM8K",
        "seed": 7,
        "num_samples": 1,
        "temperature": 0.7,
        "top_p": None,
        "max_tokens": 12000,
        "requested_top_logprobs": 10,
        "stop_token_ids": [106],
        "generation_protocol": runner.generator.generation_protocol("GSM8K"),
        "system_prompt_sha256": hashlib.sha256(
            runner.generator._system_prompt("GSM8K").encode("utf-8")
        ).hexdigest(),
        "input_sha256": "not-reached",
        "dataset": dataset,
    }
    with pytest.raises(ValueError, match="sampling_top_k"):
        runner._validate_cache_header(
            header, "GSM8K", samples, 1, dataset, "smoke"
        )


def test_normalizer_preserves_source_and_binds_transform(tmp_path):
    source = tmp_path / "legacy.jsonl"
    output = tmp_path / "normalized.jsonl"
    header = {
        "record_type": "meta",
        "schema_version": 2,
        "task": "GSM8K",
        "num_samples": 2,
        "model": {"id": "google/gemma-3-12b-it"},
    }
    trace = {
        "text": "answer",
        "logprobs": [
            {"token": "answer", "logprob": -0.1},
            {"token": "<end_of_turn>", "logprob": 0.0},
            {"token": "<end_of_turn>", "logprob": 0.0},
        ],
    }
    record = {
        "record_type": "sample",
        "sample_id": "gsm8k_test_0",
        "prompt_sha256": "x",
        "traces": [trace, trace],
    }
    source_bytes = (
        json.dumps(header, sort_keys=True) + "\n" + json.dumps(record) + "\n"
    ).encode("utf-8")
    source.write_bytes(source_bytes)

    summary = normalize_cache(source, output, (106,), ("<end_of_turn>", "<eos>"))
    assert source.read_bytes() == source_bytes
    lines = output.read_text(encoding="utf-8").splitlines()
    normalized_header = json.loads(lines[0])
    normalized_record = json.loads(lines[1])
    assert normalized_header["stop_token_ids"] == [106]
    assert normalized_header["cache_transform"]["source_cache_sha256"] == (
        hashlib.sha256(source_bytes).hexdigest()
    )
    assert normalized_record["traces"][0]["logprobs"] == [
        {"token": "answer", "logprob": -0.1}
    ]
    assert summary["removed_logprob_items"] == 4


def test_normalizer_preserves_only_exact_length_capped_trace(tmp_path):
    source = tmp_path / "length.jsonl"
    output = tmp_path / "length-normalized.jsonl"
    header = {
        "record_type": "meta",
        "schema_version": 2,
        "task": "GSM8K",
        "num_samples": 1,
        "max_tokens": 2,
        "model": {"id": "google/gemma-3-12b-it"},
    }
    record = {
        "record_type": "sample",
        "sample_id": "gsm8k_test_0",
        "prompt_sha256": "x",
        "traces": [
            {
                "text": "unfinished",
                "logprobs": [
                    {"token": "one", "logprob": -0.1},
                    {"token": "two", "logprob": -0.2},
                ],
            }
        ],
    }
    source.write_text(
        json.dumps(header) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )
    summary = normalize_cache(source, output, (106,), ("<end_of_turn>",))
    assert summary["length_capped_traces"] == 1
    normalized = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
    assert len(normalized["traces"][0]["logprobs"]) == 2
