import hashlib
import json

import pytest

from config.settings import ExperimentSettings
from run_pipeline import (
    CACHE_SCHEMA_VERSION,
    PWSC_CI_NAME,
    AcademicExperimentRunner,
    _input_sha256,
    _redact_command,
    build_parser,
    parse_ensemble_sizes,
    validate_protocol,
)


def test_cli_defaults_are_the_full_paper_protocol():
    args = build_parser().parse_args([])
    assert args.mode == "paper"
    assert args.limit == 0
    assert args.num_samples == 16
    assert parse_ensemble_sizes(args.ensemble_sizes) == (1, 4, 8, 16)
    assert args.top_p is None
    assert args.cache_output_dir == ""
    assert args.request_timeout == 3600.0
    assert args.max_retries == 0


def test_transport_settings_are_explicit_and_validated():
    args = build_parser().parse_args(
        ["--request_timeout", "1800", "--max_retries", "1"]
    )
    runner = AcademicExperimentRunner(
        settings=ExperimentSettings(random_seed=19),
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    assert runner.generator.request_timeout == 1800.0
    assert runner.generator.max_retries == 1
    with pytest.raises(ValueError, match="request_timeout"):
        AcademicExperimentRunner(request_timeout=0)
    with pytest.raises(ValueError, match="max_retries"):
        AcademicExperimentRunner(max_retries=-1)


def test_explicit_top_p_and_cache_destination_are_validated(tmp_path):
    args = build_parser().parse_args(
        ["--top_p", "0.9", "--cache_output_dir", str(tmp_path)]
    )
    runner = AcademicExperimentRunner(
        settings=ExperimentSettings(random_seed=19),
        model_name="deepseek",
        temperature=0.7,
        top_p=args.top_p,
        max_tokens=128,
        cache_output_dir=args.cache_output_dir,
    )
    samples = runner.loader.load_any_dataset("GSM8K", 1)
    dataset = runner.loader.dataset_fingerprint("GSM8K")
    header = {
        "run_mode": "smoke",
        "task": "GSM8K",
        "seed": 19,
        "num_samples": 1,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 128,
        "requested_top_logprobs": 10,
        "generation_protocol": runner.generator.generation_protocol("GSM8K"),
        "system_prompt_sha256": hashlib.sha256(
            runner.generator._system_prompt("GSM8K").encode("utf-8")
        ).hexdigest(),
        "input_sha256": _input_sha256(samples),
        "dataset": dataset,
    }
    runner._validate_cache_header(
        header, "GSM8K", samples, 1, dataset, "smoke"
    )
    header.pop("top_p")
    try:
        runner._validate_cache_header(
            header, "GSM8K", samples, 1, dataset, "smoke"
        )
        assert False, "explicit top_p must be present in a reusable cache"
    except ValueError as exc:
        assert "top_p" in str(exc)
    assert runner.cache_output_dir == tmp_path.resolve()


def test_paper_mode_rejects_partial_or_noncanonical_runs():
    for limit, num_samples, sizes in [
        (15, 16, (1, 4, 8, 16)),
        (0, 8, (1, 4, 8)),
    ]:
        try:
            validate_protocol("paper", "GSM8K", limit, num_samples, sizes)
            assert False, "noncanonical paper run must be rejected"
        except ValueError:
            pass
    validate_protocol("paper", "GSM8K", 0, 16, (1, 4, 8, 16))
    validate_protocol("paper", "AIME2025", 0, 16, (1, 4, 8, 16))
    validate_protocol("paper", "AIME2026", 0, 16, (1, 4, 8, 16))


def test_manifest_command_redacts_api_keys():
    command = ["run_pipeline.py", "--api_key", "secret", "--api_key=second"]
    assert _redact_command(command) == [
        "run_pipeline.py",
        "--api_key",
        "<redacted>",
        "--api_key=<redacted>",
    ]


def test_schema_v2_cache_runs_offline_with_only_pwsc_ci(tmp_path):
    runner = AcademicExperimentRunner(
        settings=ExperimentSettings(random_seed=11),
        model_name="deepseek",
        temperature=0.7,
        max_tokens=128,
    )
    samples = runner.loader.load_any_dataset("GSM8K", 2)
    dataset = runner.loader.dataset_fingerprint("GSM8K")
    cache_path = tmp_path / "cache.jsonl"
    header = {
        "record_type": "meta",
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": "PWSC-CI-v1.0",
        "run_mode": "smoke",
        "task": "GSM8K",
        "seed": 11,
        "num_samples": 2,
        "temperature": 0.7,
        "max_tokens": 128,
        "requested_top_logprobs": 10,
        "generation_protocol": runner.generator.generation_protocol("GSM8K"),
        "system_prompt_sha256": hashlib.sha256(
            runner.generator._system_prompt("GSM8K").encode("utf-8")
        ).hexdigest(),
        "input_sha256": _input_sha256(samples),
        "dataset": dataset,
        "model": {"id": "deepseek", "root": "/models/test"},
    }
    incompatible_header = dict(header)
    incompatible_header.pop("generation_protocol")
    try:
        runner._validate_cache_header(
            incompatible_header,
            "GSM8K",
            samples,
            2,
            dataset,
            "smoke",
        )
        assert False, "a cache without the frozen protocol tag must be rejected"
    except ValueError as exc:
        assert "generation_protocol" in str(exc)

    with cache_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        for sample in samples:
            traces = [
                {
                    "text": rf"reasoning \boxed{{{sample.gold_answer}}}",
                    "logprobs": [{"token": "answer", "logprob": -0.1}],
                },
                {
                    "text": r"reasoning \boxed{999999}",
                    "logprobs": [{"token": "answer", "logprob": -0.2}],
                },
            ]
            record = {
                "record_type": "sample",
                "sample_id": sample.sample_id,
                "prompt_sha256": hashlib.sha256(
                    sample.raw_question.encode("utf-8")
                ).hexdigest(),
                "traces": traces,
            }
            handle.write(json.dumps(record) + "\n")

    report, diagnostics, _correlation, rows = runner.run_benchmark_experiment(
        task_name="GSM8K",
        sample_limit=2,
        num_samples=2,
        ensemble_sizes=(1, 2),
        run_mode="smoke",
        cache_path=str(cache_path),
    )
    assert report["method"] == "PWSC-CI-v1.0"
    assert not report["publication_eligible"]
    assert report["evaluation_target"].startswith("correctness of the fixed first")
    assert list(report["Methods by N"]) == ["N=1", "N=2"]
    assert PWSC_CI_NAME in report["Methods by N"]["N=2"]
    assert "Worst-k NLL (k=5)" in report["Methods by N"]["N=2"]
    legacy_label = "Worst-" + str(5) + " NLL"
    assert legacy_label not in report["Methods by N"]["N=2"]
    assert all(
        not name.startswith("PWSC-") or name == PWSC_CI_NAME
        for methods in report["Methods by N"].values()
        for name in methods
    )
    assert report["run_metadata"]["generation_protocol"] == "proof-carrying-pc-v1"
    assert report["run_metadata"]["pwsc_ci_contract"] == (
        "pwsc-ci-normalized-holevo-v1"
    )
    assert "PWSC-CI Information Diagnostics by N" in report
    assert (
        report["PWSC-CI Information Diagnostics by N"]["N=2"]
        ["maximum_holevo_bound_excess"]
        <= 1e-10
    )
    assert set(report["Math Equivalence Runtime Diagnostics"]) == {
        "gold_label_verification",
        "task_answer_equivalence",
    }
    assert "pwsc_ci_risk_error_correlation" in diagnostics
    assert len(rows) == 4
    assert all(row["base_is_correct"] for row in rows)
    assert all("pwsc_ci_risk" in row for row in rows)
    assert all("pwsc_ci_holevo_information" in row for row in rows)
    assert all("pwsc_ci_partition_entropy" in row for row in rows)
    assert all(
        not key.startswith("pwsc_") or key.startswith("pwsc_ci_")
        for row in rows
        for key in row
    )


def test_partial_cache_resumes_only_after_strict_prefix_validation(tmp_path, monkeypatch):
    runner = AcademicExperimentRunner(
        settings=ExperimentSettings(random_seed=23),
        model_name="deepseek",
        temperature=0.7,
        max_tokens=128,
        generation_batch_size=1,
    )
    samples = runner.loader.load_any_dataset("GSM8K", 2)
    dataset = runner.loader.dataset_fingerprint("GSM8K")
    path = tmp_path / "partial.jsonl"
    header = {
        "record_type": "meta",
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": "PWSC-CI-v1.0",
        "run_mode": "smoke",
        "task": "GSM8K",
        "seed": 23,
        "num_samples": 2,
        "temperature": 0.7,
        "top_p": None,
        "max_tokens": 128,
        "requested_top_logprobs": 10,
        "generation_protocol": runner.generator.generation_protocol("GSM8K"),
        "system_prompt_sha256": hashlib.sha256(
            runner.generator._system_prompt("GSM8K").encode("utf-8")
        ).hexdigest(),
        "input_sha256": _input_sha256(samples),
        "dataset": dataset,
        "model": {"id": "deepseek", "root": "/models/test"},
    }

    def record(sample):
        return {
            "record_type": "sample",
            "sample_id": sample.sample_id,
            "prompt_sha256": hashlib.sha256(
                sample.raw_question.encode("utf-8")
            ).hexdigest(),
            "traces": [
                {
                    "text": rf"answer \boxed{{{sample.gold_answer}}}",
                    "logprobs": [{"token": "answer", "logprob": -0.1}],
                },
                {
                    "text": r"answer \boxed{0}",
                    "logprobs": [{"token": "answer", "logprob": -0.2}],
                },
            ],
        }

    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        handle.write(json.dumps(record(samples[0])) + "\n")

    with pytest.raises(ValueError, match="control character"):
        runner._prepare_generation_cache(
            "GSM8K",
            samples,
            2,
            "smoke",
            dataset,
            f"{path}\n  ",
            resume_cache=True,
        )

    with pytest.raises(ValueError, match="--resume_cache"):
        runner._prepare_generation_cache(
            "GSM8K", samples, 2, "smoke", dataset, str(path)
        )

    monkeypatch.setattr(
        runner.generator,
        "generate_consistency_samples",
        lambda _prompts, **_kwargs: [record(samples[1])["traces"]],
    )
    resumed, _digest, _model = runner._prepare_generation_cache(
        "GSM8K", samples, 2, "smoke", dataset, str(path), resume_cache=True
    )
    assert resumed == path.resolve()
    assert runner._cache_record_count(resumed, samples, 2) == 2


def test_incomplete_generation_preserves_prefix_and_reports_resume(tmp_path, monkeypatch):
    runner = AcademicExperimentRunner(
        settings=ExperimentSettings(random_seed=23),
        model_name="deepseek",
        temperature=0.7,
        max_tokens=128,
        generation_batch_size=2,
    )
    samples = runner.loader.load_any_dataset("GSM8K", 2)
    complete = [
        {
            "text": r"answer \boxed{1}",
            "logprobs": [{"token": "answer", "logprob": -0.1}],
        },
        {
            "text": r"answer \boxed{2}",
            "logprobs": [{"token": "answer", "logprob": -0.2}],
        },
    ]
    incomplete = [
        {
            "text": "",
            "logprobs": [],
            "error": "generation_error:APIConnectionError",
        }
        for _ in range(2)
    ]
    monkeypatch.setattr(
        runner.generator,
        "generate_consistency_samples",
        lambda _prompts, **_kwargs: [complete, incomplete],
    )
    path = tmp_path / "partial.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        with pytest.raises(RuntimeError) as exc_info:
            runner._generate_cache_records(handle, "GSM8K", samples, 2)

    message = str(exc_info.value)
    assert samples[1].sample_id in message
    assert "traces[0-1]" in message
    assert "generation_error:APIConnectionError" in message
    assert str(path.resolve()) in message
    assert "--resume_cache" in message
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["sample_id"] for record in records] == [samples[0].sample_id]
