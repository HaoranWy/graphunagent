# PWSC-CI code structure

Last synchronized: 2026-07-27 (Asia/Shanghai).

This document is the complete implementation map for the active PWSC-CI
research package. It covers runtime source files, orchestration scripts, tests,
claim-bearing data artifacts, and the boundary to archived or external
evidence.

`../PWSC_ALGORITHM_FLOW.md` is the normative algorithm specification. This
file describes where that specification is implemented and how data moves
through the repository. If prose here and the normative flow ever disagree,
the normative flow and executable tests must be reconciled before producing
new results.

Location check: the main algorithm code is physically under
`/data/newmodel/ISE/code`. The canonical numerical scorer is
`/data/newmodel/ISE/code/core/risk_estimator.py`; its task representations and
public code-behavior kernel are in the adjacent `core/adapters.py` and
`core/behavioral_kernel.py`. The supported CLI remains
`/data/newmodel/ISE/code/run_pipeline.py`. Repository-wide Python source search
finds the PWSC-CI contract constant, the two target-conditional-information
functions, and `PWSCCIRiskEstimator` only in `code/core/risk_estimator.py`.

## 1. Canonical boundary

- The sole canonical PWSC method is `PWSC-CI-v1.0`.
- The score contract is `pwsc-ci-normalized-holevo-v1`.
- The sole scorer class is `core.risk_estimator.PWSCCIRiskEstimator`.
- The sole scalar risk entry point is
  `core.risk_estimator.target_conditional_information_risk`; its component
  function exposes the same calculation for auditing.
- `core/behavioral_kernel.py` is the code-domain representation used by that
  scorer. It is not another uncertainty method.
- `routing/baselines.py` and the external-baseline scripts implement
  explicitly named non-PWSC comparators. They do not define another PWSC
  branch.
- `routing/calibration.py` is a development-only generic calibrator and is
  not imported by the canonical runner.
- The active claim-bearing result API is
  `diagnostics/pwsc_ci_frozen/pwsc_ci_results_20260717.json`. Historical raw
  files are immutable provenance, not a general active result API; the paper's
  separately labelled Qwen development table is a whitelisted read-only view
  over four archived reports and three non-PWSC sidecars.
- There is no physical or duplicate `code/ise_flow/` source tree. `code/` is
  the import root: tests add it through `pythonpath=["code"]`, while direct
  invocation of `python3 code/run_pipeline.py` adds the script directory to
  `sys.path`. Runtime imports are therefore `core.*`, `data.*`, `llm.*`,
  `routing.*`, `diagnostics.*`, and `config.*`; the `ise_flow.cli` logger name
  is not a second Python package.
- Top-level `scripts/` files orchestrate runs, validate evidence, or render
  paper artifacts. They may replay frozen equations for audits and controls,
  but none is the canonical runtime scorer.

## 2. Complete active tree

Every active file under `code/`, plus every first-party execution script and
test, is listed below. Python cache directories and transient run outputs are
not source files and are intentionally omitted.

```text
code/
|-- .vscode/
|   `-- settings.json
|-- code_structure.md
|-- config/
|   |-- __init__.py
|   `-- settings.py
|-- core/
|   |-- __init__.py
|   |-- adapters.py
|   |-- behavioral_kernel.py
|   `-- risk_estimator.py
|-- data/
|   |-- __init__.py
|   `-- loaders.py
|-- diagnostics/
|   |-- README.md
|   |-- __init__.py
|   |-- analyzer.py
|   |-- paper_cache/
|   |   |-- generations_FOLIO_1784135527551762941_fda05befd9.jsonl
|   |   |-- generations_GSM8K_1784181874597686128_b59de1d9fc.jsonl
|   |   |-- generations_HumanEvalPlus_1784134518412631283_e949a5adca.jsonl
|   |   `-- generations_MATH-500_1784139098232168849_6954315d92.jsonl
|   `-- pwsc_ci_frozen/
|       `-- pwsc_ci_results_20260717.json
|-- llm/
|   |-- __init__.py
|   `-- generator.py
|-- routing/
|   |-- __init__.py
|   |-- baselines.py
|   |-- calibration.py
|   `-- verifier_manager.py
`-- run_pipeline.py

scripts/
|-- analyze_ci_independent_promotion.py
|-- build_overview_compliant.py
|-- build_overview_motivation_innovations.py
|-- build_overview_v8_draft.py
|-- build_q_rasc_algorithm_overview.py
|-- check_nccl_tp4.py
|-- evaluate_blackbox_2026.py
|-- evaluate_csa_2026.py
|-- evaluate_pwsc_ci_controls.py
|-- evaluate_sese_2026.py
|-- normalize_gemma_partial_cache_20260721.py
|-- plot_pwsc_ci_ablation.py
|-- plot_q_rasc_motivation.py
|-- plot_qrasc_main_evidence.py
|-- probe_gemma_stop_20260721.py
|-- refine_qrasc_figures.py
|-- resume_gemma_gsm8k_20260716.sh
|-- resume_llama_csa_20260716.sh
|-- resume_llama_math500_20260716.sh
|-- run_gemma_main_20260716_clean.sh
|-- run_llama70b_main_20260723.sh
|-- run_llama_sese_20260716.sh
|-- run_paper_matrix.sh
|-- run_qwen_main_20260721.sh
|-- seal_ci_promotion_comparator.py
|-- serve_llama70b_8gpu_20260723.sh
|-- serve_llama_csa_20260716.sh
|-- summarize_gemma_main_20260722.sh
|-- summarize_llama70b_main_20260723.sh
|-- summarize_llama_supplement_20260716_18.sh
|-- summarize_pwsc_ci_supplement.py
|-- summarize_qwen_main_20260721.sh
|-- validate_pwsc_ci_code_replay.py
`-- validate_pwsc_ci_freeze.py

tests/
|-- test_analyze_ci_independent_promotion.py
|-- test_blackbox_2026.py
|-- test_data_and_verifiers.py
|-- test_gemma_stop_contract.py
|-- test_metrics.py
|-- test_paper_runner.py
|-- test_pwsc_ci.py
|-- test_pwsc_ci_frozen_replay.py
|-- test_pwsc_ci_supplement.py
|-- test_seal_ci_promotion_comparator.py
|-- test_sese_2026.py
`-- test_validate_pwsc_ci_freeze.py
```

Related but isolated directories:

```text
external_baselines/
|-- README.md
|-- manifest.json
|-- RDS/
|-- SDLG/
|-- SeSE/
|-- TokUR/
|-- TruthTorchLM/
|-- clustered_self_assessment/
|-- lm-polygraph/
`-- uncertainty_in_the_wild/

.pipeline/archive/pwsc_retired_results_20260717/
|-- immutable_provenance/              # hash-bound source evidence
`-- supplemental_experiments_20260718/ # isolated post-review results
```

The package `__init__.py` files are whitespace-only namespace markers. They
establish importable namespaces without re-exporting hidden compatibility APIs.
`.vscode/settings.json` affects editor behavior only. It is not imported by
the runtime.

The synchronized inventory contains exactly 25 non-transient files under
`code/` (including this map, four immutable generation caches, and one result
freeze), 34 top-level first-party scripts, and 12 test files. Every path appears
explicitly in the tree above. Three of the 34 scripts
(`build_overview_compliant.py`, `build_overview_motivation_innovations.py`, and
`build_overview_v8_draft.py`) are historical PowerPoint-draft builders rather
than current paper or experiment entry points; they remain listed so the path
audit is complete.

## 3. Runtime architecture

```text
CLI arguments and PYTHONHASHSEED
              |
              v
run_pipeline.validate_protocol
              |
              +----> UniversalDatasetLoader
              |          |-- TaskSample objects
              |          `-- dataset file/revision/SHA-256 fingerprint
              |
              +----> model metadata and revision validation
              |
              v
schema-v2 generation cache
  |-- reuse: validate all header fields and align every sample
  `-- create: APIGenerator -> text + chosen-token logprobs
              (optional explicit top_p; configurable cache destination)
              |
              v
for each question and each prefix N in {1,4,8,16}
              |
              v
PWSCCIRiskEstimator.score
  |-- retain chosen-token logprobs
  |-- weakest-k action and confidence mass (canonical k=5)
  |-- task adapter validity and structural features
  |-- code only: public behavior kernel or AST fallback
  |-- PSD Gram validation
  |-- normalized target-versus-peer Holevo information
  `-- independent confidence-centrality answer selection
              |
              +----> VerifierManager -> fixed-first error labels and
              |                         answer-selection accuracy
              |
              +----> SCSVBaselines -> non-PWSC paired comparators
              |
              v
ScientificSignalAnalyzer
  |-- AUROC, AUPRC, AURC, selective accuracy
  `-- bootstrap intervals and paired differences
              |
              v
sample CSV + report JSON + run manifest
              |
              v
promotion scripts -> CI-only result freeze -> provenance/replay validators
supplement scripts -> hash-bound controls/cross-seed summaries (isolated)
```

The verifier path is data-separated from the scorer: gold answers and released
tests are passed only to verifier objects. No verifier result, gold answer, or
test case is an argument to `PWSCCIRiskEstimator.score`. Verifier labels are
used downstream for metrics and selection-accuracy reporting.

## 4. Module ownership

### 4.1 Configuration

#### `code/config/settings.py`

Owns immutable method constants and runner-level reproducibility settings.

- `PWSCCIConfig`
  - `weakest_k = 5`
  - `method_version = "PWSC-CI-v1.0"`
  - `contract_version = "pwsc-ci-normalized-holevo-v1"`
- `ExperimentSettings`
  - wraps `PWSCCIConfig`
  - stores `random_seed`
  - keeps `use_calibration=False`
  - serializes through `to_dict()` and `to_json()`

No dataset, model, or result-specific value is silently inferred here.

### 4.2 Canonical scoring core

#### `code/core/risk_estimator.py`

This is the only PWSC implementation.

Public API:

```python
PWSC_CI_CONTRACT_VERSION
PathAction
PWSCCIRiskEstimator
retain_scoring_tokens
target_conditional_information_components
target_conditional_information_risk
```

`PWSCCIRiskEstimator.path_action()` exposes the frozen weakest-token quality
primitive, and `PWSCCIRiskEstimator.score()` is the sole end-to-end scorer.
Private entropy/kernel validation helpers remain implementation details within
this module and do not define additional scoring APIs.

Responsibilities:

1. Remove empty, whitespace, control-token, non-finite, and malformed chosen
   token logprob entries.
2. Compute each path action from the mean of the largest
   `min(weakest_k, token_count)` token NLL values.
3. Convert action to confidence `exp(-action)`; invalid paths receive zero
   mass.
4. Ask the task adapter for a gold-free validity mask and normalized feature
   rows.
5. Build and validate a symmetric, unit-active-diagonal PSD Gram matrix.
6. Replace the code AST kernel with the public behavior kernel when active
   probes exist.
7. Compute the fixed-index-zero target-versus-peer normalized Holevo score.
8. Return both the scalar risk and complete audit diagnostics.

Important edge contracts:

- A single path returns `1 - q_0`.
- Missing/invalid target mass, missing peer mass, or zero total mass fails
  closed to risk `1.0` for multi-path scoring.
- A non-symmetric, non-PSD, non-finite, or non-unit active kernel is rejected.
- The risk lies in `[0,1]` and is a ranking score, not a calibrated
  probability.
- Selection uses `argmax_i q_i (K w)_i` and is separate from fixed-first UQ
  labels.

The score diagnostics contain:

```text
method, contract_version, weakest_k
path_actions, path_confidences, valid_mask, q, valid_count
domain_family, outcomes, raw_outcomes
task_kernel, kernel, kernel_source, kernel_diagnostics
weights, centrality, candidate_scores
selected_index, selected_outcome, selected_raw_outcome
ci_components, risk
```

The `ci_components` mapping contains
`normalized_information`, `holevo_information`,
`partition_entropy`, `target_mass_ratio`, `peer_mass_ratio`,
`peer_entropy`, `mixture_entropy`, `evidence_deficit`, and
`degenerate`.

#### `code/core/adapters.py`

Owns deterministic task interpretation. It never reads a gold label while
building the PWSC-CI representation.

Public API groups:

- Extraction: `extract_python_code`, `extract_last_boxed`.
- Canonicalization: `canonical_logic_label`,
  `canonical_math_answer`, `normalized_code_key`.
- Code representation: `code_ast_feature`.
- Bounded math equivalence: `bounded_math_equivalent`,
  `bounded_math_verify`, `math_equivalence_runtime_diagnostics`, and
  `official_math_verifier_runtime_diagnostics`.
- Equivalence construction: `transitive_classes`.
- Adapter object: `TaskAdapter` and `make_adapter`.

`TaskAdapter` maps schema names into `math`, `logic`, `code`, or
generic `text` families and exposes:

- `outcome_keys(texts)`
- `semantic_class_keys(texts, valid_mask)`
- `valid_mask(texts)`
- `feature_matrix(texts, valid_mask)`

Math equivalence is bounded to conservative inputs and a 0.5-second attempt;
timeouts and failures fail closed. Logic uses the last explicit
True/False/Uncertain label. Code validity requires parseable Python, a
non-zero AST feature, and the public entry point.

#### `code/core/behavioral_kernel.py`

Builds the code-domain Gram representation from public information only.

Public API:

```python
generate_behavior_probes
build_behavior_kernel
```

The probe generator reads the public prompt, expected function signature, type
hints, parameter names, and docstring call inputs. It emits at most 16
deterministic calls. It never reads expected outputs, benchmark tests, private
tests, or reference solutions.

Candidate execution is isolated by a child worker with:

- 1 second CPU limit
- 2 GiB address-space limit
- zero-byte output-file size limit
- at most 32 open file descriptors
- 2.5 second parent wall timeout
- an import/builtin allowlist
- blocked dunder-attribute access and isolated stdout/stderr
- normalized and hashed return values

A probe is active only when at least two valid candidates execute it
successfully. Exact output-hash agreement produces a scaled one-hot feature
Gram. Candidate-specific failure categories prevent shared failure from
creating false agreement. If no probe is active, the scorer retains the AST
kernel.

The canonical first pass evaluates the complete 16-probe set. If exactly one probe is
unsupported by every candidate, all-error candidates are retried once on the
supported calls. Multiple universally unsupported probes are not recovered.
The deterministic probe-plan LRU is bounded to 512 entries, and the program
fingerprint LRU is bounded to 256 entries.

`build_behavior_kernel(..., max_probes=16,
recover_single_unsupported=True)` exposes those defaults as keyword-only
diagnostic controls. The canonical scorer never overrides them. The
supplemental control driver may set `{4,8,16}` and recovery on/off to isolate
representation behavior without defining another scorer.

### 4.3 Dataset layer

#### `code/data/loaders.py`

Defines the immutable input object:

```python
TaskSample(
    sample_id: str,
    task_type: str,
    prompt_data: dict,
    raw_question: str,
    gold_answer: str,
    difficulty: int = 3,
    metadata: dict = {},
)
```

`DatasetLoader` provides:

- `dataset_fingerprint(task_name)`
- `load_gsm8k_split()`
- `load_humanevalplus_test()`
- `load_mbppplus_test()`
- `load_math500_split()`
- `load_folio_split()`
- `load_aime2025_split()`
- `load_aime2026_split()`
- `load_p_folio_split()`
- `load_livecodebench_split()`

Fingerprints contain dataset ID, optional upstream revision, absolute source
paths, byte counts, and SHA-256 values. Missing files fail explicitly.
`UniversalDatasetLoader` in `run_pipeline.py` supplies the single
`load_any_dataset()` dispatch used by the CLI.

P-FOLIO is loaded from its question-level `FOLIO.csv`; the derivation table
is not silently treated as labeled questions. LiveCodeBench private cases are
stored in sample metadata for post-score evaluation. Although that metadata
object accompanies a scoring call, the adapter and behavior kernel read only
public prompt/entry-point fields and never consume `input_output` test cases.

MBPP+ is an optional stress task with 378 rows. Its function name, arity, and
public calls are derived only from released `test_list` assertions. The loader
constructs a public signature stub and never copies the reference `code` field
into the prompt or metadata. The full plus-test program is retained only as
the post-score gold verifier input.

### 4.4 Generation and cache layer

#### `code/llm/generator.py`

`APIGenerator` provides:

- `describe_model()`
- `generation_protocol(task_type)`
- `generate_consistency_samples(...)`

It calls an OpenAI-compatible local endpoint with deterministic seed,
temperature, optional explicit `top_p`, `n`, chosen-token logprobs, requested
top-logprob alternatives, an explicit request timeout, and an explicit retry
limit. Local clients disable inherited proxies; expensive generation defaults
to a 3,600-second timeout with zero automatic request replay. The supplemental
Llama matrix explicitly permits one SDK retry for transient connection/timeout
failures; the slower Gemma recovery keeps zero retries. Each returned trace has:

```text
text
logprobs[] -> {token, logprob, top_logprobs}
attentions  -> always null in this cache contract
error       -> present only for failed or incomplete generation
```

`PROOF_CARRYING_GENERATION_PROTOCOL = "proof-carrying-pc-v1"` is retained
solely to replay immutable schema-v2 caches. The math answer canonicalizer
removes the tagged certificate suffix, but no certificate field, proof step,
or certificate verdict contributes to PWSC-CI. Some frozen cache headers still
record the historical metadata string `"method": "PWSC-v1.0"`; that header
value is immutable provenance and does not identify executable scorer code.

### 4.5 Evaluation, baselines, and statistics

#### `code/routing/verifier_manager.py`

`VerifierSpec` owns verifier identity, TPR/FPR metadata, and the cost vector;
`BaseVerifier.execute()` defines the common fail-closed interface. Concrete
post-generation correctness evaluators are:

- `HumanEvalPlusVerifier`: executes released tests in a timed child.
- `MBPPPlusVerifier`: executes released MBPP+ tests in a timed child.
- `MATHOfficialVerifier`: exact normalization, bounded official
  `math-verify`, then conservative numeric comparison.
- `FOLIOVerifier`: normalized three-way label comparison.
- `LiveCodeBenchVerifier`: invokes the local official test runner in a timed
  child.
- `GSM8KOfficialVerifier`: boxed/final numeric comparison.
- `VerifierManager`: verifier registry and legacy calibrated-EVV helper.

`VerifierManager.execute()` dispatches a named evaluator.
`VerifierManager.compute_evv()` requires an already calibrated error
probability. Raw PWSC-CI risk is not passed to it by the canonical runner.

#### `code/routing/baselines.py`

`SCSVBaselines` contains non-PWSC comparison formulas: mean NLL, worst-K NLL,
self-consistency, semantic entropy, MBR, target consistency, kernel language
entropy, outcome-kernel CoCoA-MSP/PPL adaptations, self-certainty, DeepConf,
LL, P(True), Inside EigenScore, SAR, U-Ecc, and U-Deg.

Exact public methods are:

```text
calculate_nll, calculate_worst_k_nll
calculate_self_consistency, calculate_semantic_entropy
calculate_mbr, calculate_target_consistency
calculate_kernel_language_entropy
calculate_cocoa_msp, calculate_cocoa_ppl
calculate_self_certainty, calculate_deep_conf, calculate_ll, calculate_p_true
calculate_inside_eigen_score, calculate_sar, calculate_u_ecc, calculate_u_deg
```

Only methods for which the runner has required inputs are emitted. KLE,
target-consistency, and CoCoA variants are explicitly labeled as task/outcome
kernel adaptations. Hidden-state Inside EigenScore and unavailable SAR inputs
return `NaN` rather than fabricated scores. This module does not expose a PWSC
scorer.

#### `code/routing/calibration.py`

`DevPlattCalibrator` implements `fit`, `calibrate`, and
`calibrate_many` for development-only analysis. It is tested for monotonic
behavior but is not imported by `run_pipeline.py`, is not fitted on frozen
test labels, and is not part of `pwsc_ci_risk`.

#### `code/diagnostics/analyzer.py`

`ScientificSignalAnalyzer` owns:

- AUROC and AUPRC with explicit single-class undefined behavior
- risk-coverage and AURC
- selective and top-50 accuracy
- ECE and Brier score helpers
- Pearson correlation
- AST entropy diagnostic
- bootstrap confidence intervals
- paired bootstrap metric differences
- scientific-signal audit output

Its exact public methods are:

```text
calculate_auroc, calculate_auprc
risk_coverage, calculate_aurc
calculate_selective_accuracy, calculate_top_50_acc
calculate_ece, calculate_brier_score
calculate_pearson_correlation, compute_ast_entropy
bootstrap_ci, paired_bootstrap_difference
audit_m2_scientific_signal
```

The publication runner uses fixed-first error labels for every paired UQ
method. Answer-selection accuracy is reported in a separate block.

### 4.6 End-to-end runner

#### `code/run_pipeline.py`

This file preserves the established CLI while composing all active layers.

Key constants:

```python
CORE_TASKS = ("GSM8K", "MATH-500", "HumanEvalPlus", "FOLIO")
SECONDARY_TASKS = ("AIME2025", "AIME2026")
PAPER_TASKS = CORE_TASKS + SECONDARY_TASKS
PRIMARY_TASKS = PAPER_TASKS  # compatibility alias; not the frozen claim scope
DEFAULT_ENSEMBLE_SIZES = (1, 4, 8, 16)
CACHE_SCHEMA_VERSION = 2
BOOTSTRAP_SAMPLES = 1000
PWSC_CI_NAME = "PWSC-CI"
```

Main public/compatibility functions:

- `filter_reasoning_logprobs_full()`: retains the complete chosen-token
  stream for current scoring.
- `extract_homotopy_outcome()`: old call-name compatibility wrapper that
  delegates to the current task adapter; it is not an old scorer.
- `parse_ensemble_sizes()` and `validate_protocol()`.
- `compute_academic_metrics()`.
- `UniversalDatasetLoader.load_any_dataset()`.
- `AcademicExperimentRunner.preflight()`.
- `AcademicExperimentRunner.run_benchmark_experiment()`.
- `print_final_academic_table()`.
- `print_hypothesis_proof_table()`: retained four-value runner output hook;
  it prints current CI diagnostics only.
- `save_results_to_csv()`.
- `build_parser()` and `main()`.

`AcademicExperimentRunner` executes these stages:

1. Validate mode, task, dataset limit, sample count, and ensemble prefixes.
2. Load the complete/local split and compute its fingerprint.
3. Resolve server or cache model metadata and enforce the requested revision.
4. Validate and reuse a schema-v2 cache, or generate a new one in the default
   or explicitly selected cache directory.
5. Enforce one aligned cache record per dataset sample and reject extras.
6. Abort a paper run on missing response text or chosen-token logprobs.
7. Build the fixed-first correctness label through the separate task verifier;
   the value is retained only for downstream evaluation.
8. Score prefixes through the sole `PWSCCIRiskEstimator`, which receives no
   label or gold answer.
9. Compute available non-PWSC baselines on the same fixed labels.
10. Compute per-N metrics, paired bootstrap differences, CI diagnostics, and
    separate selection accuracy.
11. Return `(report, diagnostics, correlation, rows)`.
12. Write sample CSV, report JSON, and provenance manifest, including cache/
    generation, scoring/verification, metric-analysis, and total wall time.

The CLI process re-executes itself with
`PYTHONHASHSEED = seed mod 2**32` when needed. API keys are redacted from the
saved command.

## 5. Canonical scoring flow

For paths `i = 0,...,N-1`, with path zero fixed as the UQ target:

```text
chosen-token logprobs
  -> remove controls/whitespace/non-finite entries
  -> token NLL values
  -> a_i = mean(largest weakest_k NLL values)
  -> confidence c_i = exp(-a_i)
  -> q_i = c_i * logprob_valid_i * domain_valid_i

responses + public task metadata
  -> canonical outcomes and normalized features Phi
  -> task Gram K_task = Phi Phi^T
  -> code ensemble: replace with K_behavior when active public probes exist
  -> symmetric/unit-diagonal/PSD checks

q and K
  -> target mass versus confidence-weighted peer density operators
  -> peer entropy and mixture entropy
  -> Holevo information
  -> divide by natural target/peer binary partition entropy
  -> pwsc_ci_risk in [0,1]
```

This is the only call path that produces a field named
`pwsc_ci_risk`.

## 6. Task capability versus evidence status

Code support and publication admission are different contracts.

| Task | Family | Loader | Verifier | Runner capability | Active frozen claim |
|---|---|---|---|---|---|
| GSM8K | math | parquet test split | GSM8K numeric | full `paper`, `smoke` | yes |
| MATH-500 | math | JSONL test split | bounded math official | full `paper`, `smoke` | yes |
| HumanEvalPlus | code | parquet test split | released tests | full `paper`, `smoke` | yes |
| MBPPPlus | code | parquet test split | released plus tests | optional `stress` or `smoke` | no |
| FOLIO | logic | validation JSONL | three-way label | full `paper`, `smoke` | yes |
| AIME2025 | math | two JSONL releases | bounded math official | secondary `paper`, `smoke` | no |
| AIME2026 | math | JSONL release | bounded math official | secondary `paper`, `smoke` | no |
| P-FOLIO | logic | paired CSV release | three-way label | optional `stress` or `smoke` | no |
| LiveCodeBench | code | local JSONL releases | official local runner | optional `stress` or `smoke` | no |
| SMT | logic adapter token only | none | none | no CLI task | no |

The active freeze contains only GSM8K, MATH-500, HumanEvalPlus, and FOLIO at
`N=16`. A runner field such as `benchmark_tier=secondary` or a
protocol-complete optional run does not authorize a paper claim. The current
experiment plan, result summary, evidence gate, and freeze determine admission.
Unverified SMT and LiveCodeBench values must not enter the main table.

## 7. Data contracts

### 7.1 Generation cache schema v2

The first JSONL record has `record_type="meta"` and includes:

```text
schema_version, method, run_mode, task
seed, python_hash_seed
num_samples, temperature, top_p, max_tokens, requested_top_logprobs
generation_protocol
system_prompt, system_prompt_sha256
input_sha256
dataset
model
```

Each following record has:

```text
record_type="sample"
sample_id
prompt_sha256
traces[]
```

Cache reuse validates run mode, task, seed, number of samples, temperature,
optional explicit top-p, max tokens, requested top-logprobs, generation protocol, system-prompt hash,
ordered input hash, Python hash seed when present, dataset fingerprint, model
ID, and expected model revision. It also validates exact sample order, prompt
hashes, record count, and absence of trailing records.

The ordered input hash binds only `sample_id` and public prompt. Dataset
fingerprints separately bind the underlying release files.

### 7.2 Per-sample output rows

Every row contains:

```text
sample_id, task, ensemble_size
base_is_correct, fixed_first_error
pwsc_ci_risk
pwsc_ci_holevo_information
pwsc_ci_partition_entropy
pwsc_ci_target_mass_ratio
pwsc_ci_peer_mass_ratio
pwsc_ci_peer_entropy
pwsc_ci_mixture_entropy
pwsc_ci_evidence_deficit
pwsc_ci_degenerate
target_path_confidence, mean_path_confidence, valid_path_count
pwsc_ci_kernel_source
pwsc_ci_probe_count, pwsc_ci_active_probe_count, pwsc_ci_successful_calls
pwsc_ci_selected_index, pwsc_ci_selected_is_correct
self_consistency_selected_index, self_consistency_selected_is_correct
mbr_selected_index, mbr_selected_is_correct
mean_nll, worst_5_nll
baseline_* fields for explicitly named non-PWSC methods available at that N
```

No old PWSC variant score is emitted.

### 7.3 Report JSON

The report records:

- method, task, benchmark tier, mode, and protocol eligibility
- requested/scored/skipped counts and class counts
- the fixed-first evaluation target
- `Methods by N`
- `Selection Accuracy by N`
- PWSC-CI paired bootstrap differences
- PWSC-CI information-component summaries
- bounded-math runtime diagnostics
- correct/incorrect group diagnostics
- complete run metadata and methodological notes
- phase-separated wall-time diagnostics and cache-reuse status

`publication_eligible` means the runner's mechanical protocol checks passed.
It does not supersede the project evidence gate or freeze admission policy.

### 7.4 Run manifest

`save_results_to_csv()` writes:

```text
sample_results_<TASK>_<RUN_ID>.csv
report_<TASK>_<RUN_ID>.json
manifest_<TASK>_<RUN_ID>.json
```

The manifest binds:

- method, task, mode, tier, eligibility, class counts, and caveats
- redacted command, Python/platform, and package versions
- exact model ID, resolved revision, and local descriptor fingerprint
- dataset ID/revision/file byte counts/file SHA-256 values
- seed, Python hash seed, sampling settings, prompt/protocol hashes
- PWSC-CI contract and bootstrap count
- timing fields and whether the generation cache was reused
- ordered input hash and generation-cache path/hash
- sample count and CSV/report paths/hashes

### 7.5 Active freeze

`code/diagnostics/pwsc_ci_frozen/pwsc_ci_results_20260717.json` uses schema
`pwsc-ci-result-freeze-v1` and contains:

```text
schema, method, contract, score_column, status, frozen_at
evidence_scope, datasets, task_metrics, macro_metrics
promotion_gates, decision, provenance, audit_note
```

Its canonical scope is Llama-3.1-8B-Instruct, seed/Python hash seed
`20260716`, `N=16`, temperature 0.7, four complete splits, and 2,187
questions. `score_column` is exactly `pwsc_ci_risk`.

The freeze is a promotion artifact assembled from hash-bound manifests, sample
rows, external comparator records, preregistration, and decision data. It is
not silently regenerated by a routine runner invocation.

## 8. CLI contract

Supported arguments:

```text
--mode {paper,stress,smoke}
--task {GSM8K,HumanEvalPlus,MBPPPlus,AIME2025,AIME2026,FOLIO,P-FOLIO,
        LiveCodeBench,MATH-500}
--limit
--seed
--model_url
--model_name
--expected_model_revision
--api_key
--dataset_root
--top_k
--num_samples
--ensemble_sizes
--temperature
--top_p
--max_tokens
--max_workers
--generation_batch_size
--request_timeout
--max_retries
--cache_path
--cache_output_dir
--resume_cache
--output_dir
--preflight
```

Mode invariants:

- `paper`: task must be in `PAPER_TASKS`, `--limit 0`, 16 paths, and
  prefixes `1,4,8,16`.
- `smoke`: requires a positive limit.
- `stress`: accepts optional domains and rejects declared paper tasks.
- Any mode requires `num_samples >= max(ensemble_sizes)`.
- `--top_k` controls API alternatives stored per token. It does not change
  the frozen weakest-token count of 5.
- `--top_p` is optional. Omitting it preserves old server-default behavior;
  supplemental cross-seed runs set it explicitly and bind it in the cache and
  manifest.
- `--request_timeout` and `--max_retries` are transport controls, defaulting to
  3,600 seconds and zero retries. They do not change the sampling contract;
  preflight, reports, and manifests expose them for runtime provenance.
- `--preflight` checks protocol, model, dataset, and optional cache without
  generation or scoring.
- `--cache_path` enables exact offline replay when every schema field
  matches.
- `--cache_output_dir` redirects only newly created schema-v2 caches; the
  established default path and old invocations remain unchanged.
- `--resume_cache` requires `--cache_path`, validates a complete aligned
  record prefix, and appends only missing samples. Malformed, misordered,
  incomplete-record, prompt-mismatched, or extra-record caches fail closed.
- Cache paths containing CR, LF, or NUL control characters are rejected before
  path resolution, so a copied line wrap cannot masquerade as a missing file.
- Incomplete live generations are never written as sample records. The runner
  flushes the complete prefix and reports the failed trace indices/error class,
  exact cache path, and `--resume_cache` invocation hint.

Canonical preflight example:

```bash
PYTHONHASHSEED=20260716 python3 code/run_pipeline.py \
  --mode paper \
  --task GSM8K \
  --limit 0 \
  --seed 20260716 \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --expected_model_revision /data/newmodel/IE/LLM/Llama-3.1-8B-Instruct \
  --dataset_root /data/newmodel/graphunagent/datasets \
  --num_samples 16 \
  --ensemble_sizes 1,4,8,16 \
  --temperature 0.7 \
  --max_tokens 12000 \
  --cache_path code/diagnostics/paper_cache/generations_GSM8K_1784181874597686128_b59de1d9fc.jsonl \
  --preflight
```

## 9. Supporting scripts

### Publication execution

#### `scripts/check_nccl_tp4.py`

Runs a four-rank local NCCL all-reduce with a 30-second process-group timeout
and verifies the exact sum before a 70B service loads weights. It is a
deployment correctness probe, not a benchmark or a source of paper results.

#### `scripts/serve_llama70b_8gpu_20260723.sh`

Validates the frozen 70B model index, 30 weight shards, eight idle A800 GPUs,
and ports 8002/8003, then runs the NCCL probe on GPU islands 0--3 and 4--7. It
forces P2P to NVLink links only and shared-memory transport between the two
NVLink pairs, then starts two independent TP=4 vLLM services. It never starts
the experiment client.

#### `scripts/run_llama70b_main_20260723.sh`

Requires both TP=4 endpoints, their exact model/revision/serving arguments and
NCCL environment, then runs chosen-token-logprob and paper preflights. It
deterministically divides the three-seed by four-task matrix across the two
endpoints, supports strict cache-prefix resume, and requires exactly twelve
manifests before reporting completion.

#### `scripts/summarize_llama70b_main_20260723.sh`

Checks exactly one manifest for every predeclared 70B seed/task pair before
calling the shared whitelist summarizer with the frozen bootstrap contract. It
does not admit the aggregate to the paper; a separate evidence gate remains
mandatory.

#### `scripts/run_paper_matrix.sh`

Runs preflight and full CLI execution over environment-selected models, seeds,
and tasks. It preserves generic defaults, including secondary AIME tasks and
an older default seed. Therefore it is an execution wrapper, not by itself a
recipe for the current Llama freeze. Canonical reproduction must explicitly
set `MODEL_NAME`, `MODEL_REVISION`, `SEEDS=20260716`, and
`TASKS="GSM8K MATH-500 HumanEvalPlus FOLIO"`.
Optional `TOP_P`, `CACHE_ROOT`, `OUTPUT_ROOT`, `MAX_WORKERS`,
`GENERATION_BATCH_SIZE`, `REQUEST_TIMEOUT`, and `MAX_RETRIES` variables bind
supplemental sampling/transport and keep generated evidence inside its
isolated result subtree.

#### `scripts/run_qwen_main_20260721.sh`

Binds the post-freeze Qwen2.5-7B revision, explicit `top_p=0.9`, seeds
20260721--20260723, and the four core tasks before delegating to
`run_paper_matrix.sh`. Outputs and caches stay under the dedicated Qwen main
partition; they are never pooled with Qwen development or Llama evidence.

#### `scripts/summarize_qwen_main_20260721.sh`

Requires the complete post-freeze Qwen seed/task manifest matrix and calls the
shared whitelist summarizer into the Qwen-only aggregate directory. Its
`--check-only` path validates inputs without rebuilding the aggregate.

#### `scripts/probe_gemma_stop_20260721.py`

Sends a bounded request to the Gemma service and fails unless explicit
end-of-turn token ID 106 and the declared sampling top-k contract terminate the
response correctly. It is a serving-contract probe, not a scorer.

#### `scripts/normalize_gemma_partial_cache_20260721.py`

Creates a provenance-bound forensic copy of the paused Gemma cache by trimming
tokens before the first declared stop marker. It never mutates the source cache.
Because the source service used a different model-native sampling top-k, the
normalized copy is not publication-admissible and must not be resumed.

#### `scripts/run_gemma_main_20260716_clean.sh`

Runs or resumes the clean three-seed Gemma matrix only after checking the exact
model revision, explicit stop ID 106, `top_p=0.9`, sampling `top_k=0`, dataset,
and cache contracts. It writes exclusively to the dedicated Gemma main
partition and does not alter the canonical scorer.

#### `scripts/resume_gemma_gsm8k_20260716.sh`

This incident-specific wrapper is intentionally disabled. It documents the
sampling-contract mismatch in the older Gemma prefix, points to the forensic
normalized cache, and exits before generation so incompatible samples cannot
be appended.

#### `scripts/summarize_gemma_main_20260722.sh`

Validates the clean Gemma seed/task manifests and invokes the shared whitelist
summarizer into a Gemma-only aggregate. `--check-only` performs admission
checks without changing aggregate artifacts.

#### `scripts/resume_llama_math500_20260716.sh`

Incident-specific recovery wrapper for the validated Llama seed-20260716
MATH-500 schema-v2 prefix. It ignores stale shell path variables, binds the
known cache/output/log locations, validates Python and cache line-count bounds,
runs live model/data/cache preflight, and only then invokes `--resume_cache`.
`--check-only` stops after preflight and never starts generation. The script
contains orchestration only and does not alter sampling or scoring logic.

#### `scripts/resume_llama_csa_20260716.sh`

Incident-specific recovery wrapper for the Llama seed-20260716 CSA sidecar. It
binds the supplemental archive paths absolutely, checks all four task manifests,
requires the complete NLI cache and a bounded assessment prefix, probes the
8000 `/v1/models` endpoint, and then resumes only the assessment stage with
single-prompt transport. `--check-only` performs those checks without invoking
the evaluator; the wrapper prevents stale shell variables from redirecting the
run into the retired archive root.

#### `scripts/serve_llama_csa_20260716.sh`

Starts the fixed Llama CSA logprob service on port 8000 through the dedicated
vLLM environment. It leaves `max-model-len=24000` intact but limits each
full-vocabulary prompt-logprob prefill chunk to 1,024 tokens, reserves additional
activation headroom with `gpu-memory-utilization=0.70`, admits one sequence,
and uses eager execution. The wrapper refuses to replace an already responsive
port-8000 service and does not change model weights, prompts, or scoring.

#### `scripts/run_llama_sese_20260716.sh`

Runs the source-locked SeSE replay for the Llama seed-20260716 supplemental
sidecar. It binds the supplemental run/cache paths, requires the exact
DeBERTa-v2-XLarge-MNLI and static embedding revisions, maps one or more free
baseline GPUs to logical `cuda:0,...` runtimes, and supports a non-executing
`--check-only` validation before resuming or creating the source cache. It does
not substitute the CSA NLI
checkpoint and preserves the released raw/sign-corrected SeSE labels.

### Supplemental diagnostic execution

#### `scripts/evaluate_pwsc_ci_controls.py`

Validates one manifest/report/CSV/cache/dataset bundle per task, replays the
canonical row within `1e-10`, then evaluates only predeclared single-factor
controls: weakest-k, raw versus normalized Holevo, confidence versus uniform
peer mass, CI versus target consistency on the same kernel, AST versus public
behavior, probe count, recovery, and fixed versus rotated targets. All-target
bootstrap resamples questions as clusters. It writes a sample CSV, summary
JSON, and source/output manifest; it never calls the generation API. The same
driver also supports manifest-bound fixed and rotated prefix replays for the
`N=4` and `N=8` path-count controls. `N=1` fixed risk is read directly from the
core-manifest-bound reports because a one-path ensemble has no peer rotation.

#### `scripts/summarize_pwsc_ci_supplement.py`

Reads only the canonical `pwsc_ci_risk` and an explicit non-PWSC baseline
column whitelist. It rejects mixed revisions or sampling contracts, never
pools model families, computes task/seed/model summaries and paired intervals,
writes risk-coverage bands and failure-case IDs, and produces pure-vector
PDF/SVG plus a hash manifest. Retired PWSC columns present in immutable source
CSVs are ignored by construction.

#### `scripts/summarize_llama_supplement_20260716_18.sh`

Runs the fixed three-seed Llama supplemental summary without inheriting shell
path state. It binds the absolute supplemental archive root, requires exactly
one manifest per seed/task across 20260716--20260718, supports `--check-only`,
and then invokes the canonical summarizer with the predeclared bootstrap,
risk-coverage, task, and output settings. This wrapper prevents an archive-level
`SUP_ROOT` from silently redirecting aggregation to a nonexistent `core/` tree.

### Publication figure generation

#### `scripts/plot_pwsc_ci_ablation.py`

Verifies the active independent freeze and its four manifest/CSV hashes,
computes equal-weight task-macro AUROC with 2,000 paired within-task bootstrap
draws, and produces the component-evidence figure as vector PDF/SVG plus a
machine-readable statistics JSON. It reads only `pwsc_ci_risk` and non-PWSC
mean-NLL, worst-k-NLL (k=5), and target-consistency controls from the bound Llama
sample rows; it never reads Qwen development values or retired PWSC variants.

#### `scripts/plot_q_rasc_motivation.py`

Selects the real-data local-confidence and public code-behavior examples only
from the immutable Llama promotion freeze, verifies all input hashes, and emits
the motivation figure plus the `q-rasc-motivation-figure-v8` claim sidecar. It
does not implement or score a new uncertainty method.

#### `scripts/build_q_rasc_algorithm_overview.py`

Builds the algorithm-only vector overview from the maintained TeX source and
the user-provided draft, records source/output hashes, and keeps independent
answer selection visually separate from fixed-target uncertainty scoring. It
is a renderer for the normative flow, not an executable scorer.

#### `scripts/refine_qrasc_figures.py`

Renders the current native-size method and motivation figures from the frozen
motivation sidecar, writes editable and outlined vector artifacts, and records
their manifest. It changes presentation only.

#### `scripts/plot_qrasc_main_evidence.py`

Hash-checks the admitted Llama-8B, post-freeze Qwen-7B, and Llama-70B metric
aggregates, the Llama risk-coverage source, independent component statistics,
and manifest-bound `N=1/4/8/16` controls. It emits the main evidence figures and
`qrasc-main-paper-evidence-figures-v8` sidecar. Llama-70B is added only to the
complete taskwise table contract; existing two-model taskwise figures remain
unchanged. This script consumes results and never defines `pwsc_ci_risk`.

### Historical figure-draft tooling

`scripts/build_overview_compliant.py`,
`scripts/build_overview_motivation_innovations.py`, and
`scripts/build_overview_v8_draft.py` are retained only to reproduce superseded
PowerPoint design drafts. Active TeX does not invoke them, and they contain no
algorithm or experiment entry point.

### External non-PWSC comparator sidecars

#### `scripts/evaluate_blackbox_2026.py`

Evaluates source-locked black-box RDS formulas from sampled text, external
sentence embeddings, and optional path likelihood weights. It records encoder
identity/revision and labels task-output adaptations. It does not fabricate
white-box inputs.

#### `scripts/evaluate_csa_2026.py`

Runs the source-compatible Clustered Self-Assessment adaptation using
bidirectional NLI clusters, fixed-first target option A, and a separate
same-model multiple-choice probability assessment. NLI and assessment caches
are explicit inputs/outputs. Model and revision CLI defaults are convenience
values only; canonical runs must override and bind them.

#### `scripts/evaluate_sese_2026.py`

Replays the released short-form SeSE source pipeline with an external NLI
model. It preserves both the literal released-source entropy sign and the
named risk-oriented sign correction, so the reproduction boundary remains
visible.

These scripts are comparator evidence tools. They do not add PWSC
implementations under `code/core/` or alter `pwsc_ci_risk`.

### Promotion and freeze

#### `scripts/seal_ci_promotion_comparator.py`

Uses development-only non-PWSC results to freeze the preregistered strongest
single comparator and per-task envelope before independent evaluation. It
records source hashes and deterministic tie breaking.

#### `scripts/analyze_ci_independent_promotion.py`

Executes the one-shot independent promotion gate. It verifies task provenance,
joins fixed sample IDs, performs paired within-task bootstrap followed by
four-task macro aggregation, checks AUROC and AURC gates, and writes the
decision artifact.

### Validators

#### `scripts/validate_pwsc_ci_freeze.py`

Recomputes and checks freeze schema, sole score column, decision and
preregistration hashes, task manifests, dataset fingerprints, generation
caches, sample CSVs, metrics, and promotion values.

#### `scripts/validate_pwsc_ci_code_replay.py`

Loads immutable cache/sample pairs, reruns the current sole scorer row by row,
and compares `pwsc_ci_risk` plus Holevo information, partition entropy,
target mass ratio, and evidence deficit at tolerance `1e-10`. It also counts
behavior-kernel versus AST-fallback rows.

## 10. Test inventory

The active suite has 71 tests:

| Test file | Count | Coverage |
|---|---:|---|
| `tests/test_pwsc_ci.py` | 17 | token retention, weakest-k confidence, degenerate limits, bounds, invariance, PSD rejection, math equivalence, behavior kernel, configurable probes/recovery, comparator labeling |
| `tests/test_paper_runner.py` | 8 | CLI defaults, explicit transport validation, explicit sampling/cache binding, strict partial-cache resume, incomplete-generation prefix preservation/recovery diagnostics, paper protocol rejection, API-key redaction, schema-v2 offline CI-only report |
| `tests/test_data_and_verifiers.py` | 11 | P-FOLIO/LiveCodeBench/MBPP+ loaders and fail-closed logic, code, math, and official-runner verification |
| `tests/test_gemma_stop_contract.py` | 5 | explicit stop/top-$k$ request binding, legacy-header rejection, and provenance-preserving forensic normalization boundaries |
| `tests/test_metrics.py` | 6 | ranking/selective metrics, ties, undefined single-class behavior, dev calibration, paired bootstrap |
| `tests/test_blackbox_2026.py` | 6 | RDS definitions/weights, CSA clustering/assessment prompt boundary, and single-prompt resumable assessment transport |
| `tests/test_sese_2026.py` | 5 | devices, ordered runtime, released clustering/graph repair, entropy sign naming |
| `tests/test_seal_ci_promotion_comparator.py` | 2 | comparator-only selection, task envelope, deterministic ties |
| `tests/test_analyze_ci_independent_promotion.py` | 1 | reproducible joint AUROC/AURC promotion gate |
| `tests/test_validate_pwsc_ci_freeze.py` | 1 | complete freeze/provenance validation |
| `tests/test_pwsc_ci_frozen_replay.py` | 1 | current code against a frozen HumanEvalPlus row |
| `tests/test_pwsc_ci_supplement.py` | 8 | single-factor controls, model separation, paired/curve reproducibility, retired-column exclusion, stale-root diagnostics, relocated-bundle cache resolution, and stop/top-$k$ aggregate binding |

Tests import from `code/` through `pyproject.toml`'s pytest
`pythonpath=["code"]`. They do not use a second installed implementation.

## 11. Dependencies and external boundaries

`pyproject.toml` declares Python 3.10+ and these runtime dependencies:
`numpy`, `scipy`, `pandas`, `sympy`, `openai`, `httpx`, and
`tqdm`. Optional groups provide `pytest`, `math-verify`, `evalplus`, and
`matplotlib` for deterministic vector figure generation.

External baseline source snapshots are isolated under
`external_baselines/` and pinned by `external_baselines/manifest.json`.
The canonical runner does not import them. Current sidecar admission rules are:

- RDS: source formula with an explicitly versioned external sentence encoder.
- CSA: bidirectional NLI plus separate same-model self-assessment.
- SeSE: released-source replay with explicit semantic/NLI model.
- Methods needing hidden states, attentions, altered decoding, perturbed
  weights, full-vocabulary distributions, or fitted test-label calibration are
  blocked rather than approximated.

The active generation caches contain text, chosen-token logprobs, top-logprob
dictionaries, and null attention fields. They cannot support a method that
requires unavailable white-box state.

## 12. Artifact and archive boundary

`code/diagnostics/paper_cache/` contains exactly four claim-bearing immutable
generation caches, one for each frozen task. It is an input store, not a result
API.

`code/diagnostics/pwsc_ci_frozen/` contains exactly the active CI-only
machine-readable result freeze.

`code/diagnostics/README.md` records this boundary for tools that enter the
diagnostics directory directly.

The source decision, manifests, raw multi-column sample CSVs, comparator
sidecars, and 18 noncanonical generation caches are retained under
`.pipeline/archive/pwsc_retired_results_20260717/immutable_provenance/`.
Their original hashes remain auditable. They must not be scanned as a general
active result source. The sole exception is the explicitly labelled complete
Qwen development table, which reads the frozen Qwen `N=16` report metrics plus
RDS/CSA/SeSE sidecars, excludes every retired PWSC row, and is never combined
with the independent Llama freeze.

`.pipeline/archive/pwsc_retired_results_20260717/
supplemental_experiments_20260718/` is the only destination authorized for the
post-review runs in `../SUPPLEMENTAL_EXPERIMENT_COMMANDS.md`. It has separate
core, cache, control, MBPP+, external-baseline, aggregate, and log areas. It is
noncanonical until a new Evidence Gate explicitly admits an artifact.

## 13. Sole-PWSC implementation audit

The following facts distinguish support code from an alternative PWSC method:

1. Repository-wide active Python source search finds
   `PWSC_CI_CONTRACT_VERSION`, `target_conditional_information_components`,
   `target_conditional_information_risk`, and `PWSCCIRiskEstimator` only in
   `code/core/risk_estimator.py`.
2. `code/core/adapters.py` and `code/core/behavioral_kernel.py` return
   representations, masks, or kernels, never a second uncertainty score.
3. `code/run_pipeline.py` constructs only `PWSCCIRiskEstimator` and exposes
   only the `PWSC-CI` row among PWSC-named methods.
4. `routing/baselines.py` emits only non-PWSC comparators.
5. `routing/calibration.py` is not imported by the runner.
6. Compatibility hook names and immutable historical cache metadata do not
   dispatch to retired implementations.
7. The active freeze accepts exactly `score_column="pwsc_ci_risk"`.
8. Retired modules and results are outside the active source/result surface.
9. No `code/ise_flow/` directory exists; the compatibility-preserving physical
   import root and CLI remain `code/` and `code/run_pipeline.py`.

## 14. Validation commands

```bash
PYTHONPATH=code TQDM_DISABLE=1 pytest -q
python3 scripts/validate_pwsc_ci_freeze.py
python3 scripts/validate_pwsc_ci_code_replay.py --task GSM8K
python3 scripts/validate_pwsc_ci_code_replay.py --task MATH-500
python3 scripts/validate_pwsc_ci_code_replay.py --task HumanEvalPlus
python3 scripts/validate_pwsc_ci_code_replay.py --task FOLIO
```

The immutable 2026-07-17 freeze still passes all provenance checks. The
current package passes 71 tests. Full
current-code replay matched all 2,187 frozen `N=16` rows within `1e-10`.
HumanEvalPlus reproduced 155 public behavior-kernel rows and 9 AST fallbacks.
Because code replay intentionally fails closed on child-process wall-clock
timeouts, run it serially rather than concurrently with worker-intensive test
suites. A concurrent resource-contention failure is not accepted as a pass;
the required clean serial replay must match the frozen rows.

## 15. Completeness audit

For this document, "complete" has three concrete meanings:

1. **Path complete:** every non-transient file under `code/`, every top-level
   first-party script, and every test file is named in Section 2.
2. **Interface complete:** every exported/top-level public class, function,
   constant, and public class method is owned by a module in Section 4 or the
   runner/script sections. Private underscore-prefixed helpers are not a second
   API; their behavior is covered by the owning module's responsibilities and
   contracts.
3. **Contract complete:** runtime flow, task capability versus evidence scope,
   cache/sample/report/manifest/freeze schemas, CLI invariants, external
   baselines, immutable archives, tests, and validation commands all have an
   explicit owner.

The active inventory contains 25 non-transient `code/` files, 34 top-level
first-party scripts (including 3 historical figure-draft builders), and 12 test
files. The 2026-07-27 AST audit found 127 exported/top-level public classes,
functions, constants, or public class methods in the active Python runtime
surface, with no unowned public symbol. The sole-definition source audit found
the canonical PWSC-CI contract, risk functions, and scorer class only in
`code/core/risk_estimator.py`. This is an implementation map, not a second
algorithm specification: equations and semantic precedence remain exclusively
in `../PWSC_ALGORITHM_FLOW.md`.
