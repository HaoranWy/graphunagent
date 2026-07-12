# PWSC v1.0 code map

The canonical formula and end-to-end algorithm flow are maintained in
`../PWSC_ALGORITHM_FLOW.md`; this file only maps repository ownership.

```text
code/
├── config/
│   ├── settings.py          # ACTIVE: PWSC constants and experiment seed/settings
│   └── schemas.py           # LEGACY COMPAT: retired graph enums for optional router import
├── core/
│   ├── adapters.py          # answer classes, code validity, AST feature map
│   └── risk_estimator.py    # unified N=1/N>1 PWSC score
├── data/
│   ├── loaders.py           # local benchmark schemas and LCB test decoding
│   └── parsers.py           # legacy parsing helpers; not used by PWSC scoring
├── llm/
│   ├── generator.py         # OpenAI-compatible generation with chosen logprobs
│   └── prompts.py           # optional prompt templates
├── routing/
│   ├── baselines.py         # NLL, worst-K, SC, SE, and MBR baselines
│   ├── calibration.py       # labeled dev-only monotone Platt calibrator
│   ├── verifier_manager.py  # fail-closed benchmark verifiers
│   └── router.py            # optional deployment router; excluded from tables
├── diagnostics/
│   ├── analyzer.py          # ACTIVE: paper metrics and bootstrap statistics
│   ├── tracker.py           # OPTIONAL COMPAT: legacy-schema JSONL runtime tracker
│   └── monitor.py           # DEPRECATED FOR PWSC: legacy ISE-Flow v6 dashboard
└── run_pipeline.py          # paper/smoke/stress CLI with streamed cache and N-scaling
```

## Lifecycle classification

| File | Status | Evidence and allowed use |
|---|---|---|
| `config/settings.py` | **Active** | `run_pipeline.py` and paper-runner tests import `ExperimentSettings`; the formal path reads `random_seed`, `pwsc.weakest_k`, and `pwsc.method_version`. `RISEWeightsConfig`, `CostBudgetConfig`, and `SafetyGateConfig` remain compatibility/deployment fields and are not PWSC paper equations. |
| `config/schemas.py` | **Legacy compatibility, deprecated for PWSC** | It only defines `NodeType.TOKEN` and `EdgeType.ATTENTION`. The sole repository dependency is an unused `NodeType` import in optional `routing/router.py`; `EdgeType` has no consumer, and `run_pipeline.py` never imports this module. Keep it only while the optional router retains that import; it must not be presented as a PWSC graph/schema layer. |
    | `diagnostics/tracker.py` | **Optional compatibility, not paper-active** | `TrajectoryTracker` is still callable, and `ScientificSignalAnalyzer.audit_m2_scientific_signal` accepts its `get_session_logs()` interface. Nothing instantiates it in `run_pipeline.py` or tests, and its JSONL schema retains legacy `r_ise_score`, routing, cache-state, and entropy-curve fields. It must not be cited as the source of paper artifacts. |
| `diagnostics/monitor.py` | **Deprecated for PWSC v1.0** | No executable import/call exists in the repository; the only reference is a commented integration example. Its dashboard is hard-coded to `ISE-Flow v6`, `R_ISE`, calibrated routing actions, and the legacy tracker schema. Re-enable only after a separate PWSC report-schema migration and tests. |

The publication runner writes its own schema-v2 generation cache, long-form
CSV, JSON report, and checksum manifest. These artifacts do not pass through
`TrajectoryTracker` or `ExecutionMonitor`.

`run_pipeline.py` defaults to the complete paper protocol and writes a streamed
schema-v2 generation cache, long-form sample CSV, JSON report, and run manifest.
Subset runs require explicit smoke mode. Raw PWSC scores are not probabilities;
ECE/Brier remain unset unless `routing/calibration.py` is fitted on a separate
labeled dev set.

SMT is not an enabled task because the workspace has no declared SMT dataset
and trusted verifier. LiveCodeBench is optional and uses its bundled official
test utility after decoding both public and private cases.
