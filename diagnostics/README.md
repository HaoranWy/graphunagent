# Diagnostics boundary

The active claim-bearing result is
`pwsc_ci_frozen/pwsc_ci_results_20260717.json`.

`paper_cache/` contains exactly the four immutable claim-bearing generation
inputs whose hashes are recorded by the promotion decision. The other 18
generation caches, source decision, manifests, and raw multi-column
CSVs were moved to
`.pipeline/archive/pwsc_retired_results_20260717/immutable_provenance/`.
They remain hash-verifiable evidence but are not active result APIs. The
canonical code and paper read only the frozen `pwsc_ci_risk` result and
non-PWSC comparator fields.

Retired experiment directories were moved to
`.pipeline/archive/pwsc_retired_results_20260717/` on 2026-07-17.
