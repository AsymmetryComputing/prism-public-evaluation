# prism-public-evaluation

Public claim-bearing artifacts for:

**PRISM: A CPU/GPU Portfolio Optimization Engine for Deadline-Bounded
Institutional Rebalancing** (Ghosh, 2026).

PRISM is evaluated only through externally observed input-output behavior:
public problem data, returned weights, status codes, timings, feasibility
diagnostics, memory class, failure modes, and audit records. Implementation
details remain outside the public artifact.

## Contents

- `results/` — sanitized claim-bearing result tables (CSV), one file per evidence artifact.
- `tables/` — published paper tables in machine-readable form.
- `figures/` — figure files as published (vector PDF).
- `validation/` — scripts that recompute external feasibility residuals from public inputs and returned weights.
- `evidence_ledger/` — claim-to-evidence mapping and artifact policy.
- `environment/` — hardware and software disclosure for the recorded package.
- `paper/` — arXiv version and source.

## Evidence artifacts

| Artifact | Generated | Use |
|---|---|---|
| `e1_multisolver_public_rows` | 2026-05-09 | multi-solver timings, versions, CPU/GPU frontier |
| `e2_small_problem_agreement` | 2026-05-09 | FF30 small-problem comparator objective agreement |
| `e3_external_diagnostics_and_gap_checks` | 2026-05-09 | KKT-style checks, gap and covariance sensitivity, workflow checks |
| `e4_deadline_completion_grid` | 2026-05-09 | deadline-completion grid, memory class |
| `e5_production_queue_500x10000` | 2026-05-09 | production queue benchmark, constraint burden |
| `e6_gpu_timing_lane_separation` | 2026-06-09 | GPU timing-lane separation; timing/status only |

## Claim policy (summary)

- Objective gaps are claim-bearing only where a reference solver completed on the same public objective.
- Rows without a completed reference carry feasibility, timing, memory-class, and failure-status claims only.
- Queue results carry throughput, completion, p50/p99, missed-deadline, and audit-coverage claims only.
- One commercial reference solver is anonymized; its timing rows are unpublished consistent with its end-user license terms, and it contributes completed reference objectives only.

See `evidence_ledger/artifact_policy.md` for the full policy.
