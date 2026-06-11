# Artifact policy

1. Every numerical claim in the paper maps to an evidence artifact listed in
   `evidence_id_manifest.csv`.
2. Objective gaps are claim-bearing only where a reference solver completed on
   the same public objective.
3. Rows without a completed reference carry feasibility, timing, memory-class,
   and failure-status claims only — never certified-optimality claims.
4. Queue results carry throughput, completion, p50/p99, missed-deadline, and
   audit-coverage claims only.
5. Speedups compare rows within a single declared timing lane.
6. One commercial reference solver is anonymized: its timing rows are
   unpublished consistent with its end-user license terms; it contributes
   completed reference objectives only.
7. Comparator rows were produced through the CVXPY interface with model
   construction inside the timed call; PRISM rows are measured at PRISM's
   public solver-call boundary. This harness asymmetry is disclosed as a
   threat to validity in the paper.
8. Backtest and Monte Carlo results are workflow-correctness checks, not
   investment-performance claims.
9. PRISM implementation details remain outside the public artifact. The
   public reproducibility standard is evidence-level: external checks on
   returned weights are reproducible from this repository; the engine is not.
