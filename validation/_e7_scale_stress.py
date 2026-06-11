#!/usr/bin/env python3
"""
E7 addendum S7: scale stress on a real-calibrated extension of the universe.

The 481-name real factor model (PCA k=20 on 2019-2025 daily returns) is
extended to larger cross-sections by resampling real factor-loading rows with
replacement and bootstrapping idiosyncratic variances from the real
distribution; momentum tilts are resampled from the real cross-section. This
preserves the empirical risk geometry while testing larger N. Disclosed as a
real-calibrated extension, not raw constituent data.

Lanes identical to E7: cvxpy incumbent parametrized with only prob.solve()
timed; PRISM public solve() wall time (GPU host wall, warm session).
"""
import json
import sys
import time
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from solver_comparison.solvers.prism_transition_qp import solve as prism_tr_solve  # noqa: E402
import cvxpy as cp  # noqa: E402

CACHE = HERE / "evidence" / "real_prices_sp500_2019_2025.pkl"
E7 = HERE / "evidence" / "PRISM_EVIDENCE_E7_HARD_SCENARIOS_2026-06-11.json"
RNG = np.random.default_rng(11)

px = pd.read_pickle(CACHE)
rets = px.pct_change().dropna()
R = rets.values - rets.values.mean(0)
cov = R.T @ R / (len(R) - 1)
vals, vecs = np.linalg.eigh(cov)
idx = np.argsort(vals)[::-1][:20]
B0 = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0.0))
D0 = np.maximum(np.diag(cov) - (B0 ** 2).sum(1), 1e-8)
mom = (rets.iloc[-252:-21].add(1).prod() - 1).values
mu0 = 0.02 * (mom - mom.mean()) / (mom.std() + 1e-12)
n0 = len(mu0)
print(f"[base] real factor model N={n0}")


def extend(nbig):
    rows = RNG.integers(0, n0, nbig)
    B = B0[rows] * RNG.normal(1.0, 0.08, (nbig, 1))
    D = D0[RNG.integers(0, n0, nbig)] * RNG.uniform(0.8, 1.25, nbig)
    mu = mu0[RNG.integers(0, n0, nbig)]
    wc = RNG.dirichlet(np.ones(nbig) * 5.0)
    return (B.astype(np.float64), D.astype(np.float64), mu.astype(np.float64),
            wc.astype(np.float64))


def objective(w, B, D, mu, g, wc):
    Btw = B.T @ w
    return float(Btw @ Btw + D @ (w ** 2) - mu @ w + g * np.abs(w - wc).sum())


class P:
    def __init__(self, B, D, mu, wc, g, pmax):
        self.B, self.D, self.mu = B, D, mu
        self.w_current, self.w_target = wc, np.ones(len(mu)) / len(mu)
        self.gamma, self.position_max = g, pmax
        self.n_assets, self.k_factors = len(mu), B.shape[1]
        self.seed, self.crisis_mode, self.source, self.metadata = 11, False, "s7", {}


results = []
GAMMA = 0.005
for nbig in (1924, 4810, 9620):
    B, D, mu, wc = extend(nbig)
    pmax = 0.01
    print(f"[S7] N={nbig}")
    # incumbents: parametrized, solve-only timed
    w = cp.Variable(nbig)
    mu_p, wc_p = cp.Parameter(nbig), cp.Parameter(nbig)
    prob = cp.Problem(cp.Minimize(
        cp.sum_squares(B.T @ w) + cp.sum(cp.multiply(D, cp.square(w)))
        - mu_p @ w + GAMMA * cp.norm1(w - wc_p)),
        [cp.sum(w) == 1, w >= 0, w <= pmax])
    mu_p.value, wc_p.value = mu, wc
    for nm, sv, kw in [("Clarabel", cp.CLARABEL, {}),
                       ("OSQP", cp.OSQP, dict(max_iter=200000, eps_abs=1e-6,
                                              eps_rel=1e-6))]:
        t0 = time.perf_counter()
        try:
            prob.solve(solver=sv, **kw)
            ms = (time.perf_counter() - t0) * 1e3
            st = prob.status
            wv = np.asarray(w.value)
            obj = objective(wv, B, D, mu, GAMMA, wc)
            ok = abs(wv.sum() - 1) < 1e-4 and wv.min() > -1e-6 and wv.max() < pmax + 1e-6
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1e3
            st, obj, ok = f"error:{type(e).__name__}", None, False
        results.append(dict(N=nbig, solver=nm, wall_ms=round(ms, 1),
                            status=str(st), feasible=bool(ok), objective=obj))
        print(f"   {nm:10s} {ms:9.1f} ms {st} feas={ok} obj={obj}")
    p = P(B, D, mu, wc, GAMMA, pmax)
    # GPU warm-up at this size (disclosed warm session)
    _ = prism_tr_solve(p, engine="factor-gpu", settings={"mode": "quality"})
    for eng, label in [("factor-cpu", "PRISM-CPU"), ("factor-gpu", "PRISM-GPU")]:
        r = prism_tr_solve(p, engine=eng, settings={"mode": "quality"})
        wall = r.extra.get("wall_ms", r.time_ms) if r.extra else r.time_ms
        wv = np.asarray(r.weights)
        obj = objective(wv, B, D, mu, GAMMA, wc)
        ok = abs(wv.sum() - 1) < 1e-4 and wv.min() > -1e-6 and wv.max() < pmax + 1e-6
        results.append(dict(N=nbig, solver=label, wall_ms=round(float(wall), 1),
                            status=r.status, feasible=bool(ok), objective=obj))
        print(f"   {label:10s} {wall:9.1f} ms {r.status} feas={ok} obj={obj}")

e7 = json.loads(E7.read_text())
e7["s7_scale_stress"] = dict(
    description="real-calibrated extension: factor-loading rows resampled "
                "from the real cross-section, idiosyncratic variances and "
                "momentum tilts bootstrapped from the real distributions; "
                "gamma=0.005, caps 1%",
    results=results)
E7.write_text(json.dumps(e7, indent=1))
print("S7 appended to", E7.name)
