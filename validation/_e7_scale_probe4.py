#!/usr/bin/env python3
"""
E7 addendum: (a) S7 round at N=48,100 (and 96,200 PRISM-only probe) with
declared 600 s incumbent ceilings; (b) S5b batch of 50 personalized accounts
at N=4,810. Same real-calibrated extension, lanes, and external objective as
the recorded E7 protocol. Seeded; appends to the E7 evidence artifact.
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
GAMMA = 0.005
CEIL_S = 600.0

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
print(f"[base] N={n0}", flush=True)


def extend(nbig, rng):
    rows = rng.integers(0, n0, nbig)
    B = B0[rows] * rng.normal(1.0, 0.08, (nbig, 1))
    D = D0[rng.integers(0, n0, nbig)] * rng.uniform(0.8, 1.25, nbig)
    mu = mu0[rng.integers(0, n0, nbig)]
    wc = rng.dirichlet(np.ones(nbig) * 5.0)
    return (B.astype(np.float64), D.astype(np.float64),
            mu.astype(np.float64), wc.astype(np.float64))


def objective(w, B, D, mu, g, wc):
    Btw = B.T @ w
    return float(Btw @ Btw + D @ (w ** 2) - mu @ w + g * np.abs(w - wc).sum())


def okfeas(w, pmax):
    return abs(float(w.sum()) - 1) < 1e-4 and float(w.min()) > -1e-6 and float(w.max()) < pmax + 1e-6


class P:
    def __init__(self, B, D, mu, wc, g, pmax):
        self.B, self.D, self.mu = B, D, mu
        self.w_current, self.w_target = wc, np.ones(len(mu)) / len(mu)
        self.gamma, self.position_max = g, pmax
        self.n_assets, self.k_factors = len(mu), B.shape[1]
        self.seed, self.crisis_mode, self.source, self.metadata = 11, False, "s7", {}


def prism(p, eng):
    r = prism_tr_solve(p, engine=eng, settings={"mode": "quality"})
    wall = r.extra.get("wall_ms", r.time_ms) if r.extra else r.time_ms
    return np.asarray(r.weights), float(wall), r.status


s7_new = []

for nbig, pmax in ((192400, 0.0015), (384800, 0.00075)):
    B, D, mu, wc = extend(nbig, RNG)
    print(f"[S7] N={nbig} (PRISM-only probe)", flush=True)
    p = P(B, D, mu, wc, GAMMA, pmax)
    _ = prism(p, "factor-gpu")  # disclosed warm-up per size
    for eng, label in [("factor-cpu", "PRISM-CPU"), ("factor-gpu", "PRISM-GPU")]:
        wv, ms, st = prism(p, eng)
        obj = objective(wv, B, D, mu, GAMMA, wc)
        s7_new.append(dict(N=nbig, solver=label, wall_ms=round(ms, 1), status=st,
                           feasible=bool(okfeas(wv, pmax)), objective=obj))
        print(f"   {label:10s} {ms:11.1f} ms {st} feas={okfeas(wv, pmax)} obj={obj}", flush=True)

import json as _json
e7 = _json.loads(E7.read_text())
e7["s7_scale_stress"]["results"].extend(s7_new)
e7["s7d_meta"] = dict(note="PRISM-only routing probes at N=192,400 and N=384,800; "
                           "caps 0.15%/0.075%; incumbents not run, no incumbent claim")
E7.write_text(_json.dumps(e7, indent=1))
print("appended", flush=True)
