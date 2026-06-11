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

# ---- S7 N=48,100 with incumbents under declared 600 s ceiling
nbig = 48100
B, D, mu, wc = extend(nbig, RNG)
pmax = 0.005
print(f"[S7] N={nbig}", flush=True)
w = cp.Variable(nbig)
prob = cp.Problem(cp.Minimize(
    cp.sum_squares(B.T @ w) + cp.sum(cp.multiply(D, cp.square(w)))
    - mu @ w + GAMMA * cp.norm1(w - wc)),
    [cp.sum(w) == 1, w >= 0, w <= pmax])
for nm, sv, kw in [("Clarabel", cp.CLARABEL, dict(time_limit=CEIL_S)),
                   ("OSQP", cp.OSQP, dict(max_iter=200000, eps_abs=1e-6,
                                          eps_rel=1e-6, time_limit=CEIL_S))]:
    t0 = time.perf_counter()
    try:
        prob.solve(solver=sv, **kw)
        ms = (time.perf_counter() - t0) * 1e3
        st = prob.status
        wv = None if w.value is None else np.asarray(w.value)
        obj = None if wv is None else objective(wv, B, D, mu, GAMMA, wc)
        ok = wv is not None and okfeas(wv, pmax)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1e3
        st, obj, ok = f"error:{type(e).__name__}", None, False
    s7_new.append(dict(N=nbig, solver=nm, wall_ms=round(ms, 1), status=str(st),
                       feasible=bool(ok), objective=obj,
                       ceiling_s=CEIL_S))
    print(f"   {nm:10s} {ms:11.1f} ms {st} feas={ok}", flush=True)

p = P(B, D, mu, wc, GAMMA, pmax)
_ = prism(p, "factor-gpu")  # disclosed warm-up
for eng, label in [("factor-cpu", "PRISM-CPU"), ("factor-gpu", "PRISM-GPU")]:
    wv, ms, st = prism(p, eng)
    obj = objective(wv, B, D, mu, GAMMA, wc)
    s7_new.append(dict(N=nbig, solver=label, wall_ms=round(ms, 1), status=st,
                       feasible=bool(okfeas(wv, pmax)), objective=obj))
    print(f"   {label:10s} {ms:11.1f} ms {st} obj={obj}", flush=True)

# ---- S7 N=96,200 PRISM-only probe (incumbents outside declared ceiling)
nbig = 96200
B, D, mu, wc = extend(nbig, RNG)
pmax = 0.003
print(f"[S7] N={nbig} (PRISM-only probe)", flush=True)
p = P(B, D, mu, wc, GAMMA, pmax)
_ = prism(p, "factor-gpu")
for eng, label in [("factor-cpu", "PRISM-CPU"), ("factor-gpu", "PRISM-GPU")]:
    wv, ms, st = prism(p, eng)
    obj = objective(wv, B, D, mu, GAMMA, wc)
    s7_new.append(dict(N=nbig, solver=label, wall_ms=round(ms, 1), status=st,
                       feasible=bool(okfeas(wv, pmax)), objective=obj))
    print(f"   {label:10s} {ms:11.1f} ms {st} obj={obj}", flush=True)

# ---- S5b: 50-account batch at N=4,810
nb = 4810
print(f"[S5b] 50-account batch at N={nb}", flush=True)
Bb, Db, mub, wcb = extend(nb, np.random.default_rng(23))
pmax = 0.01
accounts = []
rng = np.random.default_rng(29)
for _ in range(50):
    wci = np.maximum(wcb + rng.normal(0, 0.25 / nb, nb), 0)
    accounts.append((wci / wci.sum(), mub * rng.uniform(0.6, 1.4),
                     float(rng.uniform(0.002, 0.008))))
s5b = {}
w = cp.Variable(nb)
mu_p, wc_p, g_p = cp.Parameter(nb), cp.Parameter(nb), cp.Parameter(nonneg=True)
probb = cp.Problem(cp.Minimize(
    cp.sum_squares(Bb.T @ w) + cp.sum(cp.multiply(Db, cp.square(w)))
    - mu_p @ w + g_p * cp.norm1(w - wc_p)),
    [cp.sum(w) == 1, w >= 0, w <= pmax])
for nm, sv, kw in [("Clarabel", cp.CLARABEL, {}),
                   ("OSQP", cp.OSQP, dict(max_iter=200000, eps_abs=1e-6,
                                          eps_rel=1e-6))]:
    t0 = time.perf_counter()
    per, done = [], 0
    for wci, mui, gi in accounts:
        mu_p.value, wc_p.value, g_p.value = mui, wci, gi
        t1 = time.perf_counter()
        try:
            probb.solve(solver=sv, **kw)
            wv = np.asarray(w.value)
            done += int(okfeas(wv, pmax))
        except Exception:
            pass
        per.append((time.perf_counter() - t1) * 1e3)
    s5b[nm] = dict(total_wall_s=round(time.perf_counter() - t0, 2),
                   completed=f"{done}/50",
                   p50_ms=round(float(np.percentile(per, 50)), 1),
                   p99_ms=round(float(np.percentile(per, 99)), 1))
    print("  ", nm, s5b[nm], flush=True)
for eng, label in [("factor-cpu", "PRISM-CPU"), ("factor-gpu", "PRISM-GPU")]:
    t0 = time.perf_counter()
    per, done = [], 0
    for wci, mui, gi in accounts:
        p = P(Bb, Db, mui, wci, gi, pmax)
        wv, ms, st = prism(p, eng)
        done += int(okfeas(wv, pmax))
        per.append(ms)
    s5b[label] = dict(total_wall_s=round(time.perf_counter() - t0, 2),
                      completed=f"{done}/50",
                      p50_ms=round(float(np.percentile(per, 50)), 1),
                      p99_ms=round(float(np.percentile(per, 99)), 1))
    print("  ", label, s5b[label], flush=True)

e7 = json.loads(E7.read_text())
e7["s7_scale_stress"]["results"].extend(s7_new)
e7["s7c_meta"] = dict(note="N=48,100 round with declared 600 s incumbent "
                           "ceiling; N=96,200 PRISM-only probe; position caps "
                           "0.5%/0.3% at the two sizes")
e7["s5b_batch_4810"] = dict(n_accounts=50, n_universe=nb, results=s5b)
E7.write_text(json.dumps(e7, indent=1))
print("appended to", E7.name, flush=True)
