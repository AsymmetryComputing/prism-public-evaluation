#!/usr/bin/env python3
"""Recompute external feasibility residuals from public inputs and returned
weights. These are the public residual definitions from the paper (Sec. 4.4);
they require no access to PRISM internals.
"""
import numpy as np


def budget_residual(w):
    return abs(float(np.sum(w)) - 1.0)


def box_residual(w, lo, hi):
    return max(float(np.max(np.clip(lo - w, 0, None), initial=0.0)),
               float(np.max(np.clip(w - hi, 0, None), initial=0.0)))


def exposure_residual(w, A, bmin, bmax):
    e = A @ w
    return max(float(np.max(np.clip(bmin - e, 0, None), initial=0.0)),
               float(np.max(np.clip(e - bmax, 0, None), initial=0.0)))


def turnover_residual(w, w0, D, tau):
    return max(0.0, float(np.sum(np.abs(D @ (w - w0)))) - tau)


GATES = {"budget": 1e-4, "bounds": 1e-6}

if __name__ == "__main__":
    print("Public residual definitions; import and apply to returned weights.")
    print("Gates:", GATES)
