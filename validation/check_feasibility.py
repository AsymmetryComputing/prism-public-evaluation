#!/usr/bin/env python3
"""Gate check: apply public residuals to a returned weight vector and report
pass/fail against the declared gates (budget 1e-4, bounds 1e-6)."""
import numpy as np
from compute_residuals import budget_residual, box_residual, GATES


def check(w, lo, hi):
    rb = budget_residual(w)
    rx = box_residual(w, lo, hi)
    return {
        "budget_residual": rb, "budget_pass": rb <= GATES["budget"],
        "bound_residual": rx, "bound_pass": rx <= GATES["bounds"],
    }


if __name__ == "__main__":
    w = np.full(100, 0.01)
    print(check(w, np.zeros(100), np.full(100, 0.05)))
