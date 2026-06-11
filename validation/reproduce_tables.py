#!/usr/bin/env python3
"""Reproduce the published paper tables from the sanitized results CSVs.

Reads results/*.csv and prints the claim-bearing tables (multi-solver,
GPU timing lanes, production queue) exactly as published, so a reviewer can
diff repository data against the paper without any access to PRISM internals.
"""
import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def show(name):
    p = ROOT / "results" / name
    print("=" * 8, name)
    with open(p) as f:
        for row in csv.reader(f):
            print(", ".join(row))
    print()


if __name__ == "__main__":
    show("e1_multisolver_public_rows.csv")
    show("e6_gpu_timing_lane_separation.csv")
    show("e5_production_queue_500x10000.csv")
