#!/usr/bin/env python3
# =============================================================================
# scripts/09_latency_benchmark.py
#
# What this script does:
#   Measures per-component and end-to-end inference latency over N warm-started
#   passes (manuscript Fig. 7a / Table S1: ~8.8 / 3.4 / 4.5 ms -> 17.0 ms total).
#   Excludes model load and graph construction (one-time init), matching the
#   persistent-service deployment assumption.
# =============================================================================
import importlib, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
common = importlib.import_module("00_common")
import torch
from argus import ARGUS
from argus.graph import build_graph
import numpy as np


def synthetic_event(n_sta=8):
    nf = np.random.randn(n_sta, 22).astype("float32")
    coords = np.random.uniform([33.0, 130.0], [33.5, 131.0], size=(n_sta, 2))
    parr = np.sort(np.random.uniform(0, 2, n_sta))
    g = build_graph(nf, coords, parr)
    wf = torch.randn(n_sta, 3, 1024)
    cond = torch.randn(1, 4, 128, 128)
    return g, wf, cond


def main(n=100):
    cfg = common.load_cfg()
    argus = ARGUS(cfg)  # random-init weights are fine for a timing benchmark
    g, wf, cond = synthetic_event()
    for _ in range(10):  # warm-up
        argus(g, wf, cond, profile=True)
    tot = {"gnn_locator": [], "swift_cmt": [], "fno_nami": [], "total": []}
    for _ in range(n):
        lat = argus(g, wf, cond, profile=True)["latency_ms"]
        for k in tot:
            tot[k].append(lat[k])
    print(f"device={argus.device}  ({n} warm passes)")
    for k, v in tot.items():
        print(f"  {k:12s} median {statistics.median(v):6.2f} ms")


if __name__ == "__main__":
    main()
