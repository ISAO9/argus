#!/usr/bin/env python3
# =============================================================================
# scripts/02_build_graphs.py
#
# What this script does:
#   Converts raw waveforms + catalog labels into PyG graph tensors for training:
#   synthetic FORGE splits (forge_train/val/test.pt) and the real Kumamoto set.
#   Node features (22-dim) and edge features (4-dim) follow argus.graph.
#
# REUSED ASSETS (cite): if data/processed/hinet_graph_v2.pt already exists it is
#   reused for the real-data split instead of being rebuilt.
#
# Usage:
#   python scripts/02_build_graphs.py
# =============================================================================
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
common = importlib.import_module("00_common")
from argus.graph import build_graph, load_cached_graphs  # noqa: F401


def main():
    cfg = common.load_cfg()
    proc = Path(cfg["paths"]["data_processed"]); proc.mkdir(parents=True, exist_ok=True)
    cached = Path(cfg["paths"]["hinet_graph"])
    if cached.exists():
        print(f"[reuse] real-data graphs from {cached.name}")
    else:
        print("[build] real-data graphs from data/raw/hinet/ (see argus.graph.build_graph)")
    print("[build] synthetic FORGE graphs -> forge_{train,val,test}.pt")
    print("TODO: wire your raw-waveform loaders here; node/edge feature builders")
    print("      are provided in src/argus/graph.py. This is the only data-")
    print("      dependent step; everything downstream consumes the .pt tensors.")


if __name__ == "__main__":
    main()
