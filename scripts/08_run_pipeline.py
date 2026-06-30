#!/usr/bin/env python3
# =============================================================================
# scripts/08_run_pipeline.py
#
# What this script does:
#   Loads the three trained checkpoints + conformal layer and runs the full
#   ARGUS inference on one event, printing location (+ conformal radius),
#   mechanism / f_ISO / Mw, the PGA-map summary, and per-stage latency.
#
# REUSED ASSETS (cite): best_locator.pt, checkpoint_epoch_080.pt, fno_best.pth,
#   models/conformal.json.
# =============================================================================
import importlib, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
common = importlib.import_module("00_common")
import numpy as np, torch
from argus import ARGUS
from argus.graph import build_graph


def demo_event(n_sta=8):
    nf = np.random.randn(n_sta, 22).astype("float32")
    coords = np.random.uniform([33.0, 130.0], [33.5, 131.0], size=(n_sta, 2))
    parr = np.sort(np.random.uniform(0, 2, n_sta))
    return build_graph(nf, coords, parr), torch.randn(n_sta, 3, 1024), torch.randn(1, 4, 128, 128)


def main():
    cfg = common.load_cfg()
    conf = None
    cpath = Path(cfg["paths"]["models"]) / "conformal.json"
    if cpath.exists():
        d = json.load(open(cpath)); conf = (d["q_hat"], d["sigma_hat"])
    argus = ARGUS(cfg, conformal=conf)
    if all(Path(cfg["paths"][k]).exists() for k in ("locator_ckpt", "swift_ckpt", "fno_ckpt")):
        argus.load_weights()
    else:
        print("[warn] one or more checkpoints missing; running with random init (demo).")
    g, wf, cond = demo_event()
    out = argus(g, wf, cond, profile=True)
    print("location (norm):", np.round(out["location"], 4))
    print("location radius km:", out.get("location_radius_km"))
    print("mechanism:", int(np.argmax(out["mechanism_logits"])), "| f_iso:",
          round(out["f_iso"], 3), "| Mw:", round(out["mw"], 2))
    print("PGA map gal: max=%.1f mean=%.1f" % (out["pga_map_gal"].max(), out["pga_map_gal"].mean()))
    print("latency ms:", {k: round(v, 2) for k, v in out["latency_ms"].items()})


if __name__ == "__main__":
    main()
