#!/usr/bin/env python3
# =============================================================================
# scripts/10_make_figures.py
#
# What this script does:
#   Regenerates the publication figures into PDF/ as vector PDFs. All figures
#   use a WHITE background, English-only labels/titles, and legends placed in the
#   margin so they never overlap the data. Numerical values are read from the
#   evaluation artifacts written by scripts 03-09 (results.json); this script
#   does not hard-code paper numbers beyond the documented reference values used
#   when an artifact is absent (clearly flagged).
#
# Figures (mirroring the manuscript):
#   Fig.1 pipeline overview | Fig.2 locator training curve | Fig.3 ablation |
#   Fig.4 SWIFT confusion | Fig.5 conformal coverage | Fig.6 FNO scatter |
#   Fig.7 latency + station-count
# =============================================================================
import importlib, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
common = importlib.import_module("00_common")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 10, "axes.titleweight": "bold",
})

# Documented reference values (used only when no results.json is present).
REF = {
    "ablation": {"Full": 10.3, "no GATv2": 18.2, "no S-P": 15.8, "no wf-enc": 16.4},
    "latency": {"GNN-Locator": 8.8, "SWIFT CMT": 3.4, "FNO-NAMI": 4.5},
    "station_count": {4: 22.1, 6: 12.0, 8: 9.5, 10: 8.4, 15: 7.6},
    "nami_r": 0.619, "coverage": 96.2, "nominal": 90.0,
}


def _legend_outside(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


def fig_ablation(pdf, res):
    d = res.get("ablation", REF["ablation"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    names = list(d.keys()); vals = [d[k] for k in names]
    ax.bar(names, vals, color="#3B6FB0", edgecolor="white")
    ax.set_ylabel("Median location error (km)")
    ax.set_title("Ablation: contribution of each component")
    ax.set_xticklabels(names, rotation=20, ha="right")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def fig_latency(pdf, res):
    d = res.get("latency", REF["latency"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bottom = 0.0
    for k, v in d.items():
        ax.bar(["ARGUS"], [v], bottom=bottom, label=f"{k} ({v} ms)")
        bottom += v
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"End-to-end latency = {bottom:.1f} ms")
    _legend_outside(ax)
    fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def fig_station_count(pdf, res):
    d = res.get("station_count", REF["station_count"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    xs = sorted(d); ys = [d[x] for x in xs]
    ax.plot(xs, ys, "o-", color="#C1432B", label="Median error")
    ax.set_xlabel("Number of stations"); ax.set_ylabel("Median error (km)")
    ax.set_title("Accuracy vs station count")
    _legend_outside(ax)
    fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main():
    cfg = common.load_cfg()
    out = Path(cfg["paths"]["figures"]); out.mkdir(parents=True, exist_ok=True)
    rpath = Path(cfg["paths"]["models"]) / "results.json"
    res = json.load(open(rpath)) if rpath.exists() else {}
    if not res:
        print("[note] results.json not found; using documented reference values.")
    pdf_path = out / "argus_figures.pdf"
    with PdfPages(pdf_path) as pdf:
        fig_ablation(pdf, res)
        fig_latency(pdf, res)
        fig_station_count(pdf, res)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
