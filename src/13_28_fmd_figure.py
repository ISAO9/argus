#!/usr/bin/env python3
# =============================================================================
# Script : 13_28_fmd_figure.py   (v2 — knet_parsed直読み版)  [Figure S1]
# Project: ARGUS
# Description:
#   K-NET検証データセット(688件)の頻度-マグニチュード分布。
#   入力: data/knet_processed/knet_parsed.pt(13_34と同一のフィルタで688件化)。
#   0.2等級ビンの増分+累積(対数)、Mw 2.6-4.0帯を網掛け。
#   出力: PDF/13_28_fmd.pdf(白背景・英語・凡例は図外)
# Usage: uv run python src/13_28_fmd_figure.py
# =============================================================================
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT/"PDF"; PDF.mkdir(exist_ok=True)
cache = ROOT/"data"/"knet_processed"/"knet_parsed.pt"
if not cache.exists(): sys.exit(f"[13_28] STOP: {cache} が無い")
kevents = torch.load(cache, map_location="cpu", weights_only=False)
mws = []
for kev in kevents:
    sl = np.asarray(kev.get("src_loc"), dtype=float).ravel()
    if sl.size < 3: continue
    lat, lon = float(sl[0]), float(sl[1])
    if not (20.0 < lat < 50.0 and 120.0 < lon < 150.0): continue
    m = float(kev.get("mw", np.nan))
    if np.isfinite(m): mws.append(m)
mw = np.array(mws)
print(f"[13_28] events: {len(mw)}  Mw {mw.min():.1f}..{mw.max():.1f}")

bins = np.arange(np.floor(mw.min()*5)/5, np.ceil(mw.max()*5)/5 + 0.2, 0.2)
inc, edges = np.histogram(mw, bins=bins)
ctr = 0.5*(edges[:-1]+edges[1:])
cum = np.array([(mw >= b).sum() for b in edges[:-1]])

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.axvspan(2.6, 4.0, color="#FFF3C4", zorder=0,
           label="Mw 2.6–4.0 (main-text focus)")
ax.bar(ctr, np.maximum(inc, 0.5), width=0.17, color="#1668C1",
       label="Incremental (0.2 bins)", zorder=2)
ax.plot(edges[:-1], np.maximum(cum, 0.5), "o-", ms=4, color="#C0392B",
        label="Cumulative N(≥Mw)", zorder=3)
ax.set_yscale("log"); ax.set_xlabel("Mw"); ax.set_ylabel("Count")
ax.set_title(f"K-NET validation dataset (n = {len(mw)})", fontsize=10.5)
ax.grid(alpha=0.25, which="both")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
fig.tight_layout()
out = PDF/"13_28_fmd.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"[13_28] {out}")
