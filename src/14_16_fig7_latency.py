#!/usr/bin/env python3
# =============================================================================
# Script : 14_16_fig7_latency.py        [Figure 7 再生成 — (c)なし・実測版]
# Project: ARGUS
# Description:
#   入力: models/14_15_latency.json(無ければ停止 — 先に14_15を実行)
#   パネル: (a) ステージ別+エンドツーエンド中央値(p95エラーバー, 主デバイス)
#           (b) デバイス比較(MPS vs CPU, エンドツーエンド)
#   出力: PDF/14_16_fig7_latency.pdf
# Usage: uv run python src/14_16_fig7_latency.py
# =============================================================================
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT/"PDF"; PDF.mkdir(exist_ok=True)
p = ROOT/"models"/"14_15_latency.json"
if not p.exists(): sys.exit("[14_16] STOP: 先に 14_15_latency_benchmark.py を実行")
D = json.load(open(p))
devs = [k for k in D if k != "provenance"]
main_dev = "mps" if "mps" in devs else devs[0]
r = D[main_dev]
plt.rcParams.update({"font.size": 9.5, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.6))
stages = ["locator", "swift", "fno", "end_to_end"]
labels = ["GNN-Locator", "SWIFT CMT", "FNO (PGA)", "End-to-end"]
med = [r[s]["median_ms"] for s in stages]
p95 = [r[s]["p95_ms"] for s in stages]
cols = ["#1668C1", "#2F7D32", "#E8871E", "#5F7080"]
ax[0].bar(labels, med, yerr=[np.zeros(4), np.array(p95)-np.array(med)],
          color=cols, width=0.6, capsize=4)
for x, (m, q) in enumerate(zip(med, p95)):
    ax[0].text(x, q+0.15, f"{m:.1f}", ha="center", fontsize=9.5)
ax[0].set_ylabel("Latency (ms)"); ax[0].grid(axis="y", alpha=0.25)
ax[0].set_title(f"(a) Per-stage and end-to-end ({main_dev.upper()}; "
                f"median, whisker = p95)", fontsize=10)
x2 = np.arange(len(devs))
e2e = [D[d]["end_to_end"]["median_ms"] for d in devs]
ax[1].bar([d.upper() for d in devs], e2e, color="#5F7080", width=0.45)
for x, v in enumerate(e2e):
    ax[1].text(x, v*1.02, f"{v:.1f} ms", ha="center", fontsize=9.5)
ax[1].set_ylabel("End-to-end median (ms)"); ax[1].grid(axis="y", alpha=0.25)
ax[1].set_title("(b) Device comparison", fontsize=10)
fig.tight_layout()
out = PDF/"14_16_fig7_latency.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"[14_16] {out}")
