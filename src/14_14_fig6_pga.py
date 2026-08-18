#!/usr/bin/env python3
# =============================================================================
# Script : 14_14_fig6_pga.py            [Figure 6 再生成 — PGA帰属訂正版]
# Project: ARGUS
# Description:
#   入力(無ければ停止): models/13_29_fno_vs_gmpe.json,
#                        models/locator/13_08_bias_coeff.json
#   パネル:
#     (a) 観測PGAとの相関r: FNO(zero-shot) vs SM1999無補正(同一1,008セル)
#         + 補正済みGMPE(2,892ペア・パイプライン出力)を参考群で併記
#     (b) σ_log10(FNO vs 無補正GMPE)
#   出力: PDF/14_14_fig6_pga.pdf
# Usage: uv run python src/14_14_fig6_pga.py
# =============================================================================
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT/"PDF"; PDF.mkdir(exist_ok=True)
def need(p):
    if not p.exists(): sys.exit(f"[14_14] STOP: {p} が無い")
    return json.load(open(p))
E2 = need(ROOT/"models"/"13_29_fno_vs_gmpe.json")
B = need(ROOT/"models"/"locator"/"13_08_bias_coeff.json")

plt.rcParams.update({"font.size": 9.5, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.7))
labels = ["FNO-NAMI\n(zero-shot)", "SM1999 GMPE\n(uncorrected)",
          "Corrected GMPE\n(pipeline output)"]
rs = [E2["fno_vs_obs"]["r"], E2["gmpe_vs_obs"]["r"], B["r_corr"]]
cols = ["#D1495B", "#1668C1", "#5F7080"]
ax[0].bar(labels, rs, color=cols, width=0.55)
for x, v in enumerate(rs):
    ax[0].text(x, v+0.015, f"{v:.2f}", ha="center", fontsize=9.5)
ax[0].set_ylim(0, 0.95); ax[0].set_ylabel("Pearson r vs observed K-NET PGA (log10)")
ax[0].grid(axis="y", alpha=0.25)
ax[0].set_title("(a) Agreement with observed PGA", fontsize=10)
ax[0].text(0.5, -0.34,
           f"left/center: 5 events Mw 6.8–9.0, n = {E2['n']:,} grid cells (same pixels)\n"
           f"right: Kumamoto sequence, n = {B['n_pairs']:,} station pairs "
           f"(test r = {B['r_test']:.2f}); archived run: used_nami = false",
           transform=ax[0].transAxes, ha="center", fontsize=7.8, color="#444")
s = [E2["fno_vs_obs"]["sigma_log10"], E2["gmpe_vs_obs"]["sigma_log10"]]
ax[1].bar(labels[:2], s, color=cols[:2], width=0.5)
for x, v in enumerate(s):
    ax[1].text(x, v+0.03, f"{v:.2f}", ha="center", fontsize=9.5)
ax[1].set_ylabel("σ of log10 residual (orders of magnitude)")
ax[1].grid(axis="y", alpha=0.25)
ax[1].set_title("(b) Residual spread (same pixels)", fontsize=10)
fig.tight_layout()
out = PDF/"14_14_fig6_pga.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"[14_14] {out}")
