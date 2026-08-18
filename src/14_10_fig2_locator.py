#!/usr/bin/env python3
# =============================================================================
# Script : 14_10_fig2_locator.py        [Figure 2 再生成 — Locator主結果]
# Project: ARGUS
# Description:
#   入力(全て監査済みアーティファクト。無ければ停止):
#     models/locator/13_34_random_results.json          (seed42, per-event)
#     models/locator/13_34_temporal_results_seed42.json (per-event+conformal)
#     models/locator/13_37_seed_summary.json            (全seed要約)
#   パネル:
#     (a) zero-shot vs fine-tuned のCDF(random/temporal, seed42 per-event)
#         ※zero-shotはtemporalのzero_shot_test_median値を縦線で表示
#     (b) seed別median(random 5seed / temporal 3seed)ストリップ+要約
#     (c) conformal: seed別の経験被覆 vs 名目90% + q̂注記
#   出力: PDF/14_10_fig2_locator.pdf(白背景・英語・凡例は余白)
# Usage: uv run python src/14_10_fig2_locator.py
# =============================================================================
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT/"models"/"locator"; PDF = ROOT/"PDF"; PDF.mkdir(exist_ok=True)

def need(p):
    if not p.exists(): sys.exit(f"[14_10] STOP: {p} が無い")
    return json.load(open(p))

r42 = need(MD/"13_34_random_results.json")
t42 = need(MD/"13_34_temporal_results_seed42.json")
S = need(MD/"13_37_seed_summary.json")

plt.rcParams.update({"font.size": 9.5, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "axes.facecolor": "white"})
fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))

# (a) CDF
for d, lab, c in ((r42, "Random split, fine-tuned (seed 42)", "#1668C1"),
                  (t42, "Temporal split, fine-tuned (seed 42)", "#D1495B")):
    e = np.sort(np.asarray(d["per_event_horiz_km"]))
    ax[0].plot(e, np.arange(1, len(e)+1)/len(e), lw=1.8, color=c, label=lab)
for d, c in ((r42, "#1668C1"), (t42, "#D1495B")):
    ax[0].axvline(d["zero_shot_test_median_km"], color=c, ls=":", lw=1.2)
ax[0].text(0.98, 0.06, "dotted: zero-shot medians\n(90.3 / 95.7 km)",
           transform=ax[0].transAxes, ha="right", fontsize=8, color="#444")
ax[0].set_xscale("log"); ax[0].set_xlabel("Horizontal error (km)")
ax[0].set_ylabel("CDF"); ax[0].grid(alpha=0.25, which="both")
ax[0].set_title("(a) Zero-shot vs fine-tuned", fontsize=10)
ax[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), frameon=False, fontsize=8)

# (b) seed別 median
for k, (split, c, x0) in enumerate((("random", "#1668C1", 0), ("temporal", "#D1495B", 1))):
    per = S[split]["per_seed"]
    xs = x0 + (np.arange(len(per)) - (len(per)-1)/2)*0.09
    ax[1].scatter(xs, [p["median"] for p in per], s=42, color=c, zorder=3,
                  label=f"{split} ({len(per)} seeds)")
    med = S[split]["median_km"]["median"]
    ax[1].hlines(med, x0-0.28, x0+0.28, color=c, lw=2)
    ax[1].text(x0, med+1.2, f"{med:.1f} km", ha="center", fontsize=8.5, color=c)
ax[1].set_xticks([0, 1], ["Random\n(n=138)", "Temporal\n(n=206)"])
ax[1].set_ylabel("Median error (km)"); ax[1].grid(axis="y", alpha=0.25)
ax[1].set_title("(b) Fine-tuned medians by seed", fontsize=10)
ax[1].set_ylim(0, None)
ax[1].legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), frameon=False, fontsize=8)

# (c) conformal
per = S["temporal"]["per_seed"]
seeds = [str(p["seed"]) for p in per]
cov = [p["conformal"]["empirical_coverage"]*100 for p in per]
qs = [p["conformal"]["q_hat"] for p in per]
xs = np.arange(len(per))
ax[2].bar(xs, cov, width=0.55, color="#2F7D32")
ax[2].axhline(90, color="#333", ls="--", lw=1.2)
ax[2].text(len(per)-0.45, 90.6, "nominal 90%", fontsize=8, color="#333")
for x, cv, q in zip(xs, cov, qs):
    ax[2].text(x, cv+0.4, f"{cv:.1f}%", ha="center", fontsize=8.5)
    ax[2].text(x, 82, f"q̂={q:.2f}", ha="center", fontsize=8, color="white")
ax[2].set_xticks(xs, [f"seed {s}" for s in seeds])
ax[2].set_ylim(78, 100); ax[2].set_ylabel("Empirical coverage (%)")
ax[2].grid(axis="y", alpha=0.25)
ax[2].set_title(f"(c) Conformal (n_cal = {per[0]['conformal']['n_cal']})", fontsize=10)
fig.tight_layout()
out = PDF/"14_10_fig2_locator.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"[14_10] {out}")
