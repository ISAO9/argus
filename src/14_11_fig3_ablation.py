#!/usr/bin/env python3
# =============================================================================
# Script : 14_11_fig3_ablation.py       [Figure 3 再生成 — アブレーション]
# Project: ARGUS
# Description:
#   入力: logs/13_05_rerun.log(2026-08-17 再実行ログ)。無ければ停止。
#   ログの各条件行「<Name>: horiz=XXkm  depth=YYkm」を直接パースする
#   (数値の転記はしない=出所とコードが一体)。
#   パネル: (a) 水平中央値(zero-shot転移条件), (b) 深さ中央値, (c) Δ vs Full
#   旧図の他手法参照線(EQTransformer等・出所なし)は描かない。
#   出力: PDF/14_11_fig3_ablation.pdf
# Usage: uv run python src/14_11_fig3_ablation.py
# =============================================================================
import re, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT/"logs"/"13_05_rerun.log"; PDF = ROOT/"PDF"; PDF.mkdir(exist_ok=True)
if not LOG.exists(): sys.exit(f"[14_11] STOP: {LOG} が無い")
txt = open(LOG, encoding="utf-8", errors="ignore").read()
pat = re.compile(r'(Full ARGUS|No GATv2|No S-P|No WaveEnc):\s*horiz=([\d.]+)km\s+depth=([\d.]+)km')
rows = {m.group(1): (float(m.group(2)), float(m.group(3))) for m in pat.finditer(txt)}
need = ["Full ARGUS", "No GATv2", "No S-P", "No WaveEnc"]
if any(k not in rows for k in need):
    sys.exit(f"[14_11] STOP: ログに条件行が揃わない: {list(rows)}")
n_m = re.search(r'K-NET test:\s*(\d+)\s*events', txt)
n_ev = n_m.group(1) if n_m else "?"
labels = ["Full ARGUS", "w/o GATv2\n(MLP)", "w/o S–P", "w/o WaveEnc"]
keys = need
H = [rows[k][0] for k in keys]; D = [rows[k][1] for k in keys]
dH = [100*(h-H[0])/H[0] for h in H]
colors = ["#2F7D32", "#D1495B", "#E8871E", "#1668C1"]

plt.rcParams.update({"font.size": 9.5, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6))
for a, vals, ttl, unit in ((ax[0], H, "(a) Horizontal error (median)", "km"),
                           (ax[1], D, "(b) Depth error (median)", "km")):
    a.bar(labels, vals, color=colors, width=0.62)
    for x, v in enumerate(vals):
        a.text(x, v*1.01, f"{v:.1f}", ha="center", fontsize=9)
    a.set_ylabel(f"Error ({unit})"); a.grid(axis="y", alpha=0.25)
    a.set_title(ttl, fontsize=10)
ax[2].bar(labels, dH, color=colors, width=0.62)
ax[2].axhline(0, color="#333", lw=1)
for x, v in enumerate(dH):
    ax[2].text(x, v + (3 if v >= 0 else -9), f"{v:+.0f}%", ha="center", fontsize=9)
ax[2].set_ylabel("Δ horizontal vs Full (%)"); ax[2].grid(axis="y", alpha=0.25)
ax[2].set_title("(c) Transfer degradation vs Full", fontsize=10)
fig.suptitle(f"Ablation under synthetic→real zero-shot transfer "
             f"(K-NET, n = {n_ev} events)", fontsize=10.5, y=1.02)
fig.tight_layout()
out = PDF/"14_11_fig3_ablation.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"[14_11] {out}  (parsed from {LOG.name}: "
      + ", ".join(f"{k}={rows[k][0]}" for k in keys) + ")")
