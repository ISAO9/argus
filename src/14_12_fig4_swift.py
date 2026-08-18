#!/usr/bin/env python3
# =============================================================================
# Script : 14_12_fig4_swift.py          [Figure 4 再生成 — SWIFT三段結果]
# Project: ARGUS
# Description:
#   入力(無ければ停止):
#     models/05_test_results.json        (合成テスト 99.9%)
#     models/13_36_kumamoto_transfer.json(熊本 zero-shot 100% Shear)
#     models/13_35_forge_mt_eval.json    (FORGE zero-shot 崩壊)
#     models/13_39_swift_forge_ft.json   (FORGE 現地FT κ≈0)
#   パネル:
#     (a) 合成テスト: クラス別F1(または利用可能な指標を自動選択)
#     (b) 実データzero-shot: 予測クラス分布(熊本/FORGE、真分布との対比)
#     (c) FORGE FT混同行列(κ注記)
#   出力: PDF/14_12_fig4_swift.pdf
# Usage: uv run python src/14_12_fig4_swift.py
# =============================================================================
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT/"models"; PDF = ROOT/"PDF"; PDF.mkdir(exist_ok=True)
CLS = ["Shear", "Mixed", "Tensile"]
def need(p):
    if not p.exists(): sys.exit(f"[14_12] STOP: {p} が無い")
    return json.load(open(p))
syn = need(MD/"05_test_results.json")
kum = need(MD/"13_36_kumamoto_transfer.json")
fz = need(MD/"13_35_forge_mt_eval.json")
ft = need(MD/"13_39_swift_forge_ft.json")

plt.rcParams.update({"font.size": 9.5, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))

# (a) 合成: 指標を内省して描く
def get_synth_bars(d):
    for key in ("per_class_f1", "f1_per_class", "class_f1"):
        if key in d: return [d[key][c] for c in CLS], "F1"
    if "classification_report" in d:
        cr = d["classification_report"]
        return [cr[c]["f1-score"] for c in CLS], "F1"
    if "per_class" in d:
        pc = d["per_class"]
        k0 = list(pc[CLS[0]].keys())[0]
        return [pc[c][k0] for c in CLS], k0
    print(f"[14_12] 05_test_results.json keys: {list(d.keys())}")
    sys.exit("[14_12] STOP: 合成クラス別指標のキーが見つからない。上のkeysを報告してください。")
vals, metric = get_synth_bars(syn)
acc = syn.get("accuracy", syn.get("test_accuracy", None))
ax[0].bar(CLS, vals, color="#2F7D32", width=0.55)
for x, v in enumerate(vals):
    ax[0].text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax[0].set_ylim(0.9, 1.005); ax[0].set_ylabel(metric); ax[0].grid(axis="y", alpha=0.25)
ttl = "(a) Synthetic test (n = 2,000"
ax[0].set_title(ttl + (f", acc {acc*100:.1f}%)" if acc else ")"), fontsize=10)

# (b) 実データzero-shot分布
def find_counts(d, depth=0):
    """13_36のJSON(checkpoints入れ子)からクラス件数dictを探索。"""
    if isinstance(d, dict):
        for key in ("pred_class_counts", "pred_counts", "class_counts",
                    "pred_distribution", "counts"):
            v = d.get(key)
            if isinstance(v, dict) and set(CLS) <= set(v): return v
        # epoch_080 を優先して降下
        items = sorted(d.items(), key=lambda kv: ("080" not in str(kv[0])))
        for _, v in items:
            if depth < 4:
                r = find_counts(v, depth+1)
                if r: return r
    return None
kn = find_counts(kum)
if kn is None:
    import json as _j
    sys.exit("[14_12] STOP: 13_36からクラス件数を特定できない。構造:\n"
             + _j.dumps(kum, indent=1, default=str)[:1200])
fzc = fz["pred_class_counts"]
fzt = {c: sum(fz["confusion_matrix_true_x_pred"][k]) for k, c in enumerate(CLS)}
x = np.arange(3); w = 0.26
ax[1].bar(x-w, [kn[c] for c in CLS], w, label="Kumamoto pred (n=300)", color="#5F7080")
ax[1].bar(x,   [fzc[c] for c in CLS], w, label="FORGE pred (n=148)", color="#D1495B")
ax[1].bar(x+w, [fzt[c] for c in CLS], w, label="FORGE true (Niemz MT)", color="#2F7D32")
ax[1].set_xticks(x, CLS); ax[1].set_ylabel("Events")
ax[1].grid(axis="y", alpha=0.25)
ax[1].set_title("(b) Real-data zero-shot: all-Shear collapse", fontsize=10)
ax[1].legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), frameon=False, fontsize=8)

# (c) FT混同行列
cm = np.array(ft["finetuned_test"]["confusion_true_x_pred"])
im = ax[2].imshow(cm, cmap="Blues", vmin=0)
for i in range(3):
    for j in range(3):
        ax[2].text(j, i, str(cm[i, j]), ha="center", va="center",
                   color="black" if cm[i, j] < cm.max()*0.6 else "white", fontsize=10)
ax[2].set_xticks(range(3), CLS); ax[2].set_yticks(range(3), CLS)
ax[2].set_xlabel("Predicted"); ax[2].set_ylabel("True (Niemz MT)")
acc_ft = ft["finetuned_test"]["acc"]
ax[2].set_title(f"(c) FORGE fine-tuned (n = {ft['n_test']}): "
                f"acc {acc_ft*100:.1f}%, κ ≈ 0", fontsize=10)
fig.tight_layout()
out = PDF/"14_12_fig4_swift.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"[14_12] {out}")
