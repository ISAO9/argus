#!/usr/bin/env python3
# =============================================================================
# Script : 13_35_swift_forge_mt_eval.py   [E1b — SWIFT×FORGE2024 真値MT照合]
# Project: ARGUS
#
# Description:
#   SWIFT CMT(公開重み checkpoint_epoch_080.pt)を FORGE 2024 実測波形に適用し、
#   Niemz et al. 2026 の真のMT解由来ラベル(forge2024_dataset.h5, label_source=
#   "True MT inversion (full non-DC)")と照合する。R2-Major3(tensile識別能力の
#   実データ検証)への直接回答。熊本(全Shear環境)と異なり、本データは非DC成分
#   を含むため、判別力そのものを初めて実データで測定できる。
#
#   報告内容(この出力JSONが原稿E1b転記の唯一の出所):
#     - 3クラス精度・混同行列(真値ラベル vs 予測)
#     - Shear-vs-非Shear の2値精度
#     - f_ISO 回帰: 予測 frac_pred[:,1] vs 真値 real/f_iso の Pearson r / RMSE
#       (mt_fracs の並びは 01 生成器ヘッダ準拠で (f_dc, f_iso, f_clvd))
#     - DC/非DC 層別(真値ラベル別の正解率)
#
#   Usage:
#     uv run python src/13_35_swift_forge_mt_eval.py
#     uv run python src/13_35_swift_forge_mt_eval.py --h5 data/real/forge2024_21sta_dataset.h5
# =============================================================================
import sys, json, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLS = ["Shear", "Mixed", "Tensile"]

def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--h5", default=str(PROJECT_ROOT/"data/real/forge2024_dataset.h5"))
    pa.add_argument("--ckpt", default=str(PROJECT_ROOT/"models/checkpoint_epoch_080.pt"))
    pa.add_argument("--batch", type=int, default=16)
    a = pa.parse_args()
    import h5py
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    mod05 = _lm("swift05", PROJECT_ROOT/"src"/"05_swift_evaluation_3.py")
    cfg = json.load(open(PROJECT_ROOT/"models"/"swift_architecture_config.json"))
    model = mod05.load_model(cfg.get("model", cfg), Path(a.ckpt), device=dev)
    model.eval()

    with h5py.File(a.h5) as f:
        g = f["real"]
        W = g["waveforms"][:]
        y = np.asarray(g["labels"][:], int).ravel()
        fiso_t = np.asarray(g["f_iso"][:], float).ravel() if "f_iso" in g \
                 else np.asarray(g["mt_fracs"][:], float)[:, 1]
        meta = dict(f["metadata"].attrs) if "metadata" in f else {}
    N = len(W)
    print(f"[13_35] {Path(a.h5).name}: n={N}  label_source="
          f"{meta.get('label_source','?')}")
    print(f"[13_35] true label counts: "
          f"{ {c: int((y==k).sum()) for k,c in enumerate(CLS)} }")
    print(f"[13_35] true f_iso: min={fiso_t.min():.3f} med={np.median(fiso_t):.3f} "
          f"max={fiso_t.max():.3f}")

    preds, fiso_p = [], []
    with torch.no_grad():
        for i in range(0, N, a.batch):
            w = torch.nan_to_num(torch.from_numpy(W[i:i+a.batch]).float().to(dev)
                                 ).clamp(-10, 10)
            out = model(w)
            preds += out["class_logits"].softmax(-1).argmax(-1).cpu().tolist()
            fiso_p += out["frac_pred"][:, 1].cpu().tolist()
    preds = np.array(preds); fiso_p = np.array(fiso_p)

    acc = float((preds == y).mean())
    conf = [[int(((y == t) & (preds == p)).sum()) for p in range(3)]
            for t in range(3)]
    bin_acc = float(((preds == 0) == (y == 0)).mean())
    ok = np.isfinite(fiso_t) & np.isfinite(fiso_p)
    r = float(np.corrcoef(fiso_t[ok], fiso_p[ok])[0, 1]) if ok.sum() > 2 else None
    rmse = float(np.sqrt(np.mean((fiso_t[ok]-fiso_p[ok])**2))) if ok.sum() > 2 else None
    strat = {CLS[k]: {"n": int((y == k).sum()),
                      "acc": float((preds[y == k] == k).mean()) if (y == k).any()
                             else None}
             for k in range(3)}
    pred_dist = {c: int((preds == k).sum()) for k, c in enumerate(CLS)}

    res = {"provenance": {"script": "13_35_swift_forge_mt_eval.py",
                          "h5": a.h5, "ckpt": a.ckpt,
                          "label_source": str(meta.get("label_source")),
                          "fiso_pred_def": "frac_pred[:,1] "
                          "(01生成器ヘッダの(f_dc,f_iso,f_clvd)順序に基づく)"},
           "n": int(N), "accuracy_3class": acc,
           "confusion_matrix_true_x_pred": conf,
           "shear_vs_rest_accuracy": bin_acc,
           "pred_class_counts": pred_dist,
           "per_true_class": strat,
           "f_iso_pearson_r": r, "f_iso_rmse": rmse}
    out = PROJECT_ROOT/"models"/"13_35_forge_mt_eval.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[13_35] 3-class acc={acc*100:.1f}%  shear-vs-rest={bin_acc*100:.1f}%")
    print(f"[13_35] pred counts: {pred_dist}")
    print(f"[13_35] per-true-class acc: "
          f"{ {k: (None if v['acc'] is None else round(v['acc']*100,1)) for k,v in strat.items()} }")
    print(f"[13_35] f_iso: r={None if r is None else round(r,3)}  "
          f"RMSE={None if rmse is None else round(rmse,3)}")
    print(f"[13_35] confusion (rows=true, cols=pred): {conf}")
    print(f"[13_35] JSON: {out} ← 原稿E1b転記の唯一の出所")

if __name__ == "__main__":
    main()
