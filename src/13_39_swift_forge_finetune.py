#!/usr/bin/env python3
# =============================================================================
# Script : 13_39_swift_forge_finetune.py   [E1b-FT — SWIFT現地適応の検証]
# Project: ARGUS
#
# Description:
#   SWIFT CMT を FORGE 2024 の真値MTラベル(Niemz et al. 2026 由来)で
#   現地fine-tuneし、機構判別力が実データ適応で回復するかを検証する。
#   13_35(zero-shot: 全件Shear崩壊, 3クラス26.4%)の対をなす実験。
#
#   ◆ 事前宣言プロトコル(2026-08-17 承認済み。結果を見ての変更はしない):
#     - 分割: h5行順で時系列70/30(train=104, test=44)。val=trainの末尾15%。
#       ※h5に時刻列が無いため「06cのファイル名(YYYYMMDDHHMMSS)順=時系列」
#         と仮定(来歴に明記)。仮定が外れても行順分割としてリークは無い。
#     - 学習: ヘッド層のみ解凍(name に class/head/frac/fc/out を含む層。
#       該当ゼロなら層名一覧を表示して停止)。AdamW lr=1e-4, wd=1e-4,
#       batch=16, ≤50ep, patience=10。
#     - 損失: クラス重み付きCE(train内逆頻度) + 0.5×SmoothL1(f_ISO回帰)。
#     - モデル選択: val macro-recall 最大。テスト評価は選択後に1回のみ。
#     - 結果は値の如何を問わず報告(数値台帳ルール)。
#
#   出力:
#     models/13_39_swift_forge_best.pt
#     models/13_39_swift_forge_ft.json   ← 原稿転記の唯一の出所
#
#   Usage: uv run python src/13_39_swift_forge_finetune.py
# =============================================================================
import sys, json, argparse, shutil, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLS = ["Shear", "Mixed", "Tensile"]
DRIVE = Path("/content/drive/MyDrive/ARGUS_backup")

def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def drive_backup(*files, tag=""):
    if Path("/content/drive").exists():
        DRIVE.mkdir(parents=True, exist_ok=True)
        for f in files:
            if Path(f).exists(): shutil.copy2(f, DRIVE/Path(f).name)
        print(f"[backup->Drive] {tag}")

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    acc = float((y == p).mean())
    rec = [float((p[y == k] == k).mean()) if (y == k).any() else np.nan
           for k in range(3)]
    conf = [[int(((y == t) & (p == q)).sum()) for q in range(3)]
            for t in range(3)]
    return dict(acc=acc, macro_recall=float(np.nanmean(rec)),
                per_class_recall={CLS[k]: (None if np.isnan(rec[k]) else rec[k])
                                  for k in range(3)},
                shear_vs_rest=float(((p == 0) == (y == 0)).mean()),
                confusion_true_x_pred=conf,
                pred_counts={c: int((p == k).sum()) for k, c in enumerate(CLS)})

@torch.no_grad()
def infer(model, W, dev, batch=16):
    preds, fiso = [], []
    for i in range(0, len(W), batch):
        w = torch.nan_to_num(torch.from_numpy(W[i:i+batch]).float().to(dev)
                             ).clamp(-10, 10)
        out = model(w)
        preds += out["class_logits"].softmax(-1).argmax(-1).cpu().tolist()
        fiso += out["frac_pred"][:, 1].cpu().tolist()
    return np.array(preds), np.array(fiso)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--h5", default=str(PROJECT_ROOT/"data/real/forge2024_dataset.h5"))
    pa.add_argument("--ckpt", default=str(PROJECT_ROOT/"models/checkpoint_epoch_080.pt"))
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--batch", type=int, default=16)
    pa.add_argument("--unfreeze_pat", type=str,
                    default="class,head,frac,fc,out")
    pa.add_argument("--seed", type=int, default=42)
    a = pa.parse_args()
    import h5py
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    mod05 = _lm("swift05", PROJECT_ROOT/"src"/"05_swift_evaluation_3.py")
    cfg = json.load(open(PROJECT_ROOT/"models"/"swift_architecture_config.json"))
    model = mod05.load_model(cfg.get("model", cfg), Path(a.ckpt), device=dev)

    with h5py.File(a.h5) as f:
        g = f["real"]
        W = g["waveforms"][:]
        y = np.asarray(g["labels"][:], int).ravel()
        fiso_t = np.asarray(g["f_iso"][:], float).ravel()
    N = len(W)
    n_tr = int(round(N*0.70)); n_val = int(round(n_tr*0.15))
    tr_i = np.arange(0, n_tr-n_val); va_i = np.arange(n_tr-n_val, n_tr)
    te_i = np.arange(n_tr, N)
    print(f"[13_39] n={N} -> train={len(tr_i)} val={len(va_i)} test={len(te_i)}")
    for nm, ii in (("train", tr_i), ("val", va_i), ("test", te_i)):
        print(f"        {nm} labels: "
              f"{ {c: int((y[ii]==k).sum()) for k,c in enumerate(CLS)} }")

    # ---- freeze all, unfreeze head-like layers ----
    pats = [p.strip().lower() for p in a.unfreeze_pat.split(",") if p.strip()]
    n_all, n_open, opened = 0, 0, set()
    for name, prm in model.named_parameters():
        n_all += prm.numel()
        if any(p in name.lower() for p in pats):
            prm.requires_grad = True; n_open += prm.numel()
            opened.add(name.split(".")[0])
        else:
            prm.requires_grad = False
    if n_open == 0:
        print("[13_39] STOP: 解凍対象が0。モジュール名一覧:")
        for name, _ in model.named_children(): print("   ", name)
        sys.exit(1)
    print(f"[13_39] unfrozen: {n_open:,}/{n_all:,} params  "
          f"(top-level: {sorted(opened)})")

    # ---- class weights (train inverse frequency) ----
    cnt = np.array([(y[tr_i] == k).sum() for k in range(3)], float)
    wts = torch.tensor((cnt.sum()/np.maximum(cnt, 1)) /
                       (cnt.sum()/np.maximum(cnt, 1)).mean(),
                       dtype=torch.float32, device=dev)
    print(f"[13_39] class weights: {wts.cpu().numpy().round(3).tolist()}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=1e-4)
    ck_out = PROJECT_ROOT/"models"/"13_39_swift_forge_best.pt"
    best, bad, patience = -np.inf, 0, 10
    rng = np.random.RandomState(a.seed)
    for ep in range(1, a.epochs+1):
        model.train()
        order = rng.permutation(tr_i)
        tot, nb = 0.0, 0
        for i0 in range(0, len(order), a.batch):
            ii = order[i0:i0+a.batch]
            w = torch.nan_to_num(torch.from_numpy(W[ii]).float().to(dev)
                                 ).clamp(-10, 10)
            yy = torch.from_numpy(y[ii]).long().to(dev)
            ff = torch.from_numpy(fiso_t[ii]).float().to(dev)
            opt.zero_grad()
            out = model(w)
            loss = (F.cross_entropy(out["class_logits"], yy, weight=wts)
                    + 0.5*F.smooth_l1_loss(out["frac_pred"][:, 1], ff))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step(); tot += float(loss.detach()); nb += 1
        model.eval()
        pv, _ = infer(model, W[va_i], dev, a.batch)
        mv = metrics(y[va_i], pv)
        mark = ""
        if mv["macro_recall"] > best:
            best, bad = mv["macro_recall"], 0
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_macro_recall": best}, ck_out)
            drive_backup(ck_out, tag=f"ep{ep}"); mark = "*"
        else:
            bad += 1
        print(f"[ep{ep:02d}] loss={tot/max(1,nb):.4f}  "
              f"val acc={mv['acc']*100:.1f}%  macroR={mv['macro_recall']*100:.1f}% {mark}")
        if bad >= patience:
            print(f"[13_39] early stop ep{ep}"); break

    # ---- single test evaluation ----
    model.load_state_dict(torch.load(ck_out, map_location=dev)["model_state"])
    model.eval()
    p0, f0 = infer(mod05.load_model(cfg.get("model", cfg), Path(a.ckpt),
                                    device=dev), W[te_i], dev, a.batch)
    pt, ft_ = infer(model, W[te_i], dev, a.batch)
    m_zs, m_ft = metrics(y[te_i], p0), metrics(y[te_i], pt)
    ok = np.isfinite(fiso_t[te_i])
    r_zs = float(np.corrcoef(fiso_t[te_i][ok], f0[ok])[0, 1])
    r_ft = float(np.corrcoef(fiso_t[te_i][ok], ft_[ok])[0, 1])
    res = {"provenance": {"script": "13_39_swift_forge_finetune.py",
             "h5": a.h5, "ckpt": a.ckpt, "seed": a.seed,
             "protocol": "PRE-DECLARED: chronological(row-order) 70/30, "
                         "val=last15% of train, heads-only FT "
                         f"(pat={a.unfreeze_pat}), AdamW lr={a.lr} wd=1e-4 "
                         f"batch={a.batch} <= {a.epochs}ep patience=10, "
                         "loss=weighted CE + 0.5*SmoothL1(f_iso), "
                         "select by val macro-recall, single test eval",
             "assumption": "h5 row order = chronological (06c filename order)"},
           "n_train": int(len(tr_i)), "n_val": int(len(va_i)),
           "n_test": int(len(te_i)),
           "test_true_counts": {c: int((y[te_i] == k).sum())
                                for k, c in enumerate(CLS)},
           "zero_shot_test": {**m_zs, "f_iso_r": r_zs},
           "finetuned_test": {**m_ft, "f_iso_r": r_ft,
             "f_iso_rmse": float(np.sqrt(np.mean(
                 (fiso_t[te_i][ok]-ft_[ok])**2)))}}
    out = PROJECT_ROOT/"models"/"13_39_swift_forge_ft.json"
    json.dump(res, open(out, "w"), indent=2)
    print("="*62)
    print(f"[13_39] TEST n={len(te_i)}  true={res['test_true_counts']}")
    print(f"[13_39] zero-shot : acc={m_zs['acc']*100:.1f}%  "
          f"macroR={m_zs['macro_recall']*100:.1f}%  f_iso r={r_zs:.3f}")
    print(f"[13_39] fine-tuned: acc={m_ft['acc']*100:.1f}%  "
          f"macroR={m_ft['macro_recall']*100:.1f}%  f_iso r={r_ft:.3f}")
    print(f"[13_39] FT pred counts: {m_ft['pred_counts']}")
    print(f"[13_39] FT confusion (true x pred): {m_ft['confusion_true_x_pred']}")
    print(f"[13_39] JSON: {out} ← 値の如何を問わずこのまま原稿へ転記")
    drive_backup(out, tag="end")

if __name__ == "__main__":
    main()
