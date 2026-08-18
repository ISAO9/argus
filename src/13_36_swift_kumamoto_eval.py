#!/usr/bin/env python3
# =============================================================================
# Script : 13_36_swift_kumamoto_eval.py     [再実験C — 熊本転移の実測]
# Project: ARGUS
#
# Description:
#   原稿の「Kumamoto transfer: 95.1% Shear classification」に対応する実測を
#   行う。監査(2026-08-17)により、95.1% は図生成スクリプト(14_04/15_01)への
#   ハードコードのみで、実験アーティファクトが存在しないことが確定した。
#   本スクリプトの出力が唯一の正典となり、原稿の該当記述は**この結果で
#   置き換える**(値がいくつであっても)。
#
#   評価内容(正解ラベル無しの実データのため「クラス分布」を報告):
#     - checkpoint_epoch_080.pt(公開重み・ゼロショット)…原稿の主張に対応
#     - swift_kumamoto_finetune.pt(参考: fine-tune済みが存在するため併記)
#     各: 3クラス判定分布 / Shear率 / f_ISO 統計 / 平均レイテンシ
#
#   モデル構築は 05_swift_evaluation_3.py の load_model をそのまま再利用
#   (models/swift_architecture.py + swift_architecture_config.json)。
#   kumamoto_dataset.h5 のスキーマは実行時に表示し、波形配列が特定できない
#   場合は何も計算せず停止する(推定・補完はしない)。
#
#   Usage:
#     uv run python src/13_36_swift_kumamoto_eval.py
#     uv run python src/13_36_swift_kumamoto_eval.py --h5 data/real/kumamoto_cmt_dataset.h5
# =============================================================================
import sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLS = ["Shear", "Mixed", "Tensile"]

def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def find_waveforms(h5):
    """h5内の (N, S, 3, T) 形状の配列を探す。曖昧なら候補を表示して停止。"""
    cands = []
    def visit(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset) and obj.ndim == 4 and obj.shape[2] == 3:
            cands.append((name, obj.shape))
    h5.visititems(visit)
    return cands

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--h5", default=str(PROJECT_ROOT/"data/real/kumamoto_dataset.h5"))
    pa.add_argument("--batch", type=int, default=16)
    a = pa.parse_args()
    import h5py
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

    mod05 = _lm("swift05", PROJECT_ROOT/"src"/"05_swift_evaluation_3.py")
    cfg = json.load(open(PROJECT_ROOT/"models"/"swift_architecture_config.json"))
    cfg_model = cfg.get("model", cfg)

    with h5py.File(a.h5) as f:
        print("[h5] tree:")
        f.visit(lambda n: print("   ", n))
        cands = find_waveforms(f)
        print("[h5] waveform candidates (N,S,3,T):", cands)
        if len(cands) != 1:
            sys.exit("[13_36] STOP: 波形配列を一意に特定できない。")
        name, shape = cands[0]
        W = f[name][:]
        grp = name.rsplit("/", 1)[0]
        labels = f[grp + "/labels"][:] if (grp + "/labels") in f else None
        mt_fr  = f[grp + "/mt_fracs"][:] if (grp + "/mt_fracs") in f else None
    print(f"[13_36] waveforms: {name} {W.shape}  labels: "
          f"{None if labels is None else labels.shape}")
    N = W.shape[0]

    results = {"h5": a.h5, "n_events": int(N), "waveform_key": name,
               "checkpoints": {}}
    for ck_name in ("checkpoint_epoch_080.pt", "swift_kumamoto_finetune.pt"):
        ck_path = PROJECT_ROOT/"models"/ck_name
        if not ck_path.exists():
            print(f"[13_36] skip (not found): {ck_name}"); continue
        model = mod05.load_model(cfg_model, ck_path, device=dev)
        model.eval()
        preds, fisos, lat = [], [], []
        with torch.no_grad():
            for i in range(0, N, a.batch):
                w = torch.from_numpy(W[i:i+a.batch]).float().to(dev)
                w = torch.nan_to_num(w).clamp(-10, 10)
                t0 = time.perf_counter()
                out = model(w)
                if dev.type == "mps": torch.mps.synchronize()
                lat.append((time.perf_counter()-t0)*1000/len(w))
                if isinstance(out, dict):
                    if i == 0:
                        print("[13_36] model output keys:",
                              {k: tuple(v.shape) for k, v in out.items()
                               if torch.is_tensor(v)})
                    mech = fiso = None
                    for k, v in out.items():
                        kl = k.lower()
                        if mech is None and any(s in kl for s in
                                ("mech", "class", "logit", "label")):
                            mech = v
                        if fiso is None and "iso" in kl and "frac" not in kl:
                            fiso = v
                    if fiso is None:
                        for k, v in out.items():
                            if "frac" in k.lower() and v.shape[-1] == 3:
                                fiso = v[..., 1]      # (f_dc, f_iso, f_clvd)想定
                                if i == 0:
                                    print(f"[13_36] f_iso <- {k}[...,1] "
                                          "(f_dc,f_iso,f_clvd想定。要確認)")
                    if mech is None:
                        sys.exit(f"[13_36] STOP: 機構ロジットのキーを特定できない: "
                                 f"{list(out.keys())} — キー名を教えてください。")
                else:
                    mech = out[0]
                    fiso = out[1] if len(out) > 1 else None
                preds += mech.softmax(-1).argmax(-1).cpu().tolist()
                if fiso is not None:
                    fisos += torch.as_tensor(fiso).flatten().cpu().tolist()
        preds = np.array(preds)
        dist = {c: int((preds == k).sum()) for k, c in enumerate(CLS)}
        entry = {"class_counts": dist,
                 "n": int(N),
                 "shear_fraction": float((preds == 0).mean()),
                 "mixed_fraction": float((preds == 1).mean()),
                 "tensile_fraction": float((preds == 2).mean()),
                 "latency_ms_mean": float(np.mean(lat))}
        if fisos:
            entry["fiso_mean"] = float(np.mean(fisos))
            entry["fiso_median"] = float(np.median(fisos))
        if labels is not None:
            lab = np.asarray(labels).astype(int).ravel()[:len(preds)]
            entry["accuracy_vs_h5_labels"] = float((preds == lab).mean())
            entry["h5_label_counts"] = {c: int((lab == k).sum())
                                        for k, c in enumerate(CLS)}
            entry["note_labels"] = ("labels/mt_fracs は 06 前処理由来"
                                    "(F-net MT分解)。定義の確認要。")
        results["checkpoints"][ck_name] = entry
        print(f"[13_36] {ck_name}: shear={entry['shear_fraction']*100:.1f}%  "
              f"mixed={entry['mixed_fraction']*100:.1f}%  "
              f"tensile={entry['tensile_fraction']*100:.1f}%  (n={N})")

    out = PROJECT_ROOT/"models"/"13_36_kumamoto_transfer.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"[13_36] JSON: {out}")
    print("[13_36] 原稿の『95.1% Shear』はこのJSONの checkpoint_epoch_080 の")
    print("        shear_fraction で置換する。値がいくつであってもそのまま採用。")

if __name__ == "__main__":
    main()
