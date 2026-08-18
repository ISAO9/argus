#!/usr/bin/env python3
# =============================================================================
# Script : 13_29_fno_vs_gmpe.py                             [E2 -> Table S3]
# Project: ARGUS
# Description:
#   R2-Major2 対応: FNO-NAMI と Si & Midorikawa (1999) GMPE を、knet_dataset.h5
#   の「観測」PGA(pga_maps の obs_masks 画素)に対して同一画素集合で直接比較。
#   スキーマ(13_16 準拠): features(N,4,H,W), pga_maps(N,H,W), obs_masks(N,H,W),
#   event_{i:02d}.attrs (mw, depth ほか)。
#   FNO 順伝播は 13_17 → 13_14 の順でモデルクラスを import 試行し、見つから
#   なければモジュール内容を表示して停止(捏造しない)。
#   GMPE 距離: features の距離チャネル(第3ch, d/dmax 正規化)× attrs の
#   dmax_km。attrs に無い場合は --dmax_km で指定。
#   SM1999: log10 A = 0.50Mw + 0.0043D + 0.61 - log10(X+0.0055*10^(0.50Mw)) - 0.003X
#   ※Figure 6b の参照曲線実装と一致するか --selftest で照合してから本実行。
# Usage:
#   uv run python src/13_29_fno_vs_gmpe.py --selftest
#   uv run python src/13_29_fno_vs_gmpe.py --fno_ckpt models/fno_best.pth
# =============================================================================
import sys, json, argparse, importlib.util
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def sm1999(mw, D, X):
    return 10**(0.50*mw + 0.0043*D + 0.61
                - np.log10(X + 0.0055*10**(0.50*mw)) - 0.003*X)

def load_fno(ckpt, device):
    import torch
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    if hasattr(ck, "eval"):                      # モデル本体のpickle
        print("[fno] checkpoint is a pickled model"); ck.to(device); ck.eval()
        return ck
    sd = ck.get("model_state_dict", ck.get("model_state", ck))
    cfg = ck.get("model_config", {})
    if cfg: print(f"[fno] model_config: {cfg}")
    kw = {"modes1": cfg.get("modes", 16), "modes2": cfg.get("modes", 16),
          "width": cfg.get("width", 64), "n_layers": cfg.get("n_layers", 4),
          "in_channels": cfg.get("in_channels", 4)}
    cfg = kw
    # キー名変換: checkpoint(fno_blocks.N.spectral_conv.*)-> 13_17系(bs.N.c.*)
    def remap(k):
        k = k.replace("fno_blocks.", "bs.")
        k = k.replace(".spectral_conv.weights1", ".c.w1")
        k = k.replace(".spectral_conv.weights2", ".c.w2")
        k = k.replace(".conv.weight", ".r.weight")
        k = k.replace(".conv.bias", ".r.bias")
        return k
    sd = { remap(k): v for k, v in sd.items() }
    print(f"[fno] remapped keys 例: {list(sd.keys())[:4]}")
    for f in ("13_17_fno_knet_retrain.py",
              "13_14_argus_true_nami.py", "13_08_nami_pga_validation_v2.py"):
        p = PROJECT_ROOT/"src"/f
        if not p.exists(): continue
        s = importlib.util.spec_from_file_location("fno_mod", p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        for cls in ("FNO2d", "FNONami", "NAMI", "FNO"):
            if hasattr(m, cls):
                model = None
                for kws in (cfg,
                            {"modes": cfg["modes1"], "width": cfg["width"],
                             "n_layers": cfg["n_layers"],
                             "in_channels": cfg["in_channels"]},
                            {}):
                    try:
                        model = getattr(m, cls)(**kws); break
                    except TypeError:
                        continue
                if model is None:
                    print(f"[fno] {cls} from {f}: コンストラクタ不適合"); continue
                try:
                    model.load_state_dict(sd)
                except RuntimeError as e:
                    print(f"[fno] {cls} from {f}: 不一致 -> 次の定義を試行")
                    continue
                print(f"[fno] {cls} from {f}")
                return model.to(device).eval()
        print(f"[fno] {f} にモデルクラスが見つからない。定義一覧:")
        print("     ", [x for x in dir(m) if x[0].isupper()][:20])
    sys.exit("[E2] FNO モデルを import できません。クラス名を教えてください。")

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--h5", default=str(PROJECT_ROOT/"data/knet/processed/knet_dataset.h5"))
    pa.add_argument("--fno_ckpt", default=str(PROJECT_ROOT/"models/fno_best.pth"))
    pa.add_argument("--dmax_km", type=float, default=None)
    pa.add_argument("--selftest", action="store_true")
    a = pa.parse_args()
    if a.selftest:
        for X in (10, 50, 100):
            print(f"SM1999 Mw6 D10 X={X:>3} -> {sm1999(6,10,X):8.2f} gal")
        return
    import h5py, torch
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    model = load_fno(a.fno_ckpt, dev)
    obs_all, fno_all, gm_all = [], [], []
    with h5py.File(a.h5) as f:
        print("[h5] keys:", list(f.keys()))
        feats, maps, masks = f["features"][:], f["pga_maps"][:], f["obs_masks"][:]
        N = feats.shape[0]
        for i in range(N):
            grp = f.get(f"event_{i:02d}")
            at = dict(grp.attrs) if grp else {}
            mwv = float(at.get("mw", np.nan))
            Dv  = float(at.get("depth", at.get("depth_km", 10.0)))
            dmax= float(at.get("dmax_km", a.dmax_km or np.nan))
            if not np.isfinite(mwv):
                print(f"[E2] event_{i:02d}: mw が attrs に無い -> skip"); continue
            # dmax は不要(距離はグリッド定義から直接計算)
            with torch.no_grad():
                x = torch.from_numpy(feats[i:i+1]).float().to(dev)
                pred = model(x).cpu().numpy().squeeze()
            # 予測のみ13_16/FORGE規約で逆正規化(ln-SI空間)。観測はgal生値。
            LOG_PGA_MEAN, LOG_PGA_STD = -18.031818389892578, 5.287197589874268
            pred = np.exp(pred*LOG_PGA_STD + LOG_PGA_MEAN) * 100.0
            m = masks[i] > 0
            obs = maps[i][m]
            # 距離: グリッド定義(震源中心, ±extent度, 13_16 L305-306)から直接計算
            H, W = masks[i].shape
            rr, cc = np.where(m)
            ext = 2.0
            dlat = ext - (rr + 0.5)/H * 2*ext
            dlon = -ext + (cc + 0.5)/W * 2*ext
            slat = float(at.get("lat", 35.0))
            X = np.hypot(dlat*111.0, dlon*111.0*np.cos(np.radians(slat)))
            if i == 0:
                print(f"  [sanity] obs gal med={np.median(obs):.1f} max={obs.max():.0f}"
                      f" | pred med={np.median(pred[m]):.2f}"
                      f" | X: {X.min():.0f}-{X.max():.0f} km")
            gm = sm1999(mwv, Dv, np.maximum(X, 1.0))
            obs_all.append(obs); fno_all.append(pred[m]); gm_all.append(gm)
            print(f"  event_{i:02d}: n_px={m.sum()}  Mw={mwv:.1f}")
    obs = np.concatenate(obs_all); fno = np.concatenate(fno_all)
    gm = np.concatenate(gm_all)
    ok = (obs > 0) & (fno > 0) & (gm > 0)
    obs, fno, gm = obs[ok], fno[ok], gm[ok]
    def score(p):
        lo, lp = np.log10(obs), np.log10(p)
        return dict(r=float(np.corrcoef(lo, lp)[0, 1]),
                    rmse_gal=float(np.sqrt(((p-obs)**2).mean())),
                    sigma_log10=float((lp-lo).std()))
    res = {"n": int(len(obs)), "fno_vs_obs": score(fno), "gmpe_vs_obs": score(gm)}
    out = PROJECT_ROOT/"models"/"13_29_fno_vs_gmpe.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[E2] n={res['n']}")
    print(f"     FNO  vs obs: {res['fno_vs_obs']}")
    print(f"     GMPE vs obs: {res['gmpe_vs_obs']}")
    print(f"[save] {out}  -> Table S3。解釈文は結果どおりに(GMPE優位もあり得る)。")
    print("[NOTE] 距離ch=index2 / log線形の自動判定を上の出力で必ず目視確認。")

if __name__ == "__main__":
    main()
