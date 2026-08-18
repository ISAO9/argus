#!/usr/bin/env python3
# =============================================================================
# Script : 13_30_spatial_pga_map.py   (v2 — 13_29 v4規約準拠)  [Figure S2]
# Project: ARGUS
# Description:
#   代表イベント(既定: 2016熊本 event_03)の空間PGA比較 4パネル:
#     (a) 観測K-NET PGA(gal生値・観測セルのみ)
#     (b) FNO-NAMI予測(13_29 v4と同一の逆正規化: exp(ln-SI)×100)
#     (c) SM1999 GMPE(13_29 v4と同一のグリッド距離計算)
#     (d) log10(FNO/観測) 残差(観測セルのみ)
#   モデルのロードは 13_29 v4 の load_fno(remap込み)を再利用。
#   出力: PDF/13_30_spatial_pga.pdf
# Usage:
#   uv run python src/13_30_spatial_pga_map.py
#   uv run python src/13_30_spatial_pga_map.py --event 3
# =============================================================================
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT/"PDF"; PDF.mkdir(exist_ok=True)
def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--h5", default=str(Path.home()/
        "FNO-ANN/data/knet/processed/knet_dataset.h5"))
    pa.add_argument("--fno_ckpt", default=str(Path.home()/
        "FNO-ANN/models/fno_best.pth"))
    pa.add_argument("--event", type=int, default=3,
                    help="既定3 = 2016熊本 (20160416012500)")
    a = pa.parse_args()
    import h5py
    if not Path(a.h5).exists(): sys.exit(f"[13_30] STOP: {a.h5} が無い")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m29 = _lm("m29", ROOT/"src"/"13_29_fno_vs_gmpe.py")
    model = m29.load_fno(a.fno_ckpt, dev)
    with h5py.File(a.h5) as f:
        feats = f["features"][a.event]
        obs_map = f["pga_maps"][a.event]           # gal生値(未観測=0)
        mask = f["obs_masks"][a.event] > 0
        at = dict(f[f"event_{a.event:02d}"].attrs)
    mwv, Dv = float(at["mw"]), float(at.get("depth_km", 10.0))
    slat = float(at["lat"])
    print(f"[13_30] event_{a.event:02d}  {at.get('event_id')}  Mw={mwv}  "
          f"obs cells={int(mask.sum())}")
    with torch.no_grad():
        pred = model(torch.from_numpy(feats[None]).float().to(dev)
                     ).cpu().numpy().squeeze()
    LOG_PGA_MEAN, LOG_PGA_STD = -18.031818389892578, 5.287197589874268
    pred = np.exp(pred*LOG_PGA_STD + LOG_PGA_MEAN) * 100.0     # gal
    H, W = mask.shape; ext = 2.0
    rr_, cc_ = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dlat = ext - (rr_+0.5)/H*2*ext
    dlon = -ext + (cc_+0.5)/W*2*ext
    X = np.hypot(dlat*111.0, dlon*111.0*np.cos(np.radians(slat)))
    gm = m29.sm1999(mwv, Dv, np.maximum(X, 1.0))               # gal
    vmax = max(np.nanmax(obs_map[mask]), 1.0)
    vmin = max(np.nanmin(obs_map[mask][obs_map[mask] > 0]), 0.1)

    plt.rcParams.update({"font.size": 9.5, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig, ax = plt.subplots(1, 4, figsize=(14.6, 3.6))
    obs_show = np.where(mask, obs_map, np.nan)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    for a_, dat, ttl in ((ax[0], obs_show, "(a) Observed K-NET PGA"),
                         (ax[1], pred, "(b) FNO-NAMI (zero-shot)"),
                         (ax[2], gm, "(c) SM1999 GMPE")):
        im = a_.imshow(np.maximum(dat, 1e-3), norm=norm, cmap="viridis")
        a_.set_title(ttl, fontsize=10); a_.set_xticks([]); a_.set_yticks([])
        fig.colorbar(im, ax=a_, fraction=0.046, label="PGA (gal)")
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(mask & (obs_map > 0) & (pred > 0),
                       np.log10(pred/np.maximum(obs_map, 1e-6)), np.nan)
    im = ax[3].imshow(res, cmap="RdBu_r", vmin=-3, vmax=3)
    ax[3].set_title("(d) log10(FNO / observed)", fontsize=10)
    ax[3].set_xticks([]); ax[3].set_yticks([])
    fig.colorbar(im, ax=ax[3], fraction=0.046, label="orders of magnitude")
    fig.suptitle(f"Event {at.get('event_id')} (Mw {mwv:.1f}); grid ±{ext}° "
                 f"around epicenter", fontsize=10, y=1.03)
    fig.tight_layout()
    out = PDF/"13_30_spatial_pga.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"[13_30] {out}")

if __name__ == "__main__":
    main()
