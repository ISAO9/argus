#!/usr/bin/env python3
# =============================================================================
# Script : 13_19_forge_field_validation.py   (v3 — 既存h5直読み・E1本実験)
# Project: ARGUS
#
# Description:
#   Utah FORGE 実測マイクロサイズミックによる GNN-Locator 検証(SRL改訂 E1)。
#   v3: ダウンロード経路を廃し、06c前処理済みh5を直接読む。
#
#   設計(クロスキャンペーン・リークなし):
#     学習: forge_dataset.h5        (2019年, 300件, カタログ震源)
#     検証: 学習データ末尾15%
#     テスト: forge2024_dataset.h5  (2024年, 148件, Niemz et al. 2026 真値震源)
#
#   評価:
#     [A] zero-shot strict     — 熊本/日本統計のまま(真のゼロショット)
#     [B] zero-shot recentered — lat/lon_mean のみFORGE網重心へ平行移動
#     [C] 2019→FT→2024テスト   — Bの座標系。プロトコルは13_34 v3と同一
#         (AdamW lr=5e-5, ≤30ep, patience=15, heteroscedastic NLL, clip1.0)
#     [D] コンフォーマル        — 2019検証分をscore較正に流用せず、
#         2024テストを時系列前半/後半に分けて較正74/評価74(13_09方式)
#
#   グラフ特徴量: waveforms/snr/sta_locsはh5から。tp(STA/LTA)とS-P
#   (水平動エンベロープ)は波形から算出 — 13_19 v2と同一のヒューリスティック。
#   座標単位はh5値から自動判定(|値|>90 → 06系のm投影とみなし
#   lat=y/111000, lon=x/111000, depth=|z|/1000)。変換後の妥当性を検査し、
#   範囲外なら停止(推定はしない)。
#
#   出力:
#     models/locator/13_19_forge_results.json  ← 原稿E1転記の唯一の出所
#     models/locator/13_19_forge_finetune_best.pt
#     PDF/13_19_forge_field_validation.pdf
#
#   Usage:
#     uv run python src/13_19_forge_field_validation.py
#     uv run python src/13_19_forge_field_validation.py --seed 42 --epochs 30
# =============================================================================
import sys, json, argparse, shutil, importlib.util
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT/"models"/"locator"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR = PROJECT_ROOT/"PDF"; PDF_DIR.mkdir(exist_ok=True)
DRIVE = Path("/content/drive/MyDrive/ARGUS_backup")
FS = 100.0

def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def load_mod01():
    for f in ("13_01_build_hinet_graph.py", "13_01_build_hinet_graph_1.py"):
        p = PROJECT_ROOT/"src"/f
        if p.exists(): return _lm("gnn01", p)
    sys.exit("13_01 not found")

def load_mod02():
    for f in ("13_02_gnn_locator_model.py", "13_02_gnn_locator_model_4.py"):
        p = PROJECT_ROOT/"src"/f
        if p.exists(): return _lm("gnn02", p)
    sys.exit("13_02 not found")

def drive_backup(*files, tag=""):
    if Path("/content/drive").exists():
        DRIVE.mkdir(parents=True, exist_ok=True)
        for f in files:
            if Path(f).exists(): shutil.copy2(f, DRIVE/Path(f).name)
        print(f"[backup->Drive] {tag}")

def _utm_inverse(easting, northing, zone=12, northern=True):
    """UTM(WGS84)→緯度経度。コンパクト逆変換(小域で誤差<1e-6度)。"""
    a, f = 6378137.0, 1/298.257223563
    k0 = 0.9996
    e2 = f*(2-f); ep2 = e2/(1-e2)
    x = np.asarray(easting, np.float64) - 500000.0
    y = np.asarray(northing, np.float64)
    if not northern: y = y - 10000000.0
    m = y/k0
    mu = m/(a*(1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    e1 = (1-np.sqrt(1-e2))/(1+np.sqrt(1-e2))
    phi = (mu + (3*e1/2 - 27*e1**3/32)*np.sin(2*mu)
           + (21*e1**2/16 - 55*e1**4/32)*np.sin(4*mu)
           + (151*e1**3/96)*np.sin(6*mu)
           + (1097*e1**4/512)*np.sin(8*mu))
    sp, cp, tp_ = np.sin(phi), np.cos(phi), np.tan(phi)
    c1 = ep2*cp**2
    t1 = tp_**2
    n1 = a/np.sqrt(1-e2*sp**2)
    r1 = a*(1-e2)/(1-e2*sp**2)**1.5
    d = x/(n1*k0)
    lat = phi - (n1*tp_/r1)*(d**2/2
          - (5 + 3*t1 + 10*c1 - 4*c1**2 - 9*ep2)*d**4/24
          + (61 + 90*t1 + 298*c1 + 45*t1**2 - 252*ep2 - 3*c1**2)*d**6/720)
    lon = (d - (1 + 2*t1 + c1)*d**3/6
           + (5 - 2*c1 + 28*t1 - 3*c1**2 + 8*ep2 + 24*t1**2)*d**5/120)/cp
    lon0 = np.radians((zone-1)*6 - 180 + 3)
    return np.degrees(lat), np.degrees(lon + lon0)

def to_deg(locs, utm_zone=12):
    """(…,3) 座標をdeg/kmへ。度 / 熊本式m / UTM を自動判定。検査つき。"""
    a = np.asarray(locs, np.float64)
    xy_max = np.nanmax(np.abs(a[..., :2]))
    x_med = float(np.nanmedian(a[..., 0])); y_med = float(np.nanmedian(a[..., 1]))
    if xy_max <= 180.0:                                   # 度
        lat, lon = a[..., 0], a[..., 1]
    elif 1e5 < abs(x_med) < 9e5 and 1e6 < abs(y_med) < 1e7:   # UTM (E,N)
        lat, lon = _utm_inverse(a[..., 0], a[..., 1], zone=utm_zone)
    elif abs(x_med) > 5e6:                                # 熊本式 [lon*111e3, lat*111e3]
        lon = a[..., 0]/111000.0; lat = a[..., 1]/111000.0
    else:
        sys.exit(f"[13_19] STOP: 座標系を判定できない (x_med={x_med:.3g}, "
                 f"y_med={y_med:.3g})。src_locs の定義を教えてください。")
    dep = np.abs(a[..., 2])
    if np.nanmax(dep) > 700: dep = dep/1000.0
    return lat, lon, dep

def pick_onset(w3, fs=FS):
    """3成分エンベロープのベースライン閾値ピッカー(診断2026-08-17に基づく)。
    返り値: (tp_sec, used_fallback)。検出不可なら(None, False)。
    フォールバック=エネルギー最大時刻-0.5s(使用数はJSONに記録・透明化)。"""
    env = np.abs(w3).sum(axis=0)
    if len(env) < int(0.5*fs): return None, False
    noise = np.percentile(env, 20)                 # ノイズ床の頑健推定
    thr = noise*6.0 + 1e-12
    idx = np.where(env > thr)[0]
    if len(idx) and idx[0] >= 3:
        return float(idx[0])/fs, False
    # フォールバック: エネルギー最大の0.5s手前(下限0.05s)
    t_amax = float(np.argmax(env))/fs
    tp = max(t_amax - 0.5, 0.05)
    if t_amax <= 0.1: return None, False           # 冒頭飽和は不採用
    return tp, True

def load_forge_h5(path, region_name):
    import h5py
    with h5py.File(path) as f:
        g = f["real"]
        W = g["waveforms"][:]; SNR = g["snr"][:]
        SL = g["sta_locs"][:]; EL = g["src_locs"][:]
        MW = g["mw"][:]
    slat, slon, selev = to_deg(SL)      # selev: 深さ由来だが局は標高扱い(km)
    elat, elon, edep = to_deg(EL)
    ok = (37.0 < np.nanmedian(elat) < 40.0) and (-115.0 < np.nanmedian(elon) < -110.0)
    print(f"[h5] {region_name}: n={len(W)}  ev lat med={np.nanmedian(elat):.3f} "
          f"lon med={np.nanmedian(elon):.3f}  dep med={np.nanmedian(edep):.2f} km"
          f"  Mw {np.nanmin(MW):.2f}..{np.nanmax(MW):.2f}")
    if not ok:
        sys.exit(f"[13_19] STOP: {region_name} の座標変換結果がFORGE域外。"
                 "src_locs の単位/並びを確認してください(推定はしない)。")
    evs = []
    for i in range(len(W)):
        evs.append(dict(waveforms=W[i].astype(np.float32),
                        snr=np.asarray(SNR[i], np.float32),
                        sta_lat=slat[i], sta_lon=slon[i],
                        sta_elev=np.clip(selev[i], -5, 5),
                        ev=dict(lat=float(elat[i]), lon=float(elon[i]),
                                depth_km=float(max(edep[i], 0.05)),
                                mw=float(MW[i]))))
    return evs

def event_features(rec):
    """h5レコード→(feats入力一式)。tp/S-Pは波形から算出。"""
    W = rec["waveforms"]                        # (12,3,1024)
    keep, tp, sp = [], [], []
    n_fb = 0
    for s in range(W.shape[0]):
        w = W[s]
        if not np.isfinite(w).all() or np.abs(w).max() < 1e-9:
            continue
        p, fb = pick_onset(w)
        if p is None: continue
        n_fb += int(fb)
        h = np.abs(w[1])+np.abs(w[2]); tail = h[int(p*FS):]
        s_p = float(np.argmax(tail[int(0.05*FS):])/FS + 0.05) \
              if len(tail) > int(0.3*FS) else 0.3
        keep.append(s); tp.append(p); sp.append(s_p)
    if len(keep) < 4: return None
    m = np.abs(W[keep]).max(axis=(1, 2), keepdims=True)+1e-9
    return dict(n_fallback=n_fb,
                wavs=(W[keep]/m).astype(np.float32),
                locs=np.stack([rec["sta_lat"][keep], rec["sta_lon"][keep],
                               rec["sta_elev"][keep]], 1).astype(np.float32),
                tp=np.asarray(tp, np.float32), sp=np.asarray(sp, np.float32),
                snr=np.clip(np.asarray(rec["snr"][keep], np.float32), 0, 40),
                ev=rec["ev"])

def build_graphs(feats, mod01, enc, ns, dev):
    gs = []
    with torch.no_grad():
        for f in feats:
            wt = torch.nan_to_num(torch.from_numpy(f["wavs"]).to(dev)).clamp(-10, 10)
            emb = np.nan_to_num(enc(wt).cpu().numpy())
            ei, ea = mod01.build_edges_knn(f["locs"][:, :2], k=4)
            ev = f["ev"]
            g = mod01.build_graph_for_event(
                waveforms=f["wavs"], sta_locs=f["locs"], tp_times=f["tp"],
                sp_diffs=f["sp"], snr=f["snr"],
                src_loc=np.array([ev["lat"], ev["lon"], ev["depth_km"]], np.float32),
                mw=ev["mw"], edge_index=ei, edge_attr_geo=ea,
                norm_stats=ns, wave_emb=emb)
            for k in ("x", "edge_attr", "y"):
                g[k] = np.nan_to_num(np.asarray(g[k], np.float32),
                                     nan=0.0, posinf=0.0, neginf=0.0)
            gs.append(g)
    return gs

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--train_h5", default=str(PROJECT_ROOT/"data/real/forge_dataset.h5"))
    pa.add_argument("--test_h5", default=str(PROJECT_ROOT/"data/real/forge2024_dataset.h5"))
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--epochs", type=int, default=30)
    pa.add_argument("--lr", type=float, default=5e-5)
    pa.add_argument("--local_scale", action="store_true",
                    help="recenterに加えlat/lon_stdを2019震源分布スケールへ再設定"
                         "(座標スケール不整合の対照実験)")
    a = pa.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[13_19v3] device={dev}  seed={a.seed}")
    mod01, mod02 = load_mod01(), load_mod02()
    m34eval = _lm("m34e", PROJECT_ROOT/"src"/"13_34_random_split_eval.py") \
        if False else None  # 評価関数は下で自前定義(13_03/13_34と同一式)

    def evaluate(model, gs, ns):
        model.eval()
        H, Z, SH = [], [], []
        with torch.no_grad():
            for g in gs:
                x = torch.from_numpy(g["x"]).to(dev)
                ei = torch.from_numpy(g["edge_index"]).to(dev)
                ea = torch.from_numpy(g["edge_attr"]).to(dev)
                mu, lv = model(x, ei, ea)
                mu = mu.cpu().numpy(); sg = np.exp(0.5*lv.cpu().numpy())
                lat_p = mu[0]*ns["lat_std"]+ns["lat_mean"]
                lon_p = mu[1]*ns["lon_std"]+ns["lon_mean"]
                dep_p = float(np.exp(mu[2]*np.log(60.0)))
                lt, ln, dp = map(float, g["src_loc"])
                dlat = (lat_p-lt)*111.0
                dlon = (lon_p-ln)*111.0*np.cos(np.radians(lt))
                H.append(float(np.hypot(dlat, dlon)))
                Z.append(abs(dep_p-dp))
                sh = np.sqrt((sg[0]*ns["lat_std"]*111.0)**2 +
                             (sg[1]*ns["lon_std"]*111.0*np.cos(np.radians(lat_p)))**2)
                SH.append(max(float(sh), 0.01))
        return {k: np.array(v) for k, v in
                zip(("horiz", "dep_err", "sigma_h"), (H, Z, SH))}

    pdata = torch.load(PROJECT_ROOT/"data/locator/hinet_graph_v2.pt",
                       map_location="cpu", weights_only=False)
    ns0 = dict(pdata["norm_stats"])
    if "encoder_state" not in pdata:
        sys.exit("[13_19] STOP: encoder_state 無し。")
    enc = mod01.WaveformEncoder(3, int(pdata.get("wave_emb_dim", 16)), 1024).to(dev)
    enc.load_state_dict(pdata["encoder_state"]); enc.eval()

    tr_rec = load_forge_h5(a.train_h5, "FORGE2019(train)")
    te_rec = load_forge_h5(a.test_h5, "FORGE2024(test/Niemz)")
    tr_f = [f for f in (event_features(r) for r in tr_rec) if f]
    te_f = [f for f in (event_features(r) for r in te_rec) if f]
    fb_tr = sum(f["n_fallback"] for f in tr_f)
    fb_te = sum(f["n_fallback"] for f in te_f)
    print(f"[13_19v3] usable: train2019={len(tr_f)}  test2024={len(te_f)}  "
          f"(fallback picks: train {fb_tr}, test {fb_te})")
    if len(te_f) < 50 or len(tr_f) < 100:
        sys.exit("[13_19] STOP: 使用可能イベントが少なすぎる。ピック閾値を確認。")

    all_lat = np.concatenate([f["locs"][:, 0] for f in tr_f+te_f])
    all_lon = np.concatenate([f["locs"][:, 1] for f in tr_f+te_f])
    ns_rec = dict(ns0)
    ns_rec["lat_mean"], ns_rec["lon_mean"] = float(all_lat.mean()), float(all_lon.mean())
    if a.local_scale:
        ev_lat = np.array([f["ev"]["lat"] for f in tr_f])
        ev_lon = np.array([f["ev"]["lon"] for f in tr_f])
        ns_rec["lat_std"] = max(float(ev_lat.std()), 1e-4)
        ns_rec["lon_std"] = max(float(ev_lon.std()), 1e-4)
        print(f"[13_19v3] local_scale: lat_std->{ns_rec['lat_std']:.5f}deg "
              f"lon_std->{ns_rec['lon_std']:.5f}deg")
    print(f"[13_19v3] recenter: lat {ns0['lat_mean']:.2f}->{ns_rec['lat_mean']:.3f}  "
          f"lon {ns0['lon_mean']:.2f}->{ns_rec['lon_mean']:.3f}")

    G_te_strict = build_graphs(te_f, mod01, enc, ns0, dev)
    G_te = build_graphs(te_f, mod01, enc, ns_rec, dev)
    G_tr = build_graphs(tr_f, mod01, enc, ns_rec, dev)
    n_val = int(round(len(G_tr)*0.15))
    G_va, G_tr = G_tr[-n_val:], G_tr[:-n_val]

    model = mod02.GNNLocator().to(dev)
    ck = torch.load(MODEL_DIR/"best_locator.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck.get("model_state", ck))
    zsA = evaluate(model, G_te_strict, ns0)
    zsB = evaluate(model, G_te, ns_rec)
    print(f"[A strict    ] median={np.median(zsA['horiz']):.2f} km")
    print(f"[B recentered] median={np.median(zsB['horiz']):.2f} km  "
          f"P90={np.percentile(zsB['horiz'],90):.2f}")

    rng = np.random.RandomState(a.seed)
    nll = mod02.heteroscedastic_nll
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    ckpt = MODEL_DIR/"13_19_forge_finetune_best.pt"
    best, bad, patience = np.inf, 0, 15
    for ep in range(1, a.epochs+1):
        model.train(); tot, cnt = 0.0, 0
        for i in rng.permutation(len(G_tr)):
            g = G_tr[i]
            x = torch.from_numpy(g["x"]).to(dev)
            ei = torch.from_numpy(g["edge_index"]).to(dev)
            ea = torch.from_numpy(g["edge_attr"]).to(dev)
            y = torch.from_numpy(g["y"]).to(dev)
            opt.zero_grad(); mu, lv = model(x, ei, ea)
            loss = nll(mu, lv, y)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss.detach()); cnt += 1
        vmed = float(np.median(evaluate(model, G_va, ns_rec)["horiz"]))
        mark = ""
        if vmed < best:
            best, bad = vmed, 0
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_median_km": vmed}, ckpt)
            drive_backup(ckpt, tag=f"ep{ep}"); mark = "*"
        else:
            bad += 1
        print(f"[ep{ep:02d}] NLL={tot/max(1,cnt):.4f}  val2019 median={vmed:.3f} km {mark}")
        if bad >= patience:
            print(f"[13_19v3] early stop ep{ep}"); break

    model.load_state_dict(torch.load(ckpt, map_location=dev)["model_state"])
    ft = evaluate(model, G_te, ns_rec)
    dh = ft["horiz"]
    n_cal = len(dh)//2
    s_cal = dh[:n_cal]/ft["sigma_h"][:n_cal]
    q_hat = float(np.sort(s_cal)[min(n_cal-1, int(np.ceil(0.90*(n_cal+1)))-1)])
    cov = float(np.mean(dh[n_cal:] <= q_hat*ft["sigma_h"][n_cal:]))
    mw_te = np.array([f["ev"]["mw"] for f in te_f])

    res = {
      "provenance": {"script": "13_19_forge_field_validation.py (v3)",
        "train": a.train_h5, "test": a.test_h5,
        "targets_test": "Niemz et al. 2026 MT catalog locations (h5 src_locs)",
        "encoder": "trained encoder_state (hinet_graph_v2)",
        "seed": a.seed,
        "protocol": "cross-campaign 2019->2024; AdamW lr=%g; <=%dep; "
                    "patience=15; heteroscedastic NLL; clip1.0; "
                    "recentered lat/lon means (stds unchanged)" % (a.lr, a.epochs)},
      "n_train2019": len(G_tr)+len(G_va), "n_test2024": int(len(dh)),
      "picker": {"method": "3-comp envelope, 6x 20th-pct baseline; "
                 "fallback=energy-max minus 0.5 s",
                 "n_fallback_train": int(fb_tr), "n_fallback_test": int(fb_te)},
      "mw_test_range": [float(mw_te.min()), float(mw_te.max())],
      "zero_shot_strict_median_km": float(np.median(zsA["horiz"])),
      "zero_shot_recentered": {"median_km": float(np.median(zsB["horiz"])),
                               "p90_km": float(np.percentile(zsB["horiz"], 90))},
      "finetune_test2024": {"median_km": float(np.median(dh)),
                            "p90_km": float(np.percentile(dh, 90)),
                            "p95_km": float(np.percentile(dh, 95)),
                            "depth_median_km": float(np.median(ft["dep_err"]))},
      "conformal": {"n_cal": int(n_cal), "n_eval": int(len(dh)-n_cal),
                    "q_hat": q_hat, "empirical_coverage": cov, "nominal": 0.90},
      "per_event_horiz_km": dh.tolist(),
    }
    res["provenance"]["local_scale"] = bool(a.local_scale)
    out = MODEL_DIR/("13_19_forge_results_localscale.json" if a.local_scale
                     else "13_19_forge_results.json")
    json.dump(res, open(out, "w"), indent=2)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for arr, lab, c in ((zsB["horiz"], f"Zero-shot recentered (n={len(dh)})", "#5F7080"),
                        (dh, f"Fine-tuned 2019->2024 (n={len(dh)})", "#1668C1")):
        s = np.sort(arr)
        ax[0].plot(s, np.arange(1, len(s)+1)/len(s), lw=1.8, color=c, label=lab)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Horizontal error (km)")
    ax[0].set_ylabel("CDF"); ax[0].grid(alpha=0.25, which="both")
    ax[0].set_title("(a) FORGE 2024 test", fontsize=10)
    ax[0].legend(loc="upper left", bbox_to_anchor=(0, -0.2), frameon=False,
                 fontsize=8.5)
    ax[1].bar(["Nominal", "Empirical"], [90, cov*100], color=["#9EB3C2", "#2F7D32"])
    for i, v in enumerate([90, cov*100]):
        ax[1].text(i, v+1.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax[1].set_ylim(0, 105); ax[1].set_ylabel("Coverage (%)")
    ax[1].set_title(f"(b) Conformal (n_cal={n_cal}, q={q_hat:.2f})", fontsize=10)
    ax[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PDF_DIR/"13_19_forge_field_validation.pdf", bbox_inches="tight")
    print("="*62)
    print(f"[13_19v3] zero-shot strict {res['zero_shot_strict_median_km']:.1f} km | "
          f"recentered {res['zero_shot_recentered']['median_km']:.2f} km | "
          f"FT {res['finetune_test2024']['median_km']:.2f} km "
          f"(P90 {res['finetune_test2024']['p90_km']:.2f})")
    print(f"[13_19v3] conformal q={q_hat:.2f} coverage={cov*100:.1f}%  "
          f"n_test={len(dh)}  Mw {res['mw_test_range'][0]:.1f}..{res['mw_test_range'][1]:.1f}")
    print(f"[13_19v3] JSON: {out}")
    drive_backup(out, tag="end")

if __name__ == "__main__":
    main()
