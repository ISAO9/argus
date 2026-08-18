#!/usr/bin/env python3
# =============================================================================
# Script : 13_34_random_split_eval.py   (v3 — random/temporal 統一再生成版)
# Project: ARGUS
#
# Description:
#   Locator系の論文数値を単一の再現可能プロトコルで再生成する。
#   v3 = これまでの修正を全て統合した完全版:
#     - ターゲットは K-NET ヘッダ震源(13_12系譜。F-net照合なし)
#     - グラフ構築は 13_01 関数 + 学習済み encoder_state(13_04系譜の正しい形。
#       13_11/13_12 にあった load_state_dict 欠落バグは含まない)
#     - fine-tune: best_locator.pt から AdamW lr=5e-5, 早期終了 patience=15,
#       heteroscedastic NLL(13_02定義), grad clip 1.0
#     - --split random  : 80/20 無作為(val=train末尾15%)
#       --split temporal: 発生時刻順 70/30(13_12と同一比率, val=train末尾15%)
#         + テスト207を時系列前半103較正/後半104評価に分けてコンフォーマル
#           (score = err_h/sigma_h, 13_09方式)
#     - Mw層別統計(テスト分割のみ。13_18の全688件評価はFT学習イベントを含む
#       リークがあったため、本スクリプトのテスト限定値で置き換える)
#
#   出力(seed・split別。既存アーティファクトは上書きしない):
#     models/locator/13_34_{split}_finetune_best_seed{N}.pt
#     models/locator/13_34_{split}_results_seed{N}.json  ← 原稿転記の唯一の出所
#     PDF/13_34_{split}_eval_seed{N}.pdf
#
#   Usage:
#     uv run python src/13_34_random_split_eval.py --split random  --seed 42
#     uv run python src/13_34_random_split_eval.py --split temporal --seed 42
# =============================================================================
import sys, json, argparse, shutil, importlib.util
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT/"models"/"locator"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR = PROJECT_ROOT/"PDF"; PDF_DIR.mkdir(exist_ok=True)
DRIVE = Path("/content/drive/MyDrive/ARGUS_backup")

def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def load_first(*names):
    for n in names:
        p = PROJECT_ROOT/"src"/n
        if p.exists(): return _lm(n.replace(".py", ""), p), n
    sys.exit(f"[13_34] not found: {names}")

def drive_backup(*files, tag=""):
    if Path("/content/drive").exists():
        DRIVE.mkdir(parents=True, exist_ok=True)
        for f in files:
            if Path(f).exists(): shutil.copy2(f, DRIVE/Path(f).name)
        print(f"[backup->Drive] {tag}")

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--split", choices=["random", "temporal"], default="random")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--epochs", type=int, default=30)
    pa.add_argument("--lr", type=float, default=5e-5)
    a = pa.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[13_34] split={a.split}  seed={a.seed}  device={dev}")

    mod02, used02 = load_first("13_02_gnn_locator_model.py",
                               "13_02_gnn_locator_model_4.py")
    m19 = _lm("m19", PROJECT_ROOT/"src"/"13_19_forge_field_validation.py")
    mod01 = m19.load_mod01()
    print(f"[13_34] reuse: {used02}, 13_01(graph), 13_19.evaluate")

    # ---- events: K-NETキャッシュ。教師 = ヘッダ震源(13_12系譜) ----
    cache = PROJECT_ROOT/"data"/"knet_processed"/"knet_parsed.pt"
    if not cache.exists():
        sys.exit(f"[13_34] STOP: {cache} が無い。")
    kevents = torch.load(cache, map_location="cpu", weights_only=False)
    print(f"[13_34] knet_parsed: {len(kevents)} events")
    matched, bad = [], 0
    for kev in kevents:
        sl = np.asarray(kev.get("src_loc"), dtype=float).ravel()
        if sl.size < 3: bad += 1; continue
        lat, lon, dep = float(sl[0]), float(sl[1]), float(abs(sl[2]))
        if not (20.0 < lat < 50.0 and 120.0 < lon < 150.0): bad += 1; continue
        if dep > 700.0: dep /= 1000.0
        kev = dict(kev)
        kev["fnet"] = {"lat": lat, "lon": lon, "depth_km": dep,
                       "mw": float(kev.get("mw", float("nan")))}
        matched.append(kev)
    print(f"[13_34] usable: {len(matched)}  rejected: {bad}")
    if len(matched) < 500:
        sys.exit("[13_34] STOP: src_loc 形式が想定外。")

    # ---- graphs: 13_01関数 + 学習済みエンコーダ(13_04系譜の正しい形) ----
    pdata = torch.load(PROJECT_ROOT/"data"/"locator"/"hinet_graph_v2.pt",
                       map_location="cpu", weights_only=False)
    ns = pdata["norm_stats"]
    if "encoder_state" not in pdata:
        sys.exit("[13_34] STOP: encoder_state が無い(埋め込み再現不可)。")
    T = np.asarray(matched[0]["waveforms"]).shape[-1]
    enc = mod01.WaveformEncoder(3, int(pdata.get("wave_emb_dim", 16)), T).to(dev)
    enc.load_state_dict(pdata["encoder_state"]); enc.eval()
    graphs = []
    with torch.no_grad():
        for ev in matched:
            wavs = np.asarray(ev["waveforms"], np.float32)
            w_t = torch.nan_to_num(torch.from_numpy(wavs).to(dev)).clamp(-10, 10)
            emb = np.nan_to_num(enc(w_t).cpu().numpy())
            sta = np.asarray(ev["sta_locs"], np.float32)
            ei, ea = mod01.build_edges_knn(sta, k=4)
            f = ev["fnet"]
            g = mod01.build_graph_for_event(
                waveforms=wavs, sta_locs=sta,
                tp_times=np.asarray(ev["tp_times"], np.float32),
                sp_diffs=np.asarray(ev["sp_diffs"], np.float32),
                snr=np.asarray(ev["snr"], np.float32),
                src_loc=np.array([f["lat"], f["lon"], f["depth_km"]], np.float32),
                mw=f["mw"], edge_index=ei, edge_attr_geo=ea,
                norm_stats=ns, wave_emb=emb)
            for k in ("x", "edge_attr", "y"):
                g[k] = np.nan_to_num(np.asarray(g[k], np.float32),
                                     nan=0.0, posinf=0.0, neginf=0.0)
            graphs.append(g)
    n = len(graphs)

    # ---- split ----
    rng = np.random.RandomState(a.seed)
    if a.split == "random":
        idx = rng.permutation(n)
        n_test = int(round(n*0.20))
        te_i = idx[:n_test]; tr_all = idx[n_test:]
        n_val = int(round(len(tr_all)*0.15))
        va_i, tr_i = tr_all[:n_val], tr_all[n_val:]
    else:
        ot = [m["origin_time"] for m in matched]
        arr = (np.array(ot, dtype="datetime64[s]")
               if not isinstance(ot[0], str) else np.array(ot))
        order = np.argsort(arr)
        n_train = int(round(n*0.70))
        tr_all = order[:n_train]; te_i = order[n_train:]
        n_val = int(round(len(tr_all)*0.15))
        va_i, tr_i = tr_all[-n_val:], tr_all[:-n_val]
    tr = [graphs[i] for i in tr_i]; va = [graphs[i] for i in va_i]
    te = [graphs[i] for i in te_i]
    mw_te = np.array([matched[i]["fnet"]["mw"] for i in te_i])
    print(f"[13_34] split={a.split}: train={len(tr)} val={len(va)} test={len(te)}")

    # ---- model + zero-shot ----
    model = mod02.GNNLocator().to(dev)
    ck = torch.load(MODEL_DIR/"best_locator.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck.get("model_state", ck))
    zs = m19.evaluate(model, te, ns, dev)
    print(f"[13_34] zero-shot on test: median={np.median(zs['horiz']):.2f} km")

    # ---- fine-tune ----
    nll = mod02.heteroscedastic_nll
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    ckpt = MODEL_DIR/f"13_34_{a.split}_finetune_best_seed{a.seed}.pt"
    best, bad_ep, patience = np.inf, 0, 15
    for ep in range(1, a.epochs+1):
        model.train(); order = rng.permutation(len(tr)); tot, cnt = 0.0, 0
        for i in order:
            g = tr[i]
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
        vmed = float(np.median(m19.evaluate(model, va, ns, dev)["horiz"]))
        mark = ""
        if vmed < best:
            best, bad_ep = vmed, 0
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_median_km": vmed, "seed": a.seed,
                        "split": a.split}, ckpt)
            drive_backup(ckpt, tag=f"{a.split} s{a.seed} ep{ep}"); mark = "*"
        else:
            bad_ep += 1
        print(f"[ep{ep:02d}] NLL={tot/max(1,cnt):.4f}  val median={vmed:.2f} km {mark}")
        if bad_ep >= patience:
            print(f"[13_34] early stop ep{ep}"); break

    # ---- final eval + conformal(temporal) + stratified ----
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model_state"])
    R = m19.evaluate(model, te, ns, dev)
    dh = R["horiz"]
    extra = {}
    if a.split == "temporal":
        n_cal = len(te)//2
        s_cal = dh[:n_cal]/R["sigma_h"][:n_cal]
        q_hat = float(np.sort(s_cal)[min(n_cal-1,
                      int(np.ceil(0.90*(n_cal+1)))-1)])
        cov = float(np.mean(dh[n_cal:] <= q_hat*R["sigma_h"][n_cal:]))
        extra["conformal"] = {"n_cal": int(n_cal), "n_eval": int(len(te)-n_cal),
                              "q_hat": q_hat, "empirical_coverage": cov,
                              "nominal": 0.90}
        print(f"[13_34] conformal: n_cal={n_cal}  q_hat={q_hat:.2f}  "
              f"coverage={cov*100:.1f}% (nominal 90%)")
    strat = []
    for lo, hi in [(2.6, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 7.4)]:
        m = (mw_te >= lo) & (mw_te < hi)
        if m.sum() >= 5:
            strat.append({"mw": f"{lo}-{hi}", "n": int(m.sum()),
                          "median_km": float(np.median(dh[m])),
                          "p90_km": float(np.percentile(dh[m], 90))})
    extra["mw_stratified_test_only"] = strat

    res = {
      "provenance": {
        "script": "13_34_random_split_eval.py (v3)",
        "inputs": [str(cache), "data/locator/hinet_graph_v2.pt",
                   "models/locator/best_locator.pt"],
        "targets": "K-NET header hypocenters (13_12 lineage)",
        "encoder": "trained encoder_state (13_04 lineage; 13_11/13_12 bug fixed)",
        "seed": a.seed, "split": a.split,
        "protocol": f"AdamW lr={a.lr}, epochs<={a.epochs}, patience=15, "
                    f"heteroscedastic NLL, grad clip 1.0"},
      "n_total": n, "n_test": int(len(dh)),
      "zero_shot_test_median_km": float(np.median(zs["horiz"])),
      "test_median_km": float(np.median(dh)),
      "test_p90_km": float(np.percentile(dh, 90)),
      "test_p95_km": float(np.percentile(dh, 95)),
      "test_depth_median_km": float(np.median(R["dep_err"])),
      "per_event_horiz_km": dh.tolist(),
      **extra,
    }
    out = MODEL_DIR/f"13_34_{a.split}_results_seed{a.seed}.json"
    json.dump(res, open(out, "w"), indent=2)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(6, 4))
    s = np.sort(dh); ax.plot(s, np.arange(1, len(s)+1)/len(s), color="#1668C1",
                             label=f"Fine-tuned test (n={len(s)})")
    z = np.sort(zs["horiz"]); ax.plot(z, np.arange(1, len(z)+1)/len(z),
                                      color="#5F7080", label="Zero-shot")
    ax.set_xlabel("Horizontal error (km)"); ax.set_ylabel("CDF"); ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PDF_DIR/f"13_34_{a.split}_eval_seed{a.seed}.pdf", bbox_inches="tight")
    print("="*60)
    print(f"[13_34] RESULT split={a.split} seed={a.seed} n={res['n_test']}  "
          f"median={res['test_median_km']:.2f} km  P90={res['test_p90_km']:.2f}  "
          f"P95={res['test_p95_km']:.2f}")
    print(f"[13_34] JSON: {out}")
    drive_backup(out, tag="end")

if __name__ == "__main__":
    main()
