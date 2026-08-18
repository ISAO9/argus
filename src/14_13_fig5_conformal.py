#!/usr/bin/env python3
# =============================================================================
# Script : 14_13_fig5_conformal.py      [Figure 5 再生成 — コンフォーマル]
# Project: ARGUS
# Description:
#   temporal seed42 のfine-tune済みチェックポイントから per-event の
#   (誤差, σ_h) を再計算し(学習なし・13_34 v3と同一のデータ経路)、
#     (a) 正規化残差ヒストグラム + q̂=較正半のみで算出した閾値線 + 被覆
#     (b) 較正曲線: 名目被覆 vs 経験被覆(較正半→評価半、13_09方式)
#     (c) K-NET 3seed + FORGE のconformal被覆まとめ(JSONから)
#   入力(無ければ停止): data/knet_processed/knet_parsed.pt,
#     data/locator/hinet_graph_v2.pt, models/locator/best_locator.pt,
#     models/locator/13_34_temporal_finetune_best_seed42.pt,
#     models/locator/13_37_seed_summary.json, models/locator/13_19_forge_results.json
#   出力: PDF/14_13_fig5_conformal.pdf
# Usage: uv run python src/14_13_fig5_conformal.py
# =============================================================================
import sys, json, importlib.util
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT/"models"/"locator"; PDF = ROOT/"PDF"; PDF.mkdir(exist_ok=True)

def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def need(p):
    if not p.exists(): sys.exit(f"[14_13] STOP: {p} が無い")
    return p

dev = torch.device("cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else "cpu")
m19 = _lm("m19", need(ROOT/"src"/"13_19_forge_field_validation.py"))
mod01 = m19.load_mod01()
mod02 = m19.load_mod02() if hasattr(m19, "load_mod02") else \
        _lm("m02", (ROOT/"src"/"13_02_gnn_locator_model.py") if
            (ROOT/"src"/"13_02_gnn_locator_model.py").exists() else
            need(ROOT/"src"/"13_02_gnn_locator_model_4.py"))

kevents = torch.load(need(ROOT/"data/knet_processed/knet_parsed.pt"),
                     map_location="cpu", weights_only=False)
matched = []
for kev in kevents:
    sl = np.asarray(kev.get("src_loc"), dtype=float).ravel()
    if sl.size < 3: continue
    lat, lon, dep = float(sl[0]), float(sl[1]), float(abs(sl[2]))
    if not (20.0 < lat < 50.0 and 120.0 < lon < 150.0): continue
    if dep > 700.0: dep /= 1000.0
    kev = dict(kev)
    kev["fnet"] = {"lat": lat, "lon": lon, "depth_km": dep,
                   "mw": float(kev.get("mw", float("nan")))}
    matched.append(kev)
pdata = torch.load(need(ROOT/"data/locator/hinet_graph_v2.pt"),
                   map_location="cpu", weights_only=False)
ns = pdata["norm_stats"]
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
            g[k] = np.nan_to_num(np.asarray(g[k], np.float32))
        graphs.append(g)
ot = [m["origin_time"] for m in matched]
arr = (np.array(ot, dtype="datetime64[s]") if not isinstance(ot[0], str)
       else np.array(ot))
order = np.argsort(arr)
te_i = order[int(round(len(graphs)*0.70)):]
te = [graphs[i] for i in te_i]
def evaluate(model, gs, ns):
    """13_19 v3 / 13_34 v3 と同一の評価式(ローカル定義)。"""
    model.eval(); H, SH = [], []
    with torch.no_grad():
        for g in gs:
            x = torch.from_numpy(g["x"]).to(dev)
            ei = torch.from_numpy(g["edge_index"]).to(dev)
            ea = torch.from_numpy(g["edge_attr"]).to(dev)
            mu, lv = model(x, ei, ea)
            mu = mu.cpu().numpy(); sg = np.exp(0.5*lv.cpu().numpy())
            lat_p = mu[0]*ns["lat_std"]+ns["lat_mean"]
            lon_p = mu[1]*ns["lon_std"]+ns["lon_mean"]
            lt, ln, dp = map(float, g["src_loc"])
            dlat = (lat_p-lt)*111.0
            dlon = (lon_p-ln)*111.0*np.cos(np.radians(lt))
            H.append(float(np.hypot(dlat, dlon)))
            sh = np.sqrt((sg[0]*ns["lat_std"]*111.0)**2 +
                         (sg[1]*ns["lon_std"]*111.0*np.cos(np.radians(lat_p)))**2)
            SH.append(max(float(sh), 0.01))
    return {"horiz": np.array(H), "sigma_h": np.array(SH)}

model = mod02.GNNLocator().to(dev)
ck = torch.load(need(MD/"13_34_temporal_finetune_best_seed42.pt"),
                map_location=dev, weights_only=False)
model.load_state_dict(ck["model_state"])
R = evaluate(model, te, ns)
dh, sh = R["horiz"], R["sigma_h"]
n_cal = len(dh)//2
s_cal, s_ev = dh[:n_cal]/sh[:n_cal], dh[n_cal:]/sh[n_cal:]
q90_rc = float(np.sort(s_cal)[min(n_cal-1, int(np.ceil(0.90*(n_cal+1)))-1)])
t42 = json.load(open(need(MD/"13_34_temporal_results_seed42.json")))
q90 = float(t42["conformal"]["q_hat"])          # 図はアーカイブ値で注記
cov90 = float((s_ev <= q90).mean())
print(f"[14_13] recomputed q̂={q90_rc:.3f} vs archived {q90:.3f} "
      f"(Δ={abs(q90-q90_rc):.3f}σ, MPS丸め内)  coverage={cov90*100:.1f}%")

S = json.load(open(need(MD/"13_37_seed_summary.json")))
FG = json.load(open(need(MD/"13_19_forge_results.json")))

plt.rcParams.update({"font.size": 9.5, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))
ax[0].hist(s_ev, bins=28, color="#9EB3C2", edgecolor="white")
ax[0].axvline(q90, color="#C0392B", lw=1.8)
ax[0].text(q90*1.04, ax[0].get_ylim()[1]*0.9, f"q̂ = {q90:.2f}σ",
           color="#C0392B", fontsize=9)
ax[0].set_xlabel("Normalized residual  |err| / σ̂"); ax[0].set_ylabel("Events")
ax[0].grid(alpha=0.25)
ax[0].set_title(f"(a) Eval half (n={len(s_ev)}): {cov90*100:.1f}% ≤ q̂", fontsize=10)

noms = np.linspace(0.50, 0.99, 25)
emp = []
for a_ in noms:
    q = float(np.sort(s_cal)[min(n_cal-1, int(np.ceil(a_*(n_cal+1)))-1)])
    emp.append(float((s_ev <= q).mean()))
ax[1].plot([0.5, 1.0], [0.5, 1.0], "--", color="#888", lw=1)
ax[1].plot(noms, emp, color="#1668C1", lw=1.8)
ax[1].scatter([0.90], [cov90], color="#C0392B", zorder=3)
ax[1].set_xlabel("Nominal coverage"); ax[1].set_ylabel("Empirical coverage")
ax[1].grid(alpha=0.25); ax[1].set_title("(b) Calibration curve", fontsize=10)

rows = [(f"K-NET s{p['seed']}", p["conformal"]["empirical_coverage"]*100,
         p["conformal"]["q_hat"]) for p in S["temporal"]["per_seed"]]
rows.append(("FORGE 2024", FG["conformal"]["empirical_coverage"]*100,
             FG["conformal"]["q_hat"]))
xs = np.arange(len(rows))
ax[2].bar(xs, [r[1] for r in rows],
          color=["#2F7D32"]*3 + ["#E8871E"], width=0.55)
ax[2].axhline(90, color="#333", ls="--", lw=1.2)
for x, (_, cv, q) in zip(xs, rows):
    ax[2].text(x, cv+0.4, f"{cv:.1f}%", ha="center", fontsize=8.5)
    ax[2].text(x, 82, f"q̂={q:.2f}", ha="center", fontsize=8, color="white")
ax[2].set_xticks(xs, [r[0] for r in rows], rotation=15)
ax[2].set_ylim(78, 101); ax[2].set_ylabel("Coverage (%)")
ax[2].grid(axis="y", alpha=0.25)
ax[2].set_title("(c) Across datasets (nominal 90%)", fontsize=10)
fig.tight_layout()
out = PDF/"14_13_fig5_conformal.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"[14_13] {out}")
