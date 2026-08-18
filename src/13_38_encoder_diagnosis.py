#!/usr/bin/env python3
# =============================================================================
# Script : 13_38_encoder_diagnosis.py
# Project: ARGUS
# Description:
#   ゼロショット誤差 30 vs 90 km の乖離原因を切り分ける診断。同一の
#   熊本688イベント(先頭200件)に対し、best_locator.pt で以下3構成を評価:
#     A: 学習済みencoder_state + クランプなし   (13_34の構成)
#     B: 学習済みencoder_state + x∈[-5,5]      (13_12のクランプを追加)
#     C: 乱数エンコーダ(seed=0固定) + x∈[-5,5] (13_12の実挙動の再現)
#   出力: 各構成の median / P90。models/locator/13_38_diagnosis.json
# Usage: uv run python src/13_38_encoder_diagnosis.py
# =============================================================================
import sys, json, importlib.util
from pathlib import Path
import numpy as np, torch
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT/"models"/"locator"

def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m19 = _lm("m19", PROJECT_ROOT/"src"/"13_19_forge_field_validation.py")
    mod01 = m19.load_mod01(); mod02 = m19.load_mod02()
    evs = torch.load(PROJECT_ROOT/"data/knet_processed/knet_parsed.pt",
                     map_location="cpu", weights_only=False)[:200]
    pdata = torch.load(PROJECT_ROOT/"data/locator/hinet_graph_v2.pt",
                       map_location="cpu", weights_only=False)
    ns = pdata["norm_stats"]
    model = mod02.GNNLocator().to(dev)
    ck = torch.load(MODEL_DIR/"best_locator.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck.get("model_state", ck)); model.eval()

    def build_graphs(enc, clamp5):
        gs = []
        with torch.no_grad():
            for ev in evs:
                w = np.asarray(ev["waveforms"], np.float32)
                wt = torch.nan_to_num(torch.from_numpy(w).to(dev)).clamp(-10, 10)
                emb = np.nan_to_num(enc(wt).cpu().numpy())
                sta = np.asarray(ev["sta_locs"], np.float32)
                ei, ea = mod01.build_edges_knn(sta, k=4)
                sl = np.asarray(ev["src_loc"], np.float32)
                g = mod01.build_graph_for_event(
                    waveforms=w, sta_locs=sta,
                    tp_times=np.asarray(ev["tp_times"], np.float32),
                    sp_diffs=np.asarray(ev["sp_diffs"], np.float32),
                    snr=np.asarray(ev["snr"], np.float32),
                    src_loc=sl, mw=float(ev["mw"]),
                    edge_index=ei, edge_attr_geo=ea,
                    norm_stats=ns, wave_emb=emb)
                g["x"] = np.nan_to_num(g["x"], nan=0.0, posinf=0.0, neginf=0.0)
                if clamp5:
                    g["x"] = np.clip(g["x"], -5, 5)
                g["edge_attr"] = np.nan_to_num(g["edge_attr"])
                gs.append(g)
        return gs

    T = np.asarray(evs[0]["waveforms"]).shape[-1]
    enc_tr = mod01.WaveformEncoder(3, int(pdata["wave_emb_dim"]), T).to(dev)
    enc_tr.load_state_dict(pdata["encoder_state"]); enc_tr.eval()
    torch.manual_seed(0)
    enc_rnd = mod01.WaveformEncoder(3, int(pdata["wave_emb_dim"]), T).to(dev)
    enc_rnd.eval()

    out = {}
    for name, enc, clamp5 in (("A_trained_noclamp", enc_tr, False),
                              ("B_trained_clamp5", enc_tr, True),
                              ("C_random_clamp5", enc_rnd, True)):
        gs = build_graphs(enc, clamp5)
        dh = m19.evaluate(model, gs, ns, dev)["horiz"]
        out[name] = {"median_km": float(np.median(dh)),
                     "p90_km": float(np.percentile(dh, 90)), "n": len(dh)}
        print(f"[{name:>18}] median={out[name]['median_km']:6.2f} km  "
              f"P90={out[name]['p90_km']:6.2f}")
    json.dump(out, open(MODEL_DIR/"13_38_diagnosis.json", "w"), indent=2)
    print("[13_38] JSON:", MODEL_DIR/"13_38_diagnosis.json")

if __name__ == "__main__":
    main()
