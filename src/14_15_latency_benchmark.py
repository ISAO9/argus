#!/usr/bin/env python3
# =============================================================================
# Script : 14_15_latency_benchmark.py   [レイテンシ実測 — 17ms系の一次生成]
# Project: ARGUS
# Description:
#   原稿のレイテンシ(8.8/3.4/4.5/17.0 ms)は図スクリプトへのハードコードで
#   一次計測が存在しないことが監査で判明したため、本スクリプトで実測する。
#   - GNN-Locator: hinet_graph_v2.pt 内の合成グラフ1件(8局)で推論
#   - SWIFT CMT : (1,12,3,1024) 波形で推論(checkpoint_epoch_080)
#   - FNO(PGA)  : (1,4,128,128) 入力で推論(fno_best.pth, 13_29と同じremap)
#   計測: warmup 20回 → 200回反復、各ステージ個別+直列エンドツーエンド。
#   デバイス: MPS(あれば)と CPU の両方。中央値とp95を記録。
#   出力: models/14_15_latency.json ← Fig 7 とTable 1 レイテンシ行の唯一の出所
# Usage: uv run python src/14_15_latency_benchmark.py
#        uv run python src/14_15_latency_benchmark.py --fno_ckpt ~/FNO-ANN/models/fno_best.pth
# =============================================================================
import sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def need(p):
    if not Path(p).exists(): sys.exit(f"[14_15] STOP: {p} が無い")
    return Path(p)

def bench(fn, dev, n_warm=20, n_rep=200):
    for _ in range(n_warm): fn()
    if dev.type == "mps": torch.mps.synchronize()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        if dev.type == "mps": torch.mps.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    a = np.array(ts)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "n_rep": n_rep}

def load_all(dev, fno_ckpt):
    m19 = _lm("m19", need(ROOT/"src"/"13_19_forge_field_validation.py"))
    mod01 = m19.load_mod01(); mod02 = m19.load_mod02()
    pdata = torch.load(need(ROOT/"data/locator/hinet_graph_v2.pt"),
                       map_location="cpu", weights_only=False)
    # 合成グラフを1件取得(格納形式を内省)
    g = None
    pool = pdata.get("graph_data", pdata)
    for key in ("test", "val", "train", "graphs", "data"):
        v = pool.get(key) if isinstance(pool, dict) else None
        if v is not None and len(v):
            g = v[0]; break
    if g is None and isinstance(pool, dict):
        for v in pool.values():
            if isinstance(v, (list, tuple)) and len(v):
                g = v[0]; break
    if g is None:
        cands = {k: type(v).__name__ for k, v in
                 (pool.items() if isinstance(pool, dict) else [])}
        sys.exit(f"[14_15] STOP: graph_data内にグラフlistが見つからない: {cands}")
    loc = mod02.GNNLocator().to(dev)
    ck = torch.load(need(ROOT/"models/locator/best_locator.pt"),
                    map_location=dev, weights_only=False)
    loc.load_state_dict(ck.get("model_state", ck)); loc.eval()
    x = torch.from_numpy(np.asarray(g["x"], np.float32)).to(dev)
    ei = torch.from_numpy(np.asarray(g["edge_index"])).to(dev)
    ea = torch.from_numpy(np.asarray(g["edge_attr"], np.float32)).to(dev)

    mod05 = _lm("swift05", need(ROOT/"src"/"05_swift_evaluation_3.py"))
    cfg = json.load(open(ROOT/"models"/"swift_architecture_config.json"))
    swift = mod05.load_model(cfg.get("model", cfg),
                             need(ROOT/"models/checkpoint_epoch_080.pt"), device=dev)
    swift.eval()
    w = torch.randn(1, 12, 3, 1024, device=dev)

    ckf = torch.load(need(fno_ckpt), map_location=dev, weights_only=False)
    sd = ckf.get("model_state_dict", ckf)
    def remap(k):
        return (k.replace("fno_blocks.", "bs.")
                 .replace(".spectral_conv.weights1", ".c.w1")
                 .replace(".spectral_conv.weights2", ".c.w2")
                 .replace(".conv.weight", ".r.weight")
                 .replace(".conv.bias", ".r.bias"))
    sd = {remap(k): v for k, v in sd.items()}
    mcfg = ckf.get("model_config", {})
    m17 = _lm("m17", need(ROOT/"src"/"13_17_fno_knet_retrain.py"))
    fno = None
    base = dict(width=mcfg.get("width", 64), n_layers=mcfg.get("n_layers", 4),
                in_channels=mcfg.get("in_channels", 4))
    for kws in ({"modes": mcfg.get("modes", 16), **base},
                {"modes1": mcfg.get("modes", 16),
                 "modes2": mcfg.get("modes", 16), **base}, {}):
        try:
            fno = m17.FNO2d(**kws); break
        except TypeError:
            continue
    if fno is None: sys.exit("[14_15] STOP: FNO2dのコンストラクタ不適合")
    fno = fno.to(dev)
    fno.load_state_dict(sd); fno.eval()
    z = torch.randn(1, 4, 128, 128, device=dev)
    return (loc, x, ei, ea), (swift, w), (fno, z)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--fno_ckpt",
        default=str(Path.home()/"FNO-ANN/models/fno_best.pth"))
    a = pa.parse_args()
    res = {"provenance": {"script": "14_15_latency_benchmark.py",
           "protocol": "warmup 20, 200 reps, median/p95, per-stage + "
                       "sequential end-to-end; inputs: synthetic 8-sta graph, "
                       "(1,12,3,1024) waveform, (1,4,128,128) grid"}}
    devs = [torch.device("cpu")]
    if torch.backends.mps.is_available(): devs.insert(0, torch.device("mps"))
    for dev in devs:
        print(f"=== device: {dev} ===")
        (loc, x, ei, ea), (sw, w), (fn_, z) = load_all(dev, a.fno_ckpt)
        with torch.no_grad():
            r = {}
            r["locator"] = bench(lambda: loc(x, ei, ea), dev)
            r["swift"] = bench(lambda: sw(w), dev)
            r["fno"] = bench(lambda: fn_(z), dev)
            def e2e():
                loc(x, ei, ea); sw(w); fn_(z)
            r["end_to_end"] = bench(e2e, dev)
        for k, v in r.items():
            print(f"  {k:>10}: median={v['median_ms']:.2f} ms  p95={v['p95_ms']:.2f} ms")
        res[str(dev)] = r
    out = ROOT/"models"/"14_15_latency.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[14_15] JSON: {out} ← Fig 7 / Table 1 / タイトル数値の唯一の出所")

if __name__ == "__main__":
    main()
