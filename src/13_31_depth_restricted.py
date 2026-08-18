#!/usr/bin/env python3
# =============================================================================
# Script : 13_31_depth_restricted.py                                      [E6]
# Project: ARGUS
# Description:
#   R1-Major6 対応: 合成テストのうち深さ 2-4 km(および 2.5-3.5 km)に制限した
#   場合の測位精度を全深さと比較し、Data節のシアン【E6】用の1数値を出す。
#   入力: data/locator/synthetic_locator_dataset.h5(13_00 v2 の出力)。
#   スキーマは introspect して表示。想定: グラフ化済み .pt があればそれを優先
#   (--graphs_pt)。h5 のみの場合は depth 配列とイベント誤差の対応付けに
#   13_03 実行時の合成評価出力(per-event errors)が必要 → 無ければ手順を表示。
# Usage:
#   uv run python src/13_31_depth_restricted.py --graphs_pt data/locator/synthetic_test_graphs.pt
# =============================================================================
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--graphs_pt", default=None,
                    help="合成テストのグラフ .pt (13_01/13_00 形式の dict/list)")
    pa.add_argument("--ckpt", default=str(PROJECT_ROOT/"models/locator/best_locator.pt"))
    a = pa.parse_args()
    m19 = _lm("m19", PROJECT_ROOT/"src"/"13_19_forge_field_validation.py")
    mod02 = m19.load_mod02()
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    if not a.graphs_pt:
        h5p = PROJECT_ROOT/"data/locator/synthetic_locator_dataset.h5"
        print("[E6] --graphs_pt 未指定。h5 を introspect します:", h5p)
        import h5py
        with h5py.File(h5p) as f:
            f.visit(lambda n: print("   ", n))
        print("[E6] 合成テスト分割をグラフ化した .pt のパスを --graphs_pt で指定"
              "してください(13_03 の合成評価で使ったもの)。")
        sys.exit(0)
    pdata = torch.load(a.graphs_pt, map_location="cpu", weights_only=False)
    if isinstance(pdata, dict):
        gkey = next(k for k in ("graph_data", "graphs", "test", "data") if k in pdata)
        graphs, ns = pdata[gkey], pdata["norm_stats"]
    else:
        sys.exit("[E6] dict 形式(norm_stats 同梱)が必要です。")
    model = mod02.GNNLocator().to(dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck.get("model_state", ck))
    R = m19.evaluate(model, graphs, ns, dev)
    depth = np.array([float(g["src_loc"][2]) for g in graphs])
    dh = R["horiz"]
    full = float(np.median(dh))
    print(f"[E6] full depth 1-8 km: n={len(dh)}  H median={full:.3f} km")
    for lo, hi in ((2.0, 4.0), (2.5, 3.5)):
        m = (depth >= lo) & (depth <= hi)
        if m.sum() < 10:
            print(f"[E6] {lo}-{hi} km: n={m.sum()} (<10 skip)"); continue
        med = float(np.median(dh[m]))
        print(f"[E6] {lo}-{hi} km: n={m.sum()}  H median={med:.3f} km  "
              f"delta={med-full:+.3f} km ({(med/full-1)*100:+.1f}%)")
    print("[E6] -> 2-4 km の行を Data 節シアン【E6】へ。")

if __name__ == "__main__":
    main()
