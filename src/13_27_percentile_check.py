#!/usr/bin/env python3
# =============================================================================
# Script : 13_27_percentile_check.py                                       [E7]
# Project: ARGUS
# Description:
#   R1-Major5 対応: 22.1 km が 90th / 95th どちらのパーセンタイルかを確定。
#   hinet_graph_v2.pt の graph_data 構造(list / dict分割)を自動判定して
#   全体および各分割の P50/P90/P95 を出力する。
# Usage:
#   uv run python src/13_27_percentile_check.py
#   uv run python src/13_27_percentile_check.py --ckpt models/locator/best_locator_finetune.pt
# =============================================================================
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _lm(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def as_splits(gd):
    """graph_data -> {split_name: [graph_dict, ...]} に正規化"""
    if isinstance(gd, list):
        return {"all": gd}
    if isinstance(gd, dict):
        # dict of lists (train/val/test 等)
        if all(isinstance(v, list) for v in gd.values()):
            return {str(k): v for k, v in gd.items()}
        # dict of graphs (event_id -> graph)
        if all(isinstance(v, dict) and "x" in v for v in gd.values()):
            return {"all": list(gd.values())}
    sys.exit(f"[E7] 未知の graph_data 構造: {type(gd)} / "
             f"要素例: {type(next(iter(gd.values())) if isinstance(gd, dict) else gd[0])}")

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--graph_pt", default=str(PROJECT_ROOT/"data/locator/hinet_graph_v2.pt"))
    pa.add_argument("--ckpt", default=str(PROJECT_ROOT/"models/locator/best_locator.pt"))
    a = pa.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    m19 = _lm("m19", PROJECT_ROOT/"src"/"13_19_forge_field_validation.py")
    mod02 = m19.load_mod02()
    pdata = torch.load(a.graph_pt, map_location="cpu", weights_only=False)
    splits = as_splits(pdata["graph_data"])
    ns = pdata["norm_stats"]
    print("[splits]", {k: len(v) for k, v in splits.items()})
    model = mod02.GNNLocator().to(dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck.get("model_state", ck))
    print(f"[ckpt] {Path(a.ckpt).name}")
    t = 22.1
    all_err = []
    for name, graphs in splits.items():
        dh = m19.evaluate(model, graphs, ns, dev)["horiz"]
        all_err.append(dh)
        p50, p90, p95 = (float(np.percentile(dh, q)) for q in (50, 90, 95))
        print(f"[{name:>6} n={len(dh):>3}] P50={p50:6.2f}  P90={p90:6.2f}  "
              f"P95={p95:6.2f} km")
    dh = np.concatenate(all_err)
    p50, p90, p95 = (float(np.percentile(dh, q)) for q in (50, 90, 95))
    print(f"[   ALL n={len(dh):>3}] P50={p50:6.2f}  P90={p90:6.2f}  P95={p95:6.2f} km")
    which = "90th" if abs(p90-t) <= abs(p95-t) else "95th"
    print(f"[verdict] 22.1 km は {which} percentile に最も近い "
          f"(|P90-22.1|={abs(p90-t):.2f} / |P95-22.1|={abs(p95-t):.2f})")
    print("[note   ] 論文の random split n=138 に対応する分割の行を採用すること。")
    print("[action ] 90th なら改訂原稿のまま。95th なら Table 1/本文を95thへ。")

if __name__ == "__main__":
    main()
