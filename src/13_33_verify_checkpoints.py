#!/usr/bin/env python3
# =============================================================================
# Script : 13_33_verify_checkpoints.py
# Project: ARGUS
# Description:
#   Zenodo/ローカルの学習済み重みが、リポジトリ収載の正準モデル定義
#   (src/13_02_gnn_locator_model.py) で完全ロードできることを検証する。
#   SRL改訂で「コードと重みで結果を再現できる」と主張する前の必須チェック。
#   missing/unexpected キーが1つでもあれば FAIL。
# Usage:
#   uv run python src/13_33_verify_checkpoints.py \
#       --ckpt models/locator/best_locator.pt
# =============================================================================
import sys, argparse, importlib.util
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", default=str(PROJECT_ROOT/"models/locator/best_locator.pt"))
    a = pa.parse_args()
    spec = importlib.util.spec_from_file_location(
        "gnn02", PROJECT_ROOT/"src"/"13_02_gnn_locator_model.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    model = mod.GNNLocator()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck)
    r = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {a.ckpt}")
    print(f"  params in file : {len(sd)}")
    print(f"  missing keys   : {len(r.missing_keys)}  {r.missing_keys[:4]}")
    print(f"  unexpected keys: {len(r.unexpected_keys)}  {r.unexpected_keys[:4]}")
    ok = not r.missing_keys and not r.unexpected_keys
    print("  =>", "PASS — 重みとモデル定義は完全互換" if ok else
          "FAIL — 定義と重みが不一致。収載する13_02の版を確認")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
