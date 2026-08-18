#!/usr/bin/env python3
# =============================================================================
# Script : 13_37_seed_aggregate.py   (v3 — split判定修正版)
# Project: ARGUS
# Description:
#   13_34 の全seed結果を split 別に集計する。v3修正点:
#     - split は provenance から読む(v3のJSONはトップレベルに無いため)
#     - seedなし旧ファイル名(13_34_random_results.json)もglobで拾う
#   出力: models/locator/13_37_seed_summary.json
# Usage: uv run python src/13_37_seed_aggregate.py
# =============================================================================
import json, glob
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT/"models"/"locator"

def collect():
    rows, seen = [], set()
    for f in sorted(glob.glob(str(MODEL_DIR/"13_34_*results*.json"))):
        if "summary" in f: continue
        d = json.load(open(f))
        prov = d.get("provenance", {})
        split = d.get("split") or prov.get("split") or "random"
        seed = prov.get("seed", -1)
        key = (split, seed)
        if key in seen: continue
        seen.add(key)
        rows.append(dict(split=split, seed=seed, n=d["n_test"],
                         median=d["test_median_km"], p90=d["test_p90_km"],
                         p95=d["test_p95_km"],
                         zs=d["zero_shot_test_median_km"],
                         conformal=d.get("conformal"),
                         strat=d.get("mw_stratified_test_only"),
                         file=Path(f).name))
    return rows

def main():
    rows = collect()
    summary = {}
    for split in ("random", "temporal"):
        rs = sorted([r for r in rows if r["split"] == split],
                    key=lambda r: r["seed"])
        if not rs: continue
        print(f"\n=== {split} ({len(rs)} seeds) ===")
        print(f"{'seed':>6} {'n':>4} {'median':>8} {'P90':>8} {'P95':>8} "
              f"{'zero-shot':>10}  file")
        for r in rs:
            print(f"{r['seed']:>6} {r['n']:>4} {r['median']:>8.2f} "
                  f"{r['p90']:>8.2f} {r['p95']:>8.2f} {r['zs']:>10.2f}  "
                  f"{r['file']}")
        med = [r["median"] for r in rs]; p90 = [r["p90"] for r in rs]
        zs = [r["zs"] for r in rs]
        sec = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs],
               "per_seed": rs,
               "median_km": {"median": float(np.median(med)),
                             "min": float(min(med)), "max": float(max(med))},
               "p90_km": {"median": float(np.median(p90)),
                          "min": float(min(p90)), "max": float(max(p90))},
               "zero_shot_median_km": {"median": float(np.median(zs)),
                                       "min": float(min(zs)),
                                       "max": float(max(zs))}}
        if split == "temporal" and rs[0]["conformal"]:
            qs = [r["conformal"]["q_hat"] for r in rs]
            cv = [r["conformal"]["empirical_coverage"] for r in rs]
            sec["conformal"] = {"n_cal": rs[0]["conformal"]["n_cal"],
                "q_hat_median": float(np.median(qs)),
                "q_hat_range": [float(min(qs)), float(max(qs))],
                "coverage_median": float(np.median(cv)),
                "coverage_range": [float(min(cv)), float(max(cv))]}
            print(f"  conformal: q_hat {np.median(qs):.2f} "
                  f"({min(qs):.2f}-{max(qs):.2f})  coverage "
                  f"{np.median(cv)*100:.1f}% ({min(cv)*100:.1f}-{max(cv)*100:.1f}%)"
                  f"  n_cal={sec['conformal']['n_cal']}")
            bands = {}
            for r in rs:
                for b in (r["strat"] or []):
                    bands.setdefault(b["mw"], []).append(b)
            sec["mw_stratified_median_across_seeds"] = [
                {"mw": k, "n": bs[0]["n"],
                 "median_km": float(np.median([b["median_km"] for b in bs])),
                 "p90_km": float(np.median([b["p90_km"] for b in bs]))}
                for k, bs in bands.items()]
            print("  Mw stratified (median across seeds):")
            for b in sec["mw_stratified_median_across_seeds"]:
                print(f"    Mw {b['mw']:>8}  n={b['n']:>3}  "
                      f"median={b['median_km']:6.2f}  P90={b['p90_km']:6.2f}")
        summary[split] = sec
        print(f"  -> {split}: median {sec['median_km']['median']:.2f} km "
              f"(range {sec['median_km']['min']:.2f}-{sec['median_km']['max']:.2f})"
              f", P90 {sec['p90_km']['median']:.2f} "
              f"({sec['p90_km']['min']:.2f}-{sec['p90_km']['max']:.2f}), "
              f"zero-shot {sec['zero_shot_median_km']['median']:.2f}")
    out = MODEL_DIR/"13_37_seed_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n[13_37] JSON: {out}")

if __name__ == "__main__":
    main()
