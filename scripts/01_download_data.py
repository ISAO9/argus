#!/usr/bin/env python3
# =============================================================================
# scripts/01_download_data.py
#
# What this script does:
#   Prints (and optionally verifies) the public sources for every dataset used
#   by ARGUS. All data are public-domain; this script does NOT scrape restricted
#   endpoints. It checks which expected files are already present locally and
#   tells you exactly what to fetch and where to put it.
#
# Public sources (see also data/README.md and the manuscript Data and Resources):
#   - K-NET / KiK-net strong motion : https://www.kyoshin.bosai.go.jp/
#   - Hi-net high-sensitivity       : https://www.hinet.bosai.go.jp/
#   - JMA unified hypocenter catalog: https://www.data.jma.go.jp/svd/eqev/data/bulletin/
#   - Utah FORGE Phase 2C + vel.    : https://gdr.openei.org/  (DOE GDR Sub. 1107)
#   - J-SHIS subsurface velocity    : https://www.j-shis.bosai.go.jp/
# =============================================================================
import importlib.util, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
common = importlib.import_module("00_common") if False else None
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "data/raw/knet/":            "K-NET strong-motion records (Kumamoto 2016 + 5 broadband events)",
    "data/raw/hinet/":           "Hi-net velocity records (688 Kumamoto aftershocks)",
    "data/raw/jma_catalog.txt":  "JMA unified hypocenter catalog (training labels)",
    "data/raw/forge/":           "Utah FORGE Phase 2C microseismic data + 3D velocity model",
    "data/processed/hinet_graph_v2.pt": "[reused] precomputed station-graph tensor",
    "data/processed/knet_dataset.h5":   "[reused] FNO-NAMI validation database",
}


def main():
    print("ARGUS data inventory")
    print("=" * 60)
    missing = []
    for rel, desc in EXPECTED.items():
        p = ROOT / rel
        ok = p.exists()
        print(f"[{'OK ' if ok else '   '}] {rel:38s} {desc}")
        if not ok:
            missing.append(rel)
    print("=" * 60)
    if missing:
        print(f"{len(missing)} item(s) missing. Fetch from the public sources listed")
        print("in the header / data/README.md, then re-run 02_build_graphs.py.")
    else:
        print("All datasets present.")


if __name__ == "__main__":
    main()
