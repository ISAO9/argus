#!/usr/bin/env python3
# =============================================================================
# scripts/07_conformal_calibration.py
#
# What this script does:
#   Computes the split-conformal location radius from a 138-event calibration
#   set (manuscript) and reports empirical coverage on the test set. Saves
#   (q_hat, sigma_hat) to models/conformal.json for use at inference.
#
# REUSED ASSETS (cite): best_locator.pt, processed calibration/test graphs.
# =============================================================================
import importlib, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
common = importlib.import_module("00_common")
import numpy as np, torch
from torch_geometric.loader import DataLoader
from argus.models import build_gnn_locator
from argus.conformal import fit_conformal, empirical_coverage


def predict(model, data, dev):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for b in DataLoader(data, batch_size=64):
            b = b.to(dev)
            P.append(model(b.x, b.edge_index, b.edge_attr, b.batch).cpu().numpy())
            Y.append(b.y.view(-1, 3).cpu().numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    cfg = common.load_cfg(); dev = common.get_device(cfg["device"])
    proc = Path(cfg["paths"]["data_processed"])
    model = build_gnn_locator(cfg).to(dev)
    model.load_state_dict(torch.load(cfg["paths"]["locator_ckpt"], map_location=dev))

    cal = torch.load(proc / "calib.pt"); tst = torch.load(proc / "kumamoto_test.pt")
    Pc, Yc = predict(model, cal, dev); Pt, Yt = predict(model, tst, dev)
    q_hat, sigma_hat = fit_conformal(Pc, Yc, alpha=1 - cfg["conformal"]["nominal_level"])
    cov = empirical_coverage(Pt, Yt, q_hat, sigma_hat)
    print(f"q_hat={q_hat:.3f}  sigma_hat={sigma_hat:.3f} km  "
          f"radius={q_hat*sigma_hat:.2f} km  coverage={cov*100:.1f}%")
    json.dump({"q_hat": q_hat, "sigma_hat": sigma_hat, "coverage": cov},
              open(Path(cfg["paths"]["models"]) / "conformal.json", "w"), indent=2)


if __name__ == "__main__":
    main()
