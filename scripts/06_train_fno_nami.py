#!/usr/bin/env python3
# =============================================================================
# scripts/06_train_fno_nami.py
#
# What this script does:
#   Trains FNO-NAMI (FNO2d) to map a 4-channel source-conditioning grid to a
#   128x128 log-PGA field, then recovers PGA in gal via Eq. 5. Saves best model
#   (lowest val MSE on z-scored log-PGA) to models/fno_best.pth.
#
# REUSED ASSETS (cite): consumes knet_dataset.h5 for validation pairs.
# =============================================================================
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
common = importlib.import_module("00_common")
import torch
from torch.utils.data import DataLoader, TensorDataset
from argus.models import build_fno_nami


def load_pairs(cfg, split):
    """Return TensorDataset of (cond_grid [4,128,128], target_logpga_z [128,128]).

    Wire to your processed arrays built from knet_dataset.h5 / synthetic GMM
    fields. Expected file: data/processed/fno_<split>.pt -> dict(x=..., y=...).
    """
    d = torch.load(Path(cfg["paths"]["data_processed"]) / f"fno_{split}.pt")
    return TensorDataset(d["x"], d["y"])


def main():
    cfg = common.load_cfg(); common.set_seed(cfg["seed"])
    dev = common.get_device(cfg["device"]); c = cfg["fno_nami"]
    tr = DataLoader(load_pairs(cfg, "train"), batch_size=c["batch_size"], shuffle=True)
    va = DataLoader(load_pairs(cfg, "val"), batch_size=c["batch_size"])

    model = build_fno_nami(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    ckpt = common.BestCheckpoint(cfg["paths"]["fno_ckpt"], mode="min")

    for epoch in range(1, c["epochs"] + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward(); opt.step()
        model.eval(); vs = 0.0
        with torch.no_grad():
            for x, y in va:
                vs += torch.nn.functional.mse_loss(model(x.to(dev)), y.to(dev)).item()
        vs /= max(len(va), 1)
        print(f"epoch {epoch:3d} | val_mse {vs:.4f} {'<- best' if ckpt.update(vs, model) else ''}")
    print(f"Best val MSE {ckpt.best:.4f} -> {ckpt.path}")


if __name__ == "__main__":
    main()
