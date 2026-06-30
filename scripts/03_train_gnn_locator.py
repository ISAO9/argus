#!/usr/bin/env python3
# =============================================================================
# scripts/03_train_gnn_locator.py
#
# What this script does:
#   Trains GNN-Locator on the synthetic FORGE graph dataset (5000 events,
#   70/15/15 split) to regress normalized (lat, lon, depth). Uses AdamW with
#   cosine-decayed LR for 80 epochs and saves ONLY the best validation model to
#   models/best_locator.pt (project convention).
#
# REUSED ASSETS (cite): consumes the precomputed graph tensors in
#   data/processed/ (built by 02_build_graphs.py); the canonical Hi-net graph is
#   hinet_graph_v2.pt. Output checkpoint best_locator.pt is consumed downstream
#   by 04_finetune_locator_knet.py and 08_run_pipeline.py.
#
# Usage:
#   python scripts/03_train_gnn_locator.py
# =============================================================================
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
common = importlib.import_module("00_common")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from argus.models import build_gnn_locator


def load_split(cfg, split):
    """Load a list[Data] for split in {train,val,test}.

    Expects data/processed/forge_<split>.pt produced by 02_build_graphs.py.
    Each Data carries .y = normalized (lat, lon, depth) target.
    """
    p = Path(cfg["paths"]["data_processed"]) / f"forge_{split}.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run scripts/02_build_graphs.py first.")
    return torch.load(p)


def evaluate(model, loader, device):
    model.eval(); errs = []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            pred = model(b.x, b.edge_index, b.edge_attr, b.batch)
            errs.append(torch.norm(pred - b.y.view(pred.shape), dim=1))
    return torch.cat(errs).median().item()


def main():
    cfg = common.load_cfg(); common.set_seed(cfg["seed"])
    dev = common.get_device(cfg["device"]); c = cfg["gnn_locator"]

    tr = DataLoader(load_split(cfg, "train"), batch_size=c["batch_size"], shuffle=True)
    va = DataLoader(load_split(cfg, "val"), batch_size=c["batch_size"])

    model = build_gnn_locator(cfg).to(dev)
    opt = AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=c["epochs"])
    ckpt = common.BestCheckpoint(cfg["paths"]["locator_ckpt"], mode="min")

    for epoch in range(1, c["epochs"] + 1):
        model.train(); tot = 0.0
        for b in tr:
            b = b.to(dev); opt.zero_grad()
            pred = model(b.x, b.edge_index, b.edge_attr, b.batch)
            loss = torch.nn.functional.smooth_l1_loss(pred, b.y.view(pred.shape))
            loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        val_med = evaluate(model, va, dev)
        best = ckpt.update(val_med, model)
        print(f"epoch {epoch:3d} | train_loss {tot/len(tr):.4f} | "
              f"val_median {val_med:.4f} {'<- best' if best else ''}")
    print(f"Best validation median error: {ckpt.best:.4f}  -> {ckpt.path}")


if __name__ == "__main__":
    main()
