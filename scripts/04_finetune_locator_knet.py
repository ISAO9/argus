#!/usr/bin/env python3
# =============================================================================
# scripts/04_finetune_locator_knet.py
#
# What this script does:
#   Fine-tunes the synthetic-trained GNN-Locator on the real K-NET/Hi-net
#   Kumamoto graphs for 30 epochs at LR 5e-5 (manuscript Table S2), improving the
#   temporal-split median error. Loads models/best_locator.pt, saves the best
#   fine-tuned model back to the same path.
#
# REUSED ASSETS (cite): best_locator.pt (from 03), hinet_graph_v2.pt (from 02).
# =============================================================================
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
common = importlib.import_module("00_common")
import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from argus.models import build_gnn_locator


def main():
    cfg = common.load_cfg(); common.set_seed(cfg["seed"])
    dev = common.get_device(cfg["device"]); c = cfg["gnn_locator"]
    real = torch.load(Path(cfg["paths"]["data_processed"]) / "kumamoto_train.pt")
    loader = DataLoader(real, batch_size=c["batch_size"], shuffle=True)

    model = build_gnn_locator(cfg).to(dev)
    model.load_state_dict(torch.load(cfg["paths"]["locator_ckpt"], map_location=dev))
    opt = AdamW(model.parameters(), lr=c["finetune_lr"], weight_decay=c["weight_decay"])
    ckpt = common.BestCheckpoint(cfg["paths"]["locator_ckpt"], mode="min")

    for epoch in range(1, c["finetune_epochs"] + 1):
        model.train(); tot = 0.0
        for b in loader:
            b = b.to(dev); opt.zero_grad()
            pred = model(b.x, b.edge_index, b.edge_attr, b.batch)
            loss = torch.nn.functional.smooth_l1_loss(pred, b.y.view(pred.shape))
            loss.backward(); opt.step(); tot += loss.item()
        ckpt.update(tot / len(loader), model)
        print(f"ft-epoch {epoch:2d} | loss {tot/len(loader):.4f}")
    print(f"Fine-tuned best -> {ckpt.path}")


if __name__ == "__main__":
    main()
