#!/usr/bin/env python3
# =============================================================================
# scripts/05_train_swift_cmt.py
#
# What this script does:
#   Trains SWIFT CMT (SWIFTNetV8) on synthetic FORGE events to classify fracture
#   mechanism (3-class) and regress f_ISO and Mw, using the composite loss
#   CE + 0.5*MSE(f_ISO) + 0.1*MSE(Mw) (manuscript Eq. 3). Saves the best model to
#   models/checkpoint_epoch_080.pt.
#
# REUSED ASSETS (cite): consumes processed waveform graphs from 02; output
#   checkpoint is consumed by 08_run_pipeline.py.
# =============================================================================
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
common = importlib.import_module("00_common")
import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from argus.models import build_swift_cmt, swift_loss


def main():
    cfg = common.load_cfg(); common.set_seed(cfg["seed"])
    dev = common.get_device(cfg["device"]); c = cfg["swift_cmt"]
    proc = Path(cfg["paths"]["data_processed"])
    tr = DataLoader(torch.load(proc / "swift_train.pt"), batch_size=c["batch_size"], shuffle=True)
    va = DataLoader(torch.load(proc / "swift_val.pt"), batch_size=c["batch_size"])

    model = build_swift_cmt(cfg).to(dev)
    opt = AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    ckpt = common.BestCheckpoint(cfg["paths"]["swift_ckpt"], mode="max")

    def accuracy(loader):
        model.eval(); ok = n = 0
        with torch.no_grad():
            for b in loader:
                b = b.to(dev)
                out = model(b.waveforms, b.edge_index, b.edge_attr, b.batch)
                ok += (out["mechanism"].argmax(1) == b.mechanism).sum().item()
                n += b.mechanism.numel()
        return ok / max(n, 1)

    for epoch in range(1, c["epochs"] + 1):
        model.train()
        for b in tr:
            b = b.to(dev); opt.zero_grad()
            out = model(b.waveforms, b.edge_index, b.edge_attr, b.batch)
            tgt = {"mechanism": b.mechanism, "f_iso": b.f_iso, "mw": b.mw}
            loss = swift_loss(out, tgt, c["loss_weights"])
            loss.backward(); opt.step()
        acc = accuracy(va)
        print(f"epoch {epoch:3d} | val_acc {acc*100:.2f}% {'<- best' if ckpt.update(acc, model) else ''}")
    print(f"Best val accuracy {ckpt.best*100:.2f}% -> {ckpt.path}")


if __name__ == "__main__":
    main()
