#!/usr/bin/env python3
# =============================================================================
# scripts/00_common.py
#
# What this script does:
#   Shared utilities imported by all numbered training/eval scripts:
#   config loading, reproducible seeding, device resolution, and a
#   BestCheckpoint helper that persists ONLY the best model seen so far during
#   an epoch loop (project convention: always keep the single best weights).
# =============================================================================
from __future__ import annotations
import os, sys, random, math
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_cfg(path: str | None = None) -> dict:
    path = path or (ROOT / "configs" / "argus.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 1234):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(name: str = "auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BestCheckpoint:
    """Keep only the best model. mode='min' for loss/error, 'max' for accuracy."""
    def __init__(self, path, mode: str = "min"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf

    def update(self, metric: float, model) -> bool:
        import torch
        improved = (metric < self.best) if self.mode == "min" else (metric > self.best)
        if improved:
            self.best = metric
            torch.save(model.state_dict(), self.path)
        return improved
