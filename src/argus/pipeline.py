# =============================================================================
# src/argus/pipeline.py
#
# What this module does:
#   Orchestrates the three ARGUS components into a single low-latency inference
#   pass: waveforms -> GNN-Locator (location + conformal interval) -> SWIFT CMT
#   (mechanism / f_ISO / Mw) -> FNO-NAMI (128x128 PGA map). Models are loaded
#   once at construction (REUSED checkpoints: best_locator.pt,
#   checkpoint_epoch_080.pt, fno_best.pth) and held resident, matching the
#   manuscript's "persistent service" deployment assumption.
# =============================================================================
from __future__ import annotations
import time
import torch

from .models.gnn_locator import build_gnn_locator
from .models.swift_cmt import build_swift_cmt
from .models.fno_nami import build_fno_nami


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ARGUS:
    def __init__(self, cfg: dict, conformal=None, device: str = "auto"):
        self.cfg = cfg
        self.device = resolve_device(cfg.get("device", device))
        self.locator = build_gnn_locator(cfg).to(self.device).eval()
        self.swift = build_swift_cmt(cfg).to(self.device).eval()
        self.fno = build_fno_nami(cfg).to(self.device).eval()
        self.conformal = conformal  # (q_hat, sigma_hat) or None

    def load_weights(self):
        p = self.cfg["paths"]
        self.locator.load_state_dict(torch.load(p["locator_ckpt"], map_location=self.device))
        self.swift.load_state_dict(torch.load(p["swift_ckpt"], map_location=self.device))
        self.fno.load_state_dict(torch.load(p["fno_ckpt"], map_location=self.device))
        return self

    @torch.no_grad()
    def __call__(self, graph, waveforms, cond_grid, profile: bool = False):
        """Run one event. Returns a dict of products (+ per-stage latency)."""
        g = graph.to(self.device)
        wf = waveforms.to(self.device)
        cg = cond_grid.to(self.device)
        batch = torch.zeros(g.num_nodes, dtype=torch.long, device=self.device)

        t0 = time.perf_counter()
        loc = self.locator(g.x, g.edge_index, g.edge_attr, batch)
        t1 = time.perf_counter()
        cmt = self.swift(wf, g.edge_index, g.edge_attr, batch)
        t2 = time.perf_counter()
        pga = self.fno.predict_pga_gal(cg)
        t3 = time.perf_counter()

        out = {
            "location": loc.squeeze(0).cpu().numpy(),
            "mechanism_logits": cmt["mechanism"].cpu().numpy(),
            "f_iso": float(cmt["f_iso"]),
            "mw": float(cmt["mw"]),
            "pga_map_gal": pga.squeeze(0).cpu().numpy(),
        }
        if self.conformal is not None:
            q_hat, sigma_hat = self.conformal
            out["location_radius_km"] = q_hat * sigma_hat
        if profile:
            out["latency_ms"] = {
                "gnn_locator": (t1 - t0) * 1e3,
                "swift_cmt": (t2 - t1) * 1e3,
                "fno_nami": (t3 - t2) * 1e3,
                "total": (t3 - t0) * 1e3,
            }
        return out
