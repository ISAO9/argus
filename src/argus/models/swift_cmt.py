# =============================================================================
# src/argus/models/swift_cmt.py
#
# What this module does:
#   Defines SWIFT CMT (SWIFTNetV8), the source-mechanism network of ARGUS. It
#   combines (a) a small per-station 1-D waveform encoder, (b) a spectral branch
#   that summarizes the amplitude spectrum in a fixed number of Fourier bins, and
#   (c) a GATv2 graph branch over the station network. The fused graph embedding
#   is decoded by three heads:
#       - mechanism: 3-class (Shear / Tensile / Mixed) cross-entropy
#       - f_ISO    : scalar isotropic fraction (regression)
#       - Mw       : scalar magnitude (regression)
#
#   Composite loss (manuscript Eq. 3):
#       L = CE(mechanism) + 0.5*MSE(f_ISO) + 0.1*MSE(Mw)
#
#   Architecture matches the manuscript / Table S2:
#     waveform base channels=16, Fourier bins=32, GATv2 layers=3, hidden=128,
#     dropout=0.2.
# =============================================================================
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except Exception as e:  # pragma: no cover
    raise ImportError("torch_geometric required for SWIFT CMT.") from e


class WaveformEncoder(nn.Module):
    """3-component waveform -> compact per-station embedding."""
    def __init__(self, base_channels: int = 16, out_dim: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv1d(3, c, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(c, 2 * c, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(2 * c, 4 * c, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(4 * c, out_dim)

    def forward(self, w):                      # w: (N_sta, 3, T)
        z = self.net(w).squeeze(-1)
        return self.proj(z)                    # (N_sta, out_dim)


class SpectralBranch(nn.Module):
    """Fixed-bin amplitude-spectrum summary per station."""
    def __init__(self, n_bins: int = 32, out_dim: int = 64):
        super().__init__()
        self.n_bins = n_bins
        self.mlp = nn.Sequential(
            nn.Linear(n_bins, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, w):                      # w: (N_sta, 3, T)
        spec = torch.fft.rfft(w, dim=-1).abs().mean(dim=1)   # (N_sta, F)
        # adaptive-bin the spectrum to n_bins
        spec = F.adaptive_avg_pool1d(spec.unsqueeze(1), self.n_bins).squeeze(1)
        spec = torch.log1p(spec)
        return self.mlp(spec)


class SWIFTNetV8(nn.Module):
    def __init__(
        self,
        waveform_base_channels: int = 16,
        fourier_bins: int = 32,
        gatv2_layers: int = 3,
        hidden_dim: int = 128,
        edge_feat_dim: int = 4,
        dropout: float = 0.2,
        n_mechanism_classes: int = 3,
    ):
        super().__init__()
        self.wave = WaveformEncoder(waveform_base_channels, hidden_dim // 2)
        self.spec = SpectralBranch(fourier_bins, hidden_dim // 2)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(gatv2_layers):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False,
                          edge_dim=edge_feat_dim, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.drop = nn.Dropout(dropout)
        self.head_mech = nn.Linear(hidden_dim, n_mechanism_classes)
        self.head_fiso = nn.Linear(hidden_dim, 1)
        self.head_mw = nn.Linear(hidden_dim, 1)

    def forward(self, waveforms, edge_index, edge_attr, batch):
        node = torch.cat([self.wave(waveforms), self.spec(waveforms)], dim=-1)
        h = self.node_proj(node)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.gelu(conv(h, edge_index, edge_attr)) + h)
            h = self.drop(h)
        g = global_mean_pool(h, batch)
        return {
            "mechanism": self.head_mech(g),
            "f_iso": self.head_fiso(g).squeeze(-1),
            "mw": self.head_mw(g).squeeze(-1),
        }


def swift_loss(out, target, weights):
    L = F.cross_entropy(out["mechanism"], target["mechanism"]) * weights["mechanism_ce"]
    L = L + F.mse_loss(out["f_iso"], target["f_iso"]) * weights["fiso_mse"]
    L = L + F.mse_loss(out["mw"], target["mw"]) * weights["mw_mse"]
    return L


def build_swift_cmt(cfg: dict) -> SWIFTNetV8:
    c = cfg["swift_cmt"]
    return SWIFTNetV8(
        waveform_base_channels=c["waveform_base_channels"],
        fourier_bins=c["fourier_bins"], gatv2_layers=c["gatv2_layers"],
        hidden_dim=c["hidden_dim"], dropout=c["dropout"],
        n_mechanism_classes=c["n_mechanism_classes"],
    )
