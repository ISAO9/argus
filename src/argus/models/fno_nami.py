# =============================================================================
# src/argus/models/fno_nami.py
#
# What this module does:
#   Defines FNO-NAMI, the ground-motion network of ARGUS. It is a 2-D Fourier
#   Neural Operator (Li et al., 2021) that maps a 4-channel conditioning grid
#   (source location/magnitude/mechanism rasterized onto a 128x128 domain) to a
#   128x128 log-PGA field. The final PGA in gal is recovered with the manuscript
#   normalization (Eq. 5):  PGA[gal] = exp(z * sigma + mu) * 100.
#
#   Architecture matches the manuscript / Table S2:
#     lift channels=64, Fourier modes=16, 4 FNO layers, 128x128 output.
# =============================================================================
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2-D spectral convolution keeping the lowest `modes` Fourier modes."""
    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, modes, dtype=torch.cfloat))

    def _mul(self, a, b):
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m = self.modes
        out_ft[:, :, :m, :m] = self._mul(x_ft[:, :, :m, :m], self.w1)
        out_ft[:, :, -m:, :m] = self._mul(x_ft[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, in_channels=4, lift=64, modes=16, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, lift, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(lift, lift, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(lift, lift, 1) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv2d(lift, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):                      # x: (B, in_ch, 128, 128)
        h = self.lift(x)
        for sp, pw in zip(self.spectral, self.pointwise):
            h = F.gelu(sp(h) + pw(h))
        return self.project(h).squeeze(1)      # (B, 128, 128) -> log-PGA (z-score)


class FNONami(nn.Module):
    def __init__(self, in_channels=4, lift=64, modes=16, n_layers=4,
                 mu_logpga=-18.032, sigma_logpga=5.287):
        super().__init__()
        self.net = FNO2d(in_channels, lift, modes, n_layers)
        self.register_buffer("mu", torch.tensor(float(mu_logpga)))
        self.register_buffer("sigma", torch.tensor(float(sigma_logpga)))

    def forward(self, x):
        return self.net(x)                     # z-scored log-PGA

    @torch.no_grad()
    def predict_pga_gal(self, x):
        """Recover PGA in gal (manuscript Eq. 5)."""
        z = self.net(x)
        return torch.exp(z * self.sigma + self.mu) * 100.0


def build_fno_nami(cfg: dict) -> FNONami:
    c = cfg["fno_nami"]
    return FNONami(
        in_channels=c["in_channels"], lift=c["lift_channels"],
        modes=c["fourier_modes"], n_layers=c["n_fno_layers"],
        mu_logpga=c["mu_logpga"], sigma_logpga=c["sigma_logpga"],
    )
