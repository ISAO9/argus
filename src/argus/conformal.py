# =============================================================================
# src/argus/conformal.py
#
# What this module does:
#   Implements split (inductive) conformal prediction for GNN-Locator hypocenter
#   uncertainty (manuscript Eq. 2). A held-out calibration set of N events
#   (default 138) yields nonconformity scores s_i = ||y_hat - y||_2 / sigma_hat.
#   The (1 - alpha) location radius is the finite-sample-corrected empirical
#   quantile q_hat = s_(k), k = ceil((N+1)(1-alpha)). The reported interval is a
#   ball of radius q_hat * sigma_hat around the predicted hypocenter.
#
#   This procedure is distribution-free: coverage >= 1 - alpha holds in finite
#   samples under exchangeability (Vovk et al., 2005; Angelopoulos & Bates, 2022).
# =============================================================================
from __future__ import annotations
import numpy as np


def fit_conformal(pred_xyz: np.ndarray, true_xyz: np.ndarray, alpha: float = 0.10):
    """Return (q_hat, sigma_hat) from a calibration set.

    pred_xyz, true_xyz : (N, 3) arrays of (x, y, z) in km.
    """
    err = np.linalg.norm(pred_xyz - true_xyz, axis=1)          # km
    sigma_hat = err.std() + 1e-9
    scores = err / sigma_hat
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    q_hat = np.sort(scores)[k - 1]
    return float(q_hat), float(sigma_hat)


def conformal_radius_km(q_hat: float, sigma_hat: float) -> float:
    """Location confidence radius in km for a new prediction."""
    return q_hat * sigma_hat


def empirical_coverage(pred_xyz, true_xyz, q_hat, sigma_hat) -> float:
    """Fraction of test events whose true hypocenter lies within the ball."""
    err = np.linalg.norm(np.asarray(pred_xyz) - np.asarray(true_xyz), axis=1)
    return float((err <= q_hat * sigma_hat).mean())
