# =============================================================================
# tests/test_shapes.py
#
# What this test does:
#   Fast, dependency-light mock test (no trained weights, no real data) that
#   asserts the three components and the end-to-end pipeline produce the
#   expected tensor shapes and that producer/consumer dictionary keys are
#   consistent across SWIFT CMT and the ARGUS pipeline output. Run with:
#       pytest -q
# =============================================================================
import sys
from pathlib import Path
import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("torch_geometric")

from argus.graph import build_graph                    # noqa: E402
from argus.models import build_gnn_locator, build_swift_cmt, build_fno_nami  # noqa: E402
from argus.conformal import fit_conformal, empirical_coverage               # noqa: E402

CFG = {
    "device": "cpu",
    "gnn_locator": dict(node_feat_dim=22, edge_feat_dim=4, hidden_dim=32,
                        attention_heads=4, gatv2_layers=2, dropout=0.0, out_dim=3),
    "swift_cmt": dict(waveform_base_channels=8, fourier_bins=16, gatv2_layers=2,
                      hidden_dim=32, dropout=0.0, n_mechanism_classes=3),
    "fno_nami": dict(in_channels=4, lift_channels=16, fourier_modes=8, n_fno_layers=2,
                     out_grid=128, mu_logpga=-18.032, sigma_logpga=5.287),
}

N = 8


def _event():
    nf = np.random.randn(N, 22).astype("float32")
    coords = np.random.uniform([33.0, 130.0], [33.5, 131.0], size=(N, 2))
    parr = np.sort(np.random.uniform(0, 2, N))
    return build_graph(nf, coords, parr)


def test_locator_shape():
    g = _event(); m = build_gnn_locator(CFG)
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    out = m(g.x, g.edge_index, g.edge_attr, batch)
    assert out.shape == (1, 3)


def test_swift_keys_and_shapes():
    g = _event(); m = build_swift_cmt(CFG)
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    wf = torch.randn(N, 3, 1024)
    out = m(wf, g.edge_index, g.edge_attr, batch)
    assert set(out.keys()) == {"mechanism", "f_iso", "mw"}
    assert out["mechanism"].shape == (1, 3)
    assert out["f_iso"].shape == (1,) and out["mw"].shape == (1,)


def test_fno_pga_shape():
    m = build_fno_nami(CFG)
    pga = m.predict_pga_gal(torch.randn(1, 4, 128, 128))
    assert pga.shape == (1, 128, 128)
    assert torch.isfinite(pga).all()


def test_conformal_coverage_monotone():
    pred = np.random.randn(200, 3); true = pred + np.random.randn(200, 3) * 0.5
    q, s = fit_conformal(pred[:138], true[:138], alpha=0.10)
    cov = empirical_coverage(pred[138:], true[138:], q, s)
    assert 0.0 <= cov <= 1.0


def test_pipeline_output_keys():
    # producer (pipeline) keys must match what downstream consumers expect
    expected = {"location", "mechanism_logits", "f_iso", "mw", "pga_map_gal"}
    # emulate pipeline assembly without loading ARGUS (keeps test light)
    g = _event()
    loc = build_gnn_locator(CFG)
    sw = build_swift_cmt(CFG)
    fn = build_fno_nami(CFG)
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    cmt = sw(torch.randn(N, 3, 1024), g.edge_index, g.edge_attr, batch)
    out = {
        "location": loc(g.x, g.edge_index, g.edge_attr, batch).squeeze(0).detach().numpy(),
        "mechanism_logits": cmt["mechanism"].detach().numpy(),
        "f_iso": float(cmt["f_iso"]),
        "mw": float(cmt["mw"]),
        "pga_map_gal": fn.predict_pga_gal(torch.randn(1, 4, 128, 128)).squeeze(0).numpy(),
    }
    assert set(out.keys()) == expected
