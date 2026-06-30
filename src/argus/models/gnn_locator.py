# =============================================================================
# src/argus/models/gnn_locator.py
#
# What this module does:
#   Defines GNN-Locator, the hypocenter-location network of ARGUS. Each seismic
#   network is a fully connected graph (nodes = stations). Node features encode
#   per-station waveform statistics, P-wave arrival, S-P differential time, SNR
#   and station coordinates (22-dim); edge features encode inter-station
#   distance, azimuth and arrival-time differences (4-dim). Four GATv2 layers
#   (Brody et al., 2022) propagate information; node states are pooled by
#   concatenated mean/max/sum and decoded to (lat, lon, depth).
#
#   Architecture matches the manuscript / Table S2:
#     hidden=128, heads=4, layers=4, node_feat=22, edge_feat=4.
# =============================================================================
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
except Exception as e:  # pragma: no cover - torch_geometric required at runtime
    GATv2Conv = None
    raise ImportError(
        "torch_geometric is required for GNN-Locator. Install with "
        "`uv pip install torch-geometric`."
    ) from e


class GNNLocator(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 22,
        edge_feat_dim: int = 4,
        hidden_dim: int = 128,
        heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        out_dim: int = 3,
    ):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            # concat=False keeps width = hidden_dim across layers
            self.convs.append(
                GATv2Conv(
                    hidden_dim, hidden_dim, heads=heads,
                    concat=False, edge_dim=edge_feat_dim, dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # mean ++ max ++ sum pooling -> 3 * hidden_dim
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h = norm(F.gelu(h_new) + h)               # residual + norm
            h = F.dropout(h, p=self.dropout, training=self.training)
        pooled = torch.cat(
            [global_mean_pool(h, batch),
             global_max_pool(h, batch),
             global_add_pool(h, batch)], dim=-1,
        )
        return self.head(pooled)                       # (B, 3) normalized coords


def build_gnn_locator(cfg: dict) -> GNNLocator:
    c = cfg["gnn_locator"]
    return GNNLocator(
        node_feat_dim=c["node_feat_dim"], edge_feat_dim=c["edge_feat_dim"],
        hidden_dim=c["hidden_dim"], heads=c["attention_heads"],
        n_layers=c["gatv2_layers"], dropout=c["dropout"], out_dim=c["out_dim"],
    )
