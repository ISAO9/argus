# =============================================================================
# src/argus/graph.py
#
# What this module does:
#   Builds the fully connected station graph consumed by GNN-Locator and
#   SWIFT CMT. Node features (22-dim) and edge features (4-dim) follow the
#   manuscript definition:
#     node : station lat/lon/elev, P arrival, S-P differential time, SNR, and
#            6-band waveform-envelope statistics (+ derived terms) -> 22 dims.
#     edge : inter-station distance, azimuth (sin/cos), arrival-time difference.
#
#   REUSED ASSET: the canonical Hi-net graph tensor produced during prior ARGUS
#   development is `data/processed/hinet_graph_v2.pt` (688 events). When present
#   it is loaded directly; otherwise graphs are built on the fly from waveforms.
# =============================================================================
from __future__ import annotations
import numpy as np
import torch

try:
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    raise ImportError("torch_geometric required for graph construction.") from e

EARTH_R = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def _azimuth_rad(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    dlon = (lon2 - lon1) * p
    y = np.sin(dlon) * np.cos(lat2 * p)
    x = (np.cos(lat1 * p) * np.sin(lat2 * p)
         - np.sin(lat1 * p) * np.cos(lat2 * p) * np.cos(dlon))
    return np.arctan2(y, x)


def build_graph(node_features: np.ndarray, station_coords: np.ndarray,
                p_arrivals: np.ndarray) -> "Data":
    """Construct a fully connected PyG graph.

    node_features  : (N, 22) precomputed node feature matrix.
    station_coords : (N, 2)  station (lat, lon) in degrees.
    p_arrivals     : (N,)    P-wave arrival times in s.
    """
    n = len(station_coords)
    src, dst, eattr = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = _haversine_km(*station_coords[i], *station_coords[j])
            az = _azimuth_rad(*station_coords[i], *station_coords[j])
            dt = float(p_arrivals[i] - p_arrivals[j])
            src.append(i); dst.append(j)
            eattr.append([d / 100.0, np.sin(az), np.cos(az), dt])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.float32)
    x = torch.tensor(node_features, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def load_cached_graphs(path: str):
    """Load the reused Hi-net graph tensor (hinet_graph_v2.pt) if available."""
    return torch.load(path, map_location="cpu")
