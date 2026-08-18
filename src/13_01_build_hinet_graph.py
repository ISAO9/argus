#!/usr/bin/env python3
# =============================================================================
# Script : 13_01_build_hinet_graph.py
# Project: ARGUS (SWIFT + NAMI + GNN-Locator)
#
# Description:
#   Hi-netネットワークをグラフ構造に変換し、
#   GATv2モデルの入力形式を作成する。
#
#   グラフ設計:
#     ノード: 各観測局 (n_sta = 8 per event)
#       特徴量 (node_dim = 16):
#         [lat_norm, lon_norm, elev_norm,     # 局の位置 (3)
#          tp_norm, sp_norm,                   # 走時特徴 (2)
#          snr_norm,                           # SNR     (1)
#          wave_emb_0..9]                      # 波形埋め込み (10)
#
#     エッジ: k近傍 (k=4) の局間を接続
#       特徴量 (edge_dim = 4):
#         [dist_norm,                          # 局間距離 (1)
#          az_norm,                            # 方位角   (1)
#          dt_tp_norm,                         # P波到達時刻差 (1)
#          dt_sp_norm]                         # S-P時間差の差 (1)
#
#   出力:
#     data/locator/hinet_graph_dataset.pt
#       torch_geometric DataLoader対応形式
#     PDF/13_01_graph_stats.pdf
#
#   Usage:
#     python src/13_01_build_hinet_graph.py
#     python src/13_01_build_hinet_graph.py --k_neighbors 4 --wave_emb_dim 16
#
# =============================================================================

import sys, argparse, logging, time
from pathlib import Path

import numpy as np
import h5py
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = PROJECT_ROOT / "data" / "locator"
PDF_DIR   = PROJECT_ROOT / "PDF"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# =============================================================================
# SIMPLE WAVEFORM ENCODER (CNN)
# MPS/CPUで動作する軽量エンコーダ
# GATv2の前段として波形を埋め込みベクトルに変換
# =============================================================================

import torch.nn as nn
import torch.nn.functional as F


class WaveformEncoder(nn.Module):
    """
    1D CNNによる波形エンコーダ。

    入力: (batch, 3, n_samples)  — 3成分波形
    出力: (batch, emb_dim)       — 埋め込みベクトル

    設計方針:
      - 軽量 (< 100K params) → リアルタイム推論対応
      - P/S波の特徴を自動抽出
      - SNR変動に対してロバスト（BatchNorm使用）
    """
    def __init__(self, in_channels: int = 3,
                 emb_dim: int = 16,
                 n_samples: int = 1024):
        super().__init__()
        self.emb_dim = emb_dim

        self.conv_layers = nn.Sequential(
            # Stage 1: 局所特徴 (kernel=7)
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(4),          # 1024 → 256

            # Stage 2: 中間特徴 (kernel=5, dilated)
            nn.Conv1d(32, 64, kernel_size=5, padding=4, dilation=2, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(4),          # 256 → 64

            # Stage 3: 高次特徴 (kernel=3)
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),  # → (batch, 128, 4)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 64),
            nn.GELU(),
            nn.Linear(64, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, n_samples)
        z = self.conv_layers(x)                  # (batch, 128, 4)
        z = z.flatten(1)                          # (batch, 512)
        return self.fc(z)                         # (batch, emb_dim)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """2点間の大円距離[km]."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon/2)**2)
    return float(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def compute_azimuth(lat1, lon1, lat2, lon2) -> float:
    """方位角[degrees] (lat1,lon1) → (lat2,lon2)."""
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(np.radians(lat2))
    x = (np.cos(np.radians(lat1)) * np.sin(np.radians(lat2))
         - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(dlon))
    return float(np.degrees(np.arctan2(y, x)) % 360)


def build_edges_knn(sta_locs: np.ndarray, k: int = 4) -> tuple:
    """
    k近傍グラフのエッジを構築する。

    Returns:
      edge_index: (2, n_edges) — [src, dst]
      edge_attr:  (n_edges, 4) — [dist_km, azimuth, dt_tp, dt_sp]
                                  (未正規化、後で正規化)
    """
    n = len(sta_locs)
    k = min(k, n - 1)

    # 全局間距離行列
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = haversine_km(
                    sta_locs[i, 0], sta_locs[i, 1],
                    sta_locs[j, 0], sta_locs[j, 1]
                )
            else:
                dist_matrix[i, j] = np.inf

    # k近傍
    src_list, dst_list, dist_list, az_list = [], [], [], []
    for i in range(n):
        nn_idx = np.argsort(dist_matrix[i])[:k]
        for j in nn_idx:
            src_list.append(i)
            dst_list.append(j)
            dist_list.append(dist_matrix[i, j])
            az_list.append(compute_azimuth(
                sta_locs[i, 0], sta_locs[i, 1],
                sta_locs[j, 0], sta_locs[j, 1]
            ))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    # dt_tp, dt_sp は後でイベントごとに付与
    edge_attr_geo = np.stack([dist_list, az_list], axis=1).astype(np.float32)
    return edge_index, edge_attr_geo


def build_graph_for_event(
    waveforms: np.ndarray,       # (n_sta, 3, n_samples)
    sta_locs:  np.ndarray,       # (n_sta, 3) [lat, lon, elev]
    tp_times:  np.ndarray,       # (n_sta,)
    sp_diffs:  np.ndarray,       # (n_sta,)
    snr:       np.ndarray,       # (n_sta,)
    src_loc:   np.ndarray,       # (3,) [lat, lon, dep]
    mw:        float,
    edge_index: np.ndarray,      # (2, n_edges)
    edge_attr_geo: np.ndarray,   # (n_edges, 2)
    norm_stats: dict,
    wave_emb:  np.ndarray,       # (n_sta, emb_dim)
) -> dict:
    """
    1イベントのグラフデータを構築する。

    node_features: (n_sta, 3 + 2 + 1 + emb_dim)
    edge_features: (n_edges, 4)
    target:        (3,)  [lat_norm, lon_norm, dep_norm]
    """
    n_sta = len(sta_locs)
    ns    = norm_stats

    # ── Node features ───────────────────────────────────────────────
    lat_n = (sta_locs[:, 0] - ns["lat_mean"])  / ns["lat_std"]
    lon_n = (sta_locs[:, 1] - ns["lon_mean"])  / ns["lon_std"]
    elev_n= sta_locs[:, 2] / 1.0  # km, 大体0〜0.5

    # P波到達時刻（最初の局を0基準に相対化）
    tp_ref  = tp_times[0]
    tp_rel  = (tp_times - tp_ref)          # relative times
    tp_norm = tp_rel / ns["sp_std"]        # std正規化

    sp_norm = (sp_diffs - ns["sp_mean"])   / ns["sp_std"]
    snr_n   = (snr - 15.0) / 10.0         # rough norm

    node_feat = np.concatenate([
        lat_n.reshape(-1, 1),   # (n_sta, 1)
        lon_n.reshape(-1, 1),   # (n_sta, 1)
        elev_n.reshape(-1, 1),  # (n_sta, 1)
        tp_norm.reshape(-1, 1), # (n_sta, 1)
        sp_norm.reshape(-1, 1), # (n_sta, 1)
        snr_n.reshape(-1, 1),   # (n_sta, 1)
        wave_emb,               # (n_sta, emb_dim)
    ], axis=1).astype(np.float32)

    # ── Edge features ───────────────────────────────────────────────
    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    dist_km = edge_attr_geo[:, 0]
    az      = edge_attr_geo[:, 1]

    dist_norm = dist_km / ns["dist_std"]
    az_norm   = az / 180.0 - 1.0  # [-1, 1]

    dt_tp     = tp_times[src_idx] - tp_times[dst_idx]
    dt_sp     = sp_diffs[src_idx] - sp_diffs[dst_idx]
    dt_tp_n   = dt_tp / ns["sp_std"]
    dt_sp_n   = dt_sp / ns["sp_std"]

    edge_feat = np.stack([dist_norm, az_norm, dt_tp_n, dt_sp_n],
                          axis=1).astype(np.float32)

    # ── Target ──────────────────────────────────────────────────────
    target = np.array([
        (src_loc[0] - ns["lat_mean"]) / ns["lat_std"],   # lat
        (src_loc[1] - ns["lon_mean"]) / ns["lon_std"],   # lon
        np.log(src_loc[2]) / np.log(60.0),               # depth (log scale)
    ], dtype=np.float32)

    return {
        "x":          node_feat,     # (n_sta, node_dim)
        "edge_index": edge_index,    # (2, n_edges)
        "edge_attr":  edge_feat,     # (n_edges, 4)
        "y":          target,        # (3,)
        "src_loc":    src_loc,       # (3,) 元の座標（評価用）
        "mw":         float(mw),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ARGUS Script 13_01 — Hi-net Graph Dataset Builder")
    parser.add_argument("--in_h5",       type=str,
                        default="data/locator/synthetic_locator_dataset.h5")
    parser.add_argument("--out_pt",      type=str,
                        default="data/locator/hinet_graph_dataset.pt")
    parser.add_argument("--k_neighbors", type=int, default=4,
                        help="k-NN edges per station")
    parser.add_argument("--wave_emb_dim",type=int, default=16,
                        help="Waveform embedding dimension")
    parser.add_argument("--batch_size",  type=int, default=256,
                        help="Batch size for waveform encoding")
    parser.add_argument("--device",      type=str, default="auto")
    args = parser.parse_args()

    log.info("="*65)
    log.info("  ARGUS  |  Script 13_01  |  Hi-net Graph Dataset Builder")
    log.info("="*65)
    log.info(f"  k_neighbors   : {args.k_neighbors}")
    log.info(f"  wave_emb_dim  : {args.wave_emb_dim}")

    # ── Device ───────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"  Device: {device}")

    # ── Load HDF5 ─────────────────────────────────────────────────
    in_path = Path(args.in_h5) if Path(args.in_h5).is_absolute() \
              else PROJECT_ROOT / args.in_h5
    log.info(f"\n  Loading: {in_path}")

    with h5py.File(in_path, 'r') as f:
        meta = f["metadata"]
        norm_stats = {
            k.replace("norm_", ""): float(v)
            for k, v in meta.attrs.items()
            if k.startswith("norm_")
        }
        sta_locs_global = f["station_locs"][:]   # (n_sta_global, 3)

        splits = {}
        for split in ["train", "val", "test"]:
            g = f[split]
            splits[split] = {
                "waveforms": g["waveforms"][:],
                "sta_locs" : g["sta_locs"][:],
                "src_locs" : g["src_locs"][:],
                "tp_times" : g["tp_times"][:],
                "sp_diffs" : g["sp_diffs"][:],
                "snr"      : g["snr"][:],
                "dist_km"  : g["dist_km"][:],
                "mw"       : g["mw"][:],
            }
            log.info(f"  {split}: {len(splits[split]['mw'])} events")

    # dist_std for normalization
    if "dist_std" not in norm_stats:
        all_dist = np.concatenate([
            splits[s]["dist_km"].flatten() for s in splits
        ])
        norm_stats["dist_std"] = float(all_dist.std()) + 1e-8
        norm_stats["dist_mean"]= float(all_dist.mean())

    log.info(f"  Norm stats: {list(norm_stats.keys())}")

    # ── Waveform Encoder ─────────────────────────────────────────
    log.info("\n  Building waveform encoder...")
    n_samples = splits["train"]["waveforms"].shape[-1]
    encoder   = WaveformEncoder(in_channels=3,
                                 emb_dim=args.wave_emb_dim,
                                 n_samples=n_samples).to(device)
    encoder.eval()
    n_params = sum(p.numel() for p in encoder.parameters())
    log.info(f"  Encoder params: {n_params:,}")

    # ── Graph topology (per-event, based on station subset) ──────
    # 各イベントの局配置が異なるため、イベントごとにグラフを構築
    # ここでは固定トポロジーとして最初のイベントの局配置を使用
    log.info("\n  Pre-building edge topology (k-NN)...")

    # サンプルイベントから局配置を取得
    sample_sta = splits["train"]["sta_locs"][0]  # (max_sta, 3)
    edge_index, edge_attr_geo = build_edges_knn(sample_sta,
                                                  k=args.k_neighbors)
    n_edges = edge_index.shape[1]
    log.info(f"  Stations per event: {len(sample_sta)}")
    log.info(f"  Edges per event: {n_edges}")
    log.info(f"  Node dim: {6 + args.wave_emb_dim}")
    log.info(f"  Edge dim: 4")

    # ── Build graph dataset ───────────────────────────────────────
    log.info("\n  Building graph dataset...")
    graph_data = {}
    t0 = time.time()

    for split_name, split_data in splits.items():
        n     = len(split_data["mw"])
        wavs  = split_data["waveforms"]  # (N, max_sta, 3, n_samples)
        max_sta = wavs.shape[1]

        log.info(f"  Processing split: {split_name} ({n} events)...")

        # 波形エンコード（バッチ処理）
        all_wave_emb = np.zeros((n, max_sta, args.wave_emb_dim), np.float32)
        bs = args.batch_size

        with torch.no_grad():
            for start in range(0, n * max_sta, bs):
                end   = min(start + bs, n * max_sta)
                ev_i  = np.arange(start, end) // max_sta
                sta_i = np.arange(start, end)  % max_sta
                # (bs, 3, n_samples)
                batch_wav = torch.from_numpy(
                    wavs[ev_i, sta_i]
                ).to(device)
                emb = encoder(batch_wav).cpu().numpy()
                for idx, (ei, si) in enumerate(zip(ev_i, sta_i)):
                    all_wave_emb[ei, si] = emb[idx]

                if start % (bs * 20) == 0:
                    log.info(f"    Encoded {end}/{n*max_sta} waveforms...")

        # グラフ構築
        graphs = []
        for i in range(n):
            # イベントごとに局配置が変わる → エッジを再構築
            sta_i = split_data["sta_locs"][i]
            ei_i, ea_geo = build_edges_knn(sta_i, k=args.k_neighbors)

            g = build_graph_for_event(
                waveforms  = split_data["waveforms"][i],
                sta_locs   = sta_i,
                tp_times   = split_data["tp_times"][i],
                sp_diffs   = split_data["sp_diffs"][i],
                snr        = split_data["snr"][i],
                src_loc    = split_data["src_locs"][i],
                mw         = float(split_data["mw"][i]),
                edge_index = ei_i,
                edge_attr_geo = ea_geo,
                norm_stats = norm_stats,
                wave_emb   = all_wave_emb[i],
            )
            graphs.append(g)

        graph_data[split_name] = graphs
        log.info(f"  {split_name}: {len(graphs)} graphs built")

    # ── Save ─────────────────────────────────────────────────────
    out_path = Path(args.out_pt) if Path(args.out_pt).is_absolute() \
               else PROJECT_ROOT / args.out_pt

    save_dict = {
        "graph_data"    : graph_data,
        "norm_stats"    : norm_stats,
        "k_neighbors"   : args.k_neighbors,
        "wave_emb_dim"  : args.wave_emb_dim,
        "node_dim"      : 6 + args.wave_emb_dim,
        "edge_dim"      : 4,
        "encoder_state" : encoder.state_dict(),
    }
    torch.save(save_dict, out_path)
    size_mb = out_path.stat().st_size / 1e6
    log.info(f"\n  Saved: {out_path}  ({size_mb:.1f} MB)")
    log.info(f"  Time: {time.time()-t0:.1f}s")

    # ── Sample graph stats ────────────────────────────────────────
    g0 = graph_data["train"][0]
    log.info(f"\n  Sample graph [0]:")
    log.info(f"    x.shape      : {g0['x'].shape}")
    log.info(f"    edge_index   : {g0['edge_index'].shape}")
    log.info(f"    edge_attr    : {g0['edge_attr'].shape}")
    log.info(f"    y (target)   : {g0['y']}")
    log.info(f"    src_loc      : {g0['src_loc']}")

    # ── PDF ───────────────────────────────────────────────────────
    _generate_pdf(graph_data["train"][:500], norm_stats)

    log.info("="*65)
    log.info("  Script 13_01 complete.")
    log.info(f"  Graph dataset : {out_path}")
    log.info("  Next: python src/13_02_gnn_locator_model.py")
    log.info("="*65)


def _generate_pdf(graphs: list, norm_stats: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    DARK = '#0d1b2a'; BLUE = '#4fc3f7'; GREEN = '#a5d6a7'

    pdf_path = PDF_DIR / "13_01_graph_stats.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor(DARK)
        fig.suptitle("ARGUS Script 13_01 — Graph Dataset Statistics",
                     fontsize=12, color='white', fontweight='bold')

        def sa(ax, title):
            ax.set_facecolor('#112030')
            for sp in ax.spines.values(): sp.set_edgecolor('#2a4a6a')
            ax.tick_params(colors='#7a9ab0', labelsize=9)
            ax.xaxis.label.set_color('#7a9ab0')
            ax.yaxis.label.set_color('#7a9ab0')
            ax.set_title(title, color=BLUE, fontsize=10, pad=6)

        ns = norm_stats

        # Denorm target distribution
        lats = [g["src_loc"][0] for g in graphs]
        lons = [g["src_loc"][1] for g in graphs]
        deps = [g["src_loc"][2] for g in graphs]

        ax = axes[0, 0]
        ax.scatter(lons[:200], lats[:200], s=4, alpha=0.6, c=deps[:200],
                   cmap='plasma_r')
        sa(ax, "Epicenter Distribution")
        ax.set_xlabel("Lon (°E)"); ax.set_ylabel("Lat (°N)")

        ax = axes[0, 1]
        ax.hist(deps, bins=40, color=BLUE, edgecolor=DARK, alpha=0.85)
        sa(ax, "Depth Distribution")
        ax.set_xlabel("Depth (km)"); ax.set_ylabel("Count")

        # Node feature distribution (first 3 dims = lat/lon/elev)
        node_feat_0 = np.array([g["x"][:, 0] for g in graphs[:200]]).flatten()
        ax = axes[0, 2]
        ax.hist(node_feat_0, bins=40, color=GREEN, edgecolor=DARK, alpha=0.85)
        sa(ax, "Node Feature [lat_norm] Distribution")
        ax.set_xlabel("Normalized latitude"); ax.set_ylabel("Count")

        # Edge distance distribution
        edge_dist = np.array([g["edge_attr"][:, 0] for g in graphs[:200]]).flatten()
        ax = axes[1, 0]
        ax.hist(edge_dist, bins=40, color='#ffe082', edgecolor=DARK, alpha=0.85)
        sa(ax, "Edge Distance (normalized)")
        ax.set_xlabel("Dist norm"); ax.set_ylabel("Count")

        # S-P time feature
        sp_feat = np.array([g["x"][:, 4] for g in graphs[:200]]).flatten()
        ax = axes[1, 1]
        ax.hist(sp_feat, bins=40, color='#f4a582', edgecolor=DARK, alpha=0.85)
        sa(ax, "S-P Time Feature (normalized)")
        ax.set_xlabel("S-P norm"); ax.set_ylabel("Count")

        # Target distribution
        targets = np.array([g["y"] for g in graphs])
        ax = axes[1, 2]
        ax.scatter(targets[:200, 0], targets[:200, 1], s=4, alpha=0.5,
                   c=targets[:200, 2], cmap='coolwarm')
        sa(ax, "Target Distribution (lat/lon/dep normalized)")
        ax.set_xlabel("lat_norm"); ax.set_ylabel("lon_norm")

        fig.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close(fig)

    log.info(f"  PDF: {pdf_path}")


if __name__ == "__main__":
    main()
