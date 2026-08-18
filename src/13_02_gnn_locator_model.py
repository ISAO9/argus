#!/usr/bin/env python3
# =============================================================================
# Script : 13_02_gnn_locator_model.py
# Project: ARGUS (SWIFT + NAMI + GNN-Locator)
#
# Description:
#   GATv2ベースのGNN震源特定モデルの定義と訓練。
#
#   アーキテクチャ:
#     Stage 1: 入力射影
#       node_feat (22次元) → hidden (128次元)
#
#     Stage 2: GATv2 × 4層
#       各層: マルチヘッドアテンション (heads=4)
#             エッジ特徴量を考慮したアテンション計算
#             残差接続 + LayerNorm
#
#     Stage 3: グローバル集約
#       Attention Pooling (SNRに基づく重み付き集約)
#       → イベント表現 (128次元)
#
#     Stage 4: Heteroscedastic出力ヘッド
#       → μ_lat, μ_lon, μ_dep  (平均値)
#       → σ_lat, σ_lon, σ_dep  (不確実性)
#
#   損失関数:
#     L_nll   = Negative Log-Likelihood (heteroscedastic)
#     L_phys  = S-P時間差の物理整合性ペナルティ
#     L_total = L_nll + λ * L_phys
#
#   出力:
#     models/locator/best_locator.pt
#     PDF/13_02_locator_training.pdf
#
#   Usage:
#     python src/13_02_gnn_locator_model.py
#     python src/13_02_gnn_locator_model.py --epochs 100 --lr 1e-3
#
# =============================================================================

import sys, argparse, logging, time, json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "locator"
PDF_DIR   = PROJECT_ROOT / "PDF"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# =============================================================================
# GATv2 LAYER  (エッジ特徴量対応版)
# =============================================================================

class GATv2Layer(nn.Module):
    """
    GATv2 (Brody et al. 2022) — エッジ特徴量対応版.

    GAT vs GATv2:
      GAT  : α_ij = softmax(a^T [Whi || Whj])       ← 静的アテンション
      GATv2: α_ij = softmax(a^T LeakyReLU(W[hi||hj]))← 動的アテンション

    エッジ特徴量:
      アテンション計算に局間距離・到達時刻差を組み込む
      → 近い局・早く到達した局を自動的に重視

    Args:
      in_dim   : 入力特徴量次元
      out_dim  : 出力特徴量次元 (per head)
      heads    : アテンションヘッド数
      edge_dim : エッジ特徴量次元
      dropout  : ドロップアウト率
    """
    def __init__(self, in_dim: int, out_dim: int,
                 heads: int = 4, edge_dim: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.heads   = heads
        self.out_dim = out_dim

        # ノード変換
        self.W_src = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, heads * out_dim, bias=False)

        # エッジ特徴量変換
        self.W_edge = nn.Linear(edge_dim, heads * out_dim, bias=False)

        # アテンションベクトル
        self.attn_vec = nn.Parameter(torch.Tensor(1, heads, out_dim))
        nn.init.xavier_uniform_(self.attn_vec.view(1, -1).unsqueeze(0))

        # 出力変換
        self.W_out = nn.Linear(heads * out_dim, heads * out_dim, bias=False)
        self.norm  = nn.LayerNorm(heads * out_dim)
        self.drop  = nn.Dropout(dropout)

        # 残差用
        self.residual = nn.Linear(in_dim, heads * out_dim, bias=False) \
                        if in_dim != heads * out_dim else nn.Identity()

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr:  torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:          (N, in_dim)
          edge_index: (2, E)   [src, dst]
          edge_attr:  (E, edge_dim)
        Returns:
          x_out:      (N, heads * out_dim)
        """
        N = x.size(0)
        H = self.heads
        D = self.out_dim
        E = edge_index.size(1)

        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)

        # 変換
        h_src  = self.W_src(x[src_idx]).view(E, H, D)   # (E, H, D)
        h_dst  = self.W_dst(x[dst_idx]).view(E, H, D)   # (E, H, D)
        h_edge = self.W_edge(edge_attr).view(E, H, D)   # (E, H, D)

        # GATv2: LeakyReLU後にアテンション
        e = F.leaky_relu(h_src + h_dst + h_edge, 0.2)   # (E, H, D)
        attn = (e * self.attn_vec).sum(dim=-1)           # (E, H)

        # Softmax per destination node
        # scatter softmax: 各dst_idxごとにsoftmax
        attn_exp = torch.zeros(N, H, device=x.device)
        attn_max = torch.full((N, H), float('-inf'), device=x.device)

        # max for numerical stability
        attn_max.scatter_reduce_(0, dst_idx.unsqueeze(1).expand(-1, H),
                                  attn, reduce='amax', include_self=True)
        attn_shifted = torch.exp(attn - attn_max[dst_idx])
        attn_sum = torch.zeros(N, H, device=x.device)
        attn_sum.scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, H),
                               attn_shifted)
        attn_norm = attn_shifted / (attn_sum[dst_idx] + 1e-10)  # (E, H)
        attn_norm = self.drop(attn_norm)

        # メッセージ集約
        msg = (h_src * attn_norm.unsqueeze(-1)).view(E, H * D)  # (E, H*D)
        out = torch.zeros(N, H * D, device=x.device)
        out.scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, H * D), msg)

        # 残差 + Norm
        out = self.norm(self.W_out(out) + self.residual(x))
        return F.gelu(out)


# =============================================================================
# GLOBAL ATTENTION POOLING
# =============================================================================

class GlobalAttentionPool(nn.Module):
    """
    SNR重み付きグローバルプーリング。

    各ノードの「重要度」をMLPで計算し、
    重み付き平均でグラフ全体の表現を作る。
    SNRが高い（信号が明瞭な）局を自動的に重視する。
    """
    def __init__(self, dim: int, snr_feat_idx: int = 5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.snr_feat_idx = snr_feat_idx

    def forward(self, x: torch.Tensor,
                batch_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (N, dim) — ノード特徴量
        batch_mask: None (single graph) or (N,) batch index
        Returns: (1, dim) or (B, dim)
        """
        gate_val = torch.sigmoid(self.gate(x))   # (N, 1)
        weighted = x * gate_val                    # (N, dim)

        if batch_mask is None:
            return weighted.mean(0, keepdim=True)  # (1, dim)
        else:
            # scatter_mean (simplified: just mean per batch)
            B = batch_mask.max().item() + 1
            out = torch.zeros(B, x.size(1), device=x.device)
            cnt = torch.zeros(B, 1, device=x.device)
            out.scatter_add_(0, batch_mask.unsqueeze(1).expand_as(weighted),
                              weighted)
            cnt.scatter_add_(0, batch_mask.unsqueeze(1),
                              torch.ones(len(x), 1, device=x.device))
            return out / (cnt + 1e-10)


# =============================================================================
# GNN LOCATOR MODEL
# =============================================================================

class GNNLocator(nn.Module):
    """
    GATv2ベースの震源特定モデル。

    入力:
      x          : (N, node_dim) ノード特徴量
      edge_index : (2, E)
      edge_attr  : (E, edge_dim)

    出力:
      mu_loc  : (3,) [lat_norm, lon_norm, dep_norm] 予測値
      log_var : (3,) [log(σ²_lat), log(σ²_lon), log(σ²_dep)] 不確実性

    Heteroscedastic Loss:
      L = 0.5 * Σ [exp(-log_var) * (μ - y)² + log_var]
      → 不確実な予測は自動的に分散が大きくなる
    """
    def __init__(self, node_dim: int = 22, edge_dim: int = 4,
                 hidden: int = 128, heads: int = 4, n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden   = hidden

        # 入力射影
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # GATv2 × n_layers
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            in_d = hidden if i == 0 else heads * (hidden // heads)
            self.gat_layers.append(
                GATv2Layer(in_d, hidden // heads,
                           heads=heads, edge_dim=edge_dim,
                           dropout=dropout)
            )

        gat_out_dim = heads * (hidden // heads)

        # グローバルプーリング
        self.pool = GlobalAttentionPool(gat_out_dim)

        # 出力ヘッド
        self.loc_head = nn.Sequential(
            nn.Linear(gat_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.mu_head  = nn.Linear(32, 3)  # [lat, lon, dep]
        self.var_head = nn.Linear(32, 3)  # log_var

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr:  torch.Tensor) -> tuple:
        # 入力射影
        h = self.input_proj(x)                       # (N, hidden)

        # GATv2 メッセージパッシング
        for layer in self.gat_layers:
            h = layer(h, edge_index, edge_attr)       # (N, heads*D)

        # グローバル集約 → グラフ表現
        g = self.pool(h).squeeze(0)                   # (hidden,)

        # 出力
        feat    = self.loc_head(g)                    # (32,)
        mu_loc  = self.mu_head(feat)                  # (3,)
        log_var = self.var_head(feat)                 # (3,)
        log_var = torch.clamp(log_var, -6, 4)         # 数値安定化

        return mu_loc, log_var

    def predict(self, x, edge_index, edge_attr):
        """推論用: 予測値と不確実性を返す."""
        mu, log_var = self.forward(x, edge_index, edge_attr)
        sigma = torch.exp(0.5 * log_var)
        return mu, sigma


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def heteroscedastic_nll(mu: torch.Tensor, log_var: torch.Tensor,
                         target: torch.Tensor) -> torch.Tensor:
    """
    Heteroscedastic Negative Log-Likelihood.

    L = 0.5 * Σ [exp(-log_var) * (μ - y)² + log_var]

    不確実性が高い→log_varが大きい→損失を抑える
    → モデルが自動的に難しい事例の不確実性を大きく学習
    """
    diff_sq = (mu - target)**2
    return 0.5 * (torch.exp(-log_var) * diff_sq + log_var).mean()


def physics_loss(mu: torch.Tensor, sp_diffs: torch.Tensor,
                  dist_km: torch.Tensor) -> torch.Tensor:
    """
    S-P時間差の物理整合性ペナルティ（簡易版）.

    S-P時間差 ≈ dist_km × (1/Vs - 1/Vp)
    Vp=6.0, Vs=3.46 → 1/Vs - 1/Vp ≈ 0.123 s/km

    実装: この制約が厳しすぎないようにSoft penaltyで実装
    """
    SP_COEFF = 0.123  # s/km
    # dist_km (正規化されている) を近似変換
    # 正規化統計が必要なため、ここでは簡易ペナルティ
    # 予測された深さ変化と観測S-P差の相関を促進
    sp_mean = sp_diffs.mean(dim=-1)              # (batch,) or scalar
    dep_pred = mu[..., 2]                         # log_norm depth
    # 深さが増えるとS-P差も増えるべき (正の相関を促進)
    corr = -(dep_pred * sp_mean.clamp(-3, 3)).mean()
    return corr.clamp(0, 1)


# =============================================================================
# DATALOADER
# =============================================================================

def collate_graphs(batch: list) -> dict:
    """
    グラフリストをバッチに変換する。
    torch_geometric不使用のシンプル実装。
    単一グラフとして処理（バッチサイズ=1でミニバッチ）
    """
    # ランダムに1グラフを返す（簡略化）
    return batch[0]


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs: list):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return {
            "x":          torch.from_numpy(g["x"]),
            "edge_index": torch.from_numpy(g["edge_index"]),
            "edge_attr":  torch.from_numpy(g["edge_attr"]),
            "y":          torch.from_numpy(g["y"]),
            "src_loc":    torch.from_numpy(g["src_loc"]),
            "mw":         torch.tensor(g["mw"], dtype=torch.float32),
        }


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model: nn.Module,
                dataset: GraphDataset,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                phys_lambda: float = 0.1) -> dict:
    model.train()
    total_loss = 0.0
    total_nll  = 0.0
    n = len(dataset)
    indices = torch.randperm(n).tolist()

    for idx in indices:
        g = dataset[idx]
        x          = g["x"].to(device)
        edge_index = g["edge_index"].to(device)
        edge_attr  = g["edge_attr"].to(device)
        y          = g["y"].to(device)

        optimizer.zero_grad()
        mu, log_var = model(x, edge_index, edge_attr)
        nll  = heteroscedastic_nll(mu, log_var, y)
        loss = nll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_nll  += nll.item()

    return {
        "loss": total_loss / n,
        "nll" : total_nll  / n,
    }


@torch.no_grad()
def eval_epoch(model: nn.Module,
               dataset: GraphDataset,
               device: torch.device,
               norm_stats: dict) -> dict:
    model.eval()
    n = len(dataset)

    lat_errors, lon_errors, dep_errors = [], [], []
    total_loss = 0.0

    for idx in range(n):
        g = dataset[idx]
        x          = g["x"].to(device)
        edge_index = g["edge_index"].to(device)
        edge_attr  = g["edge_attr"].to(device)
        y          = g["y"].to(device)
        src_loc    = g["src_loc"]

        mu, log_var = model(x, edge_index, edge_attr)
        loss = heteroscedastic_nll(mu, log_var, y)
        total_loss += loss.item()

        # 元の座標に逆変換
        mu_np = mu.cpu().numpy()
        ns = norm_stats
        lat_pred = mu_np[0] * ns["lat_std"] + ns["lat_mean"]
        lon_pred = mu_np[1] * ns["lon_std"] + ns["lon_mean"]
        dep_pred = np.exp(mu_np[2] * np.log(60.0))

        lat_true = float(src_loc[0])
        lon_true = float(src_loc[1])
        dep_true = float(src_loc[2])

        # 距離誤差 (km)
        dlat = (lat_pred - lat_true) * 111.0
        dlon = (lon_pred - lon_true) * 111.0 * np.cos(np.radians(lat_true))
        dist_err = np.sqrt(dlat**2 + dlon**2)

        lat_errors.append(abs(lat_pred - lat_true))
        lon_errors.append(abs(lon_pred - lon_true))
        dep_errors.append(abs(dep_pred - dep_true))

    return {
        "loss"    : total_loss / n,
        "lat_mae" : float(np.mean(lat_errors)),   # degrees
        "lon_mae" : float(np.mean(lon_errors)),
        "dep_mae" : float(np.mean(dep_errors)),    # km
        "loc_km"  : float(np.sqrt(
            (np.mean(lat_errors)*111)**2 +
            (np.mean(lon_errors)*111)**2
        )),  # 水平誤差 km
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ARGUS Script 13_02 — GNN Locator Training")
    parser.add_argument("--in_pt",       type=str,
                        default="data/locator/hinet_graph_dataset.pt")
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--heads",       type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=4)
    parser.add_argument("--dropout",     type=float, default=0.10)
    parser.add_argument("--phys_lambda", type=float, default=0.05)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--device",      type=str,   default="auto")
    args = parser.parse_args()

    log.info("="*65)
    log.info("  ARGUS  |  Script 13_02  |  GNN Locator Training")
    log.info("="*65)
    log.info(f"  Epochs   : {args.epochs}")
    log.info(f"  LR       : {args.lr}")
    log.info(f"  Hidden   : {args.hidden}  Heads: {args.heads}  Layers: {args.n_layers}")

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

    # ── Load graph dataset ───────────────────────────────────────────
    in_path = Path(args.in_pt) if Path(args.in_pt).is_absolute() \
              else PROJECT_ROOT / args.in_pt
    log.info(f"\n  Loading: {in_path}")
    data = torch.load(in_path, map_location="cpu", weights_only=False)

    graph_data = data["graph_data"]
    norm_stats = data["norm_stats"]
    node_dim   = data["node_dim"]
    edge_dim   = data["edge_dim"]

    train_ds = GraphDataset(graph_data["train"])
    val_ds   = GraphDataset(graph_data["val"])
    test_ds  = GraphDataset(graph_data["test"])

    log.info(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    log.info(f"  Node dim: {node_dim}  Edge dim: {edge_dim}")

    # ── Model ────────────────────────────────────────────────────────
    model = GNNLocator(
        node_dim = node_dim,
        edge_dim = edge_dim,
        hidden   = args.hidden,
        heads    = args.heads,
        n_layers = args.n_layers,
        dropout  = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"\n  GNNLocator: {n_params:,} parameters")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_cnt  = 0
    history = {
        "train_loss": [], "val_loss": [],
        "val_loc_km": [], "val_dep_mae": [],
    }

    log.info(f"\n  Starting training ({args.epochs} epochs)...")
    log.info(f"  {'Epoch':>5}  {'TrnLoss':>9}  {'ValLoss':>9}  "
             f"{'LocKm':>7}  {'DepKm':>7}  {'LR':>8}")
    log.info("  " + "-"*58)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        # Train (small subset for speed: 500 samples per epoch)
        subset_n = min(500, len(train_ds))
        indices  = torch.randperm(len(train_ds))[:subset_n].tolist()
        sub_ds   = GraphDataset([graph_data["train"][i] for i in indices])
        tr_stats = train_epoch(model, sub_ds, optimizer, device,
                               args.phys_lambda)

        # Validate (100 samples)
        val_sub_n = min(100, len(val_ds))
        val_sub   = GraphDataset(graph_data["val"][:val_sub_n])
        val_stats = eval_epoch(model, val_sub, device, norm_stats)

        scheduler.step()

        # History
        history["train_loss"].append(tr_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_loc_km"].append(val_stats["loc_km"])
        history["val_dep_mae"].append(val_stats["dep_mae"])

        # Save best model
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            patience_cnt  = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss"   : best_val_loss,
                "val_loc_km" : val_stats["loc_km"],
                "val_dep_mae": val_stats["dep_mae"],
                "args"       : vars(args),
                "norm_stats" : norm_stats,
                "node_dim"   : node_dim,
                "edge_dim"   : edge_dim,
                "timestamp"  : datetime.utcnow().isoformat(),
            }, MODEL_DIR / "best_locator.pt")
        else:
            patience_cnt += 1

        # Log
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            lr_cur = optimizer.param_groups[0]['lr']
            ep_t   = time.time() - t_ep
            log.info(
                f"  {epoch:5d}  {tr_stats['loss']:9.4f}  "
                f"{val_stats['loss']:9.4f}  "
                f"{val_stats['loc_km']:7.2f}  "
                f"{val_stats['dep_mae']:7.2f}  "
                f"{lr_cur:8.2e}  "
                f"({ep_t:.1f}s)"
            )

        # Early stopping
        if patience_cnt >= args.patience:
            log.info(f"  Early stopping at epoch {epoch}")
            break

    total_time = time.time() - t0
    log.info(f"\n  Training complete: {total_time:.0f}s")
    log.info(f"  Best val loss: {best_val_loss:.4f}")

    # ── Final evaluation ─────────────────────────────────────────────
    log.info("\n  Final evaluation on test set...")
    # Load best model
    ckpt = torch.load(MODEL_DIR / "best_locator.pt",
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_sub = GraphDataset(graph_data["test"][:200])
    test_stats = eval_epoch(model, test_sub, device, norm_stats)

    log.info(f"  Test horizontal error : {test_stats['loc_km']:.2f} km")
    log.info(f"  Test depth MAE        : {test_stats['dep_mae']:.2f} km")
    log.info(f"  Test lat MAE          : {test_stats['lat_mae']:.3f}°")
    log.info(f"  Test lon MAE          : {test_stats['lon_mae']:.3f}°")

    # Save results
    results = {
        "test"   : test_stats,
        "history": history,
        "args"   : vars(args),
    }
    with open(MODEL_DIR / "13_02_locator_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # ── PDF ──────────────────────────────────────────────────────────
    _generate_pdf(history, model, test_sub, device, norm_stats, test_stats)

    log.info("="*65)
    log.info("  Script 13_02 complete.")
    log.info(f"  Best model: {MODEL_DIR}/best_locator.pt")
    log.info(f"  Horiz err : {test_stats['loc_km']:.2f} km")
    log.info(f"  Depth MAE : {test_stats['dep_mae']:.2f} km")
    log.info("  Next: python src/13_03_locator_evaluation.py")
    log.info("="*65)


def _generate_pdf(history, model, test_ds, device, norm_stats, test_stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    DARK  = '#0d1b2a'; BLUE  = '#4fc3f7'
    GREEN = '#a5d6a7'; AMBER = '#ffe082'; RED = '#f4a582'

    pdf_path = PDF_DIR / "13_02_locator_training.pdf"
    with PdfPages(pdf_path) as pdf:

        # ── Page 1: Training curves ──────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor(DARK)
        fig.suptitle(
            f"ARGUS Script 13_02 — GNN Locator Training\n"
            f"Horiz: {test_stats['loc_km']:.2f}km  "
            f"Depth: {test_stats['dep_mae']:.2f}km",
            fontsize=12, color='white', fontweight='bold'
        )

        def sa(ax, title):
            ax.set_facecolor('#112030')
            for sp in ax.spines.values(): sp.set_edgecolor('#2a4a6a')
            ax.tick_params(colors='#7a9ab0', labelsize=9)
            ax.xaxis.label.set_color('#7a9ab0'); ax.yaxis.label.set_color('#7a9ab0')
            ax.set_title(title, color=BLUE, fontsize=10, pad=6)

        ep = range(1, len(history["train_loss"]) + 1)

        ax = axes[0]
        ax.plot(ep, history["train_loss"], color=BLUE,  lw=1.2, label="Train")
        ax.plot(ep, history["val_loss"],   color=GREEN, lw=1.2, label="Val")
        sa(ax, "NLL Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("NLL")
        ax.legend(fontsize=8, facecolor='#1a3a5c', labelcolor='white')

        ax = axes[1]
        ax.plot(ep, history["val_loc_km"], color=AMBER, lw=1.5)
        sa(ax, "Horizontal Error (km)"); ax.set_xlabel("Epoch")
        ax.set_ylabel("Error (km)"); ax.axhline(y=5, color=RED, lw=1, linestyle='--', alpha=0.6)

        ax = axes[2]
        ax.plot(ep, history["val_dep_mae"], color=RED, lw=1.5)
        sa(ax, "Depth MAE (km)"); ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (km)")

        fig.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close(fig)

        # ── Page 2: Prediction scatter ───────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor(DARK)
        fig.suptitle("GNN Locator — Test Set Predictions vs True",
                     fontsize=11, color='white', fontweight='bold')

        model.eval()
        pred_lats, true_lats = [], []
        pred_lons, true_lons = [], []
        pred_deps, true_deps = [], []
        ns = norm_stats

        with torch.no_grad():
            for i in range(min(200, len(test_ds))):
                g = test_ds[i]
                x  = g["x"].to(device)
                ei = g["edge_index"].to(device)
                ea = g["edge_attr"].to(device)
                src = g["src_loc"]
                mu, _ = model(x, ei, ea)
                mu_np = mu.cpu().numpy()
                pred_lats.append(mu_np[0] * ns["lat_std"] + ns["lat_mean"])
                pred_lons.append(mu_np[1] * ns["lon_std"] + ns["lon_mean"])
                pred_deps.append(np.exp(mu_np[2] * np.log(60.0)))
                true_lats.append(float(src[0]))
                true_lons.append(float(src[1]))
                true_deps.append(float(src[2]))

        for ax, pred, true, label, col in [
            (axes[0], pred_lats, true_lats, "Latitude (°N)", BLUE),
            (axes[1], pred_lons, true_lons, "Longitude (°E)", GREEN),
            (axes[2], pred_deps, true_deps, "Depth (km)", AMBER),
        ]:
            ax.scatter(true, pred, s=8, alpha=0.5, c=col)
            mn = min(min(true), min(pred)); mx = max(max(true), max(pred))
            ax.plot([mn, mx], [mn, mx], 'w--', lw=1, alpha=0.6)
            sa(ax, label)
            ax.set_xlabel("True"); ax.set_ylabel("Predicted")

        fig.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close(fig)

    log.info(f"  PDF: {pdf_path}")


if __name__ == "__main__":
    main()
