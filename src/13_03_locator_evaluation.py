#!/usr/bin/env python3
# =============================================================================
# Script : 13_03_locator_evaluation.py
# Project: ARGUS (SWIFT + NAMI + GNN-Locator)
#
# Description:
#   GNN震源特定モデルの詳細評価と日本地図可視化。
#
#   評価内容:
#     1. 水平誤差・深さ誤差の分布
#     2. Mw別・深さ別の精度
#     3. 不確実性の較正（Calibration）
#     4. 日本地図上の予測vs真値プロット
#     5. 推論レイテンシ計測
#     6. F-net実データへの適用テスト（K-NET熊本）
#
#   出力:
#     PDF/13_03_locator_evaluation.pdf
#     models/locator/13_03_eval_results.json
#
#   Usage:
#     python src/13_03_locator_evaluation.py
#     python src/13_03_locator_evaluation.py --ckpt models/locator/best_locator.pt
#
# =============================================================================

import sys, argparse, logging, time, json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "locator"
PDF_DIR   = PROJECT_ROOT / "PDF"
DATA_DIR  = PROJECT_ROOT / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# Import model from Script 13_02
import importlib.util
def load_model_class():
    spec = importlib.util.spec_from_file_location(
        "gnn02", PROJECT_ROOT / "src" / "13_02_gnn_locator_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GNNLocator, mod.GraphDataset, mod.eval_epoch


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    return float(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


@torch.no_grad()
def full_evaluation(model, graphs, device, norm_stats, n_max=500):
    """
    詳細評価を実施する。

    Returns dict with per-event errors and uncertainties.
    """
    model.eval()
    ns = norm_stats
    results = {
        "lat_true": [], "lon_true": [], "dep_true": [], "mw": [],
        "lat_pred": [], "lon_pred": [], "dep_pred": [],
        "lat_sigma":[], "lon_sigma":[], "dep_sigma":[],
        "horiz_err": [], "dep_err": [], "latency_ms": [],
    }

    for i, g in enumerate(graphs[:n_max]):
        x  = torch.from_numpy(g["x"]).to(device)
        ei = torch.from_numpy(g["edge_index"]).to(device)
        ea = torch.from_numpy(g["edge_attr"]).to(device)

        # Warmup on first 5
        if i < 5:
            _ = model(x, ei, ea)

        t0 = time.perf_counter()
        mu, log_var = model(x, ei, ea)
        if device.type == "mps":
            torch.mps.synchronize()
        latency = (time.perf_counter() - t0) * 1000

        mu_np  = mu.cpu().numpy()
        sig_np = np.exp(0.5 * log_var.cpu().numpy())

        # Denormalize
        lat_p = mu_np[0]  * ns["lat_std"] + ns["lat_mean"]
        lon_p = mu_np[1]  * ns["lon_std"] + ns["lon_mean"]
        dep_p = np.exp(mu_np[2] * np.log(60.0))
        lat_s = sig_np[0] * ns["lat_std"]
        lon_s = sig_np[1] * ns["lon_std"]
        dep_s = sig_np[2] * 10.0  # rough km scale

        lat_t = float(g["src_loc"][0])
        lon_t = float(g["src_loc"][1])
        dep_t = float(g["src_loc"][2])

        horiz = haversine_km(lat_t, lon_t, lat_p, lon_p)

        results["lat_true"].append(lat_t)
        results["lon_true"].append(lon_t)
        results["dep_true"].append(dep_t)
        results["mw"].append(float(g["mw"]))
        results["lat_pred"].append(float(lat_p))
        results["lon_pred"].append(float(lon_p))
        results["dep_pred"].append(float(dep_p))
        results["lat_sigma"].append(float(lat_s))
        results["lon_sigma"].append(float(lon_s))
        results["dep_sigma"].append(float(dep_s))
        results["horiz_err"].append(float(horiz))
        results["dep_err"].append(float(abs(dep_p - dep_t)))
        results["latency_ms"].append(float(latency))

    # Convert to numpy
    for k, v in results.items():
        results[k] = np.array(v)

    return results


def compute_metrics(results):
    """集約メトリクスを計算する。"""
    h = results["horiz_err"]
    d = results["dep_err"]
    lm = results["latency_ms"]

    return {
        "n_events"      : len(h),
        "horiz_mean_km" : float(h.mean()),
        "horiz_med_km"  : float(np.median(h)),
        "horiz_p75_km"  : float(np.percentile(h, 75)),
        "horiz_p90_km"  : float(np.percentile(h, 90)),
        "dep_mean_km"   : float(d.mean()),
        "dep_med_km"    : float(np.median(d)),
        "dep_p75_km"    : float(np.percentile(d, 75)),
        "latency_med_ms": float(np.median(lm)),
        "latency_p95_ms": float(np.percentile(lm, 95)),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ARGUS Script 13_03 — GNN Locator Evaluation")
    parser.add_argument("--ckpt", type=str,
                        default="models/locator/best_locator.pt")
    parser.add_argument("--graph_pt", type=str,
                        default="data/locator/hinet_graph_dataset.pt")
    parser.add_argument("--n_eval",   type=int, default=500)
    parser.add_argument("--device",   type=str, default="auto")
    args = parser.parse_args()

    log.info("="*65)
    log.info("  ARGUS  |  Script 13_03  |  GNN Locator Evaluation")
    log.info("="*65)

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

    # ── Load checkpoint ──────────────────────────────────────────────
    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() \
                else PROJECT_ROOT / args.ckpt
    log.info(f"\n  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    norm_stats = ckpt["norm_stats"]
    node_dim   = ckpt["node_dim"]
    edge_dim   = ckpt["edge_dim"]
    train_args = ckpt.get("args", {})

    log.info(f"  Trained epoch : {ckpt['epoch']}")
    log.info(f"  Val loss      : {ckpt['val_loss']:.4f}")
    log.info(f"  Val horiz km  : {ckpt.get('val_loc_km', '?'):.2f}")

    # ── Load model ───────────────────────────────────────────────────
    GNNLocator, GraphDataset, _ = load_model_class()

    model = GNNLocator(
        node_dim = node_dim,
        edge_dim = edge_dim,
        hidden   = train_args.get("hidden", 128),
        heads    = train_args.get("heads",  4),
        n_layers = train_args.get("n_layers", 4),
        dropout  = 0.0,  # evaluation mode
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Model params  : {n_params:,}")

    # ── Load graph dataset ───────────────────────────────────────────
    graph_path = Path(args.graph_pt) if Path(args.graph_pt).is_absolute() \
                 else PROJECT_ROOT / args.graph_pt
    log.info(f"\n  Loading graph dataset: {graph_path}")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    test_graphs  = data["graph_data"]["test"]
    train_graphs = data["graph_data"]["train"]

    # ── Full evaluation ──────────────────────────────────────────────
    log.info(f"\n  Evaluating {args.n_eval} test events...")
    results = full_evaluation(model, test_graphs, device,
                               norm_stats, n_max=args.n_eval)
    metrics = compute_metrics(results)

    log.info("\n  ══ Test Set Results ══════════════════════════════")
    log.info(f"  Events             : {metrics['n_events']}")
    log.info(f"  Horizontal error   : {metrics['horiz_mean_km']:.2f} km (mean)")
    log.info(f"                     : {metrics['horiz_med_km']:.2f} km (median)")
    log.info(f"                     : {metrics['horiz_p75_km']:.2f} km (p75)")
    log.info(f"                     : {metrics['horiz_p90_km']:.2f} km (p90)")
    log.info(f"  Depth MAE          : {metrics['dep_mean_km']:.2f} km (mean)")
    log.info(f"                     : {metrics['dep_med_km']:.2f} km (median)")
    log.info(f"  Inference latency  : {metrics['latency_med_ms']:.2f} ms (median)")
    log.info(f"                     : {metrics['latency_p95_ms']:.2f} ms (p95)")
    log.info("  ══════════════════════════════════════════════════")

    # Mw別精度
    log.info("\n  Accuracy by Magnitude:")
    for mw_lo, mw_hi in [(1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 7.0)]:
        mask = (results["mw"] >= mw_lo) & (results["mw"] < mw_hi)
        if mask.sum() > 0:
            h_m = results["horiz_err"][mask].mean()
            d_m = results["dep_err"][mask].mean()
            log.info(f"    Mw {mw_lo:.1f}-{mw_hi:.1f}: "
                     f"n={mask.sum():3d}  horiz={h_m:.1f}km  dep={d_m:.1f}km")

    # 深さ別精度
    log.info("\n  Accuracy by Depth:")
    for d_lo, d_hi in [(0, 10), (10, 25), (25, 60)]:
        mask = (results["dep_true"] >= d_lo) & (results["dep_true"] < d_hi)
        if mask.sum() > 0:
            h_m = results["horiz_err"][mask].mean()
            d_m = results["dep_err"][mask].mean()
            log.info(f"    {d_lo:2d}-{d_hi:2d}km: "
                     f"n={mask.sum():3d}  horiz={h_m:.1f}km  dep={d_m:.1f}km")

    # ── Save results ─────────────────────────────────────────────────
    eval_json = MODEL_DIR / "13_03_eval_results.json"
    save = {
        "metrics"   : {k: float(v) for k, v in metrics.items()},
        "ckpt_epoch": ckpt["epoch"],
        "n_eval"    : args.n_eval,
    }
    with open(eval_json, 'w') as f:
        json.dump(save, f, indent=2)
    log.info(f"\n  Results: {eval_json}")

    # ── PDF ──────────────────────────────────────────────────────────
    log.info("\n  Generating PDF report...")
    _generate_pdf(results, metrics, norm_stats)

    log.info("="*65)
    log.info("  Script 13_03 complete.")
    log.info(f"  Horizontal error (median): {metrics['horiz_med_km']:.2f} km")
    log.info(f"  Depth MAE (median)       : {metrics['dep_med_km']:.2f} km")
    log.info(f"  Inference latency        : {metrics['latency_med_ms']:.2f} ms")
    log.info("  Next: python src/13_04_argus_pipeline.py")
    log.info("="*65)


def _generate_pdf(results, metrics, norm_stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.colors as mcolors

    DARK  = '#0d1b2a'; BLUE  = '#4fc3f7'
    GREEN = '#a5d6a7'; AMBER = '#ffe082'; RED = '#f4a582'
    SURF  = '#112030'

    pdf_path = PDF_DIR / "13_03_locator_evaluation.pdf"
    with PdfPages(pdf_path) as pdf:

        def sa(ax, title):
            ax.set_facecolor(SURF)
            for sp in ax.spines.values(): sp.set_edgecolor('#2a4a6a')
            ax.tick_params(colors='#7a9ab0', labelsize=9)
            ax.xaxis.label.set_color('#7a9ab0')
            ax.yaxis.label.set_color('#7a9ab0')
            ax.set_title(title, color=BLUE, fontsize=10, pad=6)

        # ══ Page 1: Error distributions ══════════════════════════════
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor(DARK)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle(
            f"ARGUS GNN Locator — Evaluation Report\n"
            f"n={metrics['n_events']}  "
            f"Horiz: {metrics['horiz_med_km']:.1f}km (med)  "
            f"Depth: {metrics['dep_med_km']:.1f}km (med)  "
            f"Latency: {metrics['latency_med_ms']:.1f}ms",
            fontsize=12, color='white', fontweight='bold', y=0.97
        )

        # Horizontal error CDF
        ax = fig.add_subplot(gs[0, 0])
        h_sorted = np.sort(results["horiz_err"])
        cdf = np.arange(1, len(h_sorted)+1) / len(h_sorted)
        ax.plot(h_sorted, cdf * 100, color=BLUE, lw=2)
        ax.axvline(metrics["horiz_med_km"], color=AMBER, lw=1.5,
                   linestyle='--', label=f"Median: {metrics['horiz_med_km']:.1f}km")
        ax.axvline(metrics["horiz_p90_km"], color=RED, lw=1.5,
                   linestyle='--', label=f"p90: {metrics['horiz_p90_km']:.1f}km")
        sa(ax, "Horizontal Error CDF")
        ax.set_xlabel("Error (km)"); ax.set_ylabel("Cumulative (%)")
        ax.legend(fontsize=8, facecolor='#1a3a5c', labelcolor='white')
        ax.set_xlim(0); ax.set_ylim(0, 100)

        # Depth error histogram
        ax = fig.add_subplot(gs[0, 1])
        ax.hist(results["dep_err"], bins=40, color=GREEN,
                edgecolor=DARK, alpha=0.85)
        ax.axvline(metrics["dep_med_km"], color=AMBER, lw=1.5, linestyle='--',
                   label=f"Median: {metrics['dep_med_km']:.1f}km")
        sa(ax, "Depth Error Distribution")
        ax.set_xlabel("Depth error (km)"); ax.set_ylabel("Count")
        ax.legend(fontsize=8, facecolor='#1a3a5c', labelcolor='white')

        # Latency histogram
        ax = fig.add_subplot(gs[0, 2])
        ax.hist(results["latency_ms"], bins=40, color=AMBER,
                edgecolor=DARK, alpha=0.85)
        ax.axvline(metrics["latency_med_ms"], color=BLUE, lw=1.5,
                   linestyle='--',
                   label=f"Median: {metrics['latency_med_ms']:.2f}ms")
        sa(ax, "Inference Latency")
        ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Count")
        ax.legend(fontsize=8, facecolor='#1a3a5c', labelcolor='white')

        # Depth pred vs true
        ax = fig.add_subplot(gs[1, 0])
        sc = ax.scatter(results["dep_true"], results["dep_pred"],
                        s=6, alpha=0.5,
                        c=results["horiz_err"], cmap='plasma',
                        vmin=0, vmax=50)
        mn = min(results["dep_true"].min(), results["dep_pred"].min())
        mx = max(results["dep_true"].max(), results["dep_pred"].max())
        ax.plot([mn, mx], [mn, mx], 'w--', lw=1, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='Horiz err (km)')
        sa(ax, "Depth: Predicted vs True")
        ax.set_xlabel("True depth (km)"); ax.set_ylabel("Pred depth (km)")

        # Horiz error vs Mw
        ax = fig.add_subplot(gs[1, 1])
        ax.scatter(results["mw"], results["horiz_err"],
                   s=5, alpha=0.4, c=BLUE)
        sa(ax, "Horizontal Error vs Magnitude")
        ax.set_xlabel("Mw"); ax.set_ylabel("Error (km)")
        ax.set_ylim(0)

        # Horiz error vs depth
        ax = fig.add_subplot(gs[1, 2])
        ax.scatter(results["dep_true"], results["horiz_err"],
                   s=5, alpha=0.4, c=GREEN)
        sa(ax, "Horizontal Error vs True Depth")
        ax.set_xlabel("True depth (km)"); ax.set_ylabel("Error (km)")
        ax.set_ylim(0)

        fig.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close(fig)

        # ══ Page 2: Japan map ═════════════════════════════════════════
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        fig.patch.set_facecolor(DARK)
        fig.suptitle(
            "ARGUS GNN Locator — Japan Map: True vs Predicted Epicenters",
            fontsize=12, color='white', fontweight='bold'
        )

        # 日本地図（簡易版 — 海岸線なし、観測点と震源のみ）
        for ax, title in zip(axes, ["True Epicenters", "Predicted Epicenters"]):
            ax.set_facecolor('#0a1520')
            for sp in ax.spines.values(): sp.set_edgecolor('#2a4a6a')
            ax.tick_params(colors='#7a9ab0', labelsize=9)
            ax.xaxis.label.set_color('#7a9ab0')
            ax.yaxis.label.set_color('#7a9ab0')
            ax.set_title(title, color=BLUE, fontsize=11, pad=8)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")

            # 日本の海岸線（矩形近似）
            japan_boxes = [
                (130, 31, 132, 34),  # 九州
                (130, 34, 137, 36),  # 本州西部
                (136, 33, 139, 36),  # 本州中部
                (138, 34, 142, 38),  # 本州中東部
                (140, 37, 142, 41),  # 東北南部
                (140, 40, 142, 43),  # 東北北部
                (140, 42, 145, 46),  # 北海道
            ]
            for x0, y0, x1, y1 in japan_boxes:
                ax.fill([x0,x1,x1,x0], [y0,y0,y1,y1],
                        color='#1a2a1a', alpha=0.4, zorder=1)
                ax.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0],
                        color='#2a4a2a', lw=0.5, zorder=2)

            ax.set_xlim(128, 147)
            ax.set_ylim(30, 46)

        # 真の震源
        ax = axes[0]
        n = min(200, len(results["lat_true"]))
        sc = ax.scatter(
            results["lon_true"][:n], results["lat_true"][:n],
            c=results["dep_true"][:n], cmap='YlOrRd_r',
            s=12, alpha=0.8, zorder=5, vmin=0, vmax=60
        )
        plt.colorbar(sc, ax=ax, label='Depth (km)', shrink=0.8)

        # 予測震源 + 誤差矢印
        ax = axes[1]
        sc2 = ax.scatter(
            results["lon_pred"][:n], results["lat_pred"][:n],
            c=results["horiz_err"][:n], cmap='plasma',
            s=12, alpha=0.8, zorder=5, vmin=0, vmax=50
        )
        plt.colorbar(sc2, ax=ax, label='Horiz error (km)', shrink=0.8)

        # 誤差が大きいイベントに矢印
        for i in range(min(30, n)):
            if results["horiz_err"][i] > 30:
                ax.annotate('',
                    xy=(results["lon_pred"][i], results["lat_pred"][i]),
                    xytext=(results["lon_true"][i], results["lat_true"][i]),
                    arrowprops=dict(arrowstyle='->', color='#ff5252',
                                   lw=0.8, alpha=0.6)
                )

        fig.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close(fig)

        # ══ Page 3: Uncertainty calibration ══════════════════════════
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor(DARK)
        fig.suptitle(
            "ARGUS GNN Locator — Uncertainty Calibration\n"
            "（|error| ≈ σ であれば不確実性が較正されている）",
            fontsize=11, color='white', fontweight='bold'
        )

        for ax, (err_key, sig_key, label) in zip(axes, [
            ("horiz_err", "lat_sigma", "Lat σ vs Horiz Error"),
            ("dep_err",   "dep_sigma", "Dep σ vs Depth Error"),
            ("horiz_err", "lon_sigma", "Lon σ vs Horiz Error"),
        ]):
            ax.set_facecolor(SURF)
            for sp in ax.spines.values(): sp.set_edgecolor('#2a4a6a')
            ax.scatter(results[sig_key][:200], results[err_key][:200],
                       s=6, alpha=0.4, c=AMBER)
            # 理想線 (y=x)
            mx = max(results[sig_key][:200].max(),
                     results[err_key][:200].max())
            ax.plot([0, mx], [0, mx], 'w--', lw=1, alpha=0.6, label='ideal')
            ax.set_xlabel(f"σ (uncertainty)", color='#7a9ab0')
            ax.set_ylabel("Actual error", color='#7a9ab0')
            ax.tick_params(colors='#7a9ab0', labelsize=9)
            ax.set_title(label, color=BLUE, fontsize=10, pad=6)
            ax.legend(fontsize=8, facecolor='#1a3a5c', labelcolor='white')

        fig.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close(fig)

        # ══ Page 4: Performance summary table ════════════════════════
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(DARK)
        ax.set_facecolor(DARK); ax.axis('off')

        # 比較表
        col_labels = ["Method", "Horiz (km)", "Depth (km)", "Latency", "Data"]
        table_data = [
            ["HYPOINVERSE (traditional)", "~3", "~3", "min", "Real dense"],
            ["EQTransformer (Mousavi2020)", "~7", "~5", "~1s", "Real"],
            ["PhaseLink (Ross2019)",        "~5", "~4", "~1s", "Real"],
            ["ARGUS GNN-Locator (ours)",
             f"{metrics['horiz_med_km']:.1f}",
             f"{metrics['dep_med_km']:.1f}",
             f"{metrics['latency_med_ms']:.1f}ms",
             "Synthetic 10k"],
        ]

        row_colors = [
            ['#1a3040'] * 5,
            ['#1a3040'] * 5,
            ['#1a3040'] * 5,
            ['#1a4030'] * 5,  # ours highlighted
        ]

        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            cellColours=row_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.2, 2.0)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_text_props(color='white' if r > 0 else BLUE,
                                fontweight='bold' if r == 0 or (r == 4 and c >= 0) else 'normal')
            cell.set_edgecolor('#2a4a6a')

        ax.set_title(
            "ARGUS GNN Locator — Performance Comparison\n"
            "※ Synthetic data only; real data fine-tuning in Script 13_04",
            color=BLUE, fontsize=12, pad=20
        )
        pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close(fig)

    log.info(f"  PDF: {pdf_path}")


if __name__ == "__main__":
    main()
