# ARGUS

**A**utomated **R**eal-time **G**eophysical **U**nderstanding **S**ystem — a 17-ms
end-to-end deep-learning pipeline that jointly estimates **hypocenter location**,
**centroid moment tensor (CMT)**, and a **128×128 peak-ground-acceleration (PGA)
field** from sparse seismic networks, for induced-seismicity monitoring in
enhanced geothermal systems (EGS) and carbon capture and storage (CCS).

This is the reference implementation accompanying:

> Kurosawa, I. *ARGUS: A 17-ms End-to-End Deep Learning Pipeline for Real-Time
> Seismic Source Characterization and Ground Motion Prediction in Sparse-Network
> EGS/CCS Environments.* Seismological Research Letters (submitted, 2026).

> **Scope note.** The real-data validation in the paper is at **regional scale**
> (median location error 10–15 km) and establishes the *feasibility* of the
> integrated low-latency architecture across the EGS-relevant magnitude range
> (Mw 2.6–4.0). It does **not** demonstrate reservoir-scale location accuracy;
> direct validation on Utah FORGE Phase 2C microseismicity is the documented
> next step. Trained weights and processed validation data are released via
> Zenodo (DOI on acceptance); exact paper numbers require those artifacts.

---

## Architecture

```
 waveforms (4–8 stations)
        │
        ▼
 GNN-Locator  ── GATv2 ×4, hidden 128, 4 heads ──► (lat, lon, depth) + conformal radius
        │
        ▼
 SWIFT CMT    ── SWIFTNetV8 (wave enc + spectral + GATv2 ×3) ──► mechanism / f_ISO / Mw
        │
        ▼
 FNO-NAMI     ── FNO2d, 64 lift ch, 16 modes ──► 128×128 PGA map (gal)
```

| Component   | File                              | Key hyperparameters            |
|-------------|-----------------------------------|--------------------------------|
| GNN-Locator | `src/argus/models/gnn_locator.py` | hidden 128, heads 4, layers 4  |
| SWIFT CMT   | `src/argus/models/swift_cmt.py`   | base ch 16, F-bins 32, layers 3|
| FNO-NAMI    | `src/argus/models/fno_nami.py`    | lift 64, modes 16, 4 layers    |
| Conformal   | `src/argus/conformal.py`          | n_cal 138, nominal 90%         |

Full hyperparameters: `configs/argus.yaml` (mirrors Table S2 in the electronic
supplement).

## Project layout

```
argus/
├── configs/argus.yaml          # all hyperparameters & paths
├── src/argus/                  # importable package (models, graph, conformal, pipeline)
├── scripts/                    # numbered, self-documenting entry points (00–10)
├── tests/test_shapes.py        # fast end-to-end shape / key-consistency test
├── data/{raw,processed}/       # public datasets (not redistributed here)
├── models/                     # trained weights land here (Zenodo)
└── PDF/                        # figures are written here as vector PDFs
```

## Setup (uv)

```bash
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e .
```

PyTorch Geometric may need wheels matched to your CUDA/torch build — see the
[PyG install guide](https://pytorch-geometric.readthedocs.io/).

## Data (all public)

| Dataset                         | Source                                            |
|---------------------------------|---------------------------------------------------|
| K-NET / KiK-net strong motion   | https://www.kyoshin.bosai.go.jp/                  |
| Hi-net high-sensitivity         | https://www.hinet.bosai.go.jp/                    |
| JMA unified hypocenter catalog  | https://www.data.jma.go.jp/svd/eqev/data/bulletin/|
| Utah FORGE Phase 2C + vel. model| https://gdr.openei.org/ (DOE GDR Sub. 1107)       |
| J-SHIS subsurface velocity      | https://www.j-shis.bosai.go.jp/                   |

Run `python scripts/01_download_data.py` to see exactly which files are expected
and where to place them.

## Reproduction

```bash
python scripts/01_download_data.py        # inventory / fetch guidance
python scripts/02_build_graphs.py         # raw -> PyG graph tensors
python scripts/03_train_gnn_locator.py    # 80 epochs -> models/best_locator.pt
python scripts/04_finetune_locator_knet.py# 30 epochs @5e-5 (real-data)
python scripts/05_train_swift_cmt.py      # -> models/checkpoint_epoch_080.pt
python scripts/06_train_fno_nami.py       # -> models/fno_best.pth
python scripts/07_conformal_calibration.py# -> models/conformal.json
python scripts/08_run_pipeline.py         # end-to-end inference (one event)
python scripts/09_latency_benchmark.py    # per-stage + total latency
python scripts/10_make_figures.py         # figures -> PDF/
pytest -q                                 # shape / key-consistency tests
```

Every training script saves **only the best checkpoint** seen during the epoch
loop (project convention).

### Reused assets

These filenames are carried over from prior ARGUS development and are referenced
in the script headers: `data/processed/hinet_graph_v2.pt` (station graph),
`data/processed/knet_dataset.h5` (FNO-NAMI validation DB), and the checkpoints
`models/best_locator.pt`, `models/checkpoint_epoch_080.pt`, `models/fno_best.pth`.

## Citation

See `CITATION.cff`.

## License

Apache License 2.0 — see `LICENSE`.
