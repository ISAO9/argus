# ARGUS

**A**utomated **R**eal-time **G**eophysical **U**nderstanding **S**ystem — a **10-ms**
end-to-end deep-learning pipeline that jointly estimates **hypocenter location**,
**source-mechanism class**, and a **128×128 peak-ground-acceleration (PGA)
field** from sparse seismic networks — a feasibility study toward
induced-seismicity monitoring in enhanced geothermal systems (EGS) and carbon
capture and storage (CCS).

This is the reference implementation accompanying:

> Kurosawa, I. *ARGUS: A 10-ms End-to-End Deep Learning Pipeline for Joint
> Earthquake Location, Source-Mechanism Classification, and Ground-Motion
> Prediction from Sparse Networks — A Feasibility Study toward
> Induced-Seismicity Monitoring.* Seismological Research Letters (in revision,
> 2026).

> **Scope note (updated in the R2 revision).** The paper establishes the
> feasibility of the integrated low-latency architecture (measured **10.1 ms**
> median end-to-end; CPU-only 75.7 ms) and provides a systematic quantification
> of its **synthetic-to-field transfer limits**: synthetic pretraining alone
> does not transfer (zero-shot ≈90 km median on K-NET); limited real-data
> fine-tuning recovers regional-scale accuracy (22.6–30.6 km median); and a
> cross-campaign Utah FORGE evaluation (train 2019 → test 2024 against the
> Niemz et al. 2026 moment-tensor catalog) shows that neither sub-kilometre
> cluster structure nor mechanism discrimination is resolved with the current
> 100-Hz feature design. It does **not** demonstrate reservoir-scale accuracy.
> Trained weights and processed validation data are archived on Zenodo
> (concept DOI [10.5281/zenodo.21051516](https://doi.org/10.5281/zenodo.21051516)).

> **Reproducibility audit (R2, 2026-08).** Every value reported in the revised
> paper was regenerated under fixed-seed, single-protocol re-runs; values that
> could not be traced to archived artifacts were replaced, and two
> implementation defects were corrected. Summary:
> [README_ADDENDUM_R2.md](README_ADDENDUM_R2.md). Claim→script→artifact map:
> [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The audit and regeneration scripts
> live under `src/` (`13_xx` experiments, `14_xx` figures/benchmarks) with their
> JSON outputs under `models/` and `logs/`.

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
| Conformal   | `src/argus/conformal.py`          | n_cal 103, nominal 90% (q̂ = 2.31σ) |

Full hyperparameters: `configs/argus.yaml` (mirrors Table S2 in the electronic
supplement).

## Project layout

```
argus/
├── configs/argus.yaml          # all hyperparameters & paths
├── src/argus/                  # importable package (models, graph, conformal, pipeline)
├── src/13_xx_*.py, 14_xx_*.py  # R2 audit: experiment / figure / benchmark scripts
├── scripts/                    # numbered, self-documenting entry points (00–10)
├── tests/test_shapes.py        # fast end-to-end shape / key-consistency test
├── models/, logs/              # trained weights (Zenodo) + audited result JSONs
├── data/{raw,processed}/       # public datasets (not redistributed here)
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
| Utah FORGE 2019/2024 stimulation| https://gdr.openei.org/ (DOE GDR) + Niemz et al. (2026) MT catalog |
| J-SHIS subsurface velocity      | https://www.j-shis.bosai.go.jp/                   |

Run `python scripts/01_download_data.py` to see exactly which files are expected
and where to place them.

## Reproduction

Pipeline training / inference (original entry points):

```bash
python scripts/01_download_data.py        # inventory / fetch guidance
python scripts/02_build_graphs.py         # raw -> PyG graph tensors
python scripts/03_train_gnn_locator.py    # 80 epochs -> models/best_locator.pt
python scripts/04_finetune_locator_knet.py# fine-tune @5e-5 (real-data)
python scripts/05_train_swift_cmt.py      # -> models/checkpoint_epoch_080.pt
python scripts/06_train_fno_nami.py       # -> models/fno_best.pth
python scripts/07_conformal_calibration.py# -> models/conformal.json
python scripts/08_run_pipeline.py         # end-to-end inference (one event)
python scripts/09_latency_benchmark.py    # per-stage + total latency
python scripts/10_make_figures.py         # figures -> PDF/
pytest -q                                 # shape / key-consistency tests
```

**Paper numbers (R2).** Every number in the revised manuscript is regenerated by
the audited scripts under `src/` — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for the one-to-one claim→script→artifact map. Key entry points:
`13_34_random_split_eval.py` (locator, both splits + conformal),
`13_05_ablation_study_2.py` (ablation), `13_19_forge_field_validation.py`
(FORGE cross-campaign + local-scale control), `13_35/13_39` (mechanism
zero-shot / fine-tune vs. true MT labels), `13_29_fno_vs_gmpe.py` (PGA vs.
observations), `14_15_latency_benchmark.py` (measured latency), and
`13_33_verify_checkpoints.py` (weight–code compatibility check; run this
first). Figures: `14_10`–`14_16`.

Every training script saves **only the best checkpoint** seen during the epoch
loop (project convention).

### Reused assets

These filenames are carried over from prior ARGUS development and are referenced
in the script headers: `data/processed/hinet_graph_v2.pt` (station graph),
`data/processed/knet_dataset.h5` (FNO-NAMI validation DB), and the checkpoints
`models/best_locator.pt`, `models/checkpoint_epoch_080.pt`, `models/fno_best.pth`.

## Citation

See `CITATION.cff`.
