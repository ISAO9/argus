# ARGUS — Reproducibility Map (R2 revision, 2026-08-18)

Every number in the revised manuscript traces to a script and an archived artifact.
This file is the public counterpart of the internal numbers ledger.

| Manuscript claim | Script | Artifact |
|---|---|---|
| Locator zero-shot 90.3 / 95.7 km; fine-tuned 22.6 km (random, 5 seeds) / 30.6 km (temporal, 3 seeds); P90s | `13_34_random_split_eval.py` (`--split random/temporal`, seeds 42/7/123/1234/2026) | `models/locator/13_34_*_results*.json`, summary `13_37_seed_summary.json` |
| Conformal q̂ = 2.31σ (2.23–2.42), coverage 96.1% (94.2–96.1), n_cal = 103 | same + `14_13_fig5_conformal.py` (independent recomputation, Δ = 0.002σ) | same JSONs |
| Mw-stratified Table 2 (temporal test only) | `13_34 --split temporal` | `13_34_temporal_results_seed*.json` (`mw_stratified_test_only`) |
| Ablation Table 3 (101.0 / 222.5 / 71.8 / 54.7 km; zero-shot transfer, n = 104) | `13_05_ablation_study_2.py --epochs 40` | `logs/13_05_rerun.log` (figure parses the log directly: `14_11`) |
| Trained-vs-random encoder (88.5 vs 192.9 km) | `13_38_encoder_diagnosis.py` | `13_38_diagnosis.json` |
| SWIFT synthetic 99.9% (n = 2,000), f_ISO R² = 0.888 | `05_swift_evaluation_3.py` | `models/05_test_results.json` |
| Kumamoto zero-shot 100.0% Shear (n = 300) | `13_36_swift_kumamoto_eval.py` | `models/13_36_kumamoto_transfer.json` |
| FORGE zero-shot collapse (26.4%, f_ISO r = 0.08) | `13_35_swift_forge_mt_eval.py` | `models/13_35_forge_mt_eval.json` |
| FORGE fine-tune κ = −0.03 (pre-declared protocol) | `13_39_swift_forge_finetune.py` | `models/13_39_swift_forge_ft.json` |
| FORGE locator: strict fail / 129.3 → 30.8 km / local-scale 0.27 vs baseline 0.24 km; q̂ = 1.89, 97.3% | `13_19_forge_field_validation.py` (v3.2; `--local_scale` for the control) | `models/locator/13_19_forge_results*.json` |
| PGA attribution (used_nami = false) and corrected-GMPE 0.62 / 55 gal / 2,892 | archived `13_08` run | `models/locator/13_08_bias_coeff.json` |
| FNO 0.38 vs GMPE 0.78 (5 events, 1,008 cells) | `13_29_fno_vs_gmpe.py` | `models/13_29_fno_vs_gmpe.json` |
| Latency 1.5 / 6.0 / 3.0 ms; end-to-end 10.1 ms (MPS), 75.7 ms (CPU) | `14_15_latency_benchmark.py` | `models/14_15_latency.json` |
| Figures 2–7 | `14_10`–`14_16` | `PDF/14_1*.pdf` (inputs = the JSONs above only) |
| Weight–code compatibility of the public release | `13_33_verify_release.py` | PASS log |

Known limitations disclosed in the manuscript: FORGE picker fallback rates
(train 53% / test 12%); h5 row-order-as-chronology assumption in `13_39`;
`10_benchmark_metrics.json` is a side run with a different f_ISO definition and
is not used in the paper.
