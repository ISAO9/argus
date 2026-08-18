# README addendum — R2 revision (reproducibility audit)

During revision we audited the provenance of every reported value. Values that
could not be traced to archived artifacts were regenerated under fixed-seed,
single-protocol re-runs and replaced; two implementation defects (an evaluation
path that skipped loading trained encoder weights, and a stratified statistic
that included fine-tuning events) were corrected. Headline changes:

- Locator: 10.3 / 14.7 km → 22.6 / 30.6 km (with 90.3 / 95.7 km zero-shot baselines)
- Conformal: q̂ 4.12σ → 2.31σ at 96.1% coverage (n_cal = 103)
- SWIFT: synthetic 99.4% → 99.9% (test partition); Kumamoto 95.1% → 100.0% Shear
  (consistency check only); FORGE mechanism transfer fails (κ ≈ 0) — reported
- PGA: r = 0.62 belongs to the bias-corrected GMPE (used_nami = false), not the
  FNO; direct comparison added (FNO 0.38 vs GMPE 0.78)
- Latency: measured primary benchmark; title corrected 17 ms → 10 ms
- Removed: former Fig. 7c; Wilcoxon statistics; two ablation rows (conditions
  not implemented in the release)

See REPRODUCIBILITY.md for the claim→script→artifact map.
