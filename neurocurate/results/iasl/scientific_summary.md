# IASL feasibility summary

## Scope

Phase 0A--0D ran on five verified THINGS-EEG2 numeric exports, 50 shared repeated test images and four repetitions per image. No new EEG or model was downloaded. The export contains samples from -200 to 790 ms; the requested 0--1000 ms window was therefore truncated to the available 0--790 ms interval.

## Results

- Median image split-half reliability by subject: 0.057, 0.058, 0.000, 0.048, 0.031; range 0.000--0.058. This is low and near the predeclared reliability stop rule.
- PCA control: 1,360 flattened dimensions -> 128. SAE widths 256 and 512 had feature max-cosine 0.258 and 0.281, with 0.000 feature overlap above cosine 0.8; both are at or near their random-direction null p95.
- Subspace CKA was high (0.973, 0.989) and exceeded the row-permutation null in the tiny frozen-convolution control (null p95 0.971/0.965). This supports a reproducible pooled subspace signal, while the very small feature overlap argues against feature identity claims.
- Intervention audit: 60 raw/normalized ablations; median C_raw 0.0041, median C_norm 2.69. The energy-matched permutation null medians are 4.06, 2.314; these small audits do not establish class selectivity.

## Decision

**NO-GO for Phase 1 training or a GPU-intensive IASL alignment claim.** The data pipeline and subspace-level audit are usable, but repeated-trial reliability is low and individual SAE matching is indistinguishable from its null. A defensible follow-up is an analysis paper on subspace reproducibility and intervention limits, followed by a separately preregistered, session-separated pilot only after reliability and preprocessing sensitivity improve. No held-out-subject generalization or data-efficiency claim is made.
