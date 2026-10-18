# IASL Phase 0C SAE reproducibility

Pooled PCA input `1360` -> `128` dimensions.
Individual latent matching is diagnostic; CKA and principal-angle summaries are the primary group/subspace checks.

- width 256: feature max-cosine 0.258 (null p95 0.250), overlap 0.000, CKA 0.973, principal angle 0.0°
- width 512: feature max-cosine 0.281 (null p95 0.268), overlap 0.000, CKA 0.989, principal angle 0.0°

Tiny temporal-convolution control (seeded frozen weights):
- width 256: feature max-cosine 0.254 (null p95 0.250), CKA 0.996 (row-permutation null p95 0.971)
- width 512: feature max-cosine 0.282 (null p95 0.268), CKA 0.998 (row-permutation null p95 0.965)
