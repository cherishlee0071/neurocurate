# Phase 1.4: real subject-01/02 replication

**GO for a limited, prospectively specified expansion to additional subjects. NO-GO for treating the current Top-20 as a validated curation list or starting training/OOD evaluation.**

## Acquisition and frozen replication

Only the official standard subject-02 ZIP was downloaded. OSF API metadata, obtained separately over verified TLS, specified 1,071,516,660 bytes and SHA256 `e1eba2d74009c6dd41becf53d84b8d47532103b41110a5855e7e286bcca6fe41`. The complete downloaded archive matched both. This is complete archive verification, not a Range/metadata-only claim. Release file version 1, dated 2022-08-12.

Only `sub-02/preprocessed_eeg_test.npy` was extracted: 217,601,432 bytes, ZIP-compressed size 208,591,064 bytes, SHA256 `e5ff0774a83beb49366a73c73fb04c9e692d5ab1f80c3d57c9fd1ff6e110d791`; ZIP CRC verification passed. The source ZIP and NPY are retained. The previously verified metadata and 200 stimulus JPEGs were reused; no training EEG, new images, new checkpoint, or additional subject was acquired.

The object-array conversion ran as UID 1000 in the same descriptor-only, default-deny seccomp process as subject-01. Before deserialization, filesystem-open, network-socket, and process-creation probes were denied. Input hashes were checked both in the parent and confined child. Limits: 4 GiB address space, 120 CPU seconds, 512 MiB/output; parent timeout 150 seconds. All subsequent NumPy loads use `allow_pickle=False`.

Both actual EEG tensors are float64 `(200,80,17,100)` with 80 repetitions/image, 17 identical posterior channel names in identical order, 100 Hz and identical times −0.20 to 0.79 s. All values are finite. Exact numeric condition order maps to 200 distinct stimulus files, identical across subjects, with no missing/duplicate IDs or metadata train/test overlap. This is protocol-derived mapping corroborated by pinned official event-sorting code; raw trial-to-event roundtrip and session labels are unavailable.

Predeclared protocol: `configs/cross_subject_01_02.yaml`, SHA256 `14e070fbdc40aff039bf9851577c16a9f430f018a00568d3d5252e6a9475f2cc`, frozen before subject-02 acquisition. The subject-01 baseline was reproduced without tuning: all 17 supplied channels; [100,500) ms; no extra baseline, bins, or standardization; trial average; 680 features; correlation Brain RDM; cosine model RDM; pinned DINOv2/CLIP embeddings reused for all 200 identical images. K=5/10/20, 30 random disjoint 40/40 repetition splits, seeds 20261011/1337, 1,000 image-label permutations. Subject-01 numeric outputs and original report hashes remain intact.

## Brain RDM replication

- sub01: median split-half Spearman **0.7065**, empirical split interval [0.6775, 0.7236]. Median-null interval [-0.0489, 0.0469], one-sided Monte Carlo p=0.000999.
- sub02: median split-half Spearman **0.6466**, empirical split interval [0.6251, 0.6694]. Median-null interval [-0.0350, 0.0390], one-sided Monte Carlo p=0.000999.

Full 80-repetition subject-01 versus subject-02 Brain RDM Spearman **0.5495** across 19,900 unique image pairs; image-label null interval [-0.0508, 0.0481], p=0.000999. Both 200×200 RDMs are finite, symmetric, zero-diagonal, and independently reproducible from saved numeric features.

The between-subject correlation is numerically lower than both within-subject repetition-split correlations. These are different estimands, and full RDMs use 80 repetitions while halves use 40. Cross-subject correlation is not split-half reliability, a session-independent estimate, or a noise ceiling.

## Disagreement and predeclared ablation

All methods are reported separately for DINOv2, CLIP, and their equal-weight score mean. Row-rank disagreement is `(1−Spearman(nonself RDM rows))/2`. Reciprocal-rank weighted Jaccard uses all 199 nonself neighbors with normalized weights 1/rank and no fitted cutoff. No final metric was selected from subject-02.

| Method | Model | Within 01 ρ | Within 02 ρ | Cross ρ | Shared Top-20 | Tie-aware null overlap fraction | Top-20 p | Cutoff ties 01/02 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| jaccard_k5 | dinov2 | 0.285 | 0.301 | 0.201 | 15/20 | 0.65 | 0.2038 | 142/127 |
| jaccard_k10 | dinov2 | 0.418 | 0.381 | 0.233 | 9/20 | 0.35 | 0.1678 | 62/71 |
| jaccard_k20 | dinov2 | 0.458 | 0.414 | 0.354 | 8/20 | 0.15 | 0.0030 | 34/24 |
| row_spearman | dinov2 | 0.703 | 0.668 | 0.603 | 6/20 | 0.10 | 0.0130 | 1/1 |
| reciprocal_rank_weighted_jaccard | dinov2 | 0.511 | 0.463 | 0.348 | 4/20 | 0.10 | 0.1129 | 1/1 |
| jaccard_k5 | clip | 0.309 | 0.370 | 0.352 | 14/20 | 0.60 | 0.2557 | 122/124 |
| jaccard_k10 | clip | 0.518 | 0.495 | 0.505 | 5/20 | 0.20 | 0.3057 | 58/58 |
| jaccard_k20 | clip | 0.632 | 0.598 | 0.521 | 4/20 | 0.10 | 0.2008 | 27/19 |
| row_spearman | clip | 0.890 | 0.855 | 0.772 | 10/20 | 0.10 | 0.0010 | 1/1 |
| reciprocal_rank_weighted_jaccard | clip | 0.694 | 0.688 | 0.632 | 7/20 | 0.10 | 0.0030 | 1/1 |
| jaccard_k5 | mean_models | 0.320 | 0.367 | 0.281 | 10/20 | 0.50 | 0.5465 | 105/98 |
| jaccard_k10 | mean_models | 0.485 | 0.464 | 0.418 | 6/20 | 0.10 | 0.0030 | 29/32 |
| jaccard_k20 | mean_models | 0.604 | 0.548 | 0.480 | 2/20 | 0.10 | 0.6513 | 15/10 |
| row_spearman | mean_models | 0.855 | 0.802 | 0.744 | 6/20 | 0.10 | 0.0060 | 1/1 |
| reciprocal_rank_weighted_jaccard | mean_models | 0.633 | 0.629 | 0.554 | 4/20 | 0.10 | 0.1219 | 1/1 |

Within columns are median score-ranking Spearman across the same 30 independent repetition halves. Exact per-split statistics, empirical intervals, full score distributions, tie counts, within-subject Top-20 overlap and nulls are in `cross_subject_disagreement.json`; all 6,000 image-level scores are in `image_level_scores.csv`. Average ranks are used for score ties. Top-20 uses stable image-ID tie-breaking; nulls permute scores and then reapply that same rule. Distance ties also use stable image IDs.

The original K=10 mean-model baseline has within-subject ranking medians 0.485/0.464, cross-subject ranking ρ=0.418, and 6/20 shared images (tie-aware Top-20 p=0.0030). Its cutoff tie sizes are 29/32. DINOv2 and CLIP baseline cross-ranking correlations are 0.233/0.505; their Top-20 overlaps (9/20 and 5/20) are not above their tie-aware nulls at the unadjusted 0.05 level.

K=5 mean-model shares 10/20, but its tie-aware null median is also 10/20 (p=0.546): the apparent overlap is explained by massive score ties and the shared ID rule. K=20 mean-model ranking correlates across subjects (ρ=0.480), yet its Top-20 overlap is only 2/20 (p=0.651). Global ranking repeatability therefore does not guarantee shortlist repeatability.

Row-rank mean-model disagreement is less tied and more stable in this two-subject experiment (within ρ=0.855/0.802, cross ρ=0.744; Top-20 6/20, p=0.006). Reciprocal-rank weighting gives within ρ=0.633/0.629 and cross ρ=0.554, but its mean-model Top-20 overlap of 4/20 is not distinguishable from its null (p=0.122). These are exploratory comparisons, not a post hoc selection of a winner. CLIP-based ranking is more stable than DINOv2 in the reported conditions; the shared fixed model geometry may contribute to cross-subject score correlation.

## Inference limits and decision

All p-values are exploratory, one-sided, unadjusted Monte Carlo values; minimum resolution is 1/1001. Repeated random splits overlap across reruns, so their 2.5–97.5% intervals measure split variability, not independent-subject population uncertainty. Image-label permutation preserves complete RDM geometry; score nulls preserve each observed score distribution and ties. Correlated RDM edges are not treated as independent samples.

**Every reliability and brain–model result is conditional on the supplied official shared MVNN whitening.** Its covariance estimation used training/test data and repetitions together. Disjoint halves have independent repetition assignments, but do not have independently fitted preprocessing and may share sessions. Metadata split identity is preserved, yet this release cannot support a claim of strict preprocessing leakage-free held-out training or session-independent reliability. Training image pixels were not downloaded, so cross-split pixel-hash aliases were not audited.

GO: replicate the same frozen protocol on a small prospectively specified set of additional subjects before fitting or selecting any scoring method. The real Brain RDM signal and cross-subject ranking evidence justify this limited scientific expansion. Keep all predeclared methods and tie-aware nulls; assess shortlists separately from global rankings. NO-GO: train a classifier, claim OOD benefit, promote this two-subject Top-20 into a validated curation rule, or acquire all remaining subjects. Independently fitted/session-separated preprocessing requires raw data or an appropriate new official export.

No remaining acquisition, exact-mapping, numeric-export, or Brain-RDM blocker was encountered. The remaining blockers concern session-independent inference, preprocessing leakage, generalization beyond two subjects, and stable shortlist selection. No further subject acquisition occurred in this task.
