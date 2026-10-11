# Phase 1.5 — Brain-specific controls and selection stability

**GO for prospective replication on subjects 03–05 using the unchanged protocol and all five budgets. Row-rank is a promising exploratory candidate, not a validated scoring winner. NO-GO for a validated small curation shortlist, classifier training, or claims of data efficiency/generalization.** No new subject, stimulus, checkpoint, or other payload was downloaded; no model was trained and no pickle was deserialized.

## Frozen inputs and definitions

The Phase 1.4 preprocessing, exact 200-image ordering, Brain correlation RDMs, model cosine RDMs, pinned DINOv2/CLIP embeddings, K=5/10/20 Jaccard, nonself RDM-row Spearman disagreement and all-neighbor reciprocal-rank weighted Jaccard are unchanged. All 15 method/model combinations are retained. The main exploratory candidate is row-rank with equal-weight mean model scores; focusing its figures does not select a final method.

`configs/brain_specific_controls.yaml` was frozen before Phase 1.5 computation, SHA256 `61d17efe3eb0842080757a396ddf0b3c1ea0afcfc91e3b4e780edb3eafc9888d`. This is an exploratory extension after seeing Phase 1.4, not independent preregistration of the original scientific hypothesis. It refers to the unchanged Phase 1.4 protocol SHA256 `14e070fbdc40aff039bf9851577c16a9f430f018a00568d3d5252e6a9475f2cc`.

Verified local numeric subject-01/02 exports and cached vision features were the only data sources. Full Brain RDMs were independently recomputed from numeric EEG; cached model RDMs from numeric features. All original subject-01 scientific output hashes and the complete Phase 1.4 output manifest remain valid. Every NumPy load uses `allow_pickle=False`.

## 1. Does row-rank disagreement depend on correct EEG-to-image alignment?

**Yes, under the tested label-null assumptions.** Observed mean-model cross-subject score Spearman is **0.7435**. Independently disrupting the two subjects' image assignments produces median -0.0054, 95% permutation interval [-0.1956, 0.1708], upper-tail p=0.000999. Applying the *same* wrong assignment to both subjects produces median 0.5113, 95% permutation interval [0.3403, 0.6620], upper-tail p=0.001998. Observed agreement exceeds both, but much of the numerical correlation survives the shared wrong correspondence.

Each permutation relabels a complete Brain RDM via `B[p,p]`, simultaneously changing both axes. Original distances, within-brain geometry and the multiset of nonself edges are preserved. Each disagreement vector is then recomputed against the **unchanged** model RDMs in their correct image coordinates. Neighbor ties are re-evaluated using the same fixed image-ID rule. No independently permuted row-only matrix or simple score shuffle is substituted.

As a descriptive alignment check, mean row-rank disagreement is 0.4514/0.4432; independent brain-label null medians are 0.5000/0.5004. Lower disagreement denotes better brain/model row alignment. Correctly aligned score ranks lose their ordering under independent brain permutations: median correlations with the aligned ranking are -0.0054/0.0022. These auxiliary summaries are descriptive, not extra confirmatory tests.

## 2. How much ranking agreement remains under brain-label controls?

| Row-rank model | Observed cross ρ | Independent brain null median | Shared brain null median | Shared-null p | Partial rank correlation | Partial shared-null p |
|---|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.6025 | -0.0016 | 0.5114 | 0.0490 | 0.5920 | 0.0669 |
| clip | 0.7720 | -0.0038 | 0.5164 | 0.0020 | 0.7085 | 0.0160 |
| mean_models | 0.7435 | -0.0054 | 0.5113 | 0.0020 | 0.6927 | 0.0080 |

The shared-null mean-model median **0.5113** is substantial. This control preserves the subjects' mutual Brain RDM structure while randomizing their correspondence to the fixed models. It diagnoses preserved cross-subject brain geometry interacting with shared model geometry; it is not a purely model-only null. The observed excess over its median is 0.2322, a descriptive difference, not a decomposition into percentages of brain-specific information.

Independent-null agreement near zero argues against unchanged model geometry alone explaining this observed ranking under these permutations. The shared null shows why a large cross-subject score correlation alone is insufficient. DINOv2's partial shared-null comparison is not clearly distinguishable at an unadjusted 0.05 level; CLIP and mean-model effects are stronger. All method/model results, including Jaccard and reciprocal-weighted alternatives, are saved in `brain_specific_nulls.json`.

## 3. Is the signal distinguishable from tested model-only baselines?

Model-only controls: DINOv2–CLIP nonself RDM-row disagreement; each model's mean distance to its 20 nearest neighbors; and each model's mean distance to all 199 other images. Atypicality uses cosine distance and excludes self. Values and EEG-derived image scores are in `model_only_controls.csv` (200 rows). Rank associations and their recomputed brain-label controls are in `model_only_associations.json`.

| Fixed model-only control | Correlation with subject-01 mean-model row score | Correlation with subject-02 mean-model row score |
|---|---:|---:|
| dinov2_clip_row_disagreement | -0.0606 | 0.0253 |
| dinov2_mean_nearest20_cosine_distance | -0.1263 | -0.0414 |
| clip_mean_nearest20_cosine_distance | 0.1998 | 0.2884 |
| dinov2_mean_nonself_cosine_distance | -0.0892 | -0.0062 |
| clip_mean_nonself_cosine_distance | -0.0562 | 0.0188 |

The five covariates together explain **17.8%/19.5%** of score-rank variance *in sample*. This is descriptive OLS fit, not out-of-sample prediction. A model-only image ranking would be identical across subjects by construction; its cross-subject correlation of one is not evidence for brain specificity.

Ranked EEG scores were residualized separately on an intercept and the same five ranked, standardized covariates. The residual Pearson correlation is a partial rank correlation, not Spearman applied a second time to residuals. Design rank is 6/6, condition number 3.478, residual degrees of freedom 194. No iid image/edge regression p-value is used.

Residual mean-model cross-subject association remains **0.6927**, versus independent brain-label median -0.0035, 95% permutation interval [-0.1725, 0.1500], upper-tail p=0.000999; shared brain-label median 0.5139, 95% permutation interval [0.3552, 0.6591], upper-tail p=0.007992. Residualization is repeated for every recomputed null score using the same fixed covariates. This supports an incremental descriptive association beyond these proxies. It does not remove all image-specific visual/semantic confounds or shared preprocessing, and establishes neither causality nor predictive usefulness.

## 4. At which selection budgets are shortlists reproducible?

Overlap fraction is `|Top-b(A) ∩ Top-b(B)| / b`. The table focuses mean-model row-rank; all five methods × three model variants × five budgets are in `selection_budget_curves.csv`.

| Budget | Within 01 median [split interval] | Within 02 median [split interval] | Full cross-subject shared | Independent brain null median | Shared brain null median | Shared-null-adjusted overlap | Shared-null p |
|---|---|---|---:|---:|---:|---:|---:|
| 10 | 0.450 [0.172, 0.700] | 0.500 [0.300, 0.700] | 2/10 | 0.000 | 0.300 | -0.143 | 0.8162 |
| 20 | 0.550 [0.400, 0.650] | 0.550 [0.400, 0.650] | 6/20 | 0.100 | 0.350 | -0.077 | 0.7862 |
| 40 | 0.712 [0.600, 0.789] | 0.650 [0.575, 0.757] | 21/40 | 0.200 | 0.450 | 0.136 | 0.2388 |
| 60 | 0.750 [0.695, 0.800] | 0.733 [0.670, 0.805] | 38/60 | 0.300 | 0.550 | 0.185 | 0.0779 |
| 100 | 0.840 [0.790, 0.873] | 0.820 [0.777, 0.860] | 81/100 | 0.500 | 0.680 | 0.406 | 0.0010 |

**Within-subject** median overlap exceeds tie-aware score-label nulls at all five budgets. These are the same 30 disjoint 40/40 repetition partitions from Phase 1.4; index correspondence is preserved across all image conditions. Their quantile intervals measure random-split variability. Across reruns, splits reuse repetitions and are not independent observations or sessions.

**Cross-subject** full-score overlaps are 2/10, 6/20, 21/40, 38/60, and 81/100. Budgets 20–100 exceed the independent brain-label and score-label references; budget 10 does not. Against the more stringent shared-brain null, **only budget 100** shows clear exploratory excess for mean-model row-rank (p=0.000999); budgets 10/20 are below its median, 40 is weak, and 60 has p≈0.078. Thus small shortlists are not established by the high full-ranking correlation.

The stronger budget-100 result is not universal: CLIP shares 85/100 (shared-null p=0.000999), while DINOv2 shares 72/100 (shared-null p≈0.161). Budget 100 selects half the stimulus panel, so high raw overlap is easier; its mean-model chance-adjusted excess over the shared-brain median is 0.406, rather than the raw 0.81. This budget is an exploratory observation, not a chosen optimal selection policy.

For each budget, score-label nulls permute scores and **then reapply** descending score/stable-ID tie ordering. These controls preserve score distributions and ties, but do not establish brain specificity. Independent/shared Brain RDM nulls additionally recompute the scores, accounting for changed score distributions and ties. Chance-adjusted overlap uses the explicitly predeclared formula `(observed − median null)/(1 − median null)` without clipping negative values. It is normalized excess above an empirical median, not a conventional mean-chance kappa. In discrete small-budget nulls the median may be zero even when expected uniform overlap is b/200; neither a zero median nor a large tied overlap implies zero chance.

Full cross-subject statistics are single observed values on a fixed panel; no population confidence interval is assigned. The CSV separately reports variability from the 60 matched half-RDM comparisons (A/A and B/B for 30 splits), which use 40 repetitions instead of full 80. These quantiles are not confidence bounds for the full-score statistic. The plot separates repetition-split variability from empirical permutation envelopes.

## 5. Is there enough evidence to prospectively replicate on subjects 03–05?

**GO for that limited scientific replication.** Correct correspondence affects the rankings, and mean-model row-rank and its residual association exceed the shared-brain control. Evidence is sufficient to test the fixed candidate prospectively, not to declare it a validated winner. Before inspecting 03–05 outcomes, lock the subject set, all existing scoring definitions/model variants/budgets, both brain-null constructions, model-only residual controls, seeds/counts and reporting rules. Report unfavorable subjects and shortlist results as well as favorable ones; avoid selecting a metric or budget from this two-subject analysis. No subject 03–05 acquisition happened here.

## Seeds, inference limits and reproducibility

- 1,000 independent brain-permutation pairs: seed **2026101501** (two independently generated permutations per draw).
- 1,000 shared brain permutations: seed **2026101502**.
- 1,000 tie-aware score-label permutations: seed **2026101503**, identical permutation index used when aggregating null medians over repetition splits.
- Original 30 repetition splits: seed **20261011**, 40/40 repetitions, reproduced exactly; model checkpoints and Phase 1.4 RDM/scoring definitions remain pinned.

All reported p-values are exploratory one-sided, unadjusted Monte Carlo values `(1 + exceedances)/(1000 + 1)`, minimum resolution 0.000999. Row-rank was identified after Phase 1.4; multiple methods, models, controls and budgets were inspected. Permutation quantiles are null envelopes, not confidence intervals. Image-label exchangeability is the tested random-assignment reference; it can disrupt shared image confounds along with genuine alignment. Evidence cannot uniquely identify neural mechanisms.

All EEG results remain **conditional on the official shared MVNN whitening**, which used training/test partitions and repetitions in covariance fitting. Disjoint repetition assignments do not create independently fitted preprocessing or session-independent reliability. Only two subjects and one fixed 200-image test panel were studied; session labels and training-image pixel-alias checks remain unavailable. Correct metadata split identity is preserved, and test images must remain outside future training/evaluation decisions. No improved data efficiency, generalization, OOD performance or causal value is claimed.

Exact permutations, raw null rank/partial/overlap statistics and observed full/half scores are saved in pickle-free `null_statistics.npz` and `observed_scores.npz`. `provenance.json` pins source inputs, unchanged Phase 1.4 manifests and executed code. Optional local tests independently recompute null samples from Brain RDMs, confirm shared permutation preserves cross-Brain geometry, validate partial associations, budget overlaps, chance adjustment, seeds and hashes. See `test_results.json` and `VALIDATION.md` for executed commands/results; `verified_outputs.json` records output sizes and SHA256 values.
