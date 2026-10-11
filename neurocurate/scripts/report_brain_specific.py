"""Generate the Phase 1.5 scientific summary from saved statistics only."""
import json
from pathlib import Path
import pandas as pd
from neurocurate.lowdisk import sha256

root=Path('results/real_phase0/brain_specific_controls')
report=json.loads((root/'brain_specific_nulls.json').read_text())
frame=pd.read_csv(root/'selection_budget_curves.csv')
associations=json.loads((root/'model_only_associations.json').read_text())['associations']
primary=report['results']['row_spearman/mean_models']

def stat(value):
    lo,hi=value['null_95_percentile_interval']
    return f"median {value['null_median']:.4f}, 95% permutation interval [{lo:.4f}, {hi:.4f}], upper-tail p={value['exploratory_one_sided_p']:.6f}"

ranking_rows=[]
for model in ['dinov2','clip','mean_models']:
    result=report['results']['row_spearman/'+model]
    ranking_rows.append(f"| {model} | {result['observed_cross_subject_spearman']:.4f} | {result['independent_brain_null']['null_median']:.4f} | {result['shared_brain_null']['null_median']:.4f} | {result['shared_brain_null']['exploratory_one_sided_p']:.4f} | {result['observed_partial_rank_correlation']:.4f} | {result['partial_shared_brain_null']['exploratory_one_sided_p']:.4f} |")
budget_rows=[]
selected=frame[(frame.method=='row_spearman')&(frame.model=='mean_models')]
for budget in [10,20,40,60,100]:
    subset=selected[selected.budget==budget]
    within=[subset[subset.comparison==s].iloc[0] for s in ['within_sub01','within_sub02']]
    cross=subset[(subset.comparison=='cross_subject_full')&(subset.null_type=='brain_shared')].iloc[0]
    nullind=subset[(subset.comparison=='cross_subject_full')&(subset.null_type=='brain_independent')].iloc[0]
    budget_rows.append(f"| {budget} | {within[0].overlap_fraction:.3f} [{within[0].split_q025:.3f}, {within[0].split_q975:.3f}] | {within[1].overlap_fraction:.3f} [{within[1].split_q025:.3f}, {within[1].split_q975:.3f}] | {round(cross.expected_shared_images)}/{budget} | {nullind.null_median:.3f} | {cross.null_median:.3f} | {cross.chance_adjusted_overlap:.3f} | {cross.exploratory_p:.4f} |")
association_rows=[]
for control in report['model_control_design']['controls']:
    values=[next(a['score_spearman'] for a in associations if a['subject']==s and a['control']==control and a['method_model']=='row_spearman/mean_models') for s in ['sub01','sub02']]
    association_rows.append(f"| {control} | {values[0]:.4f} | {values[1]:.4f} |")
subjects=primary['subjects']
text=f'''# Phase 1.5 — Brain-specific controls and selection stability

**GO for prospective replication on subjects 03–05 using the unchanged protocol and all five budgets. Row-rank is a promising exploratory candidate, not a validated scoring winner. NO-GO for a validated small curation shortlist, classifier training, or claims of data efficiency/generalization.** No new subject, stimulus, checkpoint, or other payload was downloaded; no model was trained and no pickle was deserialized.

## Frozen inputs and definitions

The Phase 1.4 preprocessing, exact 200-image ordering, Brain correlation RDMs, model cosine RDMs, pinned DINOv2/CLIP embeddings, K=5/10/20 Jaccard, nonself RDM-row Spearman disagreement and all-neighbor reciprocal-rank weighted Jaccard are unchanged. All 15 method/model combinations are retained. The main exploratory candidate is row-rank with equal-weight mean model scores; focusing its figures does not select a final method.

`configs/brain_specific_controls.yaml` was frozen before Phase 1.5 computation, SHA256 `{report['config']['config_sha256']}`. This is an exploratory extension after seeing Phase 1.4, not independent preregistration of the original scientific hypothesis. It refers to the unchanged Phase 1.4 protocol SHA256 `14e070fbdc40aff039bf9851577c16a9f430f018a00568d3d5252e6a9475f2cc`.

Verified local numeric subject-01/02 exports and cached vision features were the only data sources. Full Brain RDMs were independently recomputed from numeric EEG; cached model RDMs from numeric features. All original subject-01 scientific output hashes and the complete Phase 1.4 output manifest remain valid. Every NumPy load uses `allow_pickle=False`.

## 1. Does row-rank disagreement depend on correct EEG-to-image alignment?

**Yes, under the tested label-null assumptions.** Observed mean-model cross-subject score Spearman is **{primary['observed_cross_subject_spearman']:.4f}**. Independently disrupting the two subjects' image assignments produces {stat(primary['independent_brain_null'])}. Applying the *same* wrong assignment to both subjects produces {stat(primary['shared_brain_null'])}. Observed agreement exceeds both, but much of the numerical correlation survives the shared wrong correspondence.

Each permutation relabels a complete Brain RDM via `B[p,p]`, simultaneously changing both axes. Original distances, within-brain geometry and the multiset of nonself edges are preserved. Each disagreement vector is then recomputed against the **unchanged** model RDMs in their correct image coordinates. Neighbor ties are re-evaluated using the same fixed image-ID rule. No independently permuted row-only matrix or simple score shuffle is substituted.

As a descriptive alignment check, mean row-rank disagreement is {subjects['sub01']['observed_mean_disagreement']:.4f}/{subjects['sub02']['observed_mean_disagreement']:.4f}; independent brain-label null medians are {subjects['sub01']['independent_brain_null_mean_disagreement']['median']:.4f}/{subjects['sub02']['independent_brain_null_mean_disagreement']['median']:.4f}. Lower disagreement denotes better brain/model row alignment. Correctly aligned score ranks lose their ordering under independent brain permutations: median correlations with the aligned ranking are {subjects['sub01']['independent_brain_null_correlation_with_aligned_ranking']['median']:.4f}/{subjects['sub02']['independent_brain_null_correlation_with_aligned_ranking']['median']:.4f}. These auxiliary summaries are descriptive, not extra confirmatory tests.

## 2. How much ranking agreement remains under brain-label controls?

| Row-rank model | Observed cross ρ | Independent brain null median | Shared brain null median | Shared-null p | Partial rank correlation | Partial shared-null p |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(ranking_rows)}

The shared-null mean-model median **{primary['shared_brain_null']['null_median']:.4f}** is substantial. This control preserves the subjects' mutual Brain RDM structure while randomizing their correspondence to the fixed models. It diagnoses preserved cross-subject brain geometry interacting with shared model geometry; it is not a purely model-only null. The observed excess over its median is {primary['shared_brain_null']['excess_above_null_median']:.4f}, a descriptive difference, not a decomposition into percentages of brain-specific information.

Independent-null agreement near zero argues against unchanged model geometry alone explaining this observed ranking under these permutations. The shared null shows why a large cross-subject score correlation alone is insufficient. DINOv2's partial shared-null comparison is not clearly distinguishable at an unadjusted 0.05 level; CLIP and mean-model effects are stronger. All method/model results, including Jaccard and reciprocal-weighted alternatives, are saved in `brain_specific_nulls.json`.

## 3. Is the signal distinguishable from tested model-only baselines?

Model-only controls: DINOv2–CLIP nonself RDM-row disagreement; each model's mean distance to its 20 nearest neighbors; and each model's mean distance to all 199 other images. Atypicality uses cosine distance and excludes self. Values and EEG-derived image scores are in `model_only_controls.csv` (200 rows). Rank associations and their recomputed brain-label controls are in `model_only_associations.json`.

| Fixed model-only control | Correlation with subject-01 mean-model row score | Correlation with subject-02 mean-model row score |
|---|---:|---:|
{chr(10).join(association_rows)}

The five covariates together explain **{100*subjects['sub01']['in_sample_model_only_rank_r_squared']:.1f}%/{100*subjects['sub02']['in_sample_model_only_rank_r_squared']:.1f}%** of score-rank variance *in sample*. This is descriptive OLS fit, not out-of-sample prediction. A model-only image ranking would be identical across subjects by construction; its cross-subject correlation of one is not evidence for brain specificity.

Ranked EEG scores were residualized separately on an intercept and the same five ranked, standardized covariates. The residual Pearson correlation is a partial rank correlation, not Spearman applied a second time to residuals. Design rank is {report['model_control_design']['rank']}/{report['model_control_design']['columns']}, condition number {report['model_control_design']['condition_number']:.3f}, residual degrees of freedom {report['model_control_design']['residual_degrees_of_freedom']}. No iid image/edge regression p-value is used.

Residual mean-model cross-subject association remains **{primary['observed_partial_rank_correlation']:.4f}**, versus independent brain-label {stat(primary['partial_independent_brain_null'])}; shared brain-label {stat(primary['partial_shared_brain_null'])}. Residualization is repeated for every recomputed null score using the same fixed covariates. This supports an incremental descriptive association beyond these proxies. It does not remove all image-specific visual/semantic confounds or shared preprocessing, and establishes neither causality nor predictive usefulness.

## 4. At which selection budgets are shortlists reproducible?

Overlap fraction is `|Top-b(A) ∩ Top-b(B)| / b`. The table focuses mean-model row-rank; all five methods × three model variants × five budgets are in `selection_budget_curves.csv`.

| Budget | Within 01 median [split interval] | Within 02 median [split interval] | Full cross-subject shared | Independent brain null median | Shared brain null median | Shared-null-adjusted overlap | Shared-null p |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(budget_rows)}

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
'''
(root/'scientific_summary.md').write_text(text)
paths=[p for p in root.iterdir() if p.is_file() and p.name not in ['verified_outputs.json','analysis.log']]
manifest=dict(outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(paths)},
    report_code_sha256=sha256('scripts/report_brain_specific.py'))
(root/'verified_outputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(root/'scientific_summary.md')
