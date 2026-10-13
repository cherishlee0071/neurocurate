# NeuroCurate Phase 1.7 — consensus generalization and shortlist diagnosis

This exploratory analysis uses only the five verified subjects and frozen Phase 1.6 scores. The recorded input commit is `93593bda25399826fd3eefb84de6259c12cfe4ad`; earlier results are unchanged.

The primary mean-model row-rank score has positive leave-one-subject-out correlations for every held-out subject: sub01 0.837, sub02 0.772, sub03 0.806, sub04 0.719, and sub05 0.734 (median 0.772). Every independent brain-label permutation null had p=0.001 with 1,000 permutations. The shared-permutation null is exactly degenerate because a common image permutation preserves rank correlation; it is a geometry-invariance control, not evidence against the independent null. The five folds are dependent and are not five population replicates.

Small-budget selection is unstable despite global ranking agreement. The median score gap at the cutoff is 0.00032 for K=10, 0.00274 for K=20, 0.00137 for K=40, 0.00018 for K=60, and 0.00100 for K=100. Only 0, 4, 10, 23, and 60 images respectively reach within-subject split frequency 0.8 at K=10, 20, 40, 60, and 100. This points mainly to narrow score-boundary margins at small budgets, with genuine between-subject differences contributing as well.

The held-out repetition proxy shows weak and heterogeneous association between image score and split instability (Spearman values by subject 0.187, 0.143, 0.244, -0.001, and 0.075). This proxy uses the same repetition splits as the disagreement scores, so it is descriptive rather than independent reliability evidence.

There is no reproducible high-disagreement subset at K=10. Larger budgets contain more stable images, but this does not establish improved data efficiency. Expansion to all ten subjects is justified only as an exploratory representational replication with independently audited preprocessing, not as a selection-efficiency claim.

Recommended direction: prioritize a representational-analysis paper and a session-separated preprocessing sensitivity pilot. Defer independent-pool data-efficiency experiments until narrow margins, subject heterogeneity, and preprocessing dependence are resolved. No classifier training, additional EEG download, or causal/generalization claim is supported here.
