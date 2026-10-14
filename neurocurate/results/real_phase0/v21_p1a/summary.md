# NeuroCurate v2.1 P1-A Robustness Drift Pilot

## Objective

This exploratory sanity pilot tests whether the frozen population-consensus neural disagreement score is associated with cosine representation drift under fixed blur and crop perturbations. It does not claim model failure, robustness improvement, prediction, or causality.

## Frozen inputs

Only the read-only P0 artifacts under `results/real_phase0/v21_frozen/` were used. Disagreement was not recomputed. The five-subject, 200-image population consensus was formed from the saved subject-level mean-model row-rank scores. No uncertainty, behavior, confidence, or additional model signal entered the score.

## Results

- clip blur 1: rho=-0.046, p=0.519; high-minus-low drift=-0.00187; two-sided permutation p=0.418; random-baseline exceedance=0.441.
- clip blur 2: rho=-0.054, p=0.45; high-minus-low drift=-0.00660; two-sided permutation p=0.23; random-baseline exceedance=0.224.
- clip blur 4: rho=-0.054, p=0.448; high-minus-low drift=-0.00399; two-sided permutation p=0.653; random-baseline exceedance=0.667.
- clip crop 0.75: rho=-0.286, p=3.94e-05; high-minus-low drift=-0.02353; two-sided permutation p=0.0006; random-baseline exceedance=0.001.
- clip crop 0.9: rho=-0.163, p=0.0208; high-minus-low drift=-0.00680; two-sided permutation p=0.105; random-baseline exceedance=0.097.
- dinov2 blur 1: rho=-0.089, p=0.209; high-minus-low drift=-0.00072; two-sided permutation p=0.312; random-baseline exceedance=0.309.
- dinov2 blur 2: rho=-0.131, p=0.0643; high-minus-low drift=-0.00556; two-sided permutation p=0.326; random-baseline exceedance=0.325.
- dinov2 blur 4: rho=0.122, p=0.0844; high-minus-low drift=0.03441; two-sided permutation p=0.212; random-baseline exceedance=0.212.
- dinov2 crop 0.75: rho=-0.026, p=0.717; high-minus-low drift=-0.00300; two-sided permutation p=0.824; random-baseline exceedance=0.806.
- dinov2 crop 0.9: rho=-0.066, p=0.352; high-minus-low drift=-0.00319; two-sided permutation p=0.525; random-baseline exceedance=0.506.

## Decision

NO-GO for the claim that frozen neural disagreement predicts perturbation drift in this pilot. The only nominally strong condition is CLIP crop 0.75, with a negative association (rho=-0.286), opposite to the proposed direction; the remaining conditions are weak, null, or inconsistent. This is a weak/inconclusive diagnostic result, not evidence of predictive validity.

P1-B uncertainty matching is NO-GO as a follow-on confirmatory experiment. It may be revisited only as a separately frozen exploratory control after preregistering a broader perturbation design and independently auditing preprocessing. Do not claim model vulnerability, causal effects, or generalization.
