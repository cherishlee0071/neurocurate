# IASL Phase 0 recovery report

Status: **SUBSPACE_ONLY**

## Scope

This recovery audit preserves all prior Phase 0A–0D and NO-GO files. It uses the existing five numeric THINGS-EEG2 test exports, all 80 available repetitions, Set-A/Set-B/Set-C image subsets, and no new downloads. Requested window: 0–1000 ms; available analyzed interval: 0–790 ms; baseline interval: -200–0 ms.

## Findings

- Primary all-80 split-half reliability (mean image-wise correlation) is:

| Subject | Raw split-half | Spearman–Brown corrected | Bootstrap 95% CI | Permutation-null p95 | Single-trial to rest |
|---|---:|---:|---:|---:|---:|
| sub01 | 0.836 | 0.911 | [0.831, 0.841] | 0.745 | 0.329 |
| sub02 | 0.725 | 0.841 | [0.718, 0.732] | 0.603 | 0.224 |
| sub03 | 0.875 | 0.933 | [0.871, 0.879] | 0.772 | 0.320 |
| sub04 | 0.812 | 0.896 | [0.806, 0.818] | 0.666 | 0.315 |
| sub05 | 0.766 | 0.867 | [0.759, 0.772] | 0.645 | 0.287 |

Subject-03 is retained and analyzed separately; it is not removed to improve the result.
- Preprocessing variants A–E and fixed image subsets do not change the rule that configurations are reported without post-hoc selection. Variant F is deferred to fold-local fitting.
- Trial-average and single-trial diagnostics are separate. Trial averaging cannot be presented as single-trial generalization.
- Feature identity remains unstable; the analysis unit is a low-rank subspace or feature group.
- The energy-matched intervention audit has no task labels or readout, so target/non-target selectivity and semantic steering are unavailable. No reliable semantic intervention selectivity was established.

## Gates

```json
{
  "Gate_A_data_reliability": {
    "pass": true,
    "reason": "corrected reliability and bootstrap lower bounds"
  },
  "Gate_B_subspace_reliability": {
    "pass": true,
    "reason": "PCA CKA versus row permutation and subject pairs"
  },
  "Gate_C_intervention": {
    "pass": false,
    "reason": "task labels/selectivity and independent normalized effect unavailable"
  },
  "Gate_D_small_loso": {
    "pass": false,
    "reason": "requires A, B and C"
  },
  "status": "SUBSPACE_ONLY"
}
```

Gate D is blocked unless Gates A, B and C all pass. Full IASL and small LOSO were not run.

## Recommendation

The earlier NO-GO is not overwritten. If Gate A fails, the appropriate conclusion is `NO-GO_RELIABILITY`; if reliability improves but intervention remains unsupported, the project should be reported as `SUBSPACE_ONLY` and shifted toward a subspace reproducibility and intervention-limits analysis paper. No cross-subject generalization or data-efficiency claim is made.
