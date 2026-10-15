# Time2Depth Phase 0 scientific summary

## Objective

This exploratory sanity check tests whether the fixed temporal order of visual EEG RDMs aligns monotonically with fixed DINOv2 block depth. It is not a claim that biological time equals artificial depth.

## Results

Subject temporal-order Spearman rhos are {'sub01': 0.8, 'sub02': 0.9, 'sub03': 0.7, 'sub04': 0.9, 'sub05': 1.0}; all 5/5 are positive. The cohort median is 0.9000. The exact common 5! time-label null gives one-sided p=0.0250, with null range [-0.9000, 0.9000]. This p has 1/120 granularity and includes the identity permutation. The whole-RDM image-label null gives p=0.0110, median=0.0000, 95% interval [-0.8000, 0.8000].

## Controls and limitations

Main-window split-half reliability medians range from 0.2490 to 0.9163. The 50–100, 100–150 and 150–200 ms windows contain only five samples at 100 Hz. Across the 25 subject/window observations, reliability versus preferred layer position has descriptive rho=-0.5831; reliability versus mean raw RSA has rho=0.3792. The late 300–500 ms window has the lowest reliability (about 0.249), while late windows prefer deeper layers, so SNR confounding is not resolved. These are not independent reliability evidence. The pre-stimulus baseline was retained as a diagnostic profile and excluded from the primary order statistic. Raw RSA includes static layer alignment; centered RSA removes each layer's across-window mean. The official shared MVNN whitening used training/test partitions and repetitions, so split-half estimates are conditional and not session-independent. RDM edges are not independent observations.

## Decision: WEAK / INCONCLUSIVE

The temporal-order and stimulus-label controls are favorable, but the frozen GO rule also requires that SNR or static-layer artifacts do not clearly explain the pattern. That condition is not established: reliability is weakest in the late windows and is negatively associated with preferred depth. Therefore this is WEAK/INCONCLUSIVE, not a GO for training. A future preprocessing-separated reliability audit is needed before any Phase 1 pilot.

If a later Phase 1 is authorized after that audit, it must keep the 200 repeated-test images evaluation-only and compare fixed/random/unconstrained/monotonic time-to-layer regularizers on an independent training image pool. This report does not start Phase 1.
