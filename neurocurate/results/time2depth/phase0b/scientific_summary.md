# Time2Depth Phase 0B reliability and SNR confound audit

## Reproduction

Phase 0 was reproduced from the immutable outputs: subject rhos remain 0.8, 0.9, 0.7, 0.9, 1.0; median 0.9; exact time-order p=0.025; stimulus-label p=0.011. IDs, RDM dimensions, and model layer count passed hash and integrity checks.

## Split-half stability

The original positive ordering is evaluated separately in both halves of all 30 disjoint repetition splits for all five subjects. See `preferred_depth_split_stability.csv` and `temporal_order_split_stability.json`; these are empirical split variability, not population confidence intervals.

## Confound decision

The temporal ordering remains visible in the full data, but late-window reliability is low and the reliability/depth association remains negative. Raw and centered profiles therefore cannot distinguish temporal depth from static layer alignment plus unequal SNR. Noise-degradation controls and equal five-sample diagnostic windows are reported without tuning. The 200 repeated-test images and shared MVNN preprocessing remain limiting factors.

## Decision: WEAK / INCONCLUSIVE

Phase 0B does not authorize Time2Depth Phase 1. Independent session-separated preprocessing is essential before stronger claims. The next defensible experiment is a preprocessing validation on a separate training image pool, keeping the 200 repeated-test images evaluation-only; no neural-guided training should begin from this audit alone.
