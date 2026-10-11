# Phase 1 real-data integration

Status: **PASS_REAL_INTEGRATION**. The user explicitly approved the two verified official NPY files.

Real inputs: subject-01 official test EEG (200,80,17,100), float64; 16,000 trials; 100 Hz; 17 posterior/occipital channels; times -0.2 to 0.79 seconds. All 204 retained input files passed SHA256/size checks; no further EEG, stimulus or model weights were downloaded.

All 200 stimulus images were matched using explicit numeric metadata conditions and pinned author collection/conversion/preprocessing code. The original test split is repository sanity and cannot be used as future training data. No missing/duplicate image IDs, extra test files or train/test metadata identity overlap was found. Filenames were not sorted to infer EEG order. Original raw trial events and session labels were not independently recovered.

Existing NeuroCurate loader/preprocessing produced (200,680) features: average 80 repetitions, all 17 channels, 100 <= t < 500 ms, no additional baseline or normalization. Verified correlation-distance Brain RDMs are (200,200) and (16,16). The explicit subset is EEG indices 0..15. Frozen local DINOv2 produced (16,768) features and CLIP (16,512); both cosine RDMs are (16,16). All RDMs are finite, symmetric, zero-diagonal and independently reproduce from saved features. Saved identities preserve original EEG-axis ordering.

Results: results/phase1/smoke/. No synthetic substitution, training or OOD work occurred.

Limits: one subject; no EEG reliability/null-control or cross-subject estimate. Official MVNN uses shared covariance from training and test repetitions, so later reliability must be interpreted conditional on this transform. Strict independently fitted training evaluation requires raw-data preprocessing. Training image content hashes were not audited.

Recommended next experiment: repeated-image split-half RDM reliability with image-label permutation controls, then a second subject.
