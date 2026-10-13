# Preprocessing sensitivity plan (Phase 1.7)

No raw EEG is downloaded in Phase 1.7. Existing results use official shared MVNN preprocessing, so reliability is conditional.

Acquire one subject/session pilot with raw EEG, event triggers, montage, sampling rate, bad-channel annotations, and the verified 200-image manifest. Fit bad-channel detection, line-noise, baseline, whitening/MVNN, and dimensionality reduction on training repetitions or sessions only; apply transforms to held-out repetitions.

Use subjects 01–02, two sessions if available, 40 repetitions per image, 30 fixed splits, 200x200 Brain RDMs and frozen row-rank disagreement. Require event-to-image validation before epochs enter the analysis. Reserve 2–3x the raw input size and at least 16 GB RAM per pilot; keep raw and temporary files outside Git.

Any ranking change is preprocessing sensitivity, not evidence of improved data efficiency. An independent-pool experiment requires a separately fitted pipeline and new stimulus pool.
