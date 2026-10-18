# ECoG reliability

The official Groen2022 derivative MAT files contain concatenated tasks in the documented order `spatialpattern, temporalpattern, soc`; only the leading spatialpattern block was analyzed. Event labels and run identities were reconstructed from official NEMAR events.tsv. Requested analysis window was 0–500 ms; derivative time starts at -100 ms and is sampled at 512 Hz. Electrode identities are subject-private.

See `subject_summary.csv` for run-split reliability, Spearman–Brown correction, bootstrap intervals, permutation nulls and single-trial diagnostics.
