# IASL-ECoG DS004194 final report

Status: **ECOG_NO_GO_DATA**

## Data and acquisition

NEMAR `on004194` v1.0.0 (OpenNeuro `ds004194` v3.0.0), CC0. The dataset metadata reports 14 subjects, 209 recordings and 57.9 GB of total payload. Only four official Groen2022 derivative MAT files were downloaded: p01, p02, p05 and p06; their MD5/size records are in `reproducibility_manifest.json`. No raw `.eeg`, `.dat` or training payload was downloaded.

## Analysis

E0 found nine subjects sharing `spatialpattern` and nine sharing `temporalpattern`; four downloaded subjects were selected deterministically by smallest derivative size. The official derivative pipeline concatenates tasks in `spatialpattern, temporalpattern, soc` order, so the leading spatialpattern block was selected. Event order was reconstructed from official NEMAR events.tsv. Broadband visual-response electrodes were inherited from the official derivative. Electrode indices remain subject-private.

The derivative epoch is sampled at 512 Hz and spans -100 ms to 1.2 s; the analysis window was 0–500 ms. PCA and small SAE subspace diagnostics were run with seeds 0/1/2. Intervention was limited to energy-matched reconstruction damage because no task readout was supplied.

## Gates and decision

- Gate A reliability: passed.
- Gate B subspace: failed.
- Gate C intervention: failed; target/non-target selectivity and task damage are unavailable.
- Gate D LOSO: not run.

The result is **ECOG_NO_GO_DATA**. Gate B was evaluated on six subject pairs; the PCA temporal-group CKA exceeded its row-permutation p95 for 2/6 pairs. This supports no robust shared subspace under the current compact representation. It does not establish intervention-aware transfer or cross-subject generalization. Full IASL and LOSO remain blocked until subspace stability and an independent task readout support the intervention test.

THINGS-EEG's historical NO-GO and the ECoG result are preserved as separate studies.
