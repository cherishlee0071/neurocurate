# THINGS-EEG environment audit

Audit date: 2026-10-18 UTC. Repository: `/workspace/neurocurate/neurocurate`.

## Decision

`PAYLOAD_FOUND`

Actual THINGS-EEG2 standard preprocessed payloads are present locally. The
archives are the expected official `nm000232` release artifacts (the pinned
release path is `v1.1.1`), and their SHA-256 values match the independently
recorded acquisition manifests. No download was attempted in this audit.

The OSF metadata records `source_version: null`; therefore `v1.1.1` is
identified by the pinned NEMAR release URL and matching published subject-01
size/hash, while the local OSF mirror metadata itself does not contain a
release string. The exact archive hashes and source metadata hashes are in
`things_eeg_payload_inventory.json`.

## Path classification

1. **Actual THINGS-EEG2 payload:** five standard subject ZIP archives under
   `data/local/real_phase0/`, each containing `preprocessed_eeg_test.npy` and
   `preprocessed_eeg_training.npy`; the extracted test arrays are also present.
2. **Existing NeuroCurate frozen/derived arrays:** files under
   `results/real_phase0/`, `results/time2depth/`, and `results/time2depth/phase0b/`.
   These are derived outputs, not source archives. Their hashes and provenance
   remain unchanged.
3. **Metadata and stimuli:** `data/local/real_phase0/image_metadata.npy`,
   `test_images.zip`, and the extracted 200 ordered stimulus JPEGs. The
   metadata and image files are not EEG payloads.
4. **Metadata-only or Git-annex pointers:** no Git-annex pointer files or
   DataLad pointer payloads were found in the repository or requested scratch
   path. JSON manifests are provenance records, not pointers.
5. **Download archives:** the five subject ZIP files and `test_images.zip` are
   preserved source downloads. ZIP inspection shows exactly three members per
   subject archive: a directory entry, one test NPY, and one training NPY.
6. **Derived/unrelated EEG files:** the search also found synthetic demo files
   and Python-package test fixtures. They are unrelated and were not used.

The requested path `/workspace/scratch/12d1aaae6a45/repo/neurocurate` does not
exist in this environment. No conclusion was based on that absent path.

## Verified source and payload facts

- Dataset identifier: `nm000232`, pinned release path `v1.1.1`.
- Expected subject-01 archive: 1,071,722,467 bytes,
  SHA-256 `f25d5ea30ba6811edd7ebd460d7b271edaf527b6920ab2d3d6e7950e9ee9639f`.
- Each extracted test tensor is 217,601,432 bytes and has the verified schema
  `(200, 80, 17, 100)`, `float64`; downstream loads use `allow_pickle=False`.
- The 200 image IDs are unique and ordered, and all corresponding local
  stimulus files are present.
- Training members are present inside the archives but were not extracted or
  processed by this audit. No raw EEG, resting-state data, or new subjects
  were downloaded.

The existing local payload is sufficient for the prior real-data experiments.
It must not be confused with a fresh download performed by this environment.
