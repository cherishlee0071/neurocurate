# THINGS-EEG access decision

**PAYLOAD_FOUND**

The current environment already contains the official standard preprocessed
THINGS-EEG2 subject-01 through subject-05 ZIP archives and their extracted
test-only NPY arrays. The subject-01 archive exactly matches the requested
expected size and SHA-256. Subjects 02–05 also match the hashes recorded in
their acquisition manifests.

Because the payload is present and verified, the pinned NEMAR URL
`https://data.nemar.org/nm000232/v1.1.1/derivatives/preprocessed_eeg/sub-01.zip`
was not contacted and no resumable download was started. The prior audit's
OSF mirror range inspection is provenance evidence only; it is not a new
download in this run.

The source-version caveat is explicit: the OSF file metadata has no populated
release field (`source_version: null`). The v1.1.1 label comes from the pinned
official NEMAR release path and the matching expected subject-01 archive pin.

No IASL experiment should start from this audit alone. Existing NeuroCurate
results and source files were left untouched.
