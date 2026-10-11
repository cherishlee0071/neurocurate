# Published research artifacts

This directory contains the scientific reports, figures, tables, provenance, and compact derived arrays produced through Phase 1.6. See [../VALIDATION.md](../VALIDATION.md) for exact commands, verification results, and limitations.

Raw EEG archives, object-array source files, numeric EEG exports (`pickle_free/eeg.npy`), large repetition-half RDM caches, stimulus JPEGs, and pretrained model checkpoints remain local and are not distributed in this Git repository. Reports and acquisition manifests may reference those local files. Verification manifests describe the original complete local runs; a fresh clone alone does not contain every input listed in them. Reproduction requires the independently verified inputs and pinned model caches described in the project README.

The Phase 1.6 frozen protocol snapshot is included at `real_phase0/prospective_03_05/protocol_snapshot.tar`. The Git tag `phase16-protocol-2026-10-11` preserves the exact immutable snapshot commit recorded by the protocol; the implementation and scientific outputs are published on `main`.

All scientific findings remain conditional on the official shared preprocessing. These artifacts do not establish causal value or improved training efficiency or generalization.
