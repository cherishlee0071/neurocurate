# IASL feasibility outputs

`run_iasl_pilot.py` creates three Phase 0 outputs:

- `phase0_data_audit/`: shared-image manifest, reliability and a compact diagnostic;
- `phase0_sae_reproducibility/`: fixed PCA control, small SAE seed/width audit, feature matching and subspace diagnostics;
- `phase0_intervention/`: raw and removed-energy-normalized ablation diagnostics.

All inputs are existing pickle-free numeric exports. The pilot does not download data or fit on a held-out subject. Phase 1 LOSO retrieval is gated by the stop rules in the YAML config.
