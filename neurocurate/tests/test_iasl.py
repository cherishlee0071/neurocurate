import numpy as np

from neurocurate.iasl import (
    flatten_trials, pca_fit_transform, tiny_temporal_encoder,
    train_sae_replicates, matching_diagnostics, intervention_audit,
)


def test_iasl_representation_and_sae_shapes():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(6, 2, 3, 8)).astype("float32")
    flat = flatten_trials(x)
    z, _ = pca_fit_transform(flat, 4)
    assert z.shape == (12, 4)
    tiny = tiny_temporal_encoder(x.reshape(-1, 3, 8), 4, seed=1)
    assert tiny.shape == (12, 4)
    reps = train_sae_replicates(z, (2,), (0, 1), epochs=1)
    assert all(r["codes"].shape == (12, 8) for r in reps)
    assert all(np.isfinite(r["reconstruction"]).all() for r in reps)


def test_iasl_matching_and_intervention_are_finite():
    rng = np.random.default_rng(5)
    z, _ = pca_fit_transform(rng.normal(size=(20, 6)).astype("float32"), 6)
    reps = train_sae_replicates(z, (2,), (0, 1), epochs=1)
    diag = matching_diagnostics(reps)
    assert len(diag["widths"]) == 1
    rows = intervention_audit(reps[0], (0.5,), n_groups=2)
    assert rows and all(np.isfinite(r["C_norm"]) for r in rows)
