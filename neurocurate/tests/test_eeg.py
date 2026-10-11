import numpy as np
import pytest
from neurocurate.eeg import EEGTrials, EEGConfig, extract_eeg_features, load_eeg


def test_average_baseline_and_shape(trials):
    original = trials.data.copy()
    ids, features, counts = extract_eeg_features(trials, EEGConfig(time_window=(0, 0.2), baseline=(-0.2, 0)))
    assert ids.tolist() == ["a", "b"]
    assert features.shape == (2, 4)
    np.testing.assert_allclose(features, [[7, 9, 9, 11], [3, 5, 5, 7]])
    np.testing.assert_array_equal(counts, [2, 2])
    np.testing.assert_array_equal(trials.data, original)


def test_channel_selection_temporal_bins(trials):
    _, features, _ = extract_eeg_features(trials, EEGConfig(time_window=(0, 0.2), baseline=None,
                                                         channels=("O2",), temporal_bins=1))
    np.testing.assert_allclose(features, [[11], [7]])


def test_standardization_is_per_image(trials):
    _, features, _ = extract_eeg_features(trials, EEGConfig(time_window=(0, 0.2), baseline=None, standardize=True))
    np.testing.assert_allclose(features.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(features.std(axis=1), 1)


@pytest.mark.parametrize("options", [dict(min_repetitions=3), dict(min_repetitions=0),
    dict(channels=("missing",)), dict(temporal_bins=3), dict(temporal_bins=0),
    dict(time_window=(0.1, 0.1)), dict(time_window=(-1, 0.1)), dict(baseline=(0.5, 0.6)),
    dict(temporal_bins=True), dict(min_repetitions=True), dict(standardize="false")])
def test_invalid_preprocessing(trials, options):
    settings = dict(time_window=(0, 0.2), baseline=None)
    settings.update(options)
    with pytest.raises(ValueError):
        extract_eeg_features(trials, EEGConfig(**settings))


@pytest.mark.parametrize("kind", ["nan", "shape", "times", "channels", "split"])
def test_invalid_trials(trials, kind):
    args = [trials.data.copy(), trials.image_ids.copy(), trials.times.copy(), trials.channel_names.copy(), trials.split]
    if kind == "nan": args[0][0, 0, 0] = np.nan
    if kind == "shape": args[0] = args[0][:, 0]
    if kind == "times": args[2] = args[2][::-1]
    if kind == "channels": args[3] = np.array(["O1", "O1"])
    if kind == "split": args[4] = "test"
    with pytest.raises(ValueError): EEGTrials(*args)


def test_canonical_loader(tmp_path, trials):
    path = tmp_path / "eeg.npz"
    np.savez(path, eeg=trials.data, image_ids=trials.image_ids, times=trials.times,
             channel_names=trials.channel_names, split=np.array("sanity"))
    loaded = load_eeg(path, "sanity", np.array(["a", "b"]))
    np.testing.assert_array_equal(loaded.data, trials.data)
    with pytest.raises(ValueError, match="split mismatch"):
        load_eeg(path, "train", np.array(["a", "b"]))
    with pytest.raises(ValueError, match="exactly match"):
        load_eeg(path, "sanity", np.array(["a", "c"]))


def test_missing_npz_arrays(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, eeg=np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="missing EEG arrays"):
        load_eeg(path, "sanity", np.array(["a", "b"]))


def test_official_loader_explicit_order(tmp_path, manifest):
    path = tmp_path / "official.npy"
    array = np.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4)
    np.save(path, dict(preprocessed_eeg_data=array, times=np.arange(4) / 10, ch_names=["O1", "O2"]))
    with pytest.raises(ValueError, match="pickle"):
        load_eeg(path, "sanity", np.array(["a", "b"]), manifest=manifest)
    with pytest.raises(ValueError, match="eeg_index"):
        load_eeg(path, "sanity", np.array(["a", "b"]), trusted_pickle=True)
    loaded = load_eeg(path, "sanity", np.array(["a", "b"]), trusted_pickle=True, manifest=manifest.iloc[::-1])
    assert loaded.image_ids.tolist() == ["a"] * 3 + ["b"] * 3
    np.testing.assert_array_equal(loaded.data, array.reshape(6, 2, 4))


def test_mne_epochs_trial_metadata(tmp_path):
    import mne
    import pandas as pd
    epochs = mne.EpochsArray(np.arange(24).reshape(4, 2, 3).astype(float),
                             mne.create_info(["O1", "O2"], 10, "eeg"), tmin=-0.1, verbose=False)
    path = tmp_path / "sample-epo.fif"
    epochs.save(path, verbose=False)
    metadata = tmp_path / "trials.csv"
    pd.DataFrame({"image_id": ["a", "b", "a", "b"], "split": ["sanity"] * 4}).to_csv(metadata, index=False)
    loaded = load_eeg(path, "sanity", np.array(["a", "b"]), trial_metadata=metadata)
    assert loaded.data.shape == (4, 2, 3)
    with pytest.raises(ValueError, match="trial_metadata"):
        load_eeg(path, "sanity", np.array(["a", "b"]))
