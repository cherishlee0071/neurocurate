"""Safe trial loading and image-wise EEG preprocessing.

Canonical NPZ arrays contain epochs, not whole continuous recordings.
Official THINGS-EEG2 pickled preprocessed arrays require explicit trust consent.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .metadata import validate_ids


@dataclass
class EEGTrials:
    data: np.ndarray  # (trials, channels, time)
    image_ids: np.ndarray  # repeated image identity per trial
    times: np.ndarray  # seconds relative to stimulus onset
    channel_names: np.ndarray
    split: str

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        self.image_ids = validate_ids(self.image_ids, unique=False)
        self.times = np.asarray(self.times, dtype=float)
        self.channel_names = validate_ids(self.channel_names)
        if self.data.ndim != 3 or any(d == 0 for d in self.data.shape):
            raise ValueError("EEG must be nonempty (trials, channels, time)")
        if self.data.shape != (len(self.image_ids), len(self.channel_names), len(self.times)):
            raise ValueError("EEG shape does not match trial IDs, channels, or times")
        if self.times.ndim != 1 or len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing seconds with at least two samples")
        if not np.isfinite(self.data).all() or not np.isfinite(self.times).all():
            raise ValueError("EEG and times must be finite")
        if self.split not in {"sanity", "train", "evaluation"}:
            raise ValueError("invalid EEG split")


def _check_ids(trials, expected_ids):
    expected = set(validate_ids(expected_ids))
    if set(trials.image_ids) != expected:
        raise ValueError("EEG IDs do not exactly match selected manifest split")
    return trials


def load_eeg(path, split, expected_ids, *, trusted_pickle=False, manifest=None,
             trial_metadata=None):
    """Load canonical NPZ, official preprocessed NPY, or MNE epochs FIF.

    NPZ: eeg, image_ids, times, channel_names, split (scalar string).
    NPY: dict preprocessed_eeg_data (images,repetitions,channels,time),
         times, ch_names. The explicit manifest eeg_index defines array order.
    FIF: epochs file and trial CSV image_id,split in exactly epoch order.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            required = {"eeg", "image_ids", "times", "channel_names", "split"}
            if not required <= set(arrays.files):
                raise ValueError(f"missing EEG arrays: {sorted(required - set(arrays.files))}")
            stored_split = arrays["split"]
            if stored_split.ndim != 0 or str(stored_split.item()) != split:
                raise ValueError("EEG split mismatch; refusing possible split leakage")
            trials = EEGTrials(arrays["eeg"], arrays["image_ids"], arrays["times"],
                               arrays["channel_names"], split)
    elif path.suffix == ".npy":
        if not trusted_pickle:
            raise ValueError("official NPY uses pickle; set trusted_pickle only for a trusted source")
        if manifest is None or "eeg_index" not in manifest:
            raise ValueError("official NPY requires explicit manifest eeg_index mapping")
        group = manifest[manifest.split == split].sort_values("eeg_index")
        if len(group) == 0 or list(group.eeg_index) != list(range(len(group))):
            raise ValueError("invalid official-array image order")
        document = np.load(path, allow_pickle=True).item()
        required = {"preprocessed_eeg_data", "times", "ch_names"}
        if not isinstance(document, dict) or not required <= set(document):
            raise ValueError("missing official THINGS-EEG2 preprocessed keys")
        array = np.asarray(document["preprocessed_eeg_data"])
        if array.ndim != 4 or len(array) != len(group):
            raise ValueError("official EEG must be (images,repetitions,channels,time), matching manifest")
        trials = EEGTrials(array.reshape(-1, *array.shape[2:]),
                           np.repeat(group.image_id.to_numpy(), array.shape[1]),
                           document["times"], np.asarray(document["ch_names"]), split)
    elif path.name.endswith((".fif", ".fif.gz")):
        if trial_metadata is None:
            raise ValueError("MNE epochs require explicit trial_metadata CSV")
        import mne
        import pandas as pd
        table = pd.read_csv(trial_metadata, dtype=str)
        if not {"image_id", "split"} <= set(table) or table.isna().any().any():
            raise ValueError("missing trial metadata")
        if not (table.split == split).all():
            raise ValueError("trial metadata mixes splits")
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        trials = EEGTrials(epochs.get_data(), table.image_id.to_numpy(), epochs.times,
                           np.asarray(epochs.ch_names), split)
    else:
        raise ValueError("supported EEG formats: .npz, trusted .npy, epochs .fif/.fif.gz")
    return _check_ids(trials, expected_ids)


@dataclass(frozen=True)
class EEGConfig:
    time_window: tuple[float, float] = (0.0, 0.5)
    baseline: tuple[float, float] | None = (-0.1, 0.0)
    channels: tuple[str, ...] | None = None
    temporal_bins: int | None = None
    min_repetitions: int = 2
    standardize: bool = False


def _window(times, interval, name):
    if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] >= interval[1]:
        raise ValueError(f"invalid {name} interval")
    # Inclusive start and exclusive stop, avoiding baseline/poststimulus overlap.
    if interval[0] < times[0] - 1e-9 or interval[1] > times[-1] + np.max(np.diff(times)) + 1e-9:
        raise ValueError(f"{name} lies outside epoch time coverage")
    mask = (times >= interval[0] - 1e-9) & (times < interval[1] - 1e-9)
    if not mask.any():
        raise ValueError(f"{name} selects no samples")
    return mask


def extract_eeg_features(trials, config=EEGConfig()):
    """Baseline per trial, average repetitions, then flatten channels/time bins.

    Optional standardization is within each image's feature vector, never
    fitted across stimuli or splits. No train/evaluation data are combined.
    """
    if isinstance(config.min_repetitions, bool) or not isinstance(config.min_repetitions, int) or config.min_repetitions < 1:
        raise ValueError("min_repetitions must be a positive integer")
    if not isinstance(config.standardize, bool):
        raise ValueError("standardize must be a boolean")
    data = trials.data.copy()
    if config.channels is not None:
        channels = validate_ids(np.asarray(config.channels))
        lookup = {name: i for i, name in enumerate(trials.channel_names)}
        if not set(channels) <= set(lookup):
            raise ValueError("unknown requested EEG channels")
        data = data[:, [lookup[c] for c in channels], :]
    if config.baseline is not None:
        baseline = _window(trials.times, config.baseline, "baseline")
        data -= data[:, :, baseline].mean(axis=2, keepdims=True)
    data = data[:, :, _window(trials.times, config.time_window, "time window")]
    ids = np.unique(trials.image_ids)
    means, counts = [], []
    for image_id in ids:
        selected = data[trials.image_ids == image_id]
        if len(selected) < config.min_repetitions:
            raise ValueError(f"too few repetitions for {image_id}")
        means.append(selected.mean(axis=0))
        counts.append(len(selected))
    averaged = np.stack(means)
    if config.temporal_bins is not None:
        bins = config.temporal_bins
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1 or bins > averaged.shape[-1]:
            raise ValueError("temporal_bins must be between 1 and selected sample count")
        averaged = np.stack([part.mean(axis=-1) for part in np.array_split(averaged, bins, axis=-1)], axis=-1)
    features = averaged.reshape(len(ids), -1)
    if config.standardize:
        std = features.std(axis=1, keepdims=True)
        if np.any(std <= 1e-12):
            raise ValueError("cannot standardize constant EEG features")
        features = (features - features.mean(axis=1, keepdims=True)) / std
    return ids, features, np.asarray(counts)
