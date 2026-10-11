# Local data and minimal disk usage

Do not download the full THINGS-EEG2 / NEMAR nm000232 dataset. Keep selected
local EEG files, exact image metadata, and stimulus files under ignored
`data/local/`, or point configs to another local directory.

Inspect at most one recording's metadata first:

```bash
python scripts/query_metadata.py data/local/nemar-test-metadata.json --task test --limit 1
```

This script reads public EEGDash metadata, not EEG bytes. Check `storage`,
`annex_keys`, file sizes, and companion files before obtaining any recording.
A BrainVision `.vhdr` can refer to a much larger `.eeg` file; fetching the header
alone is not a usable EEG dataset. NEMAR native recordings may use SHA-addressed
storage, not publicly accessible BIDS-named S3 paths.

EEGDash is not guaranteed to stream bounded EEG chunks: loading a raw recording
can cache the full file plus sidecars. `cache_dir` sets a location, not a size
limit. DataLad/git-annex is an optional external way to obtain an explicitly
selected file and its companions, not part of the default installation. Never
run a dataset-wide `get` during this milestone.

The NeuroCurate loader supports local canonical NPZ trials, explicitly trusted
THINGS-EEG2 preprocessed NPY dictionaries with exact image-index metadata, and
MNE epochs FIF with trial-to-image CSV metadata. Continuous NEMAR recordings
need event-to-image mapping and epoch preparation first; see the main README.
No network access occurs inside the experiment pipeline for EEG loading.

Allowed stimulus splits are `sanity`, `train`, and `evaluation`. Official repeated
THINGS-EEG2 test stimuli belong to `sanity`. Do not repurpose them for later
training, evaluation, or model/hyperparameter selection. Concept names can locate
image folders; they are not assumed to be ImageNet-compatible class labels.
