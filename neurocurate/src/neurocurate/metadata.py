"""Explicit image identity and split membership; no class-label assumptions."""

from pathlib import Path
import numpy as np
import pandas as pd

SPLITS = {"sanity", "train", "evaluation"}


def validate_ids(ids, *, unique=True):
    values = np.asarray(ids)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("image IDs must be a nonempty one-dimensional array")
    if values.dtype.kind not in "US" and not (values.dtype.kind == "O" and all(isinstance(x, str) for x in values)):
        raise ValueError("image IDs must be strings, not labels or numeric indices")
    values = values.astype(str)
    if any(not s.strip() or s != s.strip() for s in values):
        raise ValueError("image IDs must be nonempty, trimmed strings")
    if unique and len(set(values)) != len(values):
        raise ValueError("duplicate image IDs")
    return values


def validate_manifest(table):
    table = table.copy()
    required = {"image_id", "image_path", "split"}
    if not required <= set(table.columns):
        raise ValueError(f"missing metadata columns: {sorted(required - set(table.columns))}")
    if table[list(required)].isna().any().any():
        raise ValueError("missing metadata values")
    validate_ids(table.image_id.to_numpy())
    if not set(table.split) <= SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    if any(not isinstance(p, str) or not p.strip() for p in table.image_path):
        raise ValueError("image_path must contain nonempty paths")
    # Different IDs must not conceal the same stimulus across splits.
    paths = [str(Path(p).resolve()) for p in table.image_path]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate image paths: possible alias or split leakage")
    if "eeg_index" in table:
        for _, group in table.groupby("split"):
            index = pd.to_numeric(group.eeg_index, errors="raise").to_numpy()
            if not np.isfinite(index).all() or not np.equal(index, np.floor(index)).all():
                raise ValueError("eeg_index must contain integers")
            if set(index) != set(range(len(group))):
                raise ValueError("eeg_index must be unique and contiguous within each split")
    return table


def things_metadata_manifest(path, image_root, *, trusted_pickle=False):
    """Convert the official image_metadata.npy, keeping file identity and array order.

    Official 'test' images are assigned to 'sanity', never evaluation.
    Concept strings are used only as folder paths, not supervision labels.
    """
    if not trusted_pickle:
        raise ValueError("image_metadata.npy requires trusted_pickle consent")
    document = np.load(path, allow_pickle=True).item()
    if not isinstance(document, dict):
        raise ValueError("official image metadata must be a dictionary")
    rows = []
    for prefix, split, folder in [("test", "sanity", "test_images"), ("train", "train", "training_images")]:
        key = f"{prefix}_img_files"
        if key not in document:
            raise ValueError(f"missing official metadata key: {key}")
        files = validate_ids(np.asarray(document[key]))
        concepts = document.get(f"{prefix}_img_concepts")
        for index, filename in enumerate(files):
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("official metadata image filenames must be safe relative paths")
            if len(relative.parts) == 1:
                if concepts is None or len(concepts) != len(files):
                    raise ValueError("bare filenames require one concept folder per image; supply an explicit CSV instead")
                concept = str(concepts[index])
                if not concept or Path(concept).is_absolute() or len(Path(concept).parts) != 1 or concept == "..":
                    raise ValueError("invalid concept folder in official metadata")
                relative = Path(concept) / relative
            if relative.parts[0] != folder:
                relative = Path(folder) / relative
            rows.append({"image_id": relative.as_posix(), "image_path": str((Path(image_root) / relative).resolve()),
                         "split": split, "eeg_index": index})
    return validate_manifest(pd.DataFrame(rows))


def load_manifest(path, image_root=None):
    path = Path(path)
    table = pd.read_csv(path, dtype={"image_id": str, "image_path": str, "split": str})
    if "image_path" not in table:
        return validate_manifest(table)
    root = Path(image_root) if image_root is not None else path.parent
    table["image_path"] = [str((root / p).resolve()) if isinstance(p, str) else p
                           for p in table.image_path]
    return validate_manifest(table)


def align_features(target_ids, source_ids, features):
    """Reorder by exact image identity; reject absent/duplicate IDs."""
    target = validate_ids(target_ids)
    source = validate_ids(source_ids)
    features = np.asarray(features)
    if features.ndim != 2 or len(source) != len(features):
        raise ValueError("features must be (images, features) and match source IDs")
    lookup = {key: i for i, key in enumerate(source)}
    missing = sorted(set(target) - set(source))
    if missing:
        raise ValueError(f"missing image IDs: {missing[:10]}")
    return features[[lookup[key] for key in target]]


def assert_disjoint(*id_groups):
    """Call before any future fitting; sanity stimuli cannot enter training/evaluation."""
    seen = set()
    for group in id_groups:
        current = set(validate_ids(group))
        if seen & current:
            raise ValueError(f"split leakage: overlapping image IDs {sorted(seen & current)[:10]}")
        seen |= current
