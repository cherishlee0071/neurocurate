"""THINGS-EEG2 protocol-derived test-axis alignment with strict validation."""
from pathlib import Path
import numpy as np
from .metadata import things_metadata_manifest, validate_ids


def validate_test_order(document, image_root):
    """Require explicit numeric condition prefixes in metadata, never sort files.

Author collection code labels test condition c; conversion writes behavioral
tot_img_number as event code; epoching sorts np.unique(event codes). Thus axis
i corresponds to condition i+1. Callers must retain that source provenance.
"""
    required = {"test_img_files", "test_img_concepts", "test_img_concepts_THINGS",
                "train_img_files", "train_img_concepts_THINGS"}
    if not isinstance(document, dict) or not required <= set(document):
        raise ValueError("missing official mapping metadata")
    test_files = validate_ids(document["test_img_files"])
    concepts = validate_ids(document["test_img_concepts"])
    origins = validate_ids(document["test_img_concepts_THINGS"])
    if len(test_files) != 200 or len(concepts) != 200 or len(origins) != 200:
        raise ValueError("official test metadata must contain 200 conditions")
    for index, (concept, filename) in enumerate(zip(concepts, test_files)):
        parts = str(concept).split("_", 1)
        if len(parts) != 2 or parts[0] != f"{index + 1:05d}":
            raise ValueError("metadata numeric condition order is invalid; refusing to sort it")
        if Path(str(filename)).name != filename or Path(str(concept)).name != concept:
            raise ValueError("unsafe metadata image path")
        if not (Path(image_root) / "test_images" / concept / filename).is_file():
            raise ValueError("mapped stimulus image is missing")
    train_files = validate_ids(document["train_img_files"])
    train_origins = np.asarray(document["train_img_concepts_THINGS"])
    if train_origins.ndim != 1 or len(train_origins) != len(train_files):
        raise ValueError("training metadata identity lengths differ")
    train_identity = set(zip(map(str, train_origins), map(str, train_files)))
    test_identity = set(zip(map(str, origins), map(str, test_files)))
    if train_identity & test_identity or set(train_files) & set(test_files):
        raise ValueError("test/train metadata image identity overlap")
    expected_paths = {("test_images/" + c + "/" + f) for c, f in zip(concepts, test_files)}
    actual_paths = {p.relative_to(image_root).as_posix()
                    for p in (Path(image_root) / "test_images").rglob("*")
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if actual_paths != expected_paths:
        raise ValueError("stimulus inventory does not exactly match official metadata")
    return dict(matched_images=200, numeric_condition_ids=list(range(1, 201)),
                axis_mapping="eeg_index = numeric condition ID - 1",
                metadata_order_valid=True, missing_images=0, duplicate_test_ids=0,
                extra_images=0, train_test_metadata_identity_overlap=0,
                train_test_filename_overlap=0, alphabetical_order_assumed=False,
                original_split="test", repository_split="sanity",
                raw_trial_event_roundtrip_verified=False,
                evidence_basis="official collection/conversion/preprocessing code plus explicit ordered numeric metadata",
                limitation="session/repetition event IDs absent from released array; training-image content hashes unavailable")


def load_real_manifest(metadata_path, image_root, *, trusted_pickle=False):
    if not trusted_pickle:
        raise ValueError("explicit verified official pickle approval required")
    document = np.load(metadata_path, allow_pickle=True).item()
    report = validate_test_order(document, image_root)
    manifest = things_metadata_manifest(metadata_path, image_root, trusted_pickle=True)
    selected = manifest[manifest.split == "sanity"].sort_values("eeg_index")
    if selected.eeg_index.tolist() != list(range(200)):
        raise ValueError("repository manifest changed official test-axis ordering")
    return manifest, selected, report
