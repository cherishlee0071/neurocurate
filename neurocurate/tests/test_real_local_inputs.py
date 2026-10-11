"""Optional local real-file tests; never download or deserialize pickle.

Enable with NEUROCURATE_REAL_ACQUISITION_MANIFEST pointing at a verified manifest.
These tests check acquired inputs, not EEG tensor alignment or scientific RDMs.
"""
import json
import os
from pathlib import Path
import zipfile

from PIL import Image
import pytest
from neurocurate.acquisition import inspect_npy_header, verify_artifact


@pytest.fixture(scope="module")
def real_manifest():
    configured = os.environ.get("NEUROCURATE_REAL_ACQUISITION_MANIFEST")
    if not configured:
        pytest.skip("optional local real inputs not selected; no downloads")
    path = Path(configured)
    assert path.is_file(), "explicitly selected real manifest is missing"
    return json.loads(path.read_text())


def test_retained_real_file_integrity(real_manifest):
    files = real_manifest["files"] + real_manifest["extracted"]
    assert len(files) == 204
    for artifact in files:
        verify_artifact(artifact)


def test_real_test_only_inventory(real_manifest):
    eeg = [a for a in real_manifest["extracted"] if a["archive"] == "sub-01.zip"]
    images = [a for a in real_manifest["extracted"] if a["archive"] == "test_images.zip"]
    assert len(eeg) == 1 and eeg[0]["bytes"] == 217601432
    assert len(images) == len({a["member"] for a in images}) == 200
    archive = next(a for a in real_manifest["files"] if Path(a["path"]).name == "sub-01.zip")
    with zipfile.ZipFile(archive["path"]) as z:
        assert len([m for m in z.infolist() if m.filename.endswith("/preprocessed_eeg_test.npy")]) == 1
    assert not list(Path(eeg[0]["path"]).parent.rglob("*training*"))
    for artifact in images:
        with Image.open(artifact["path"]) as image:
            image.verify()


def test_real_headers_reject_unapproved_pickle(real_manifest):
    eeg = next(a for a in real_manifest["extracted"] if a["archive"] == "sub-01.zip")
    metadata = next(a for a in real_manifest["files"] if Path(a["path"]).name == "image_metadata.npy")
    for artifact in (eeg, metadata):
        report = inspect_npy_header(artifact["path"])
        assert report["serialized_dtype"] == "object" and report["serialized_shape"] == []
        assert report["pickle_loaded"] is False
        assert "allow_pickle=False" in report["allow_pickle_false_result"]


def test_approved_real_loader_alignment_and_small_rdm(real_manifest):
    if os.environ.get("NEUROCURATE_REAL_PICKLE_APPROVED") != "1":
        pytest.skip("explicit real pickle approval not selected")
    import numpy as np
    from neurocurate.realdata import load_real_manifest
    from neurocurate.eeg import load_eeg, EEGTrials, EEGConfig, extract_eeg_features
    from neurocurate.rdm import compute_rdm, validate_rdm
    eeg = next(a for a in real_manifest["extracted"] if a["archive"] == "sub-01.zip")
    metadata = next(a for a in real_manifest["files"] if Path(a["path"]).name == "image_metadata.npy")
    # Reverify the exact approved pickle inputs before deserializing them.
    verify_artifact(eeg)
    verify_artifact(metadata)
    manifest, selected, alignment = load_real_manifest(metadata["path"], Path(metadata["path"]).parent / "stimuli",
                                                       trusted_pickle=True)
    assert alignment["matched_images"] == 200 and alignment["train_test_metadata_identity_overlap"] == 0
    ids = selected.image_id.to_numpy(dtype=str)
    trials = load_eeg(eeg["path"], "sanity", ids, trusted_pickle=True, manifest=manifest)
    assert trials.data.shape == (16000, 17, 100)
    assert np.allclose(np.diff(trials.times), 0.01)
    assert np.array_equal(trials.image_ids, np.repeat(ids, 80))
    subset = EEGTrials(trials.data[:320], trials.image_ids[:320], trials.times, trials.channel_names, "sanity")
    feature_ids, features, counts = extract_eeg_features(subset, EEGConfig(time_window=(0.1,0.5), baseline=None, min_repetitions=80))
    assert features.shape == (4, 680) and np.array_equal(feature_ids, ids[:4])
    assert np.all(counts == 80)
    rdm = validate_rdm(compute_rdm(features))
    assert rdm.shape == (4, 4)
