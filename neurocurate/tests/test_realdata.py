import numpy as np
import pytest
from neurocurate.realdata import validate_test_order


@pytest.fixture
def official_style_metadata(tmp_path):
    concepts = [f"{i+1:05d}_concept_{i}" for i in range(200)]
    files = [f"image_{i:03d}.jpg" for i in range(200)]
    for concept, filename in zip(concepts, files):
        path = tmp_path / "test_images" / concept / filename
        path.parent.mkdir(parents=True)
        path.touch()
    document = dict(test_img_concepts=concepts, test_img_files=files,
                    test_img_concepts_THINGS=[f"origin_{i}" for i in range(200)],
                    train_img_files=["training_only.jpg"], train_img_concepts_THINGS=["train_origin"])
    return document, tmp_path


def test_numeric_mapping_preserved(official_style_metadata):
    document, root = official_style_metadata
    report = validate_test_order(document, root)
    assert report["matched_images"] == 200
    assert not report["alphabetical_order_assumed"]
    assert report["train_test_metadata_identity_overlap"] == 0


def test_permuted_metadata_not_silently_sorted(official_style_metadata):
    document, root = official_style_metadata
    for key in ["test_img_files", "test_img_concepts", "test_img_concepts_THINGS"]:
        document[key][0], document[key][1] = document[key][1], document[key][0]
    with pytest.raises(ValueError, match="refusing to sort"):
        validate_test_order(document, root)


def test_missing_mapped_stimulus(official_style_metadata):
    document, root = official_style_metadata
    (root / "test_images" / document["test_img_concepts"][0] / document["test_img_files"][0]).unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_test_order(document, root)


def test_metadata_test_train_identity_contamination(official_style_metadata):
    document, root = official_style_metadata
    document["train_img_files"] = [document["test_img_files"][0]]
    document["train_img_concepts_THINGS"] = [document["test_img_concepts_THINGS"][0]]
    with pytest.raises(ValueError, match="overlap"):
        validate_test_order(document, root)


def test_unmapped_extra_stimulus(official_style_metadata):
    document, root = official_style_metadata
    (root / "test_images" / "extra.jpg").touch()
    with pytest.raises(ValueError, match="inventory"):
        validate_test_order(document, root)
