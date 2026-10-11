import numpy as np
import pytest
from neurocurate.metadata import validate_manifest, load_manifest, align_features, assert_disjoint, things_metadata_manifest


def test_alignment_by_identity():
    result = align_features(["a", "b"], ["b", "a", "c"], [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(result, [[3, 4], [1, 2]])


@pytest.mark.parametrize("source,features", [(["a", "a"], [[1], [2]]), (["a", "c"], [[1], [2]]),
                                           (["a", "b"], [[1]]), (["a", "b"], [1, 2])])
def test_alignment_failures(source, features):
    with pytest.raises(ValueError): align_features(["a", "b"], source, features)


@pytest.mark.parametrize("kind", ["column", "missing", "duplicate", "split", "alias", "index"])
def test_metadata_failures(manifest, kind):
    if kind == "column": manifest = manifest.drop(columns="image_id")
    if kind == "missing": manifest.loc[0, "image_id"] = None
    if kind == "duplicate": manifest.loc[1, "image_id"] = "a"
    if kind == "split": manifest.loc[0, "split"] = "test"
    if kind == "alias": manifest.loc[1, "image_path"] = manifest.loc[0, "image_path"]
    if kind == "index": manifest.loc[1, "eeg_index"] = 0
    with pytest.raises(ValueError): validate_manifest(manifest)


def test_relative_manifest_paths(tmp_path, manifest):
    manifest.image_path = ["a.png", "b.png"]
    path = tmp_path / "images.csv"
    manifest.to_csv(path, index=False)
    loaded = load_manifest(path)
    assert loaded.image_path[0] == str(tmp_path / "a.png")


def test_split_leakage():
    assert_disjoint(["sanity-a"], ["train-b"], ["eval-c"])
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(["a", "b"], ["b", "c"])


def test_different_ids_same_image_path_across_splits(manifest):
    manifest.loc[1, "split"] = "train"
    manifest.loc[1, "eeg_index"] = 0
    manifest.loc[1, "image_path"] = manifest.loc[0, "image_path"]
    with pytest.raises(ValueError, match="split leakage"):
        validate_manifest(manifest)


def test_official_metadata_no_class_labels(tmp_path):
    path = tmp_path / "image_metadata.npy"
    np.save(path, {"test_img_files": ["test.jpg"], "test_img_concepts": ["test-folder"],
                   "train_img_files": ["train.jpg"], "train_img_concepts": ["train-folder"]})
    with pytest.raises(ValueError, match="pickle"):
        things_metadata_manifest(path, tmp_path)
    table = things_metadata_manifest(path, tmp_path, trusted_pickle=True)
    assert table.split.tolist() == ["sanity", "train"]
    assert table.image_id.tolist() == ["test_images/test-folder/test.jpg", "training_images/train-folder/train.jpg"]
    assert "class_label" not in table


def test_missing_official_metadata(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, {"train_img_files": ["x.jpg"]})
    with pytest.raises(ValueError, match="missing official metadata"):
        things_metadata_manifest(path, tmp_path, trusted_pickle=True)
