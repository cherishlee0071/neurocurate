import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import yaml
from neurocurate.pipeline import run_experiment

spec = importlib.util.spec_from_file_location("make_synthetic", Path(__file__).parents[1] / "scripts/make_synthetic.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complete_synthetic_pipeline_and_visualizations(tmp_path):
    path = module.make_synthetic(tmp_path / "fixture")
    output = run_experiment(path)
    scores = pd.read_csv(output / "scores.csv")
    assert len(scores) == 12 and scores.score.between(0, 1).all()
    assert np.load(output / "image_ids.npy", allow_pickle=False).shape == (12,)
    assert scores.score.is_monotonic_decreasing
    for name in ["brain_subject-1", "brain_subject-2", "brain_subject-3", "dinov2", "clip"]:
        rdm = np.load(output / f"{name}_rdm.npy")
        assert rdm.shape == (12, 12)
        np.testing.assert_allclose(rdm, rdm.T)
    for name in ["rdms.png", "top_disagreement.png"]:
        assert (output / name).read_bytes().startswith(b"\x89PNG")
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["models"]["clip"]["pretrained_model_executed"] is False
    assert (output / "COMPLETE").exists()
    with pytest.raises(FileExistsError): run_experiment(path)


@pytest.mark.parametrize("problem", ["split", "duplicate_subject", "bad_k", "unknown_option", "mixed_cache"])
def test_invalid_pipeline_config(tmp_path, problem):
    path = module.make_synthetic(tmp_path / "fixture")
    config = yaml.safe_load(path.read_text())
    if problem == "split": config["split"] = "train"
    if problem == "duplicate_subject": config["subjects"][1] = config["subjects"][0].copy()
    if problem == "bad_k": config["scoring"]["k"] = 12
    if problem == "unknown_option": config["typo"] = True
    if problem == "mixed_cache": config["vision"]["clip"]["batch_size"] = 4
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError): run_experiment(path)


def test_manifest_leakage_rejected(tmp_path):
    path = module.make_synthetic(tmp_path / "fixture")
    csv = path.parent / "images.csv"
    table = pd.read_csv(csv)
    extra = table.iloc[[0]].copy()
    extra["split"] = "train"
    pd.concat([table, extra]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        run_experiment(path)


def test_feature_split_leakage_rejected(tmp_path):
    path = module.make_synthetic(tmp_path / "fixture")
    cache_path = path.parent / "clip.npz"
    with np.load(cache_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["split"] = np.array("train")
    np.savez(cache_path, **arrays)
    with pytest.raises(ValueError, match="split mismatch"):
        run_experiment(path)


def test_max_images_tracks_subset(tmp_path):
    path = module.make_synthetic(tmp_path / "fixture")
    config = yaml.safe_load(path.read_text())
    config["max_images"] = 5
    config["visualization"]["enabled"] = False
    path.write_text(yaml.safe_dump(config))
    output = run_experiment(path)
    provenance = json.loads((output / "provenance.json").read_text())
    assert len(provenance["image_ids"]) == 5
    assert len(provenance["excluded_by_max_images"]) == 7
    assert len(pd.read_csv(output / "scores.csv")) == 5
