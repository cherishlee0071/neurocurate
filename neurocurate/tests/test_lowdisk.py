import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image
from neurocurate import lowdisk
from neurocurate.vision import save_feature_cache


@pytest.fixture
def fixture(tmp_path):
    rng = np.random.default_rng(31)
    ids = np.array([f"real-format-synthetic-{i:03d}" for i in range(200)])
    rows = []
    for i, image_id in enumerate(ids):
        path = tmp_path / f"{image_id}.png"
        Image.new("RGB", (1, 1), (i, 0, 0)).save(path)
        rows.append(dict(image_id=image_id, image_path=str(path), split="sanity", eeg_index=i))
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    subjects = []
    for index in range(2):
        path = tmp_path / f"subject-{index}.npz"
        np.savez(path, eeg=rng.normal(size=(400, 2, 6)), image_ids=np.repeat(ids, 2),
                 times=np.arange(-1, 5)/10, channel_names=np.array(["O1", "O2"]), split=np.array("sanity"))
        subjects.append(dict(id=f"sub-{index}", eeg={"path": str(path), "temporary": True}))
    cache = tmp_path / "clip.npz"
    save_feature_cache(cache, ids, rng.normal(size=(200, 8)), split="sanity", model_name="clip",
                       provenance={"source": "synthetic"})
    config = dict(manifest=str(tmp_path / "manifest.csv"), work_dir=str(tmp_path / "owned"),
                  output_dir=str(tmp_path / "results"), subjects=subjects,
                  vision_features={"clip": str(cache)}, reserve_bytes=1024,
                  subject_output_budget_bytes=4*1024**2,
                  preprocessing={"baseline": [-0.1, 0], "time_window": [0, 0.4], "temporal_bins": 2})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config, ids


def write_config(path, config):
    path.write_text(yaml.safe_dump(config))


def test_dry_run_no_network_no_mutation(fixture, monkeypatch):
    path, config, _ = fixture
    monkeypatch.setattr(lowdisk, "_download", lambda *args: pytest.fail("download called during dry run"))
    report = lowdisk.run_sequential(path, dry_run=True)
    assert report["ready"] and report["estimated_peak_additional_bytes"] > 0
    assert not Path(config["output_dir"]).exists()
    assert not Path(config["work_dir"]).exists()


def test_sequential_outputs_and_single_vision_reuse(fixture, monkeypatch):
    path, config, ids = fixture
    calls = []
    original = lowdisk.load_feature_cache
    def spy(*args, **kwargs):
        calls.append(str(args[0]))
        return original(*args, **kwargs)
    monkeypatch.setattr(lowdisk, "load_feature_cache", spy)
    result = lowdisk.run_sequential(path, cleanup=True)
    assert result["subjects_verified"] == 2 and len(calls) == 2  # source + saved cache, once total
    root = Path(result["output_dir"])
    assert (root / "COMPLETE").exists()
    for subject in config["subjects"]:
        folder = root / subject["id"]
        assert lowdisk.verify_subject(folder, ids)
        assert np.load(folder / "brain_rdm.npy").shape == (200, 200)
        assert (folder / "VERIFIED").exists()
        assert Path(subject["eeg"]["path"]).exists()  # Pre-existing inputs always preserved.
    assert Path(config["vision_features"]["clip"]).exists()


@pytest.mark.parametrize("problem", ["space", "unknown_size", "archive", "checksum", "manifest", "blocked"])
def test_acquisition_blocked_before_download(fixture, monkeypatch, problem):
    path, config, _ = fixture
    artifact = config["subjects"][0]["eeg"]
    if problem == "space": config["disk_budget_bytes"] = 1
    if problem == "unknown_size": artifact["path"] = str(path.parent / "owned" / "absent.npy")
    if problem == "archive": artifact["path"] = str(path.parent / "owned" / "sub-01.zip")
    if problem == "checksum": artifact["sha256"] = "0" * 64
    if problem == "manifest": config["manifest"] = str(path.parent / "absent.csv")
    if problem == "blocked": config["availability_blockers"] = ["official array not independently accessible"]
    write_config(path, config)
    monkeypatch.setattr(lowdisk, "_download", lambda *args: pytest.fail("blocked download attempted"))
    report = lowdisk.run_sequential(path, dry_run=True)
    assert not report["ready"]
    with pytest.raises(ValueError, match="blocked"): lowdisk.run_sequential(path)
    assert not Path(config["output_dir"]).exists()


def downloadable(fixture):
    path, config, ids = fixture
    source = Path(config["subjects"][0]["eeg"]["path"])
    target = path.parent / "owned" / "subject.npz"
    config["subjects"] = [config["subjects"][0]]
    config["subjects"][0]["eeg"] = dict(path=str(target), bytes=source.stat().st_size,
                                       sha256=lowdisk.sha256(source), url="https://example.org/test-only.npz", temporary=True)
    write_config(path, config)
    return path, config, ids, source, target


def test_only_owned_reproducible_download_removed_after_verification(fixture, monkeypatch):
    path, config, ids, source, target = downloadable(fixture)
    def download(artifact, work):
        target.parent.mkdir()
        shutil.copyfile(source, target)
        return True
    monkeypatch.setattr(lowdisk, "_download", download)
    lowdisk.run_sequential(path, cleanup=True)
    assert not target.exists() and source.exists()
    root = Path(config["output_dir"]) / "sub-0"
    assert lowdisk.verify_subject(root, ids)
    assert json.loads((root / "provenance.json").read_text())["cleanup"].startswith("removed")


def test_unverified_input_never_removed(fixture, monkeypatch):
    path, config, ids, source, target = downloadable(fixture)
    def download(artifact, work):
        target.parent.mkdir()
        shutil.copyfile(source, target)
        return True
    monkeypatch.setattr(lowdisk, "_download", download)
    monkeypatch.setattr(lowdisk, "verify_subject", lambda *args: (_ for _ in ()).throw(ValueError("verification failed")))
    with pytest.raises(ValueError, match="verification failed"):
        lowdisk.run_sequential(path, cleanup=True)
    assert target.exists() and source.exists()
    assert not (Path(config["output_dir"]) / "COMPLETE").exists()


def test_saved_output_tampering_rejected(fixture):
    path, config, ids = fixture
    lowdisk.run_sequential(path)
    folder = Path(config["output_dir"]) / "sub-0"
    np.save(folder / "brain_rdm.npy", np.zeros((200, 200)))
    with pytest.raises(ValueError, match="checksum"):
        lowdisk.verify_subject(folder, ids)


def test_wrong_image_count(fixture):
    path, config, _ = fixture
    manifest = Path(config["manifest"])
    pd.read_csv(manifest).iloc[:199].to_csv(manifest, index=False)
    report = lowdisk.run_sequential(path, dry_run=True)
    assert not report["ready"] and any("exactly 200" in b for b in report["blockers"])


def test_wrong_trial_mapping_keeps_input(fixture):
    path, config, _ = fixture
    source = Path(config["subjects"][0]["eeg"]["path"])
    with np.load(source) as data: arrays = {k: data[k] for k in data.files}
    arrays["image_ids"][0] = "unknown-image"
    np.savez(source, **arrays)
    with pytest.raises(ValueError, match="exactly match"): lowdisk.run_sequential(path, cleanup=True)
    assert source.exists()


def test_download_rejects_https_downgrade(tmp_path, monkeypatch):
    import requests
    class Response:
        is_redirect = True
        headers = {"Location": "http://example.org/file"}
        def close(self): pass
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            assert url.startswith("https:") and kwargs["allow_redirects"] is False
            return Response()
    monkeypatch.setattr(requests, "Session", Session)
    with pytest.raises(ValueError, match="non-HTTPS"):
        lowdisk._download({"path": str(tmp_path / "file.npy"), "url": "https://example.org/file", "bytes": 1, "sha256": "0"*64}, tmp_path)
    assert not (tmp_path / "file.npy").exists()


@pytest.mark.parametrize("mode", ["success", "checksum", "oversize"])
def test_direct_stream_integrity_and_retained_partials(tmp_path, monkeypatch, mode):
    import hashlib
    import requests
    class Response:
        is_redirect = False
        status_code = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size):
            yield b"abc" if mode != "oversize" else b"abcdef"
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            assert kwargs.get("verify", True) is True
            assert kwargs["stream"] and not kwargs["allow_redirects"]
            return Response()
    monkeypatch.setattr(requests, "Session", Session)
    path = tmp_path / "subject.npy"
    artifact = dict(path=str(path), url="https://example.org/test.npy", bytes=3,
                    sha256=hashlib.sha256(b"abc").hexdigest() if mode != "checksum" else "0"*64)
    if mode == "success":
        assert lowdisk._download(artifact, tmp_path)
        assert path.read_bytes() == b"abc"
        assert not lowdisk._download(artifact, tmp_path)  # Existing files are not owned.
    else:
        with pytest.raises(ValueError): lowdisk._download(artifact, tmp_path)
        assert not path.exists() and list(tmp_path.glob("*.partial"))
