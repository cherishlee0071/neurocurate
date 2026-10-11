import io
import zipfile

import numpy as np
import pytest
from neurocurate.acquisition import extract_member, inspect_npy_header, safe_member_name, verify_artifact
from neurocurate.lowdisk import sha256


@pytest.mark.parametrize("name", ["../escape.npy", "/absolute.npy", "a\\escape.npy", "C:/data.npy"])
def test_archive_path_escape_rejected(name):
    with pytest.raises(ValueError, match="unsafe"):
        safe_member_name(name)


def test_object_header_does_not_unpickle(tmp_path):
    path = tmp_path / "object.npy"
    np.save(path, {"real_array": np.zeros((2, 3))})
    report = inspect_npy_header(path)
    assert report["object_dtype"] and report["serialized_shape"] == []
    assert not report["pickle_loaded"]
    assert "allow_pickle=False" in report["allow_pickle_false_result"]


def test_numeric_header(tmp_path):
    path = tmp_path / "numeric.npy"
    np.save(path, np.zeros((200, 4), dtype=np.float32))
    report = inspect_npy_header(path)
    assert report["serialized_shape"] == [200, 4]
    assert report["serialized_dtype"] == "float32" and not report["object_dtype"]


def test_only_requested_member_extracted_and_preserved(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("subject/test.npy", b"test content")
        z.writestr("subject/training.npy", b"must remain in archive")
    with zipfile.ZipFile(buffer) as z:
        member = z.getinfo("subject/test.npy")
        path = extract_member(z, member, tmp_path, reserve_bytes=0)
        assert path.read_bytes() == b"test content"
        assert not (tmp_path / "subject/training.npy").exists()
        assert extract_member(z, member, tmp_path, reserve_bytes=0) == path
        path.write_bytes(b"existing different input")
        with pytest.raises(ValueError, match="differs"):
            extract_member(z, member, tmp_path, reserve_bytes=0)
        assert path.read_bytes() == b"existing different input"


def test_checksum_mismatch_aborts(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"verified input")
    artifact = dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
    verify_artifact(artifact)
    artifact["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        verify_artifact(artifact)


def test_space_refusal_before_extraction(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("test.npy", b"test content")
    from collections import namedtuple
    usage = namedtuple("Usage", "free")
    monkeypatch.setattr("neurocurate.acquisition.shutil.disk_usage", lambda _: usage(1))
    with zipfile.ZipFile(buffer) as z:
        with pytest.raises(ValueError, match="insufficient space"):
            extract_member(z, z.getinfo("test.npy"), tmp_path, reserve_bytes=0)
    assert not (tmp_path / "test.npy.partial").exists()
