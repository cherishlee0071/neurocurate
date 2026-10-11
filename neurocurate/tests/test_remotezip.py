import io
import zipfile

import pytest
from neurocurate.remotezip import MetadataRanges, inspect_test_member, parse_directory


class LocalMetadata:
    def __init__(self, archive):
        self.archive, self.size = archive, len(archive)
        self.bytes_read, self.requests, self.etag = 0, [], '"synthetic"'

    def read(self, start, length):
        self.bytes_read += length
        self.requests.append((start, length))
        return self.archive[start:start + length]


def archive(names=("sub-01/preprocessed_eeg_test.npy",), comment=b""):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.writestr(name, b"synthetic payload" * 100)
        z.comment = comment
    return output.getvalue()


def test_metadata_ranges_never_intersect_payload():
    reader = LocalMetadata(archive())
    report = inspect_test_member(reader)
    member = report["test_member"]
    assert member["uncompressed_bytes"] == len(b"synthetic payload" * 100)
    payload = report["compressed_payload_range"]
    for start, length in reader.requests:
        assert start + length <= payload["start"] or start > payload["end"]
    assert not report["archive_sha256_verified"]
    assert not report["member_crc32_verified"]


@pytest.mark.parametrize("names", [(), ("other.npy",),
    ("a/preprocessed_eeg_test.npy", "b/preprocessed_eeg_test.npy")])
def test_unique_test_entry_required(names):
    with pytest.raises(ValueError, match="unique"):
        inspect_test_member(LocalMetadata(archive(names)))


def test_comment_stops_after_only_22_bytes():
    reader = LocalMetadata(archive(comment=b"comment"))
    with pytest.raises(ValueError, match="end record"):
        inspect_test_member(reader)
    assert reader.bytes_read == 22


def test_truncated_directory():
    with pytest.raises(ValueError, match="truncated"):
        parse_directory(b"PK", 1)


def test_local_filename_mismatch():
    data = bytearray(archive())
    data[30] = ord("X")
    with pytest.raises(ValueError, match="filename"):
        inspect_test_member(LocalMetadata(bytes(data)))


@pytest.mark.parametrize("status,headers", [
    (200, {}), (206, {"Content-Range": "bytes 0-21/100"}),
    (206, {"Content-Range": "bytes 78-99/100", "Content-Length": "23"}),
    (206, {"Content-Range": "bytes 78-99/100", "Content-Length": "22", "ETag": 'W/"weak"'}),
])
def test_invalid_http_response_body_not_read(monkeypatch, status, headers):
    class Response:
        status_code = status
        is_redirect = False
        def __init__(self):
            self.headers = headers
            self.raw = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, *args, **kwargs):
            pytest.fail("invalid HTTP body must not be read")
    monkeypatch.setattr("neurocurate.remotezip.requests.get", lambda *a, **k: Response())
    with pytest.raises(ValueError):
        MetadataRanges("https://example.org/a.zip", 100).read(78, 22)


def test_http_downgrade_no_request(monkeypatch):
    monkeypatch.setattr("neurocurate.remotezip.requests.get", lambda *a, **k: pytest.fail("HTTP requested"))
    with pytest.raises(ValueError, match="HTTPS"):
        MetadataRanges("http://example.org/a.zip", 100).read(78, 22)


def test_oversized_range_no_request():
    with pytest.raises(ValueError, match="oversized"):
        MetadataRanges("https://example.org/a.zip", 100000).read(0, 16385)
