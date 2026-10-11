"""Bounded ZIP metadata inspection. Never reads or extracts member payloads.

Deliberately supports only comment-free, single-disk, classic ZIP archives.
Unsupported layouts stop rather than probing a tail that could contain EEG.
"""
import struct
from urllib.parse import urljoin, urlparse

import requests


class MetadataRanges:
    def __init__(self, url, size):
        self.url, self.size = url, size
        self.etag = None
        self.requests = []
        self.bytes_read = 0

    def read(self, start, length):
        if length < 1 or length > 16384 or start < 0 or start + length > self.size:
            raise ValueError("invalid or oversized metadata range")
        if self.bytes_read + length > 32768:
            raise ValueError("metadata byte budget exceeded")
        end = start + length - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Match"] = self.etag
        url, redirects = self.url, []
        for _ in range(6):
            if urlparse(url).scheme != "https":
                raise ValueError("HTTPS required for every redirect")
            with requests.get(url, headers=headers, stream=True, allow_redirects=False,
                              timeout=(10, 20)) as response:
                redirects.append({"host": urlparse(url).hostname, "status": response.status_code})
                if response.is_redirect:
                    url = urljoin(url, response.headers["Location"])
                    continue  # close without consuming a redirect body
                if response.status_code != 206:
                    raise ValueError("exact HTTP 206 required; body not consumed")
                if response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                    raise ValueError("Content-Range mismatch; body not consumed")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("encoded response refused")
                if response.headers.get("Content-Length") != str(length):
                    raise ValueError("Content-Length mismatch; body not consumed")
                etag = response.headers.get("ETag")
                if not etag or etag.startswith("W/") or (self.etag and etag != self.etag):
                    raise ValueError("missing, weak or changed archive ETag")
                self.etag = etag
                data = response.raw.read(length, decode_content=False)
                if len(data) != length:
                    raise ValueError("truncated metadata response")
                self.bytes_read += length
                self.requests.append({"start": start, "end": end, "bytes": length,
                                      "content_range": response.headers["Content-Range"],
                                      "redirects": redirects})
                return data
        raise ValueError("too many redirects")


def parse_directory(data, expected_entries):
    entries, pos = [], 0
    while pos < len(data):
        if len(data) - pos < 46:
            raise ValueError("truncated central directory")
        h = struct.unpack_from("<4s6H3I5H2I", data, pos)
        stop = pos + 46 + h[10] + h[11] + h[12]
        if h[0] != b"PK\x01\x02" or stop > len(data):
            raise ValueError("invalid central directory")
        if h[13] != 0 or 0xffffffff in (h[8], h[9], h[16]):
            raise ValueError("multi-disk or ZIP64 unsupported")
        name = data[pos + 46:pos + 46 + h[10]].decode("utf8" if h[3] & 2048 else "cp437")
        entries.append(dict(name=name, flags=h[3], compression_method=h[4],
                            crc32=f"{h[7]:08x}", compressed_bytes=h[8],
                            uncompressed_bytes=h[9], local_header_offset=h[16]))
        pos = stop
    if len(entries) != expected_entries:
        raise ValueError("central directory entry count mismatch")
    return entries


def inspect_test_member(reader):
    eocd = struct.unpack("<4s4H2IH", reader.read(reader.size - 22, 22))
    if (eocd[0] != b"PK\x05\x06" or eocd[1:3] != (0, 0)
            or eocd[3] != eocd[4] or eocd[7] != 0
            or eocd[4] == 65535 or eocd[5] + eocd[6] != reader.size - 22):
        raise ValueError("unsupported ZIP end record; no further ranges requested")
    entries = parse_directory(reader.read(eocd[6], eocd[5]), eocd[4])
    matches = [e for e in entries if e["name"].split("/")[-1] == "preprocessed_eeg_test.npy"]
    if len(matches) != 1:
        raise ValueError("test member must be unique")
    member = matches[0]
    offset = member["local_header_offset"]
    if offset + 30 > eocd[6] or member["flags"] & 1 or member["compression_method"] not in (0, 8):
        raise ValueError("unsupported test member")
    h = struct.unpack("<4s5H3I2H", reader.read(offset, 30))
    if h[0] != b"PK\x03\x04" or h[2] != member["flags"] or h[3] != member["compression_method"]:
        raise ValueError("local header disagrees with central directory")
    # Check exact expected filename length before requesting variable metadata.
    encoded = member["name"].encode("utf8" if member["flags"] & 2048 else "cp437")
    if h[9] != len(encoded) or h[10] > 4096:
        raise ValueError("unexpected local metadata lengths")
    variable = reader.read(offset + 30, h[9] + h[10])
    if variable[:h[9]] != encoded:
        raise ValueError("local filename disagrees with central directory")
    if not member["flags"] & 8 and (h[6], h[7], h[8]) != (
            int(member["crc32"], 16), member["compressed_bytes"], member["uncompressed_bytes"]):
        raise ValueError("local sizes/checksum mismatch")
    start = offset + 30 + h[9] + h[10]
    end = start + member["compressed_bytes"] - 1
    next_headers = [e["local_header_offset"] for e in entries if e["local_header_offset"] > offset]
    if end >= min(next_headers + [eocd[6]]):
        raise ValueError("member overlaps another header or directory")
    return dict(entries=entries, test_member=member,
                compressed_payload_range={"start": start, "end": end, "fetched": False},
                metadata_bytes_read=reader.bytes_read, ranges=reader.requests,
                etag=reader.etag, archive_sha256_verified=False,
                member_sha256_verified=False, member_crc32_verified=False)
