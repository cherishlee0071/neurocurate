"""Inspect official subject-01 ZIP metadata only; no EEG payload acquisition."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import requests
from neurocurate.remotezip import MetadataRanges, inspect_test_member


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an audit")
    api = "https://api.osf.io/v2/files/62f6b4ff170d520c73c27c1e/"
    with requests.get(api, headers={"Accept-Encoding": "identity"}, stream=True,
                      allow_redirects=False, timeout=(10, 20)) as response:
        if response.status_code != 200:
            raise ValueError("OSF metadata endpoint unavailable")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("encoded OSF metadata refused")
        data = response.raw.read(65537, decode_content=False)
        if len(data) > 65536:
            raise ValueError("OSF metadata exceeds 64 KiB cap")
        source = json.loads(data)["data"]
    attributes = source["attributes"]
    expected_hash = "f25d5ea30ba6811edd7ebd460d7b271edaf527b6920ab2d3d6e7950e9ee9639f"
    if (attributes["name"] != "sub-01.zip" or attributes["size"] != 1071722467
            or attributes["extra"]["hashes"]["sha256"] != expected_hash):
        raise ValueError("official archive metadata changed; stop and review")
    url = "https://osf.io/download/nb8wr/"
    report = inspect_test_member(MetadataRanges(url, attributes["size"]))
    report.update(inspected_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_api=api, official_download_url=url,
                  osf_file_id=source["id"], archive_bytes=attributes["size"],
                  published_sha256=expected_hash, published_md5=attributes["extra"]["hashes"]["md5"],
                  source_version=attributes["extra"].get("version"),
                  osf_metadata_response_bytes=len(data),
                  disk_free_bytes=shutil.disk_usage(Path.cwd()).free)
    meminfo = {line.split(":")[0]: int(line.split()[1]) * 1024
               for line in Path("/proc/meminfo").read_text().splitlines() if line.endswith(" kB")}
    report["host_memory_available_bytes"] = meminfo["MemAvailable"]
    report["host_memory_total_bytes"] = meminfo["MemTotal"]
    for name in ("memory.max", "memory.current"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            value = path.read_text().strip()
            report["cgroup_" + name.replace(".", "_")] = int(value) if value != "max" else value
    report["storage_estimates"] = storage_estimates(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def storage_estimates(report):
    """Additional allocation, retaining archives; budgets are explicit assumptions."""
    common = dict(stimulus_zip_bytes=8129948, extracted_stimulus_bytes=8084268,
                  image_metadata_bytes=656579, output_budget_bytes=128 * 1024**2,
                  reserve_bytes=1024**3)
    base = sum(common.values())
    member = report["test_member"]
    return dict(common=common,
                full_zip_peak_bytes=base + report["archive_bytes"] + member["uncompressed_bytes"],
                range_staged_peak_bytes=base + member["compressed_bytes"] + member["uncompressed_bytes"],
                range_streamed_peak_bytes=base + member["uncompressed_bytes"],
                external_export_local_peak_bytes=base + member["uncompressed_bytes"],
                external_source_peak_bytes=base + report["archive_bytes"] + member["uncompressed_bytes"],
                processing_memory_provisional_budget_bytes=4 * 1024**3,
                memory_budget_verified_on_real_eeg=False)


if __name__ == "__main__":
    main()
