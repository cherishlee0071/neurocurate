"""Acquire pinned official files and inspect headers; never load pickle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image
from neurocurate.acquisition import acquire, extract_member, inspect_npy_header
from neurocurate.lowdisk import sha256


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/local/real_phase0"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/real_phase0/sub01"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    audited = json.loads(Path("data/lowdisk_source_audit.json").read_text())
    selected = [a for a in audited["preprocessed_files"] if a["name"] == "sub-01.zip"] + [a for a in audited["image_files"]
                if a["name"] in {"test_images.zip", "image_metadata.npy"}]
    if len(selected) != 3:
        raise ValueError("required audited files are missing or duplicated")
    work = args.data_dir.resolve()
    artifacts = [dict(path=str(work / item["name"]), url=item["url"],
                      bytes=item["bytes"], sha256=item["sha256"]) for item in selected]
    audit_root = work
    while not audit_root.exists():
        audit_root = audit_root.parent
    free = shutil.disk_usage(audit_root).free
    missing = sum(a["bytes"] for a in artifacts if not Path(a["path"]).exists())
    peak = missing + 217601432 + 8084268 + 128 * 1024**2 + 1024**3
    cgroup = Path("/sys/fs/cgroup")
    memory = {name: (cgroup / name).read_text().strip() for name in ("memory.max", "memory.current")}
    meminfo = {line.split(":")[0]: int(line.split()[1]) * 1024
               for line in Path("/proc/meminfo").read_text().splitlines() if line.endswith(" kB")}
    memory["host_available_bytes"] = meminfo["MemAvailable"]
    headroom = meminfo["MemAvailable"]
    if memory["memory.max"] != "max":
        headroom = min(headroom, int(memory["memory.max"]) - int(memory["memory.current"]))
    memory["effective_headroom_bytes"] = headroom
    plan = dict(artifacts=artifacts, free_disk_bytes=free, estimated_peak_additional_bytes=peak,
                cgroup_memory=memory, reserve_bytes=1024**3, approved_scope="subject-01 standard + test assets",
                excludes=["63-channel ZIPs", "other subjects", "training EEG extraction", "pickle loading"])
    print(json.dumps(plan, indent=2), flush=True)
    if peak > free:
        raise ValueError("estimated peak exceeds available disk; no downloads")
    if headroom < 4 * 1024**3:
        raise ValueError("insufficient headroom for provisional 4 GiB processing budget")
    if args.dry_run:
        return
    if (args.output_dir / "acquisition_manifest.json").exists():
        raise FileExistsError("preserve existing acquisition report; select a new output directory")
    work.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), plan=plan,
                    files=[], extracted=[], checkpoints=[], verified_zip_preserved=True,
                    status="acquisition_in_progress")
    def checkpoint(stage):
        manifest["checkpoints"].append(dict(stage=stage, disk_free_bytes=shutil.disk_usage(work).free))
        save(output / "acquisition_manifest.json", manifest)
    checkpoint("before_download")
    for artifact in artifacts:
        manifest["files"].append(acquire(artifact, work, 1024**3))
        checkpoint("verified " + Path(artifact["path"]).name)
        print("Verified " + Path(artifact["path"]).name, flush=True)
    with zipfile.ZipFile(work / "sub-01.zip") as archive:
        matches = [m for m in archive.infolist() if m.filename.split("/")[-1] == "preprocessed_eeg_test.npy"]
        if len(matches) != 1 or matches[0].file_size != 217601432:
            raise ValueError("unexpected test member inventory/size")
        member = matches[0]
        if member.compress_size != 208634917:
            raise ValueError("test member compressed size differs from metadata audit")
        eeg = extract_member(archive, member, work / "eeg")
        manifest["extracted"].append(dict(path=str(eeg), archive="sub-01.zip", member=member.filename,
                                          bytes=eeg.stat().st_size, sha256=sha256(eeg), crc32_verified=True))
    checkpoint("test EEG extracted; training not extracted")
    with zipfile.ZipFile(work / "test_images.zip") as archive:
        members = [m for m in archive.infolist() if not m.is_dir()
                   and Path(m.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and not m.filename.startswith("__MACOSX/")]
        if len(members) != 200 or len({m.filename for m in members}) != 200:
            raise ValueError("official stimulus ZIP must contain exactly 200 unique original images")
        if sum(m.file_size for m in members) + 1024**3 > shutil.disk_usage(work).free:
            raise ValueError("insufficient space for all stimulus extraction")
        for member in members:
            path = extract_member(archive, member, work / "stimuli")
            with Image.open(path) as image:
                image.verify()
            manifest["extracted"].append(dict(path=str(path), archive="test_images.zip", member=member.filename,
                                              bytes=path.stat().st_size, sha256=sha256(path), crc32_verified=True))
    checkpoint("200 original images extracted and decoded")
    reports = [inspect_npy_header(eeg), inspect_npy_header(work / "image_metadata.npy")]
    save(output / "eeg_shape_report.json", dict(subject="sub-01", header_inspections=reports,
         real_tensor_shape=None, axes=None, sampling_rate_hz=None, repetition_structure=None,
         status="blocked_pending_explicit_pickle_approval" if any(r["object_dtype"] for r in reports)
                else "headers_inspected; mapping not yet verified"))
    save(output / "image_alignment_report.json", dict(original_stimulus_count=200,
         exact_eeg_mapping_verified=False, metadata_pickle_loaded=False,
         status="blocked_pending_metadata_inspection", alphabetical_order_assumed=False))
    manifest["status"] = "verified_acquisition_complete; no_pickle_loaded"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["observed_peak_additional_disk_bytes"] = free - min(c["disk_free_bytes"] for c in manifest["checkpoints"])
    manifest["disk_peak_measurement_note"] = "filesystem checkpoint deltas, including other concurrent writes; not continuous telemetry"
    checkpoint("safe header inspection complete")
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
