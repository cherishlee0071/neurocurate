"""Re-audit retained Phase 1 inputs offline, without enabling pickle loading."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image
from neurocurate.acquisition import inspect_npy_header, verify_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path,
                        default=Path("results/real_phase0/sub01/acquisition_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase1/real_data_audit.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing audit; use a new --output path")
    acquisition = json.loads(args.acquisition.read_text())
    artifacts = acquisition["files"] + acquisition["extracted"]
    for artifact in artifacts:
        verify_artifact(artifact)
    files = {Path(a["path"]).name: a for a in acquisition["files"]}
    eeg = [a for a in acquisition["extracted"] if a["archive"] == "sub-01.zip"]
    images = [a for a in acquisition["extracted"] if a["archive"] == "test_images.zip"]
    if len(eeg) != 1 or len(images) != 200:
        raise ValueError("unexpected extracted inventory")
    with zipfile.ZipFile(files["sub-01.zip"]["path"]) as archive:
        members = [m for m in archive.infolist()
                   if m.filename.split("/")[-1] == "preprocessed_eeg_test.npy"]
        if len(members) != 1 or members[0].file_size != 217601432:
            raise ValueError("invalid verified archive test member")
    names = [a["member"] for a in images]
    if len(set(names)) != 200 or len({Path(a["path"]).resolve() for a in images}) != 200:
        raise ValueError("duplicate stimulus member/path")
    for artifact in images:
        with Image.open(artifact["path"]) as image:
            image.verify()
    headers = [inspect_npy_header(eeg[0]["path"]),
               inspect_npy_header(files["image_metadata.npy"]["path"])]
    report = dict(audited_at_utc=datetime.now(timezone.utc).isoformat(), subject="sub-01",
        dataset=dict(name="THINGS-EEG2", official_osf_project="3jk45",
            preprocessed_osf_node="anp5v", archive_file_id="62f6b4ff170d520c73c27c1e",
            archive_version=1, archive_release_date="2022-08-12",
            version_evidence="outputs/lowdisk-audit/osf-preprocessed-complete.json: attributes.current_version",
            original_split="test", repository_split="sanity", training_use_prohibited=True),
        source_manifest=str(args.acquisition.resolve()), files=list(files.values()), eeg_export=eeg[0],
        image_files=images, all_file_sizes_sha256_verified=True, verified_file_count=len(artifacts),
        real_data_operations=["rehashed retained official EEG/metadata/image payloads",
            "checked local ZIP member inventory", "decoded 200 original stimulus images",
            "inspected real NPY headers", "attempted NPY loading with allow_pickle=False"],
        access=dict(network_requests_this_run=0, downloaded_bytes_this_run=0,
                    eeg_file_access=True, eeg_tensor_loaded=False, stimulus_files_access=True,
                    metadata_file_access=True, metadata_object_loaded=False),
        npy_headers=headers, eeg_tensor_shape=None, eeg_dtype=None, sampling_rate_hz=None,
        channel_count=None, trial_count=None, image_axis_count=None,
        alignment=dict(stimulus_files_present=200, unique_image_paths=200,
            unique_image_file_hashes=len({a["sha256"] for a in images}), missing_extracted_files=0,
            duplicate_archive_paths=0, matched_eeg_images=None, eeg_image_ids=None,
            exact_mapping_verified=False, alphabetical_order_assumed=False,
            ordering_mismatches=None, missing_metadata_ids=None, duplicate_metadata_ids=None,
            train_test_metadata_overlap=None, original_split_preserved=True),
        brain_rdm=dict(status="not_run", reason="explicit pickle approval and exact mapping pending"),
        vision=dict(dinov2="not_run_on_real_images", clip="not_run_on_real_images"),
        available_disk_bytes=shutil.disk_usage(Path.cwd()).free,
        memory_cgroup_limit=Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        status="BLOCKED", blockers=[
            "Earlier instruction.md requires explicit approval before pickle-based loading; approval pending.",
            "Both official NPY files contain scalar object arrays and reject allow_pickle=False.",
            "Actual tensor/time axes and exact metadata-based EEG-to-image mapping remain unverified."],
        next_experiment="After approval, validate actual metadata ordering, then run a small real-image/EEG subset using existing loaders and frozen local checkpoints.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Phase 1 real-data audit", "", "Status: BLOCKED before pickle loading and image alignment.", "",
        "Reused the fully verified subject-01 standard archive, test-only EEG export, official image metadata and 200 test JPEGs.",
        "No network access, downloads, training EEG extraction, synthetic substitution or pickle loading occurred in this run.", "",
        f"All {len(artifacts)} file sizes and SHA256 values were rechecked; all 200 image files decoded successfully.",
        "Both NPY headers are scalar object containers; their shape is not the EEG tensor shape.",
        "Actual EEG axes, sample rate, trial count and EEG-to-image mapping are unknown until approved loading.",
        "No real Brain RDM or real-image vision embeddings were computed. The split remains test/sanity.", "",
        "The attached instruction.md explicitly requires approval for pickle loading. A narrowly scoped approval request is pending.",
        "The next experiment is a small aligned real-data smoke test after that prerequisite is resolved.", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({k: report[k] for k in ("status", "verified_file_count", "alignment", "blockers")}, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
