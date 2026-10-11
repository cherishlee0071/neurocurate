"""Bounded direct-file acquisition and sequential sanity-subject processing.

Archive acquisition/extraction is deliberately unsupported. Unknown sizes or
unavailable test-only sources block all downloads, including shared assets.
"""
from pathlib import Path
from urllib.parse import urlparse
import gc
import hashlib
import importlib.metadata
import json
import shutil
import uuid
import numpy as np
import yaml
from .eeg import EEGConfig, load_eeg, extract_eeg_features
from .metadata import load_manifest, align_features, validate_ids
from .rdm import compute_rdm, validate_rdm
from .vision import load_feature_cache, save_feature_cache, validate_cached_image_hashes


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    allowed = {"manifest", "subjects", "shared_files", "vision_features", "preprocessing", "brain_metric",
               "work_dir", "output_dir", "reserve_bytes", "subject_output_budget_bytes", "disk_budget_bytes",
               "availability_blockers", "source_inventory"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("invalid or unknown low-disk configuration fields")
    for key in ["manifest", "subjects", "work_dir", "output_dir"]:
        if key not in config:
            raise ValueError(f"missing configuration: {key}")
    def resolve(value):
        return (path.parent / value).resolve()
    config["manifest"] = str(resolve(config["manifest"]))
    for key in ["work_dir", "output_dir"]:
        config[key] = str(resolve(config[key]))
    subjects = config["subjects"]
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("at least one subject is required")
    validate_ids([s["id"] for s in subjects])
    if any(not s["id"].replace("-", "").replace("_", "").isalnum() for s in subjects):
        raise ValueError("invalid subject ID")
    for subject in subjects:
        if set(subject) - {"id", "eeg", "trusted_pickle", "trial_metadata"}:
            raise ValueError("unknown subject fields")
        if "trial_metadata" in subject:
            subject["trial_metadata"] = str(resolve(subject["trial_metadata"]))
    artifacts = list(config.get("shared_files", [])) + [s["eeg"] for s in subjects]
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) - {"path", "url", "bytes", "sha256", "temporary", "blocked_reason"}:
            raise ValueError("invalid artifact declaration")
        artifact["path"] = str(resolve(artifact["path"]))
    if len({a["path"] for a in artifacts}) != len(artifacts):
        raise ValueError("artifact paths must be distinct")
    vision = config.get("vision_features", {})
    if not isinstance(vision, dict) or set(vision) - {"dinov2", "clip"}:
        raise ValueError("vision_features must map dinov2/clip to existing caches")
    config["vision_features"] = {k: str(resolve(v)) for k, v in vision.items()}
    _integer(config.get("reserve_bytes", 1024**3), "reserve_bytes")
    _integer(config.get("subject_output_budget_bytes", 64 * 1024**2), "subject_output_budget_bytes", 1)
    return config


def _disk(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def storage_plan(config, *, cleanup=False):
    """Pure local dry run; never issues a network request or creates directories."""
    blockers = list(config.get("availability_blockers", []))
    work, output = Path(config["work_dir"]), Path(config["output_dir"])
    if work == output or work in output.parents or output in work.parents:
        blockers.append("work_dir and output_dir must not overlap")
    if output.exists():
        blockers.append("output directory already exists; select a new run")
    groups = [config.get("shared_files", [])] + [[s["eeg"]] for s in config["subjects"]]
    downloads, sizes = [], []
    for group in groups:
        group_bytes = 0
        for artifact in group:
            path = Path(artifact["path"])
            item = dict(artifact, present=path.is_file(), download_required=not path.is_file())
            downloads.append(item)
            if artifact.get("blocked_reason"):
                blockers.append(artifact["blocked_reason"])
            if path.suffix.lower() in {".zip", ".tar", ".7z"} or path.name.endswith(".tar.gz"):
                blockers.append(f"archive acquisition unsupported: {path.name}")
            if path.is_file():
                if artifact.get("bytes") is not None and path.stat().st_size != artifact["bytes"]:
                    blockers.append(f"existing file size mismatch: {path.name}")
                if artifact.get("sha256") and sha256(path) != artifact["sha256"]:
                    blockers.append(f"existing file checksum mismatch: {path.name}")
                continue
            size = artifact.get("bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                blockers.append(f"unknown download size: {path.name}")
                continue
            group_bytes += size
            digest = artifact.get("sha256", "") or ""
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                blockers.append(f"missing SHA256 for download: {path.name}")
            if urlparse(artifact.get("url") or "").scheme != "https":
                blockers.append(f"missing HTTPS direct-file URL: {path.name}")
            if work not in path.parents:
                blockers.append(f"download destination must be inside work_dir: {path.name}")
        sizes.append(group_bytes)
    manifest = Path(config["manifest"])
    image_bytes = None
    if manifest.is_file():
        table = load_manifest(manifest)
        selected = table[table.split == "sanity"].sort_values("image_id")
        if len(selected) != 200:
            blockers.append("sanity manifest must contain exactly 200 distinct images")
        declared = {a["path"] for a in config.get("shared_files", [])}
        for path in selected.image_path:
            if not Path(path).is_file() and path not in declared:
                blockers.append(f"missing stimulus with no download declaration: {path}")
        image_bytes = sum(Path(p).stat().st_size for p in selected.image_path if Path(p).is_file())
    else:
        blockers.append("exact 200-image manifest unavailable; mapping cannot be inferred from numeric labels")
    for subject in config["subjects"]:
        if subject.get("trial_metadata") and not Path(subject["trial_metadata"]).is_file():
            blockers.append(f"missing trial metadata: {subject['id']}")
        if Path(subject["eeg"]["path"]).suffix == ".npy" and subject.get("trusted_pickle") is not True:
            blockers.append(f"official NPY requires trusted_pickle consent: {subject['id']}")
    for name, cache in config["vision_features"].items():
        if not Path(cache).is_file():
            blockers.append(f"missing reusable {name} cache; extract once separately before this run")
    output_budget = config.get("subject_output_budget_bytes", 64 * 1024**2)
    # Cleanup only saves storage for subject artifacts explicitly marked temporary.
    retained = sum(size for size, s in zip(sizes[1:], config["subjects"])
                   if not (cleanup and s["eeg"].get("temporary") is True))
    ephemeral = max([size for size, s in zip(sizes[1:], config["subjects"])
                     if cleanup and s["eeg"].get("temporary") is True] or [0])
    reserve = config.get("reserve_bytes", 1024**3)
    estimated = sizes[0] + retained + ephemeral + output_budget * (len(config["subjects"]) + 1) + reserve
    free = min(_disk(work), _disk(output))
    budget = config.get("disk_budget_bytes")
    if budget is not None:
        free = min(free, _integer(budget, "disk_budget_bytes"))
    unknown = any(b.startswith("unknown download size:") for b in blockers)
    if estimated > free:
        blockers.append("estimated peak allocation plus reserve exceeds available disk")
    return {"ready": not blockers, "cleanup_requested": cleanup, "available_bytes": free,
            "estimated_peak_additional_bytes": None if unknown else estimated,
            "known_allocation_plus_reserve_bytes": estimated, "reserve_bytes": reserve,
            "subject_output_budget_bytes": output_budget, "existing_stimulus_bytes": image_bytes,
            "downloads": downloads, "source_inventory": config.get("source_inventory", []),
            "blockers": sorted(set(blockers))}


def _download(artifact, work):
    """Stream one hash-pinned file, checking space and size; retain failed partials."""
    import requests
    path = Path(artifact["path"])
    if path.exists():
        return False  # Never claim ownership of a pre-existing input.
    if Path(work) not in path.resolve().parents:
        raise ValueError("download path escapes work_dir")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    url, count, digest = artifact["url"], 0, hashlib.sha256()
    with requests.Session() as session:
        response = None
        for _ in range(10):
            if urlparse(url).scheme != "https":
                raise ValueError("refusing non-HTTPS redirect")
            response = session.get(url, stream=True, allow_redirects=False, timeout=(10, 60),
                                   headers={"Accept-Encoding": "identity"})
            if response.is_redirect:
                from urllib.parse import urljoin
                url = urljoin(url, response.headers["Location"])
                response.close()
            else:
                break
        else:
            raise ValueError("too many download redirects")
        with response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("expected a complete direct-file response")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != artifact["bytes"]:
                raise ValueError("remote size differs from audited size")
            with partial.open("xb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    count += len(chunk)
                    if count > artifact["bytes"]:
                        raise ValueError("download exceeded audited byte limit; partial retained")
                    stream.write(chunk)
                    digest.update(chunk)
        if count != artifact["bytes"] or digest.hexdigest() != artifact["sha256"]:
            raise ValueError("download size/checksum mismatch; partial retained for inspection")
        if path.exists():
            raise FileExistsError("destination appeared during download; partial retained")
        partial.rename(path)
    return True


def verify_subject(output, expected_ids):
    """Reload features and recompute the full RDM before permitting cleanup."""
    output = Path(output)
    provenance = json.loads((output / "provenance.json").read_text())
    for name, digest in provenance["output_sha256"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"saved output checksum mismatch: {name}")
    ids = np.load(output / "image_ids.npy", allow_pickle=False)
    if ids.shape != (200,) or not np.array_equal(ids, expected_ids):
        raise ValueError("saved IDs differ from the exact 200-image order")
    with np.load(output / "features.npz", allow_pickle=False) as arrays:
        features = arrays["features"]
        if not np.array_equal(arrays["image_ids"], ids) or str(arrays["split"].item()) != "sanity":
            raise ValueError("saved feature identity/split mismatch")
        if features.shape[0] != 200 or not np.isfinite(features).all():
            raise ValueError("invalid saved EEG features")
    rdm = validate_rdm(np.load(output / "brain_rdm.npy", allow_pickle=False))
    if rdm.shape != (200, 200) or not np.allclose(rdm, compute_rdm(features, provenance["metric"]), atol=1e-12):
        raise ValueError("saved RDM does not reproduce from features")
    return True


def run_sequential(config_path, *, dry_run=False, cleanup=False):
    config = _config(config_path)
    plan = storage_plan(config, cleanup=cleanup)
    if dry_run:
        return plan
    if not plan["ready"]:
        raise ValueError("acquisition blocked: " + "; ".join(plan["blockers"]))
    output = Path(config["output_dir"])
    output.mkdir(parents=True)
    (output / "storage_plan.json").write_text(json.dumps(plan, indent=2))
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    for artifact in config.get("shared_files", []):
        _download(artifact, config["work_dir"])
    table = load_manifest(config["manifest"])
    selected = table[table.split == "sanity"].sort_values("image_id")
    ids = validate_ids(selected.image_id.to_numpy())
    if len(ids) != 200 or any(not Path(p).is_file() for p in selected.image_path):
        raise ValueError("missing exact 200-image sanity inputs")
    selected.to_csv(output / "manifest.csv", index=False)
    stimulus_hashes = {i: sha256(p) for i, p in zip(ids, selected.image_path)}
    (output / "stimulus_hashes.json").write_text(json.dumps(stimulus_hashes, indent=2))
    # Vision caches are validated and persisted ONCE, outside the subject loop.
    vision_reference = {}
    for name, path in config["vision_features"].items():
        source_ids, features, provenance = load_feature_cache(path, split="sanity", model_name=name)
        if set(source_ids) != set(ids):
            raise ValueError("vision cache must cover exactly the 200 sanity images")
        features = align_features(ids, source_ids, features)
        provenance["cache_image_validation"] = validate_cached_image_hashes(ids, selected.image_path.tolist(), provenance)
        cache = output / f"{name}_features.npz"
        save_feature_cache(cache, ids, features, split="sanity", model_name=name, provenance=provenance)
        load_feature_cache(cache, split="sanity", model_name=name)
        vision_reference[name] = {"path": cache.name, "sha256": sha256(cache)}
        del features
    budget = config.get("subject_output_budget_bytes", 64 * 1024**2)
    if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > budget:
        raise ValueError("shared outputs exceed declared output budget; no EEG downloaded")
    for subject in config["subjects"]:
        artifact = subject["eeg"]
        missing = 0 if Path(artifact["path"]).is_file() else artifact["bytes"]
        if _disk(config["work_dir"]) < missing + budget + config.get("reserve_bytes", 1024**3):
            raise ValueError("disk space changed; stopping before subject download")
        created = _download(artifact, config["work_dir"])
        raw_path = Path(artifact["path"])
        input_digest = sha256(raw_path)
        trials = load_eeg(raw_path, "sanity", ids, trusted_pickle=subject.get("trusted_pickle", False),
                          manifest=table, trial_metadata=subject.get("trial_metadata"))
        feature_ids, features, counts = extract_eeg_features(trials, EEGConfig(**config.get("preprocessing", {})))
        features = align_features(ids, feature_ids, features)
        count_map = dict(zip(feature_ids, counts))
        metric = config.get("brain_metric", "correlation")
        rdm = compute_rdm(features, metric)
        folder = output / subject["id"]
        folder.mkdir()
        prospective = features.nbytes + rdm.nbytes + ids.nbytes + 1024 * 1024
        if prospective > budget:
            raise ValueError("subject outputs exceed declared output budget; input retained")
        np.save(folder / "image_ids.npy", ids)
        np.save(folder / "brain_rdm.npy", rdm)
        np.savez_compressed(folder / "features.npz", image_ids=ids, features=features, split=np.array("sanity"),
                            repetitions=np.array([count_map[i] for i in ids]))
        (folder / "config.yaml").write_text(yaml.safe_dump(config))
        provenance = {"subject": subject["id"], "split": "sanity", "input_path": str(raw_path),
                      "input_sha256": input_digest, "input_bytes": raw_path.stat().st_size,
                      "metric": metric, "image_ids": ids.tolist(), "trial_count": len(trials.image_ids),
                      "trial_mapping": "validated exact identity set; official NPY mapped by manifest eeg_index",
                      "manifest_sha256": sha256(output / "manifest.csv"), "vision": vision_reference,
                      "stimulus_hashes_sha256": sha256(output / "stimulus_hashes.json"),
                      "versions": {p: importlib.metadata.version(p) for p in ["neurocurate", "numpy", "scipy", "mne"]},
                      "trial_metadata_sha256": sha256(subject["trial_metadata"]) if subject.get("trial_metadata") else None,
                      "repetitions": {i: int(count_map[i]) for i in ids},
                      "cleanup": "input retained", "output_sha256": {}}
        provenance["output_sha256"] = {p.name: sha256(p) for p in folder.iterdir()}
        (folder / "provenance.json").write_text(json.dumps(provenance, indent=2))
        verify_subject(folder, ids)
        if sum(p.stat().st_size for p in folder.iterdir()) > budget:
            raise ValueError("actual subject output size exceeds budget; input retained")
        (folder / "VERIFIED").write_text("200-image IDs, features, checksums and recomputed RDM verified\n")
        if cleanup and created and artifact.get("temporary") is True:
            if raw_path.is_symlink() or Path(config["work_dir"]) not in raw_path.resolve().parents:
                raise ValueError("refusing cleanup outside owned work directory")
            if sha256(raw_path) != artifact["sha256"] or input_digest != artifact["sha256"]:
                raise ValueError("input changed after verification; refusing cleanup")
            raw_path.unlink()  # Only this run's hash-pinned, reproducible subject download.
            provenance["cleanup"] = "removed this run's explicitly temporary verified download"
            (folder / "provenance.json").write_text(json.dumps(provenance, indent=2))
        del trials, features, rdm
        gc.collect()
    (output / "COMPLETE").write_text("All sequential subject outputs verified\n")
    return {"output_dir": str(output), "subjects_verified": len(config["subjects"])}
