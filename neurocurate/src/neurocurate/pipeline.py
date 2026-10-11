"""One-split, strict-alignment experimental execution with recorded provenance."""

from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import json
import numpy as np
import yaml
from .metadata import load_manifest, align_features, validate_ids
from .eeg import EEGConfig, load_eeg, extract_eeg_features
from .rdm import compute_rdm, nearest_neighbors
from .scoring import neurocurate_scores
from .vision import extract_vision_features, load_feature_cache, save_feature_cache, validate_cached_image_hashes
from .visualization import plot_rdms, plot_top_disagreement


def run_experiment(config_path):
    path = Path(config_path).resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    allowed = {"manifest", "image_root", "split", "subjects", "vision", "preprocessing", "rdm",
               "scoring", "output_dir", "max_images", "visualization"}
    if set(config) - allowed:
        raise ValueError(f"unknown configuration keys: {sorted(set(config) - allowed)}")
    required = {"manifest", "split", "subjects", "vision", "output_dir"}
    if not required <= set(config):
        raise ValueError(f"missing configuration keys: {sorted(required - set(config))}")
    def resolve(value):
        return (path.parent / value).resolve()
    table = load_manifest(resolve(config["manifest"]),
                          resolve(config["image_root"]) if config.get("image_root") else None)
    split = config["split"]
    if split not in {"sanity", "train", "evaluation"}:
        raise ValueError("invalid selected split")
    selected = table[table.split == split].sort_values("image_id")
    if len(selected) < 3:
        raise ValueError("selected split requires at least three images")
    full_ids = validate_ids(selected.image_id.to_numpy())
    if config.get("max_images") is not None:
        maximum = config["max_images"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 3:
            raise ValueError("max_images must be an integer >= 3")
        selected = selected.head(maximum)
    ids = validate_ids(selected.image_id.to_numpy())
    subjects = config["subjects"]
    if not isinstance(subjects, list) or len(subjects) < 2:
        raise ValueError("at least two distinct subjects are required")
    validate_ids(np.asarray([s["id"] for s in subjects]))
    subject_paths = [resolve(s["eeg"]) for s in subjects]
    if len(set(subject_paths)) != len(subject_paths):
        raise ValueError("subjects must use distinct EEG files")
    preprocessing = EEGConfig(**config.get("preprocessing", {}))
    rdm_config = config.get("rdm", {})
    if set(rdm_config) - {"brain_metric", "vision_metric"}:
        raise ValueError("unknown RDM configuration")
    scoring = config.get("scoring", {})
    if set(scoring) - {"k", "model_weights"}:
        raise ValueError("unknown scoring configuration")
    k = scoring.get("k", 5)
    # Validate k before loading images or model weights.
    nearest_neighbors(np.zeros((len(ids), len(ids))), k, ids)
    vision_specs = config["vision"]
    if not isinstance(vision_specs, dict) or not vision_specs:
        raise ValueError("vision must be a nonempty model mapping")
    if set(vision_specs) - {"dinov2", "clip"}:
        raise ValueError("vision names must be dinov2 or clip")
    output = resolve(config["output_dir"])
    # Never overwrite a previous run or put generated outputs over input files.
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}; choose a new run directory")
    output.mkdir(parents=True)
    rdms, subject_neighbors, model_neighbors = {}, [], {}
    provenance = {"split": split, "image_ids": ids.tolist(), "excluded_by_max_images": sorted(set(full_ids) - set(ids)),
                  "preprocessing": asdict(preprocessing), "config": config, "subjects": {}, "models": {},
                  "versions": {p: importlib.metadata.version(p) for p in ["neurocurate", "numpy", "torch", "transformers", "mne"]},
                  "score_definition": "mean pairwise subject Jaccard similarity * weighted mean brain-model Jaccard disagreement",
                  "training": "none; no parameters fitted across images or splits"}
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    for subject in subjects:
        if set(subject) - {"id", "eeg", "trusted_pickle", "trial_metadata"}:
            raise ValueError("unknown subject configuration")
        name = subject["id"]
        if not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("subject ID must contain only letters, numbers, '-' or '_'")
        trials = load_eeg(resolve(subject["eeg"]), split, full_ids,
                          trusted_pickle=subject.get("trusted_pickle", False), manifest=table,
                          trial_metadata=resolve(subject["trial_metadata"]) if subject.get("trial_metadata") else None)
        subject_ids, features, counts = extract_eeg_features(trials, preprocessing)
        features = align_features(ids, subject_ids, features)
        rdm = compute_rdm(features, rdm_config.get("brain_metric", "correlation"))
        rdms[f"brain_{name}"] = rdm
        subject_neighbors.append(nearest_neighbors(rdm, k, ids))
        count_map = dict(zip(subject_ids, counts))
        provenance["subjects"][name] = {"path": str(resolve(subject["eeg"])), "repetitions": {i: int(count_map[i]) for i in ids}}
        np.savez_compressed(output / f"brain_{name}_features.npz", image_ids=ids, features=features,
                            repetitions=np.array([count_map[i] for i in ids]), split=np.array(split))
    for name, specification in vision_specs.items():
        specification = dict(specification)
        cache = specification.pop("features", None)
        if cache is not None:
            if specification:
                raise ValueError("cached features cannot be combined with extraction options")
            source_ids, features, model_provenance = load_feature_cache(resolve(cache), split=split, model_name=name)
            if not set(source_ids) <= set(full_ids):
                raise ValueError("feature cache contains IDs outside the selected split")
            features = align_features(ids, source_ids, features)
            model_provenance["cache_image_validation"] = validate_cached_image_hashes(
                ids, selected.image_path.tolist(), model_provenance)
        else:
            allowed_options = {"checkpoint", "revision", "batch_size", "device", "normalize", "local_files_only", "cache_dir"}
            if set(specification) - allowed_options:
                raise ValueError("unknown vision extraction option")
            if "cache_dir" in specification:
                specification["cache_dir"] = str(resolve(specification["cache_dir"]))
            features, model_provenance = extract_vision_features(ids, selected.image_path.tolist(), kind=name, **specification)
        provenance["models"][name] = model_provenance
        save_feature_cache(output / f"{name}_features.npz", ids, features, split=split,
                           model_name=name, provenance=model_provenance)
        rdm = compute_rdm(features, rdm_config.get("vision_metric", "cosine"))
        rdms[name] = rdm
        model_neighbors[name] = nearest_neighbors(rdm, k, ids)
    scores = neurocurate_scores(ids, subject_neighbors, model_neighbors, scoring.get("model_weights"))
    scores.to_csv(output / "scores.csv", index=False)
    np.save(output / "image_ids.npy", ids)
    for name, rdm in rdms.items():
        np.save(output / f"{name}_rdm.npy", rdm)
    np.savez_compressed(output / "neighbors.npz", image_ids=ids,
                        **{f"brain_{s['id']}": n for s, n in zip(subjects, subject_neighbors)}, **model_neighbors)
    visualization = config.get("visualization", {})
    if set(visualization) - {"enabled", "top_n"}:
        raise ValueError("unknown visualization configuration")
    if visualization.get("enabled", True):
        plot_rdms(rdms, ids, output / "rdms.png")
        plot_top_disagreement(scores, dict(zip(selected.image_id, selected.image_path)),
                              output / "top_disagreement.png", visualization.get("top_n", 10))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "COMPLETE").write_text("Analysis completed; see provenance.json for data and model origin.\n")
    return output
