"""Frozen Hugging Face DINOv2 CLS and CLIP projected image embeddings."""

import hashlib
import json
from pathlib import Path
import numpy as np
from .metadata import validate_ids

DEFAULT_CHECKPOINTS = {"dinov2": "facebook/dinov2-base", "clip": "openai/clip-vit-base-patch32"}


def save_feature_cache(path, ids, features, *, split, model_name, provenance):
    ids = validate_ids(ids)
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or len(features) != len(ids) or features.shape[1] == 0 or not np.isfinite(features).all():
        raise ValueError("invalid cached feature shape or values")
    if split not in {"sanity", "train", "evaluation"}:
        raise ValueError("invalid feature split")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("cache requires a nonempty provenance object")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image_ids=ids, features=features, split=np.array(split),
                        model_name=np.array(model_name), provenance=np.array(json.dumps(provenance)))


def load_feature_cache(path, *, split, model_name):
    with np.load(path, allow_pickle=False) as arrays:
        required = {"image_ids", "features", "split", "model_name", "provenance"}
        if not required <= set(arrays.files):
            raise ValueError("missing feature cache metadata")
        for key, expected in [("split", split), ("model_name", model_name)]:
            if arrays[key].ndim != 0 or str(arrays[key].item()) != expected:
                raise ValueError(f"feature cache {key} mismatch")
        ids = validate_ids(arrays["image_ids"])
        features = arrays["features"].copy()
        if features.ndim != 2 or features.shape[0] != len(ids) or features.shape[1] == 0 or not np.isfinite(features).all():
            raise ValueError("invalid cached features")
        provenance = json.loads(str(arrays["provenance"].item()))
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("cache requires a nonempty provenance object")
    return ids, features, provenance


def validate_cached_image_hashes(ids, paths, provenance):
    """Reject stale cached stimuli when image hashes and files are available."""
    expected = provenance.get("image_sha256")
    if expected is None:
        return {"verified": 0, "unavailable": len(ids), "reason": "cache has no stimulus hashes"}
    if not isinstance(expected, dict) or not set(ids) <= set(expected):
        raise ValueError("cached stimulus hashes do not cover selected image IDs")
    verified = 0
    for image_id, path in zip(ids, paths):
        path = Path(path)
        if path.is_file():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected[image_id]:
                raise ValueError(f"stale feature cache: stimulus hash mismatch for {image_id}")
            verified += 1
    return {"verified": verified, "unavailable": len(ids) - verified}


def _load_transformers(kind):
    # Import only when extracting, so cached-feature analyses stay offline.
    if kind == "dinov2":
        from transformers import AutoImageProcessor, Dinov2Model
        return AutoImageProcessor, Dinov2Model
    if kind == "clip":
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        return CLIPImageProcessor, CLIPVisionModelWithProjection
    raise ValueError("vision kind must be dinov2 or clip")


def extract_vision_features(ids, image_paths, *, kind, checkpoint=None, revision="main",
                            batch_size=8, device="cpu", normalize=True,
                            local_files_only=False, cache_dir=None):
    """No gradients or fitting; uses each checkpoint's official preprocessing."""
    import torch
    from PIL import Image
    ids = validate_ids(ids)
    if len(image_paths) != len(ids) or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("paths must match IDs and batch_size must be a positive integer")
    if kind not in DEFAULT_CHECKPOINTS:
        raise ValueError("vision kind must be dinov2 or clip")
    paths = [Path(p) for p in image_paths]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("missing stimulus image file")
    hashes = {str(i): hashlib.sha256(p.read_bytes()).hexdigest() for i, p in zip(ids, paths)}
    checkpoint = checkpoint or DEFAULT_CHECKPOINTS[kind]
    processor_class, model_class = _load_transformers(kind)
    options = dict(revision=revision, local_files_only=local_files_only, cache_dir=cache_dir)
    processor_options = {"use_fast": False} if kind == "dinov2" else {}
    processor = processor_class.from_pretrained(checkpoint, **options, **processor_options)
    model = model_class.from_pretrained(checkpoint, **options).to(device)
    model.requires_grad_(False)
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model(**inputs)
            features = output.last_hidden_state[:, 0] if kind == "dinov2" else output.image_embeds
            if features.ndim != 2 or not torch.isfinite(features).all():
                raise ValueError("vision model produced invalid embeddings")
            if normalize:
                norms = features.norm(dim=1, keepdim=True)
                if torch.any(norms <= 1e-12):
                    raise ValueError("vision model produced zero embeddings")
                features = features / norms
            batches.append(features.cpu().numpy())
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision is None and not Path(checkpoint).exists():
        # CLIP's nested vision config can lose the parent config's commit hash.
        # Recover it from the already downloaded Hub config, without a new request.
        from huggingface_hub import try_to_load_from_cache
        cached_config = try_to_load_from_cache(checkpoint, "config.json", cache_dir=cache_dir, revision=revision)
        if isinstance(cached_config, str) and Path(cached_config).parent.parent.name == "snapshots":
            resolved_revision = Path(cached_config).parent.name
    provenance = dict(kind=kind, checkpoint=checkpoint, revision=revision,
                      resolved_revision=resolved_revision,
                      pooling="cls" if kind == "dinov2" else "projected_image_embedding",
                      processor_class=type(processor).__name__, normalize=normalize,
                      frozen=True, image_sha256=hashes)
    return np.concatenate(batches), provenance
