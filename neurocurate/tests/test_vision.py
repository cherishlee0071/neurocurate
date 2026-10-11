from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image
from neurocurate import vision


@pytest.mark.parametrize("kind", ["dinov2", "clip"])
def test_frozen_adapter_with_fake_backend(tmp_path, monkeypatch, kind):
    class Processor:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            assert kwargs["local_files_only"] is True
            return cls()
        def __call__(self, *, images, return_tensors):
            assert all(image.mode == "RGB" for image in images)
            return {"pixel_values": torch.ones(len(images), 3, 2, 2)}
    class Model(torch.nn.Module):
        instance = None
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(_commit_hash="test-revision")
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            cls.instance = cls()
            return cls.instance
        def forward(self, pixel_values):
            assert not self.training and not torch.is_grad_enabled() and not self.weight.requires_grad
            embeddings = torch.tensor([3., 4.]).repeat(len(pixel_values), 1)
            return SimpleNamespace(last_hidden_state=embeddings[:, None, :], image_embeds=embeddings)
    monkeypatch.setattr(vision, "_load_transformers", lambda k: (Processor, Model))
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.new("L", (3, 3), i).save(path)
        paths.append(path)
    features, provenance = vision.extract_vision_features(["a", "b", "c"], paths, kind=kind,
                                                          batch_size=2, local_files_only=True)
    assert features.shape == (3, 2)
    np.testing.assert_allclose(features, np.tile([0.6, 0.8], (3, 1)))
    assert provenance["frozen"] is True
    assert provenance["resolved_revision"] == "test-revision"
    assert len(provenance["image_sha256"]) == 3


def test_cache_identity_split_provenance(tmp_path):
    path = tmp_path / "cache.npz"
    vision.save_feature_cache(path, ["a", "b"], [[1, 2], [3, 4]], split="sanity", model_name="clip",
                              provenance={"source": "synthetic"})
    ids, features, provenance = vision.load_feature_cache(path, split="sanity", model_name="clip")
    assert ids.tolist() == ["a", "b"] and features.shape == (2, 2)
    assert provenance["source"] == "synthetic"
    with pytest.raises(ValueError, match="split mismatch"):
        vision.load_feature_cache(path, split="train", model_name="clip")
    with pytest.raises(ValueError, match="model_name mismatch"):
        vision.load_feature_cache(path, split="sanity", model_name="dinov2")


def test_missing_cache_metadata(tmp_path):
    path = tmp_path / "cache.npz"
    np.savez(path, features=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="metadata"):
        vision.load_feature_cache(path, split="sanity", model_name="clip")


def test_stale_stimulus_cache_rejected(tmp_path):
    import hashlib
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(path)
    provenance = {"image_sha256": {"a": hashlib.sha256(path.read_bytes()).hexdigest()}}
    assert vision.validate_cached_image_hashes(["a"], [path], provenance)["verified"] == 1
    Image.new("RGB", (4, 4), "blue").save(path)
    with pytest.raises(ValueError, match="stale"):
        vision.validate_cached_image_hashes(["a"], [path], provenance)


def test_missing_images_rejected_before_model_load(tmp_path):
    with pytest.raises(FileNotFoundError):
        vision.extract_vision_features(["a"], [tmp_path / "absent.jpg"], kind="clip")


@pytest.mark.parametrize("kind", ["dinov2", "clip"])
def test_real_architecture_local_random_checkpoint(tmp_path, kind):
    """Exercises actual Transformers APIs, not pretrained-weight validity."""
    from transformers import (Dinov2Config, Dinov2Model, BitImageProcessor,
                              CLIPVisionConfig, CLIPVisionModelWithProjection, CLIPImageProcessor)
    checkpoint = tmp_path / "checkpoint"
    if kind == "dinov2":
        model = Dinov2Model(Dinov2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                        image_size=28, patch_size=14))
        processor = BitImageProcessor(size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28})
    else:
        model = CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, image_size=28, patch_size=14, projection_dim=8))
        processor = CLIPImageProcessor(size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28})
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32), (25, 100, 200)).save(image)
    features, provenance = vision.extract_vision_features(["a"], [image], kind=kind,
        checkpoint=str(checkpoint), local_files_only=True)
    assert features.shape == (1, 16 if kind == "dinov2" else 8)
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-6)
    assert provenance["frozen"] is True
