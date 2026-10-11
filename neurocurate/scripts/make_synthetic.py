"""Generate a small offline integration fixture, never impersonating real EEG."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from neurocurate.vision import save_feature_cache


def make_synthetic(root, seed=7, n_images=12):
    if isinstance(n_images, bool) or not isinstance(n_images, int) or n_images < 4:
        raise ValueError("n_images must be an integer >= 4")
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("synthetic directory already exists; choose a new location")
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    n, repeats, channels = n_images, 4, 4
    ids = np.array([f"synthetic-{i:03d}" for i in range(n)])
    times = np.arange(-10, 61) / 100
    signal = rng.normal(size=(n, channels, len(times)))
    rows = []
    for i, image_id in enumerate(ids):
        pixels = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / f"{image_id}.png")
        rows.append({"image_id": image_id, "image_path": f"{image_id}.png", "split": "sanity", "eeg_index": i})
    pd.DataFrame(rows).to_csv(root / "images.csv", index=False)
    for s in range(3):
        eeg = np.repeat(signal, repeats, axis=0) + rng.normal(scale=0.1, size=(n * repeats, channels, len(times)))
        np.savez_compressed(root / f"subject-{s + 1}.npz", eeg=eeg, image_ids=np.repeat(ids, repeats),
                            times=times, channel_names=np.array(["O1", "O2", "Oz", "POz"]), split=np.array("sanity"))
    for name in ["dinov2", "clip"]:
        # Random vectors exercise scoring, not a pretrained model's behavior.
        save_feature_cache(root / f"{name}.npz", ids, rng.normal(size=(n, 16)), split="sanity",
                           model_name=name, provenance={"source": "synthetic_random_features", "seed": seed,
                                                       "pretrained_model_executed": False})
    config = {"manifest": "images.csv", "split": "sanity",
              "subjects": [{"id": f"subject-{s + 1}", "eeg": f"subject-{s + 1}.npz"} for s in range(3)],
              "vision": {name: {"features": f"{name}.npz"} for name in ["dinov2", "clip"]},
              "preprocessing": {"baseline": [-0.1, 0.0], "time_window": [0.0, 0.5], "temporal_bins": 10,
                                "min_repetitions": 4, "channels": ["O1", "O2", "Oz", "POz"], "standardize": False},
              "rdm": {"brain_metric": "correlation", "vision_metric": "cosine"},
              "scoring": {"k": 3, "model_weights": {"dinov2": 0.5, "clip": 0.5}},
              "visualization": {"enabled": True, "top_n": 6}, "output_dir": "results"}
    (root / "experiment.yaml").write_text(yaml.safe_dump(config))
    if n == 200:
        sequential = {"manifest": "images.csv", "work_dir": "owned_downloads", "output_dir": "sequential-results",
                      "subjects": [{"id": s["id"], "eeg": {"path": s["eeg"], "temporary": False}}
                                   for s in config["subjects"]], "preprocessing": config["preprocessing"],
                      "brain_metric": "correlation", "reserve_bytes": 1024**3,
                      "subject_output_budget_bytes": 64 * 1024**2,
                      "vision_features": {name: f"{name}.npz" for name in ["dinov2", "clip"]}}
        (root / "lowdisk.yaml").write_text(yaml.safe_dump(sequential))
    return root / "experiment.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--images", type=int, default=12)
    args = parser.parse_args()
    print(make_synthetic(args.output, args.seed, args.images))
