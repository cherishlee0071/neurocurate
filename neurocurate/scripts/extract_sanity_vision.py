"""Extract the two cached pretrained embeddings once for exactly 200 images."""
import argparse
from pathlib import Path
import shutil
from neurocurate.metadata import load_manifest, validate_ids
from neurocurate.vision import extract_vision_features, save_feature_cache, load_feature_cache

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--cache-dir", default="/workspace/setup/neurocurate/model_cache")
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()
table = load_manifest(args.manifest)
table = table[table.split == "sanity"].sort_values("image_id")
if len(table) != 200 or any(not Path(p).is_file() for p in table.image_path):
    parser.error("exactly 200 mapped local sanity stimulus images are required")
if args.output_dir.exists():
    parser.error("output directory already exists")
parent = args.output_dir.resolve().parent
while not parent.exists():
    parent = parent.parent
if shutil.disk_usage(parent).free < 64 * 1024**2:
    parser.error("less than the conservative 64 MiB feature-output budget is free")
args.output_dir.mkdir(parents=True)
ids = validate_ids(table.image_id.to_numpy())
for name, revision in [("dinov2", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
                       ("clip", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")]:
    features, provenance = extract_vision_features(ids, table.image_path.tolist(), kind=name,
        revision=revision, cache_dir=args.cache_dir, batch_size=args.batch_size, local_files_only=True)
    path = args.output_dir / f"{name}_features.npz"
    save_feature_cache(path, ids, features, split="sanity", model_name=name, provenance=provenance)
    load_feature_cache(path, split="sanity", model_name=name)
    print(name, features.shape, "saved and reloaded; local checkpoints only")
