"""Convert only an explicitly trusted local official image_metadata.npy."""
import argparse
from pathlib import Path
from neurocurate.metadata import things_metadata_manifest

parser = argparse.ArgumentParser()
parser.add_argument("metadata", type=Path)
parser.add_argument("image_root", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--trust-pickle", action="store_true")
args = parser.parse_args()
if args.output.exists():
    parser.error("output already exists; refusing to overwrite")
table = things_metadata_manifest(args.metadata, args.image_root, trusted_pickle=args.trust_pickle)
args.output.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.output, index=False)
print(f"Wrote {len(table)} image records; official test split maps to sanity")
