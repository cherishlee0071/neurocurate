"""Read at most a few metadata entries; never fetch recording bytes."""
import argparse
import json
from pathlib import Path
from eegdash import EEGDash

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--dataset", default="nm000232")
parser.add_argument("--task", choices=["test", "train"], default="test")
parser.add_argument("--limit", type=int, default=1)
args = parser.parse_args()
if not 1 <= args.limit <= 10:
    parser.error("limit must be 1..10 for metadata inspection")
if args.output.exists():
    parser.error("output already exists")
records = EEGDash().find({"dataset": args.dataset, "task": args.task}, limit=args.limit)
if not records:
    raise RuntimeError("No matching records; check dataset/task identity")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(records, indent=2))
print(f"Saved {len(records)} metadata entries; no EEG recording bytes downloaded")
