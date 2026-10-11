"""Dry-run or execute a bounded, sequential 200-image sanity experiment."""
import argparse
import json
from neurocurate.lowdisk import run_sequential

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--cleanup", action="store_true", help="remove only this run's explicitly temporary verified EEG downloads")
args = parser.parse_args()
result = run_sequential(args.config, dry_run=args.dry_run, cleanup=args.cleanup)
print(json.dumps(result, indent=2))
if args.dry_run and not result["ready"]:
    raise SystemExit(2)
