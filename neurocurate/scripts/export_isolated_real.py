"""Launch the approved pickle exception in a kernel-confined child only."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
from neurocurate.acquisition import verify_artifact
from neurocurate.lowdisk import sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subject', choices=['sub01','sub02'], default='sub01')
subject = parser.parse_args().subject
root = Path("results/real_phase0") / subject
manifest = json.loads((root / "acquisition_manifest.json").read_text())
archive_name = 'sub-01.zip' if subject == 'sub01' else 'sub-02.zip'
eeg = next(a for a in manifest["extracted"] if a["archive"] == archive_name)
shared = json.loads(Path('results/real_phase0/sub01/acquisition_manifest.json').read_text())
metadata = next(a for a in shared["files"] if Path(a["path"]).name == "image_metadata.npy")
if subject == 'sub01':
    assert eeg["sha256"] == "08e52b18caf46b8504beb4edaadef025f114eaea25ab1f4eb8c5f9114c56ebff"
assert metadata["sha256"] == "681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c"
for artifact in [eeg, metadata]:
    verify_artifact(artifact)
archive = next(a for a in manifest["files"] if Path(a["path"]).name == archive_name)
expected_archive = {'sub01': ('https://osf.io/download/nb8wr/', 'f25d5ea30ba6811edd7ebd460d7b271edaf527b6920ab2d3d6e7950e9ee9639f'),
                    'sub02': ('https://osf.io/download/u43p5/', 'e1eba2d74009c6dd41becf53d84b8d47532103b41110a5855e7e286bcca6fe41')}[subject]
assert (archive['url'], archive['sha256']) == expected_archive
if subject == 'sub02':
    official = json.loads((root/'official_source_metadata.json').read_text())['data']['attributes']
    assert official['extra']['hashes']['sha256'] == archive['sha256'] and official['name'] == archive_name
verify_artifact(archive)
export = root / "pickle_free"
export.mkdir()  # Never overwrite an earlier export.
names = ["eeg.npy", "times.npy", "channels.npy", "metadata.json", "isolation_report.json"]
handles = [open(a["path"], "rb") for a in [eeg, metadata]] + [open(export / n, "xb") for n in names]
try:
    fds = [h.fileno() for h in handles]
    result = subprocess.run([sys.executable, "-I", "-u", "scripts/_isolated_export.py", *map(str, fds), eeg['sha256'], metadata['sha256']],
        pass_fds=fds, close_fds=True, stdin=subprocess.DEVNULL, timeout=150,
        env={"OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
             "PYTHONDONTWRITEBYTECODE":"1", "LC_ALL":"C.UTF-8"})
    if result.returncode:
        raise RuntimeError("isolated export failed; no downstream loading allowed")
finally:
    for h in handles:
        h.close()
report = json.loads((export / "isolation_report.json").read_text())
assert report["all_exports_pickle_free"] and all(v == "denied" for v in report["probes"].values())
(export / "export_manifest.json").write_text(json.dumps(dict(source_acquisition_manifest=str(root / "acquisition_manifest.json"),
    original_input_sha256=[eeg["sha256"], metadata["sha256"]], isolation=report,
    exports={name:sha256(export/name) for name in names}),indent=2)+"\n")
(export / "VERIFIED").write_text("isolated export completed; downstream pickle forbidden\n")
print(json.dumps(report, indent=2))
