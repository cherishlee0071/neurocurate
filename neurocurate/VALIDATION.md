# Initial milestone validation

Executed in the existing cloud checkout and Python 3.12.14 venv. No worktree was
created. No THINGS-EEG2 EEG recording bytes were downloaded.

## Tests and environment

`python -m pytest --junitxml=outputs/validation/pytest.xml`:

- **83 passed; 0 failed; 0 skipped** (5.51 seconds).
- Coverage includes EEG averaging/feature shapes, baseline and time-bin math,
  exact image-ID alignment, all three RDM metrics, neighbor tie handling,
  self-exclusion, Jaccard disagreement, pairwise subject consistency, weighted
  score ranking, missing metadata, invalid inputs, cross-split aliases, cache
  split mismatches, stale stimulus caches, and complete pipeline/figure outputs.
- Official NPY layout and MNE FIF loaders were exercised on generated arrays and
  epochs, not real THINGS EEG recordings.
- DINOv2/CLIP tests exercise frozen eval/inference behavior with a fake backend
  and actual Transformers architectures using tiny local random checkpoints.
- Editable installation with `.[dev,data]` succeeded. `uv pip check` verified all
  116 installed packages. Source compilation and `git diff --check` passed.

Runtime versions: NeuroCurate 0.1.0, NumPy 2.5.4, PyTorch 2.11.0+cpu,
torchaudio 2.11.0+cpu, Transformers 4.57.6, MNE 1.13.2, EEGDash 0.9.1.

## Functional runs

| Check | Evidence | Scope |
| --- | --- | --- |
| Documented generator and CLI | `outputs/synthetic-demo/results/COMPLETE` | 12 synthetic images, three synthetic EEG subjects, random vision caches; scores and both figures generated |
| Pretrained vision integration | `outputs/synthetic-pretrained/results/COMPLETE` | Same synthetic EEG/images with actual frozen pretrained model features; DINOv2 `(12,768)`, CLIP `(12,512)`, finite outputs and complete ranking/figures |
| Real dataset metadata | `scripts/query_metadata.py` succeeded | One nm000232 test-task metadata record, subject 01; no EEG bytes fetched |
| Real EEG processing | Not executed | Requires local real EEG and matching image metadata/stimuli |

Pretrained revisions:

- `facebook/dinov2-base`: `f9e44c814b77203eaa57a6bdbbd535f21ede1415`
- `openai/clip-vit-base-patch32`: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`

Both checkpoints are retained under `/workspace/setup/neurocurate/model_cache`.
The example real-data config pins those revisions and defaults to local-only
loading. Each completed run records exact input IDs, repetitions, preprocessing,
library versions, model revision and feature origin in `provenance.json`.
Generated demo artifacts are ignored by Git and remain available in this machine.

## Remaining blockers and limitations

Supply selected trusted THINGS-EEG2 preprocessed arrays or MNE epochs for at least
two independent subjects, their exact image-ID/index or trial mapping, and local
stimulus files. Configure paths, available channels, time coverage, and K in a
copy of `configs/things_sanity.yaml`, then execute it. A raw NEMAR BrainVision
recording additionally needs event-to-image mapping and epoch preparation.

Pretrained features were verified on synthetic images; no claim is made about
real EEG performance or scientific usefulness of the heuristic score. The
metadata endpoint currently works, so the earlier network-policy denial is no
longer a blocker for that operation. Any future selective dataset acquisition
must separately verify its storage access and file sizes.

The score is consistency times disagreement, not a learned quality predictor.
Every analysis uses one explicit split; official repeated test images belong to
sanity and must remain outside later training/evaluation. Duplicate-copy image
identity and truthful split declarations still require input curation. Full
training, OOD, diffusion, and online active learning were not implemented.

## Low-disk Phase 0 audit and results

Work stayed in the existing checkout/venv. No worktree was created. **No real EEG
or stimulus payloads were downloaded.** Only API inventories, the pinned Git tree,
and the small NEMAR README were retrieved. No checkpoint or unrelated file was
removed. The complete sizes, published SHA256 hashes and source URLs are in
`data/lowdisk_source_audit.json`; raw snapshots are in `outputs/lowdisk-audit/`.

Disk/cache commands executed:

```bash
df -h /workspace
df -B1 /workspace
du -sh /workspace/setup/neurocurate/*
find /workspace /tmp -type d \( -name '*eeg*cache*' -o -name '*nm000232*' \) -print
ls -la /home/agent/.cache
du -sh /home/agent/.cache/*
```

Initial free space: **32,033,169,408 bytes (29.83 GiB)**. The model cache (about
1.5 GiB) and venv (about 1.6 GiB) were preserved. Existing base-layer files are
not new allocations when using filesystem free-space measurements. No EEG cache
was found in the readable workspace/home cache. An inaccessible daemon directory
under `/tmp` was excluded; no claim is made about its contents.

Inventory requests executed with TLS verification enabled:

- `https://api.osf.io/v2/nodes/3jk45/children/?page[size]=100`
- `https://api.osf.io/v2/nodes/anp5v/files/osfstorage/?page[size]=100`
- `https://api.osf.io/v2/nodes/y63gw/files/osfstorage/?page[size]=100`
- `https://api.github.com/repos/nemarDatasets/nm000232/git/trees/c9865f3022082dd53832d6adb69f0af787eb9bcc?recursive=1`
- EEGDash `GET https://data.eegdash.org/api/eegdash/records`, filter
  `{"dataset":"nm000232","task":"test"}`, limit 100.

The initial default OSF preprocessed listing was paginated. The final 100-item
request returned **all 20 entries** with no next page: ten standard subject ZIPs
and ten 63-channel ZIPs, with no separately listed test arrays. The pinned NEMAR
tree contained 4,147 entries, was not truncated, and also listed preprocessed
subject ZIP pointers. EEGDash returned 80 test-record metadata entries across
10 subjects/source and BIDS representations; no dataset object was instantiated.

| Candidate asset | Exact bytes | Availability |
| --- | ---: | --- |
| Standard subject-01 ZIP | 1,071,722,467 | Separate subject archive, containing train/test |
| Smallest standard subject ZIP (sub-04) | 1,071,421,033 | Same archive limitation |
| 63-channel subject ZIPs | about 4.03–4.04 billion | Larger optional archives, excluded |
| Official test_images.zip | 8,129,948 | Separate stimulus archive |
| Official image_metadata.npy | 656,579 | Separate metadata artifact |
| Original-name 200 test JPEG entries | 8,084,268 total | Pinned Git inventory; renamed aliases excluded |
| Subject-01 source test EEG/header/marker, all four sessions | 1,312,140,127 | Raw alternative; additional event metadata/epochs excluded |
| Subject-01 session-01 source test EEG/header/marker | 341,649,569 | Possible 200-image, approximately 20-repeat alternative requiring event validation/preprocessing |

The smallest desired scientific subset is one subject's **test-only preprocessed
array**, exact image-axis mapping, and the corresponding 200 stimuli. The EEG
array's uncompressed size T is **unknown**: the inspected official inventories
publish archives, not standalone test-array files. A compressed archive size does
not prove the member size or peak allocation.

For a subject-01 archive alternative, the known transfer floor (subject ZIP,
stimulus ZIP, metadata) is **1,080,508,994 bytes**. A conservative retained-archive
peak would add T, 8,084,268 bytes of extracted stimuli, at least 128 MiB for
shared/subject outputs, and a 1 GiB reserve. T must be measured before claiming
this fits. No expansion ratio or guessed test-array size authorized acquisition.

**Stop decision:** acquisition stopped under the user's rule because neither
inspected official inventory lists a directly accessible test-only array. The
real one-subject EEG loader smoke test and real 200×200 Brain RDM were **not run**.
This is a source-layout/unknown-size blocker, not a credential or TLS failure.

### Concrete alternative

On a machine already holding the trusted subject-01 archive, verify SHA256
`f25d5ea30ba6811edd7ebd460d7b271edaf527b6920ab2d3d6e7950e9ee9639f`, inspect the ZIP
inventory, and export only `preprocessed_eeg_test.npy` after checking its exact
size against free space. Keep the archive and do not extract training EEG.
Transfer the test-only export with its size/SHA256, the verified image metadata,
and only the 200 stimulus files, or expose the export as an individual HTTPS
artifact. On that archive-holding machine, this inspection downloads nothing:

```python
from pathlib import Path
from zipfile import ZipFile
from neurocurate.lowdisk import sha256
archive = Path('sub-01.zip')  # Already present outside this run.
assert sha256(archive) == 'f25d5ea30ba6811edd7ebd460d7b271edaf527b6920ab2d3d6e7950e9ee9639f'
with ZipFile(archive) as z:
    members = [m for m in z.infolist() if Path(m.filename).name == 'preprocessed_eeg_test.npy']
    assert len(members) == 1 and not members[0].is_dir()
    print(members[0].filename, members[0].file_size, members[0].compress_size)
    # Check file_size + reserve, then export just this member and record its SHA256.
```

Alternatively, a future explicitly authorized task could acquire ONE verified
subject archive after measuring extraction requirements, extract only its test
member, and retain the archive until outputs are verified. That route was not
executed here. A single raw test session is another bounded alternative but needs
verified events and documented preprocessing; it is not equivalent to the authors'
four-session preprocessed test split. Neither route requires the full dataset.

### Implemented and executed workflow

`scripts/process_subjects_lowdisk.py` accepts exactly 200 mapped sanity images,
processes one subject at a time, and preserves features, RDM, IDs, config and
provenance. Each output is reloaded without pickle, checksummed and recomputed
before cleanup. Downloads require audited exact size, SHA256 and HTTPS sources;
unknown prerequisites prevent ALL payload acquisition. Space is checked before
the plan executes and before each subject. Archives are unsupported and blocked.
Cleanup only removes this run's explicitly temporary, reproducible EEG download
following successful verification. Existing files, checkpoints, shared stimuli,
vision caches, failed inputs and failed partials remain intact. Vision caches are
aligned/copied once and referenced by every subject. `extract_sanity_vision.py`
prepares them using pinned local checkpoints, without fetching model weights.

Commands executed in the existing venv:

```bash
python scripts/process_subjects_lowdisk.py configs/lowdisk_osf_audit.yaml --dry-run \
  > outputs/lowdisk-audit/dry-run.json
# Expected exit 2: acquisition blocked, peak estimate null.
python scripts/make_synthetic.py outputs/lowdisk-synthetic --images 200
python scripts/process_subjects_lowdisk.py outputs/lowdisk-synthetic/lowdisk.yaml --dry-run \
  > outputs/lowdisk-audit/synthetic-dry-run.json
python scripts/process_subjects_lowdisk.py outputs/lowdisk-synthetic/lowdisk.yaml --cleanup
python -m pytest --junitxml=outputs/lowdisk-audit/pytest.xml
UV_CACHE_DIR=/workspace/setup/uv-cache uv pip check \
  --python /workspace/setup/neurocurate/venv/bin/python
git diff --check
```

Results:

- Real-source dry run correctly blocked; total peak was null. Known output budgets
  plus reserve were 1,207,959,552 bytes, NOT a complete real-data peak estimate.
- Synthetic dry run passed with additional allocation/reserve of 1,342,177,280
  bytes and zero required downloads.
- Synthetic CLI verified THREE subjects with features `(200,40)` and Brain RDMs
  `(200,200)`. Independent saved-output re-verification succeeded.
- Two shared synthetic vision caches were reused. All three pre-existing inputs
  stayed intact even with `--cleanup`. Retained outputs totaled 1,308,182 bytes.
- Final suite: **100 passed; 0 failed; 0 skipped** (6.48 seconds), including 17
  low-disk tests for storage refusal, mapping, sequential outputs, reuse,
  checksum/byte limits, HTTPS downgrade prevention, cleanup ownership, preserved
  failures, and saved-output corruption. Dependency/whitespace checks passed.
- An object-array serialization defect was fixed in both pipelines: saved image
  IDs now reload without pickle.

The 200-image RDMs above are **synthetic**, not real EEG results. Remaining
prerequisite: a trusted, sized test-only real EEG export plus exact mapping and
stimuli. No real EEG readiness or scientific result is claimed.

## Phase 1.1 — Metadata-only remote ZIP inspection

Inspection on 2026-10-11 resolved the previously unknown test-member size. This
supersedes the unknown-size portion of the earlier stop decision, but does not
provide a verified test-only export or authorize acquisition by the existing
processor. **No complete ZIP, EEG payload or stimulus payload was downloaded;
no real EEG loader, processing or 200-image Brain RDM was run.**

### Exact archive member and HTTP Range results

The official OSF file API
`https://api.osf.io/v2/files/62f6b4ff170d520c73c27c1e/` returned bounded metadata
consistent with the previous official inventory: `sub-01.zip`, **1,071,722,467
bytes**, downloaded through `https://osf.io/download/nb8wr/`. The earlier inventory
places this file in official THINGS-EEG2 project `3jk45`, preprocessed child `anp5v`.

The official URL redirected over verified HTTPS through `files.de-1.osf.io` to
`storage.googleapis.com`. Every metadata range returned **206 Partial Content**,
the exact requested Content-Range and Content-Length, identity encoding, and the
same strong ETag `"09de4551754e18fec51e6500d64b4bb0"`. Subsequent requests sent
`If-Match` with that ETag. Signed redirect URLs are intentionally not retained.

| Metadata read | Inclusive byte range | Bytes |
| --- | --- | ---: |
| ZIP end record, no comment | 1,071,722,445–1,071,722,466 | 22 |
| Complete central directory (3 entries) | 1,071,722,136–1,071,722,444 | 309 |
| Test member fixed local header | 69–98 | 30 |
| Local filename and extra fields | 99–162 | 64 |

The reproducible run read **425 archive bytes**, all metadata. Preliminary end
record and directory probes read another 331 metadata bytes: **756 archive bytes
total during this task**. No broad suffix/tail request was used. The final bounded
OSF JSON response was 2,158 bytes. Requests stream responses, close redirects
without reading bodies, reject ignored/mismatched Range responses before body
reads, and cap each ZIP metadata read at 16 KiB with a 32 KiB total budget.
Unsupported ZIP comments, ZIP64, multi-disk layouts or missing/duplicate test
entries stop inspection. TLS certificate verification uses Requests defaults.

There is exactly one `preprocessed_eeg_test.npy` basename:

| Property | Observed value |
| --- | --- |
| Archive path | `sub-01/preprocessed_eeg_test.npy` |
| Uncompressed size | **217,601,432 bytes (207.521 MiB)** |
| Compressed size | **208,634,917 bytes (198.970 MiB)** |
| Compression | DEFLATE, method 8 |
| General-purpose flags | 8 (sizes/CRC use a data descriptor) |
| Central-directory CRC32 | `8ef08aa5` — observed, not verified |
| Local header offset | 69 |
| Compressed payload range | **163–208,635,079 inclusive — NOT fetched** |

Local and central filenames, flags and compression agree. Payload boundaries do
not overlap the next header. The other entries are `sub-01/` and
`sub-01/preprocessed_eeg_training.npy` (863,086,926 compressed / 899,777,433
uncompressed bytes); neither was extracted. NPY header, dtype, shape, repetitions,
image mapping and contents remain uninspected because they are EEG payload.

### Provenance and integrity status

The fresh official API publishes archive SHA256:
`f25d5ea30ba6811edd7ebd460d7b271edaf527b6920ab2d3d6e7950e9ee9639f`.
It agrees with our previously pinned source audit. Published MD5 agrees with the
observed ETag. This establishes **metadata consistency and an official HTTPS
acquisition path**, not cryptographic verification of downloaded archive bytes.
No full SHA256 was computed; no test-member SHA256 or CRC32 was verified. ETag,
TLS and partial metadata reads cannot verify the complete archive checksum.
The central-directory sizes are exact reported metadata, not independently
verified sizes of a downloaded EEG file. OSF version was not present in the fresh
API's `extra` object and is recorded as null rather than inferred.

Machine-readable evidence, sanitized redirect hosts, individual ranges and
resource observations: [data/test_member_range_audit.json](data/test_member_range_audit.json).

### Disk and RAM requirements

At the reproducible inspection, free disk was **32,034,488,320 bytes (29.834 GiB)**.
Host RAM total was 35,743,784,960 bytes with 34,903,105,536 bytes available. The
effective cgroup limit was **32 GiB**, current usage 145,272,832 bytes, leaving
**31.865 GiB** headroom. These observations are time-specific; recheck before any
future acquisition. Existing environment and pretrained checkpoint files remain
in place and are already excluded from free space; nothing was deleted.

The following estimates are *additional* allocation. They retain the EEG ZIP or
staged compressed member throughout extraction, and include the previously
audited stimulus ZIP (8,129,948 bytes), extracted 200 original JPEGs (8,084,268
bytes), metadata (656,579 bytes), a **128 MiB** output budget and **1 GiB** reserve.
The extracted JPEG total is a pinned repository inventory measurement, not a
measurement of a newly downloaded stimulus ZIP. Output allocation is a planning
budget; real outputs depend on preprocessing configuration. Transfer partials
are renamed in place, not copied into a second full file. Only the test member
would be extracted; training EEG is excluded.

| Future acquisition route | Peak additional disk including budgets/reserve | Acquisition RAM planning budget |
| --- | ---: | ---: |
| Full subject-01 ZIP + test-only extraction | **2,514,154,246 bytes / 2.341 GiB** | 64 MiB plus Python/OS baseline |
| Range extraction, retain compressed test member | **1,651,066,696 bytes / 1.538 GiB** | 64 MiB plus Python/OS baseline |
| Range extraction streamed directly to test file | **1,442,431,779 bytes / 1.343 GiB** | 64 MiB plus Python/OS baseline |
| External verified export: local destination | **1,442,431,779 bytes / 1.343 GiB** | 64 MiB plus Python/OS baseline |
| External verified export: source machine | **2,514,154,246 bytes / 2.341 GiB** | 64 MiB plus Python/OS baseline |

Acquisition RAM estimates assume incremental SHA256/CRC32 and DEFLATE with bounded
1–8 MiB buffers, never whole-archive or whole-member in-memory buffering. They
are implementation budgets, not measured payload extraction peaks. On an external
machine with an existing archive, additional source allocation falls by
1,071,722,467 bytes. Each machine needs its own disk check before transfer.

EEG processing has the same RAM needs for all routes. The existing loader uses
float64 arrays and preprocessing makes copies. For ordinary float32/float64 EEG,
a provisional **10× uncompressed file size = 2,176,014,320 bytes (2.027 GiB)**
allows substantial room for original arrays, float64 conversion, selection and
copy temporaries; plan **4 GiB plus Python/library baseline**. This is not a
proven upper bound: serialized dtype, array shape and object allocations have not
been inspected. Real EEG peak memory must be measured after integrity-verified
acquisition. The measured 31.865 GiB headroom exceeds the provisional budget.
Vision embeddings should be reused rather than loading both backbones during
subject EEG processing.

### Recommended acquisition route and remaining blockers

**For a future acquisition task, the safest feasible local route is one official
subject-01 ZIP, full published SHA256 verification, then streaming extraction of
only the test member**, checking output size and ZIP CRC32 and recording a new
member SHA256 before trusting the pickled NPY. Its 2.341 GiB planning peak fits
current disk. This downloads one subject archive (including compressed training
bytes), not the complete dataset. It was not executed in this metadata-only task.

If the continued policy requires downloading only test EEG bytes locally, prefer
**an external export from that fully SHA256-verified archive**, with exact member
size, member SHA256, parent archive SHA256 and extraction provenance transferred
through a trusted channel. The destination can then verify the 207.521 MiB export
without retaining the 1.0 GiB ZIP. The external machine need only inspect/extract
the member; it need not run EEG preprocessing to create this export.

Direct test-only Range extraction is technically feasible: request only the
observed payload range in bounded chunks, pin ETag/If-Match, incrementally inflate
raw DEFLATE and check compressed/uncompressed counts and CRC32. Flags 8 mean local
header sizes must not be relied on. However, **CRC32 cannot substitute for the
published archive SHA256**. A Range-only extractor cannot verify that whole-file
hash and should not feed the trusted-pickle loader without an independently
trusted test-member SHA256 (e.g. from the verified external export). This project
does not yet implement a payload Range extractor or subject ZIP acquisition.

Remaining blockers for a real smoke test: choose/authorize a future acquisition
route; obtain a cryptographically verified test export, verified metadata and
stimuli; validate exact image ordering, NPY schema and repetition counts; then
measure real memory and verify the saved 200×200 RDM. No credentials, full EEGDash
dataset instantiation, chunk-streaming assumption or TLS bypass is required.

### Reproduction and verification

Commands executed in the existing environment:

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
python scripts/inspect_remote_eeg_zip.py --output data/test_member_range_audit.json
python -m pytest tests/test_remotezip.py -q
python -m pytest --junitxml=outputs/range-audit/pytest.xml
git diff --check
```

The inspector refuses to overwrite existing reports. Repeat with a new path,
for example `--output outputs/range-audit/recheck.json`. Storage estimates in the
audit are derived offline from the inspected sizes and explicit budgets. The
metadata-only live run succeeded; **13 new synthetic tests passed**, covering
payload exclusion, ambiguous/missing entries, unsupported comments, truncated
directory, local filename mismatch, ignored/wrong Range responses, bad lengths,
weak ETags, HTTP refusal and oversized requests. No test acquired EEG payloads.
The complete regression suite passed: **113 passed, 0 failed, 0 skipped in
7.59 seconds**. Whitespace and Python compilation checks passed. This verifies
synthetic behavior and live remote metadata access, not real EEG loading or RDMs.

## Phase 1.2 — Verified acquisition; explicit format approval pending

The attached `instruction.md` authorized one-subject acquisition. We used the
existing checkout/environment without a worktree. Linux `/workspace` has no
Windows C:/D: drive. Immediately before acquisition, available disk was
**32,034,312,192 bytes**; the **2,514,154,246-byte** conservative peak estimate
fit. The 32 GiB cgroup limit minus then-current usage (228,872,192 bytes) left
**31.787 GiB** headroom. Host MemAvailable was also inspected before acquisition
(34,059,132 KiB). No pretrained checkpoint or unrelated file was removed.

### Verified real inputs

| Official download | Exact bytes | Verified full-file SHA256 |
| --- | ---: | --- |
| `https://osf.io/download/nb8wr/` → `sub-01.zip` | 1,071,722,467 | `f25d5ea30ba6811edd7ebd460d7b271edaf527b6920ab2d3d6e7950e9ee9639f` |
| `https://osf.io/download/znu7b/` → `test_images.zip` | 8,129,948 | `57ed472cd88f68ccf88e9ecbadf3da91ac222d8548684d5680be95bcf47ec14a` |
| `https://osf.io/download/qkgtf/` → `image_metadata.npy` | 656,579 | `681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c` |

These checks are **complete downloaded-file SHA256 verification**, unlike the
earlier Range metadata audit. The ZIP central directory contained exactly one
`sub-01/preprocessed_eeg_test.npy`, with the audited compressed/uncompressed
sizes. Only this EEG member was extracted. Its size is **217,601,432 bytes**,
its CRC32 passed through zipfile's EOF validation, and its computed export SHA256
is **`08e52b18caf46b8504beb4edaadef025f114eaea25ab1f4eb8c5f9114c56ebff`**.

The stimulus ZIP contained exactly **200 distinct original image members**.
All were extracted, CRC-checked, decoded with Pillow verification, sized and
SHA256-hashed. Their combined extracted size was **8,084,268 bytes**, matching
the prior pinned repository inventory. No training images were downloaded, no
training EEG was extracted, no other subject or 63-channel ZIP was downloaded.
The one subject ZIP necessarily contains compressed training EEG; it remains
unextracted and is retained until all eventual scientific outputs are validated.

Retained inputs: `data/local/real_phase0/`. Reports:
`results/real_phase0/sub01/`. An independent verification pass rehashed and checked
the sizes of **204 files** (three downloads, one EEG export, 200 images), checked
that only one EEG member was exported and that no extracted training EEG exists.
Evidence is in `acquisition_manifest.json` and
`acquisition_independent_verification.json` in that result directory.

### Actual format blocker — no object deserialization performed

Safe NumPy header inspection established the following for BOTH EEG and image
metadata: NPY version 1.0, serialized shape `()`, dtype **object**, C-order,
128-byte header. Both explicit `np.load(..., allow_pickle=False)` attempts failed
with `Object arrays cannot be loaded when allow_pickle=False` before unpickling.
The scalar shape describes the serialized container; **it is not the EEG tensor
shape**. Actual EEG dtype, tensor shape/axes, channels, times, sampling rate and
repetitions remain unverified. Actual metadata keys and EEG-to-image mapping have
not been inspected. Alphabetical order was not substituted for that mapping.

The attachment explicitly says: **“No pickle-based loading unless independently
justified and explicitly approved.”** Verified official hashes and the author's
dictionary serialization code independently justify a narrowly scoped exception,
but this run has **not** assumed that those facts supply the required explicit
approval. A request for approval to load only these two verified NPY files was
presented after acquisition and independent integrity verification. No pickle
loading, custom unpickler workaround or synthetic substitution was used.

Accordingly, exact image ordering, duplicate/missing metadata identifiers and
train/test identifier overlap remain unchecked. Brain feature/RDM computation,
reliability splits/null controls, real-image DINOv2/CLIP extraction, disagreement
scores and figures are **not run**. No empty or synthetic numeric artifacts were
created to suggest completion. `eeg_shape_report.json`,
`image_alignment_report.json`, `brain_reliability.json` and `go_no_go.json`
explicitly record unknown/not-run fields and the blocking reason.

### Author preprocessing evidence and scientific caveats

We inspected text only from the authors' `gifale95/eeg_encoding` repository at
commit **`101997d4f14c41b615bcaf8c0907740106d260c5`**, whose README links the same
official OSF project `3jk45`. Bounded source downloads and their SHA256s are
recorded in `outputs/real-phase0-source/source_provenance.json`, with summarized
evidence in `results/real_phase0/sub01/preprocessing_source_report.json`.

`02_eeg_preprocessing/preprocessing.py` defaults to four sessions and 100 Hz;
`preprocessing_utils.py` epochs −200 to +800 ms, applies baseline correction,
selects posterior/occipital channels, sorts numeric event conditions, selects
20 repetitions per test image/session, merges sessions and shuffles repetitions
with seed 20200220. `save_prepr` serializes a dictionary using `np.save` with keys
`preprocessed_eeg_data`, `ch_names`, `times`. These are **source conventions to
verify against actual arrays**, not observed array properties or proof of exact
image alignment.

Importantly for later interpretation, `mvnn` fits a session whitening covariance
averaged over **both test and training partitions**, using all repetitions. Thus
these supplied arrays are unsuitable for a claim of independently fitted
held-out preprocessing in later training evaluation. Random repetition halves
would also share that fitted transform; any later split-half result must be
described as reliability conditional on supplied preprocessing, not proof of
fully independent preprocessing. Session labels are not saved in the test dict.
Strict session-independent reliability may require raw test-session data and
new split-specific preprocessing. This run has not yet assessed EEG reliability.

### Actual observed storage and verification

Filesystem checkpoints observed **1,307,615,232 bytes (1.218 GiB)** additional
disk allocation, including report/test writes. Retained downloaded/extracted
payloads total **1,306,194,694 bytes**. Minimum sampled free disk was
**30,726,610,944 bytes (28.616 GiB)**. The 2.341 GiB preflight estimate additionally
included a 1 GiB reserve and 128 MiB output budget; those are planning allowances,
not measured payload allocations. Forty resource samples at approximately
10-second intervals observed cgroup memory usage peaking at **1,540,730,880
bytes (1.435 GiB)**, including filesystem cache. These are checkpoint/sampled
measurements, not a continuously measured upper bound; no EEG-processing memory
peak has been measured. Inputs and checkpoints were all preserved.

Commands executed:

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
python scripts/acquire_real_phase0.py --dry-run
python scripts/acquire_real_phase0.py
python -m pytest --junitxml=outputs/range-audit/phase12-pytest.xml
git diff --check
```

The new acquisition helper checks storage before downloading/extracting, verifies
exact sizes/full SHA256, refuses unsafe ZIP paths, preserves existing inputs,
retains failed partials, validates extraction CRCs and inspects NPY headers
without pickle. The latest full suite passed: **122 passed, 0 failed, 0 skipped
in 7.65 seconds**, including nine new synthetic acquisition tests for checksum
failure, path safety, selective extraction, preserved inputs, space refusal and
object-format refusal. Python compilation and whitespace checks passed.
Real verification covers source acquisition, checksums, extraction, image decode
and safe NPY headers; **real EEG loading, RDMs and vision integration are unverified**.

**Current decision: NO-GO for multi-subject Brain RDM reliability experiments.**
Acquisition integrity passes, but explicit pickle approval, actual tensor/schema
inspection, exact image mapping, a validated real RDM and defensible reliability
evidence remain prerequisites. This is not evidence that subject-01 EEG has poor
reliability. After approval, resume with these retained verified files, stop on
unresolved image mapping, validate actual time axes before choosing a window,
then perform the requested one-subject analysis and reevaluate GO/NO-GO.

## Phase 1 — Real EEG/image integration after explicit approval

The user explicitly approved loading **only the two previously SHA256-verified
official NPY files**. This resolves the earlier pickle gate. This entry supersedes
the earlier pending-approval, uninspected-tensor and unverified-mapping statuses;
the original acquisition records remain preserved as historical evidence.

### Actual files and access

We reused all retained inputs without downloading any additional EEG, stimulus
images or model weights. The three official input URLs, exact bytes and complete
SHA256 values are listed in the Phase 1.2 acquisition table above. The test-only
export still matches SHA256
`08e52b18caf46b8504beb4edaadef025f114eaea25ab1f4eb8c5f9114c56ebff`.
All **204** retained file sizes and hashes were checked again; all **200** JPEGs
decode and have distinct paths and file hashes. No source recording or raw EEG
session was required; the smallest retained suitable EEG input is the 217,601,432
byte official preprocessed test export. Its verified parent ZIP remains intact.

Dataset: THINGS-EEG2, OSF project `3jk45`, preprocessed node `anp5v`, subject-01
standard ZIP GUID `nb8wr`, release date **2022-08-12**, **archive version 1**.
Version evidence is `attributes.current_version` in the saved complete official
OSF inventory; the earlier fresh API audit's `extra.version` was null because
that was not the correct field. The checksum pins the exact release content.
Original split is **test**, represented as **sanity** in this repository.

Current evidence:

- `results/phase1/real_data_audit.json`: complete input inventory, hashes, mapping,
  observed tensor properties, model provenance, status and limitations.
- `results/phase1/real_data_audit.md`: human-readable real-operation report.
- `results/phase1/smoke/`: real features, IDs, mappings, RDMs, config, provenance
  and the completion marker.

### Observed EEG tensor and preprocessing

The actual official dictionary has `preprocessed_eeg_data`, `ch_names`, `times`.
Using the existing `load_eeg` implementation:

| Observed property | Real value |
| --- | --- |
| Released tensor | **(200, 80, 17, 100)**, float64 |
| Axes | images × repetitions × channels × time |
| Loader epochs | **(16,000, 17, 100)** |
| Sampling rate | **100 Hz**, from a uniformly spaced 0.01-second time vector |
| Time coverage | **−0.20 to +0.79 seconds** |
| Images / repetitions | **200 / 80 per image** |
| Channels | **17**, all finite EEG |

Channels: `Pz, P3, P7, O1, Oz, O2, P4, P8, P1, P5, PO7, PO3, POz, PO4, PO8,
P6, P2`. The released trial arrangement contains image-axis ordering and shuffled
repetitions, but does not embed raw per-trial stimulus/session IDs. The loader
expands validated axis identities into 80 repeated IDs per image; these are
protocol-derived identifiers, not independently read raw-event IDs.

Configured preprocessing uses **100 ≤ t < 500 ms**, all 17 channels, no extra
baseline correction (the official array is already baseline corrected and MVNN
whitened), no standardization and no temporal binning. It averages all 80
repetitions/image and flattens channels/time into **680 features/image**. Time
coverage and sampling were checked before using this interval. Configuration:
`configs/real_subset.yaml`, copied into the result directory.

### Exact image-axis mapping evidence

Actual metadata keys include `test_img_files`, `test_img_concepts`,
`test_img_concepts_THINGS` and their training equivalents. There are **200 test**
and **16,540 training** metadata entries; training payloads were not loaded.

The mapping uses explicit condition IDs and preserves metadata row order:

1. Test concept folders in metadata carry numeric prefixes **00001–00200** in
   exactly that order. For example, EEG index 0 maps to
   `test_images/00001_aircraft_carrier/aircraft_carrier_06s.jpg`; index 199 maps to
   `test_images/00200_wok/wok_12s.jpg`.
2. Authors' `get_stimuli_test.m`/`data_collection_test.m` assign numeric test
   conditions and behavioral `tot_img_number` to the corresponding stimulus.
3. `source_to_raw.py` writes behavioral `tot_img_number` into the EEG event
   channel; targets become 99999.
4. `preprocessing_utils.py` removes targets and sorts `np.unique` numeric event
   conditions before constructing the released image axis. Thus **EEG index =
   numeric condition ID − 1**. Session merging changes repetitions, not this axis.

This chain is pinned to author commit
`101997d4f14c41b615bcaf8c0907740106d260c5`; source URLs/hashes are retained in
`image_alignment_report.json`. No new alphabetical filename ordering was used
to infer the EEG axis. The implementation rejects reordered numeric metadata
rather than silently sorting it. The 200 referenced files exactly match the
extracted inventory, with **zero missing images, duplicate test IDs, extra test
files, test/train original-THINGS image-identity overlap or filename overlap**.
Concept strings are identity/folder metadata, not ImageNet classes or supervision.

The evidence validates the release's documented axis convention against the
actual ordered metadata and all 200 files. It does **not** independently reconstruct
raw session/trial events or hashes of unacquired training images. Those limits
are explicit in the audit, rather than claimed as completed checks.

### Real EEG RDM and matching vision smoke results

Existing preprocessing and `compute_rdm` produced:

| Real output | Features | RDM | Distance |
| --- | --- | --- | --- |
| EEG, all test images | **(200, 680)** | **(200, 200)** | correlation |
| EEG, explicit indices 0–15 | **(16, 680)** | **(16, 16)** | correlation |
| Frozen DINOv2, same 16 images | **(16, 768)** | **(16, 16)** | cosine |
| Frozen CLIP, same 16 images | **(16, 512)** | **(16, 16)** | cosine |

Brain RDMs and vision RDMs are finite, symmetric and zero-diagonal. The 16-image
Brain RDM equals the corresponding restriction of the 200-image RDM. Full Brain
distances range from 0 to 0.6227551232878772. Saved IDs and feature caches preserve
the explicit EEG index selection. A separate verification pass checked all output
SHA256s and independently recomputed **all four saved RDMs** from safely reloaded
features. A real integration test also extracts a four-image subset directly from
320 real trials, confirming feature shape, image IDs, repetitions and a 4×4 RDM.

Vision used the already installed frozen CPU backbones and pinned local revisions:
DINOv2 `f9e44c814b77203eaa57a6bdbbd535f21ede1415` (CLS embeddings), CLIP
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (projected image embeddings), normalized
features and each checkpoint's own processor. `local_files_only=True` plus
Hub/Transformers offline environment flags prevented model network acquisition.
These outputs use **real stimulus JPEGs**, not the earlier synthetic images.

### Commands and test results

Executed in the existing Python environment:

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
python scripts/audit_real_data.py
# Initially exit 2; wrote the safe-header audit before explicit approval.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/smoke_real_data.py --config configs/real_subset.yaml --trust-official-pickle
NEUROCURATE_REAL_ACQUISITION_MANIFEST=results/real_phase0/sub01/acquisition_manifest.json \
NEUROCURATE_REAL_PICKLE_APPROVED=1 \
  python -m pytest tests/test_real_local_inputs.py -q --junitxml=results/phase1/real_input_tests.xml
python -m pytest --junitxml=results/phase1/regression_tests.xml
git diff --check
```

Optional local real-data suite: **4 passed, 0 skipped in 3.02 seconds**.
Standard network-independent suite: **127 passed, 4 optional real-input tests
skipped in 7.16 seconds**. Five new synthetic alignment tests cover ordered numeric
mapping, refused reorder, missing/extra stimuli and split contamination. The
standard suite never requires EEG downloads or pickle approval. Output/input
hashes and real vision-RDM recomputation also passed outside pytest.

Reports and smoke directories refuse accidental overwrite. For a repeat, create
a new audit (`audit_real_data.py --output NEW_AUDIT.json`), pass `--audit
NEW_AUDIT.json` to the smoke script and select a config with a fresh output path.
Approval applies only to the two retained, verified official NPY files.

### Limits and next experiment

Current free disk after the smoke outputs was **30,721,724,416 bytes (28.612 GiB)**.
There were no additional EEG, image or checkpoint downloads and no deletions.
Only small bounded author-source text downloads supplemented mapping evidence.

**Real input integration passes.** No current engineering blocker prevents a
repeat with these retained files. Scientific reliability, null controls and
cross-subject consistency remain unmeasured; this single-subject smoke test
does not establish reliable disagreement or benefit. Author MVNN shares fitted
covariance across training/test partitions and repetitions, so supplied arrays
must remain sanity-only for later training evaluation. Future independently
preprocessed training/evaluation requires raw-data preprocessing.

Recommended next experiment: repeated random split-half RDM correlations with
image-label permutation controls, reported conditional on the supplied whitening;
then repeat with a second subject. Strict session-independent estimates require
verified session labels and independently fitted preprocessing. Full 200-image
vision disagreement/visualization and multi-subject NeuroCurate scoring are
subsequent experiments, not claimed as completed in this subset integration audit.

## Phase 1.3 — First real scientific result

**Completed using only existing subject-01 inputs; no new subjects, image payloads,
model weights, architectures, classifier training or OOD benchmarking.** All
required outputs are in `results/real_phase0/sub01/`, including the two figures,
CSV scores, reliability/stability JSON, feature/RDM arrays and
`scientific_summary.md`. This supersedes the previous “reliability unmeasured”
status; previous acquisition manifests and input archives remain intact.

### Controlled deserialization and pickle-free downstream

The two explicitly approved input hashes were checked against the acquisition
manifest and again inside the child:

- EEG test export: `08e52b18caf46b8504beb4edaadef025f114eaea25ab1f4eb8c5f9114c56ebff`.
- Image metadata: `681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c`.

The complete parent archive was also reverified against its pinned official OSF
URL/hash. `bwrap` failed with “setting up uid map: Read-only file system”; network/
mount `unshare` was denied, and Landlock returned ENOSYS. These mechanisms were
not bypassed. Instead, an unprivileged UID-1000 child used **kernel-enforced
seccomp default-deny syscall confinement**, with actual enforcement probes denying
filesystem opens, socket creation and fork **before** unpickling. No filesystem
open, network, process creation, exec, ptrace or further seccomp operation is on
the allowlist. Only existing descriptors can be used: two readonly approved
inputs, five designated writable exports and stdio. No other inherited descriptors
were allowed, the environment was sanitized, Python ran with `-I`, and filter
installation required a single-threaded process. Resource limits: **4 GiB address
space, 120 CPU seconds, 512 MiB per file, no core dumps**, 150-second parent timeout.

The first confined attempt could not import the old pickle's `numpy.core` module
after filesystem opens had been denied. It stopped safely; empty attempt outputs
were preserved as `pickle_free_failed_import_1/`. Preloading trusted legacy NumPy
runtime modules **before** confinement fixed this, without weakening the filter.
The successful export produced `eeg.npy`, `times.npy`, `channels.npy`,
`metadata.json` and `isolation_report.json` under `pickle_free/`, with hashes in
`export_manifest.json` and a completion marker. Numeric/schema validations and
metadata-string validation were performed during export; exact image mapping
was checked downstream using JSON. **This Phase 1.3 run did not deserialize
pickle in the main environment. Every downstream array loads with
`allow_pickle=False`.**

### Actual data and Brain RDM

Actual numeric export: **float64 (200,80,17,100)**, 200 images, 80 repetitions,
17 posterior/occipital channels, 100 time points. Time vector is −0.20 through
+0.79 seconds at **100 Hz**; **zero missing or nonfinite EEG values**. Shape/channel/
time details are in `eeg_shape_report.json`. Image-axis index i corresponds to
explicit numeric test condition i+1, validated against actual metadata and the
pinned author collection/conversion/preprocessing protocol, not alphabetical
filename inference. All **200** stimulus files map one-to-one, with no missing,
duplicate or extra files, zero content aliases among test JPEGs, and zero metadata
test/train identity overlap. Raw per-trial/session labels remain unavailable.

Posterior baseline: all 17 supplied P/O/PO channels, **100 ≤ t < 500 ms**, no extra
baseline correction/standardization, average repetitions then flatten. Features
are **(200,680)**. Correlation-distance **(200,200)** Brain RDM is finite, symmetric,
zero-diagonal and ordered to saved image IDs. Reloading numeric features and
independently recomputing the RDM passed. Configuration, source/export/image hashes
and output checksums are retained in `scientific_config.yaml` and
`scientific_provenance.json`.

### Reliability and image-label null

Thirty seeded random splits partition 80 repetitions into disjoint **40/40**
halves, using the same repetition-index groups for every image and identical
preprocessing. Images are never shuffled between halves. Spearman uses only the
19,900 unique off-diagonal RDM pairs.

- Median split-half Spearman: **0.7065**.
- Empirical 95% random-split variability interval: **[0.6775, 0.7236]**.
- Null median statistic: **−0.0001**; null 95% interval **[−0.0489, 0.0469]**.
- One-sided Monte Carlo **p=1/1001=0.000999**.

The null uses 1,000 image-label permutations of half B, **simultaneously permuting
rows and columns**, maintaining matrix symmetry and RDM pair dependencies. For
each null replicate, the same image permutation is used across all 30 splits;
the test statistic is the median split correlation. This is not a permutation
of independent matrix edges. Split records and null statistics are saved.
The interval describes random-split variability, **not a population confidence
interval**; splits overlap across repetitions of the experiment.

### Full 200-image model comparison and ranking stability

Pinned frozen DINOv2/CLIP produced **(200,768)** / **(200,512)** real-image features
and 200×200 cosine RDMs. The existing offline checkpoints/revisions were reused.
Features, caches, stimulus hashes and safely recomputed model RDMs were verified.
`disagreement_scores.csv` contains raw **1 − neighborhood Jaccard** scores for
each model and their equally weighted mean, at K=5/10/20. These are raw single-
subject disagreement scores, **not cross-subject NeuroCurate consensus scores**.

| Mean-model disagreement | Split score-ranking Spearman median | 95% split interval | Top-20 overlap median | Tie-aware null overlap median | Top-20 null p |
| --- | ---: | --- | ---: | ---: | ---: |
| K=5 | **0.3195** | [0.2092, 0.4426] | 0.60 | 0.55 | **0.1279** |
| K=10 | **0.4846** | [0.4020, 0.5611] | 0.25 | 0.15 | **0.0010** |
| K=20 | **0.6044** | [0.5539, 0.6576] | 0.30 | 0.10 | **0.0010** |

Score-ranking Spearman uses average ranks for ties; all nine model/K ranking
null comparisons have exploratory p=0.0010. Stable image-ID tie-breaking is used
for displayed Top-20 membership. Crucially, the Top-20 null **permutes half-B
score labels then reapplies that same tie rule**, and aggregates the median across
splits. A generic 20/200 overlap reference would miss tie-induced overlap. At K=5,
apparent shortlist stability is largely compatible with that tie-aware null.
At primary K=10, the two halves retain a median **5/20** images, and the full-score
cutoff has **29 tied maxima**: the plot shows one deterministic 20-image subset
of that group, not 20 uniquely superior candidates. Score rankings have moderate
repeatability; shortlists are not ready to be treated as definitive.

`top20_disagreement.png` shows each anchor plus its two nearest Brain, DINOv2 and
CLIP neighbors, with the cutoff tie and exploratory scope labeled. The PNG was
visually inspected. `brain_rdm.png` shows the full ordered real Brain RDM.

### Reproduction and checks

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
python scripts/export_isolated_real.py
# Export already exists: do not rerun/overwrite; reuse its verified numeric files.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib \
  python scripts/scientific_subject01.py
NEUROCURATE_PHASE13_RESULTS=results/real_phase0/sub01 \
  python -m pytest tests/test_real_scientific_outputs.py -q \
    --junitxml=results/real_phase0/sub01/phase13_real_tests.xml
python -m pytest --junitxml=results/real_phase0/sub01/phase13_tests.xml
git diff --check
```

Final standard suite: **130 passed, 7 optional real tests skipped (7.05 seconds)**.
Phase 1.3 real output tests: **3 passed (0.83 seconds)**. Those tests verify export
hashes/isolation evidence, all scientific output checksums, direct real EEG means
and saved Brain/model RDM recomputation, split disjointness/coverage, independently
recomputed first-two split Spearman correlations and the saved Monte Carlo p-value.
Three new network-free tests verify the rank-matrix permutation optimization
against SciPy Spearman, tie handling and the variability-interval calculation.
The old direct-pickle real integration tests were not enabled in this task.
No subject-02 download or training was performed; all verified inputs are preserved.

### Scientific decision

**GO to subject-02 for a narrowly scoped replication/reliability experiment.**
The real Brain RDM is valid, and repetition-split reliability is clearly above
image-permutation controls. Disagreement score rankings show moderate repeatability,
but ties and weak shortlist overlap prevent a strong curation claim.

**All EEG reliability, brain–model comparison and ranking inferences remain
exploratory and conditional on the supplied shared preprocessing.** Official MVNN
uses covariance from both training/test partitions and all repetitions. Disjoint
repetition halves are not independently preprocessed or independent sessions.
No cross-subject consensus, training benefit, OOD benefit or causality is established.
Strict session-independent reliability or held-out training evaluation needs raw
data and separately fitted preprocessing. Subject-02 replication should precede
using disagreement for selection; no additional subject was acquired in this task.

## Phase 1.4 — real cross-subject replication (2026-10-11 UTC)

Completed in the existing `/workspace/neurocurate/neurocurate` checkout with
`/workspace/setup/neurocurate/venv/bin/python`. No worktree, setup reinstall,
full nm000232 instantiation, full-dataset acquisition, new vision inference,
training, OOD evaluation, or additional subject acquisition occurred.

### Frozen protocol and acquisition evidence

Before subject-02 acquisition, the unchanged subject-01 EEG/model settings and
five scoring methods were frozen in `configs/cross_subject_01_02.yaml`, SHA256
`14e070fbdc40aff039bf9851577c16a9f430f018a00568d3d5252e6a9475f2cc`.
The timestamp, complete config text, and original subject-01 config hash are in
`results/real_phase0/cross_subject_01_02/frozen_protocol.json`.

Fresh independent trusted metadata was obtained over verified TLS from
`https://api.osf.io/v2/files/62f6b6eb170d520c73c27e35/` with a bounded 65,536-byte
metadata request. The official standard preprocessed `sub-02.zip`, file version 1,
dated 2022-08-12, was acquired from `https://osf.io/download/u43p5/`:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| Complete subject-02 ZIP | 1,071,516,660 | `e1eba2d74009c6dd41becf53d84b8d47532103b41110a5855e7e286bcca6fe41` |
| Extracted test NPY | 217,601,432 | `e5ff0774a83beb49366a73c73fb04c9e692d5ab1f80c3d57c9fd1ff6e110d791` |
| Reused image metadata NPY | 656,579 | `681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c` |

The complete archive was streamed with byte limits and verified against the
independently retrieved published SHA256. This was full archive verification,
not just metadata inspection or successful HTTP Range access. Its only extracted
member was `sub-02/preprocessed_eeg_test.npy`, compressed size 208,591,064 bytes;
ZIP CRC verification passed. The NPY hash was derived from that verified extraction
and rechecked before conversion. Training EEG remains inside the preserved ZIP.
Archive path: `data/local/real_phase0/sub02/sub-02.zip`; extraction:
`data/local/real_phase0/sub02/eeg/sub-02/preprocessed_eeg_test.npy`.
Fresh official metadata and all sizes/hashes are retained in
`results/real_phase0/sub02/{official_source_metadata,acquisition_manifest}.json`.

Before any EEG download, available disk was 30,487,580,672 bytes (28.39 GiB).
Conservative additional peak budget was 3,487,435,764 bytes (3.25 GiB): retained
archive, 512 MiB test-member allowance, 512 MiB numeric-export allowance,
256 MiB scientific outputs, and 1 GiB reserve. It fit. After extraction, free disk
was 29,198,434,304 bytes; after analysis it was 28,943,241,216 bytes. These are
observed free-space snapshots, not a measured instantaneous peak. No files were
deleted. The environment memory limit was 32 GiB; deserialization independently
had a 4 GiB address-space limit. Numeric EEG occupies 217.6 MB per subject;
one subject was processed at a time, with half-RDMs retained for verification.

### Controlled conversion and exact mapping

The authorized subject-02 object array and previously verified image metadata were
loaded only in the unprivileged UID-1000 child described in Phase 1.3, with the
same default-deny kernel seccomp allowlist. Hashes were checked before launch and
again inside the child. Filesystem opens, network sockets, and process creation
were demonstrably denied before deserialization. Only two readonly approved input
FDs and five designated output FDs plus stdio were available; CPU 120 seconds,
address space 4 GiB, output size 512 MiB/file, timeout 150 seconds, core dumps off.
`results/real_phase0/sub02/pickle_free/isolation_report.json` records all probes.
Numeric EEG/times/channel arrays and JSON metadata were immediately exported,
hashed, and marked verified. All downstream loads use `allow_pickle=False`.
Namespace support remains unavailable; the functioning kernel syscall confinement
was used, not an unconstrained fallback. No direct-pickle integration tests ran.

Actual subject-02 EEG is float64 `(200,80,17,100)`: 200 images, 80 repetitions per
image (16,000 trials), 17 posterior channels, 100 Hz, −0.20 to 0.79 seconds,
no missing/nonfinite values. Channel names/order and time values match subject-01
exactly. See `sub02/eeg_shape_report.json` for the complete channel and time lists.
Both subjects use the official repeated test split, internally labeled `sanity`.
The metadata numeric condition convention and pinned official event-sorting source
establish image-axis index i → condition i+1 → the exact metadata image path.
The 200 ordered IDs match subject-01 exactly; all 200 local files exist and have
distinct hashes. Missing IDs, duplicates, metadata split overlap, and mismatched
ordering are absent. No EEG order was inferred from alphabetical filenames.
Raw event roundtrip, session labels, and cross-split pixel-hash alias checks are
unavailable; training images were not downloaded.

### Exact commands executed

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
python scripts/acquire_subject02.py
python scripts/export_isolated_real.py --subject sub02
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib \
  python scripts/cross_subject_analysis.py \
  > results/real_phase0/cross_subject_01_02/analysis.log 2>&1
python scripts/report_cross_subject.py
python -m pytest -q
NEUROCURATE_PHASE14_RESULTS=results/real_phase0/cross_subject_01_02 \
NEUROCURATE_PHASE13_RESULTS=results/real_phase0/sub01 \
  python -m pytest tests/test_real_cross_subject_outputs.py \
    tests/test_real_scientific_outputs.py -q
```

Acquisition and isolated export are one-time commands, already completed. The
exporter intentionally refuses an existing export directory. Reproduce science
using the last two analysis commands and existing verified numeric exports.
An initial analysis reached the report-writing step but encountered YAML's bare
`null` key representation; the reader was corrected without changing any frozen
protocol bytes or parameters, and the full final analysis completed successfully.
The original subject-01 report content was preserved against its prior SHA256
manifest; no source acquisition or numeric export was replaced.

### Scientific results

Both subjects have saved `(200,680)` averaged EEG features and valid `(200,200)`
correlation Brain RDMs, finite, symmetric, zero-diagonal, correctly ordered, and
independently recomputable from numeric EEG. All 200-image frozen DINOv2 and CLIP
embeddings were reused from subject-01 with verified image hashes and pinned
checkpoint revisions. No model weights or stimulus images were downloaded.

| Brain RDM statistic | Spearman | Empirical split interval | Permutation p |
|---|---:|---|---:|
| Subject-01, disjoint 40/40 repetitions | 0.7065 | [0.6775, 0.7236] | 0.000999 |
| Subject-02, disjoint 40/40 repetitions | 0.6466 | [0.6251, 0.6694] | 0.000999 |
| Cross-subject, full 80 repetitions each | 0.5495 | Not a repetition-split interval | 0.000999 |

Cross-subject Brain RDM permutation-null 95% interval is [−0.0508, 0.0481].
Within-subject null intervals are [−0.0489, 0.0469] and [−0.0350, 0.0390].
Nulls permute image identity simultaneously on both RDM axes, preserving geometry.
Cross-subject agreement is a different estimand from within-subject reliability;
full versus half RDMs use different repetition counts, and no noise ceiling or
session-independent reliability is inferred.

Mean-model disagreement ablation, using the same independent repetition halves:

| Method | Within 01 rank ρ | Within 02 rank ρ | Cross rank ρ | Cross Top-20 shared | Tie-aware Top-20 p |
|---|---:|---:|---:|---:|---:|
| Jaccard K=5 | 0.320 | 0.367 | 0.281 | 10/20 | 0.5465 |
| Jaccard K=10 | 0.485 | 0.464 | 0.418 | 6/20 | 0.0030 |
| Jaccard K=20 | 0.604 | 0.548 | 0.480 | 2/20 | 0.6513 |
| Nonself RDM-row Spearman disagreement | 0.855 | 0.802 | 0.744 | 6/20 | 0.0060 |
| All-neighbor reciprocal-rank weighted Jaccard | 0.633 | 0.629 | 0.554 | 4/20 | 0.1219 |

Each method is also reported separately for DINOv2 and CLIP in the full report,
CSV, and JSON. At K=10 their cross-rank correlations are 0.2332/0.5052; Top-20
shared counts are 9/20 and 5/20, with tie-aware p=0.1678/0.3057. Thus model-specific
shortlist replication is weaker than the mean-model result.

Score ties are substantial for Jaccard: mean-model K=5 cutoff tie groups contain
105/98 images; K=10 groups 29/32; K=20 groups 15/10. Continuous row-rank and
reciprocal-weighted metrics have 200 unique full scores per subject/model and
cutoff tie size 1. At K=5, the mean-model 10/20 observed overlap equals its
10/20 tie-aware null median. A uniform 2/20 overlap reference would mislead.
Nulls permute image scores and then reapply the identical stable ID tie rule.
Spearman uses average ranks. No method or hyperparameter was selected from
subject-02 outcomes; row-rank's stronger repeatability is an exploratory finding.

All intervals are empirical random-split variability, not population confidence
intervals; random splits overlap across reruns. All p-values are exploratory,
one-sided, unadjusted Monte Carlo values with 1/1001 resolution. Two subjects
cannot establish population generalization or stable curation benefits. The
shared fixed vision geometry can contribute to cross-subject score agreement.

### Tests and artifacts

- Standard network-independent suite: **135 passed, 11 optional skipped**, 7.41 s.
- Opt-in local numeric real-data suite: **7 passed**, 1.73 s, no network or pickle.
- Five added synthetic metric/null tests passed: identical geometries, nonself
  row Spearman against SciPy, reciprocal-rank weighting, tie-aware permutation
  ordering, and invalid/constant inputs.
- Real tests independently recompute subject-02 features/RDM, check both export
  hash manifests and isolation probes, confirm all original subject-01 scientific
  output hashes, recompute cross-subject RDM/score correlations and Top-20 overlap,
  and validate disjoint repetitions with exact image correspondence.
- `ranking_stability.png` was visually inspected for labels and plotted values.

Required outputs in `results/real_phase0/cross_subject_01_02/`:
`cross_subject_rdm.json`, `cross_subject_disagreement.json`, `scoring_ablation.csv`,
`ranking_stability.png`, `scientific_summary.md`. Additional retained evidence:
`frozen_protocol.json`, `provenance.json`, `verified_outputs.json`,
`image_level_scores.csv`, `sub01_half_rdms.npz`, `sub02_half_rdms.npz`,
`test_results.json`, and `analysis.log`. Subject-02 shape/alignment reports,
features/RDM/IDs, reliability, score stability, figure, and numeric export are
under `results/real_phase0/sub02/`. Local data/results remain git-ignored.

### Decision and remaining blockers

**GO for a limited prospective expansion with the same frozen methods and
settings; NO-GO for promoting the current Top-20 into a validated curation list.**
Strong Brain RDM replication and cross-subject score-ranking evidence support a
small additional replication cohort. Predeclare the next subjects and keep all
methods; assess shortlist overlap separately from ranking agreement. No further
subject acquisition or classifier/OOD work was performed here.

Every EEG reliability and disagreement inference remains exploratory and
conditional on the supplied official MVNN whitening, whose covariance fitting
shared training/test data and repetitions. Disjoint repetition assignment does
not provide independently fitted preprocessing or session independence. Strict
held-out evaluation/session reliability requires raw data or a separately fitted
preprocessing export. Other remaining limits are only two subjects, high Jaccard
tie counts, inconsistent shortlist replication, and potential shared-model
contributions to score correlations. There is no unresolved acquisition, exact
mapping, conversion, or numeric Brain RDM blocker for these two subjects.

## Phase 1.5 — Brain-specific controls and selection stability (2026-10-11, UTC+8)

Completed in the existing checkout and installed Python environment. Only existing
verified numeric subject-01/02 exports, derived Phase 1.4 half-RDMs, and cached
vision features were used. No dataset/network request, new subject/image/checkpoint,
object-array deserialization, classifier training or worktree was involved.

### Protocol and reproducible commands

`configs/brain_specific_controls.yaml` was frozen before Phase 1.5 computation,
SHA256 `61d17efe3eb0842080757a396ddf0b3c1ea0afcfc91e3b4e780edb3eafc9888d`.
The complete text and UTC timestamp are retained in
`results/real_phase0/brain_specific_controls/frozen_protocol.json`. This is an
exploratory extension after Phase 1.4 observations, not an independently
preregistered confirmation. Phase 1.4 config SHA256, preprocessing, exact ordering,
RDM metrics, checkpoint revisions, all five scoring definitions, K=5/10/20,
equal-weight model aggregation and tie policies remain unchanged. Row-rank is
labeled a promising exploratory candidate; all 15 method/model variants are saved.

Executed commands:

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib \
  python scripts/brain_specific_controls.py \
  > results/real_phase0/brain_specific_controls/analysis.log 2>&1
python -m pytest tests/test_brain_controls.py -q
python -m pytest -q
NEUROCURATE_PHASE15_RESULTS=results/real_phase0/brain_specific_controls \
  python -m pytest tests/test_real_brain_specific_outputs.py -q
python scripts/report_brain_specific.py
```

Retain `frozen_protocol.json` with the local results when reproducing these
commands; its hash must agree with the config. The report generator reads saved
statistics. Prior data/outputs and their provenance manifests are unchanged.

### Null construction and model-only controls

1. **Independent brain labels:** 1,000 independently generated permutation pairs,
   seed 2026101501. Within each subject, simultaneously relabel rows/columns via
   `Brain[p,p]`; preserve full Brain RDM geometry; recompute all scores against
   unchanged, correctly ordered model RDMs. Recompute score ranks and selections.
2. **Shared brain labels:** 1,000 permutations, seed 2026101502. Apply the same
   permutation to both subjects, then recompute all scores. This preserves the
   between-subject Brain geometry while breaking its correspondence to the models.
   Shared-null agreement is not a model-only effect; preserved brain geometry can
   interact with common fixed model geometry.
3. **Score-label reference:** 1,000 permutations, seed 2026101503. Relabel B's
   existing score vector and reapply the exact score/ID tie rule. This is a
   shortlist-distribution reference, not proof of brain-specific information.
4. **Repetition splits:** the original 30 disjoint 40/40 partitions, seed 20261011,
   reproduced exactly. Every image retains its repetition correspondence. Null
   permutations use the same draw index when taking medians across these splits.

Model-only covariates are DINOv2–CLIP nonself RDM-row disagreement, each model's
mean nearest-20 cosine distance, and each model's mean distance to all 199 other
images. Atypicality excludes self. The five ranked, standardized controls plus
intercept give design rank 6/6, condition number 3.478 and residual df 194.
Partial rank association is Pearson correlation between residualized *ranked*
EEG score vectors. It is not Spearman applied again to residuals. The same fixed
projection is reapplied to every recomputed null score. No iid image/edge OLS
significance test or prediction claim is made.

### Real-data findings

Mean-model row-rank observed cross-subject Spearman: **0.7435**.

| Control | Null median | 95% null interval | Observed upper-tail p |
|---|---:|---|---:|
| Independent brain-label permutations | −0.0054 | [−0.1956, 0.1708] | 0.000999 |
| Shared brain-label permutations | 0.5113 | [0.3403, 0.6620] | 0.001998 |
| Partial-rank independent brain null | −0.0035 | [−0.1725, 0.1500] | 0.000999 |
| Partial-rank shared brain null | 0.5139 | [0.3552, 0.6591] | 0.007992 |

Observed partial rank association is **0.6927**. The tested model-only covariates
explain 17.8%/19.5% of subject-01/02 score-rank variance in sample. The largest
individual proxy correlation is CLIP nearest-neighbor atypicality, 0.1998/0.2884;
DINOv2–CLIP row-disagreement correlations are −0.0606/0.0253. These descriptive
associations do not establish causal or predictive value or exhaust image confounds.

Correctly aligned mean row disagreement is 0.4514/0.4432, versus independent
brain-null means near 0.5000/0.5004. Correlation between scrambled and correctly
aligned rankings has medians −0.0054/0.0022. Together with the recomputed nulls,
this supports dependence on correct alignment under the tested null assumptions.
However, a substantial ranking correlation (~0.51) remains under the same wrong
image permutation in both subjects. Observed excess over that median is 0.2322,
not a variance decomposition or fraction of brain-specific information.

DINOv2 row-rank observed correlation is 0.6025, shared-null median 0.5114,
unadjusted p=0.0490; its residual association 0.5920 has shared-null p=0.0669.
CLIP correlation is 0.7720, shared-null median 0.5164, p=0.0020; residual
association 0.7085 has shared-null p=0.0160. Thus the strength is model-dependent.
All other predeclared scoring variants are reported rather than discarded.

### Selection budgets and uncertainty

Mean-model row-rank shortlist overlap fraction:

| Budget | Within 01 median | Within 02 median | Full cross-subject shared | Shared-brain null median fraction | Shared-null-adjusted overlap | Shared-null p |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 0.450 | 0.500 | 2/10 | 0.300 | −0.143 | 0.8162 |
| 20 | 0.550 | 0.550 | 6/20 | 0.350 | −0.077 | 0.7862 |
| 40 | 0.713 | 0.650 | 21/40 | 0.450 | 0.136 | 0.2388 |
| 60 | 0.750 | 0.733 | 38/60 | 0.550 | 0.185 | 0.0779 |
| 100 | 0.840 | 0.820 | 81/100 | 0.680 | 0.406 | 0.000999 |

Within-subject overlap exceeds tie-aware score-label nulls at every budget;
empirical intervals and chance-adjusted intervals are retained in the CSV.
Against independent brain and score-label references, full cross-subject budget
20/40/60/100 overlaps are exploratory excesses, while budget 10 is not. Against
shared-brain controls, only budget 100 clearly exceeds the null for mean-model
row-rank. At 100, CLIP shares 85 images (shared-null p=0.000999) whereas DINOv2
shares 72 (p=0.1608). Selecting half the panel is a much broader selection task
than a Top-10 or Top-20 shortlist. No budget was chosen as an optimal policy.

Tie-aware nulls relabel scores before reapplying descending-score/stable-image-ID
ordering; brain nulls recompute scores before ordering. Average ranks are used for
Spearman. Chance adjustment is the frozen descriptive formula
`(observed overlap − median null overlap)/(1 − median null overlap)`; negatives
are retained. It is median-normalized excess, not mean-chance kappa. A discrete
budget-10 null can have median zero while expected uniform overlap is 10/200;
zero median does not mean no chance overlap.

Within-subject 2.5–97.5% quantiles measure variability across the original 30
random repetition splits, which reuse repetitions across reruns. Cross-subject
half-RDM quantiles use 60 A/A and B/B comparisons, and are reported separately
from full 80-repetition scores. Neither provides an independent-subject population
confidence interval. Null envelopes are permutation quantiles. P-values are
exploratory, one-sided, unadjusted Monte Carlo values with minimum 1/1001;
multiple methods/models/controls/budgets have been inspected.

### Validation and retained artifacts

- Network-independent standard suite: **140 passed, 15 optional skipped**, 14.87 s.
- Five new synthetic control tests: **5 passed**, 0.64 s.
- Optional local Phase 1.5 real-numeric tests: **4 passed**, 2.33 s.
- Real tests recompute null scores from permuted Brain RDMs, validate shared
  permutation invariance of cross-Brain geometry, check residual/null correlations,
  all budget overlaps and chance adjustments, seeds, input and code hashes.
- Input checks recompute full Brain RDMs from numeric EEG and model RDMs from
  cached embeddings; validate identical image ordering and exact original split
  assignments; verify all prior subject-01 and Phase 1.4 output hashes.
- `selection_budget_curves.png` was visually inspected. Its top bands show
  repetition-split variability; bottom bands show null quantiles.
- No legacy direct-pickle tests were enabled.

Required outputs in `results/real_phase0/brain_specific_controls/`:
`brain_specific_nulls.json`, `model_only_controls.csv`,
`selection_budget_curves.csv`, `selection_budget_curves.png`,
`scientific_summary.md`. Extra evidence includes `frozen_protocol.json`,
`model_only_associations.json`, `selection_split_statistics.json`,
`null_statistics.npz`, `observed_scores.npz`, `provenance.json`,
`test_results.json`, `verified_outputs.json` and `analysis.log`.
All NPZ arrays are numeric or Unicode and load without pickle. Local results
remain git-ignored and are retained in this workspace.

### Scientific decision and limits

**GO for prospectively specified replication on subjects 03–05**, keeping all
methods and budgets fixed before inspecting those subjects. This is a recommendation
only; no additional subject was acquired. Row-rank remains an exploratory candidate.
Correct alignment affects ranking agreement beyond independent and shared brain
controls, with residual association beyond the tested model-only proxies. This
justifies testing replication, not declaring a validated curation winner.

**Small shortlists remain unvalidated.** Global rank correlation is stronger than
selection overlap, particularly against the shared-brain null. Budget-100 evidence
is promising but cannot validate Top-10/20 selection or identify an optimal budget.

Every inference remains conditional on official shared MVNN whitening, whose
covariance fit mixed training/test partitions and repetitions. Disjoint repetition
assignment does not provide independently fitted preprocessing or session-independent
reliability. Brain-label permutations can disrupt image confounds along with genuine
neural alignment; five model-only proxies cannot remove every visual/semantic
confound. Only two subjects and a fixed 200-image panel are available. No causal
value, improved data efficiency, predictive usefulness, generalization or OOD
benefit is claimed. Strict future held-out evaluation requires appropriately
separated preprocessing and preservation of the original test split.

## Phase 1.6 — Prospective subjects 03–05 replication (2026-10-11, UTC+8)

The existing checkout and installed Python environment were reused. No worktree,
setup reinstall, classifier training, new model, new metric or budget was created.
This is prospective replication of a candidate discovered using subjects 01–02,
not an original preregistration.

### Immutable snapshot before new-subject access

Snapshot Git commit: `ac462a65b49158cf63311c6c2dfb596f7cc86af2`.
Parent checkout HEAD: `51793f971498dcb03271d5a9f139fa1ce5302424`.
Snapshot SHA256:
`20840d3661ce9c75e39c1290887b8e8f0b83eebe18150d93b79c268c6654be6e`.
Prospective config SHA256:
`54caa4d67ade0a4b1f90edfa1636515b1d443f575aab54634b6e9efdf9b7d070`.

Before any new-subject request or EEG inspection, all source, scripts, configs,
tests and dependency metadata were collected into deterministic
`results/real_phase0/prospective_03_05/protocol_snapshot.tar`. A content-addressed
Git commit stores that archive at `refs/neurocurate/phase16-protocol`; the branch,
checkout and existing changes were preserved. `frozen_protocol.json` records
exact per-file SHA256, timestamp, base/snapshot Git commits, snapshot Git blob,
config and failure policy. The runtime checks frozen source/config hashes before
acquisition and analysis. This snapshot is distinct from committing all pending
repository changes to the working branch.

The candidate is mean-model row-rank disagreement. DINOv2, CLIP, Jaccard K=5/10/20
and reciprocal-rank weighting are secondary; budgets remain 10/20/40/60/100.
Phase 1.5 preprocessing, ordering, checkpoints, score definitions, five model-only
covariates, OLS rank residualization, seeds and 1,000-permutation counts remain
fixed. Failed/missing subjects are retained in status/reliability outputs, with
no replacement, parameter adjustment or synthetic substitution. Complete-cohort
GO decisions require all three subjects.

The primary new-subject cohort statistic is the median of the three new-new pair
correlations, recalculated within each joint five-subject null draw. The full
and partial primary tests have a frozen two-test Holm correction. Five cohort
budget tests receive Holm correction separately. No pairwise correlation is
regarded as an independent replicate. Frozen decision rules are in the config.

### Acquisition, capacity and verified sources

Fresh trusted metadata was obtained from bounded official OSF file API requests,
over verified TLS, before each archive download. Names, sizes and published hashes
matched independently recorded official inventory pins. Complete archive byte
counts and SHA256 were then verified before extraction.

| Subject | Official standard ZIP URL | ZIP bytes | ZIP SHA256 |
|---|---|---:|---|
| 03 | https://osf.io/download/fy4n6/ | 1,071,567,672 | `ec12b31410b8161f5b274b4c61890a615e5fdbb5e58e6e9325a0188272daa936` |
| 04 | https://osf.io/download/py8jz/ | 1,071,421,033 | `eec31590d21248d9cc8caf3b93f5711b2c24540f080eab4f83f152155f5a6eb5` |
| 05 | https://osf.io/download/2cn7v/ | 1,072,655,693 | `d353e154e840ff9c07b2de19db2105626df4bc556313365cff392dd56d8df40e` |

Only `sub-XX/preprocessed_eeg_test.npy` was extracted, with ZIP CRC verification.
Each uncompressed member is 217,601,432 bytes:

| Subject | Test compressed bytes | Extracted test SHA256 |
|---|---:|---|
| 03 | 208,608,501 | `a84287154735104fce3ea830ebad0897f7b654e7812e2a9102f6917590954129` |
| 04 | 208,578,856 | `9632bbd3361a3b9cf62cb8ac3535810c51e012aa6cbe6d149a36a8f7d35e629b` |
| 05 | 208,786,039 | `4822c960746c89bceaf270b697c24c7946cb439b1223f96afda75374a9788d84` |

As explicitly requested, standard ZIPs were acquired; they physically contain
training members as well as test members. Training EEG was not separately
requested, extracted, deserialized or used. No 63-channel archives, new images,
checkpoints or subjects 06–10 were acquired. Verified source ZIPs and test NPYs
are preserved under `data/local/real_phase0/sub03`, `sub04`, `sub05`.

Initial available disk was about 32,025,940,000 bytes; the conservative whole-cohort
additional peak budget was **8,047,482,606 bytes**, including all three retained
ZIPs, conservative test/numeric exports, output allowance and reserve. Capacity
was checked before each download and before extraction. The budget fit. No cleanup
or deletion was performed. These are pre-download estimates/free-space snapshots,
not a measured instantaneous peak. The environment RAM limit was 32 GiB.

### Controlled conversion and schema

Each full archive and extracted test hash was reverified before conversion. The
unchanged isolated worker ran as UID 1000 with a default-deny kernel seccomp
allowlist and only two approved readonly input descriptors plus designated
outputs/stdio. Filesystem-open, network-socket and process-creation denial probes
all passed before deserialization. Limits remained 4 GiB address space, CPU 120 s,
file 512 MiB, timeout 150 s; core dumps disabled. Shared metadata was the originally
verified official file, SHA256
`681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c`.
Only the confined process used pickle; every subsequent NumPy load uses
`allow_pickle=False`. Per-subject export hashes/isolation reports are retained.

All three actual tensors are float64 `(200,80,17,100)`, with 80 repetitions per
image, the same 17 channel names/order, the same 100-Hz times −0.20 to 0.79 s,
and no missing/nonfinite values. All 200 condition IDs map to existing verified
stimuli in exactly the discovery ordering; the original repeated test split is
preserved. Full features are `(200,680)` and Brain RDMs `(200,200)`, using identical
[100,500) ms, posterior channels, trial average and correlation distance.
Each new subject has shape/alignment reports, features, RDM, IDs, half-RDMs,
reliability and acquisition/export provenance under the prospective output root.

### Commands executed

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
python -m pytest tests/test_prospective.py -q
python scripts/freeze_prospective.py
# One-time authorized acquisitions; all three completed successfully.
python scripts/acquire_prospective_subject.py sub03
python scripts/acquire_prospective_subject.py sub04
python scripts/acquire_prospective_subject.py sub05
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib \
  python scripts/prospective_analysis.py \
  > results/real_phase0/prospective_03_05/analysis.log 2>&1
```

The three acquisition workers ran concurrently in separate per-subject paths,
after the whole-cohort capacity check; analysis processes numeric EEG sequentially.
Snapshot/acquisition/export are one-time commands and refuse replacing the frozen
protocol or numeric export. Reproduction uses the verified existing exports and
frozen snapshot with the final analysis/report/test commands documented below.

### Prospective results and decision evidence

| Subject | Median Brain split-half Spearman | Empirical split interval | Reliability permutation p | Within primary score-rank median |
|---|---:|---|---:|---:|
| 03 | 0.8145 | [0.7974, 0.8268] | 0.000999 | 0.8387 |
| 04 | 0.7663 | [0.7477, 0.7814] | 0.000999 | 0.8694 |
| 05 | 0.7335 | [0.7172, 0.7489] | 0.000999 | 0.8335 |

All three acquisitions, isolated conversions, schema/mapping checks and analyses
succeeded; no subjects were missing, failed, replaced or silently excluded.
New-new Brain RDM correlations are 0.5676 (03–04), 0.6010 (03–05), 0.4551 (04–05).
Their primary score-ranking correlations are 0.7010, 0.6508, 0.5849.
Discovery 01–02 ranking is 0.7435; discovery-new median is 0.6728;
new-new median is 0.6508. All-five pairwise descriptive Brain/RDM ranking medians
are 0.5546/0.6728, incorporating discovery and not constituting a new cohort test.
All ten pairs and all 15 scoring variants are reported.

Primary new-new joint cohort tests:

| Statistic | Observed median | Independent null median | Shared null median | Shared null 95% interval | Shared p | Primary Holm p |
|---|---:|---:|---:|---|---:|---:|
| Mean-model row-rank correlation | 0.6508 | −0.0055 | 0.4944 | [0.3589, 0.6237] | 0.011988 | 0.023976 |
| Partial rank association | 0.6005 | −0.0028 | 0.4953 | [0.3590, 0.6204] | 0.047952 | 0.047952 |

Independent-null p=0.000999 for both statistics. Partial rank association retains
the fixed five model-only covariates and rank residualization. The shared null
preserves cross-Brain geometry; its excess is not a causal decomposition.
The cohort median is computed separately inside each joint permutation so that
shared-subject dependence among pair statistics is preserved. Pairwise tests are
secondary and unadjusted, never independent-subject observations.

**Replication is heterogeneous.** For the 03–05 pair, full primary ranking
shared-null p=0.1119; partial p=0.2677. Pair 04–05 partial p=0.1558. Only pair
03–04 has an unadjusted partial shared-null excess at p=0.0140. The aggregate
partial result is borderline: 47/1,000 shared-null draws exceed it; a descriptive
exact binomial 95% interval for the sampled tail probability is [0.0347, 0.0620],
which spans 0.05. This is Monte Carlo sampling uncertainty, not a confidence
interval for neural effects or population replication. No additional permutations,
threshold change or retuning was performed. The frozen GO rule is respected while
its fragility is disclosed. `monte_carlo_uncertainty.json` retains these diagnostics.

Prospective mean-model row-rank budget cohort:

| Budget | New-new median overlap | Joint shared-null median | Unadjusted shared p | Holm p across five budgets |
|---|---:|---:|---:|---:|
| 10 | 0.300 | 0.200 | 0.462537 | 0.569431 |
| 20 | 0.500 | 0.350 | 0.038961 | 0.194805 |
| 40 | 0.500 | 0.450 | 0.248751 | 0.569431 |
| 60 | 0.583 | 0.533 | 0.189810 | 0.569431 |
| 100 | 0.730 | 0.680 | 0.062937 | 0.251748 |

All five overlap statistics exceed the independent-brain references (p=0.000999),
but **none** passes the frozen shared-null Holm correction. Discovery's strong
budget-100 overlap does not clearly replicate. No optimal budget was selected.
The complete selection CSV has 2,625 rows: every within-subject method/model/budget,
and every cross-pair method/model/budget against score-label, independent-brain
and shared-brain nulls. It retains negative chance-adjusted overlaps, within-split
quantiles and separate cross-half variability. Quantile intervals are split
variability or permutation envelopes, not independent-subject population CIs.

### Completed commands, tests and output verification

```bash
MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib \
  python scripts/report_prospective.py
# Append preserved post-result interpretation without altering frozen analysis code.
cat results/real_phase0/prospective_03_05/scientific_interpretation.md \
  >> results/real_phase0/prospective_03_05/scientific_summary.md
python -m pytest -q
NEUROCURATE_PHASE16_RESULTS=results/real_phase0/prospective_03_05 \
  python -m pytest tests/test_real_prospective_outputs.py -q
NEUROCURATE_PHASE15_RESULTS=results/real_phase0/brain_specific_controls \
NEUROCURATE_PHASE14_RESULTS=results/real_phase0/cross_subject_01_02 \
NEUROCURATE_PHASE13_RESULTS=results/real_phase0/sub01 \
  python -m pytest tests/test_real_brain_specific_outputs.py \
    tests/test_real_cross_subject_outputs.py tests/test_real_scientific_outputs.py -q
```

Standard suite: **143 passed, 18 optional skipped**, 9.62 s.
New prospective synthetic statistical checks: **3 passed**, 0.70 s.
Prospective local real-numeric tests: **3 passed**, 1.83 s.
Discovery real-numeric regression checks: **11 passed**, 6.20 s.
No legacy direct-pickle tests ran. All tests are network independent.

The prospective real checks independently recompute numeric features/RDMs for all
new subjects, verify export and snapshot/source hashes, reproduce all pairwise
primary correlations and the joint cohort null from retained draws, and recompute
shared brain-null scores from relabeled RDMs. Additional checks validated cohort
budget null medians/p-values and exact binomial Monte Carlo diagnostics. Original
Phase 1.4/1.5 output manifests remained unchanged. The immutable Git object exactly
reproduces the on-disk snapshot archive, and all acquisition start times follow
its freeze timestamp. The four-panel figure was visually inspected.

All required artifacts are retained in `results/real_phase0/prospective_03_05/`:
`frozen_protocol.json`, `subject_reliability.csv`, `cross_subject_rdm.csv`,
`cross_subject_disagreement.csv`, `brain_specific_nulls.json`,
`selection_budget_replication.csv`, `replication_figures.png`,
`scientific_summary.md`. Additional evidence includes the snapshot archive,
environment versions, per-subject verified inputs/exports/results, subject-status
registry, raw null/score NPZs, within-rank stability CSV, analysis provenance,
Monte Carlo diagnostics, interpretation appendix, test results and output hashes.
Local results/data are git-ignored and preserved. After regenerating the report
and appending the interpretation, refresh the summary hash using the exact README
command; the retained output manifest describes this completed run.

### Frozen decisions and limitations

**A — GO for extending the reliability/ranking replication to all ten subjects.**
All three subjects verified; all Brain reliabilities passed; all new-new primary
correlations are positive; both predeclared primary Holm p-values are ≤0.05.
This is a cautious replication recommendation with borderline residual evidence,
not validation of a scoring winner or shortlists. Subjects 06–10 were not acquired.

**B — GO for a separate exploratory reliability-aware scoring investigation.**
The verified reliable EEG and between-subject variation support investigation,
without promising that weighting will remove subject differences or improve
selection. No new reliability-aware function, exploratory branch or classifier
was implemented in this task.

**C — NO-GO for advancing the independent-pool selection experiment at this stage.**
No prospective budget survives the frozen correction against shared-brain controls.
First strengthen selection stability evidence and plan independently fitted/session-
separated preprocessing; do not choose a favorable budget from these outcomes.

Every EEG inference remains conditional on official shared MVNN, whose covariance
fit combined training/test partitions and repetitions. Disjoint repetition halves
are not independently preprocessed or session-independent. Model-only rank controls
do not exhaust visual/semantic confounds; image-label nulls can disrupt confounds
along with genuine alignment. Pair correlations are dependent, secondary p-values
are unadjusted, and Monte Carlo uncertainty matters near thresholds. No causal,
predictive, data-efficiency, OOD or generalization improvement is claimed.
The original test split remains protected from future training/evaluation choices.

Final saved `reproducibility_checks.json` also independently reproduced all 30
Brain split correlations for every new subject, validated all disjoint/complete
repetition assignments, recomputed both halves of the first split directly from
numeric EEG, and verified unchanged frozen source hashes, exact immutable Git
archive content and preserved working-branch HEAD. All required output hashes
were refreshed after the post-result interpretation/diagnostics were added.
