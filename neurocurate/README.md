# NeuroCurate

Phase 1.6 prospective replication on subjects 03–05 is complete. All three
verified real datasets have valid aligned 200-image Brain RDMs; median split-half
reliability is 0.8145/0.7663/0.7335. New-pair mean-model row-rank correlation has
median 0.6508, versus shared-brain null median 0.4944 (Holm p=0.0240).
Partial rank association is 0.6005 (Holm p=0.0480, borderline). None of the five
selection budgets passes the prospective Holm correction.
See the [scientific report](results/real_phase0/prospective_03_05/scientific_summary.md)
and [replication figures](results/real_phase0/prospective_03_05/replication_figures.png).
**GO for ten-subject replication and a separate exploratory reliability-aware
investigation; NO-GO for advancing the independent-pool selection experiment.**
No subjects 06–10 were acquired and no classifier was trained.

## v2.1 P0 disagreement freeze

The v2.1 measurement primitive is frozen in
`configs/v21_disagreement_frozen.yaml`. It uses the verified five-subject,
200-image caches, correlation-distance Brain/model RDMs, and row-wise
Spearman disagreement for CLIP and DINOv2. Compact frozen features, RDMs,
per-image scores, rankings, validation, and subject correlations are under
`results/real_phase0/v21_frozen/`. Rebuild them with:

```bash
PYTHONPATH=src python scripts/freeze_v21.py
```

This P0 step does not tune the metric or test vulnerability prediction. The
Jaccard and reciprocal-rank measures remain historical baselines; later v2.1
diagnostic experiments must use this frozen definition without outcome-based
retuning.

## Time2Depth Phase 0

The temporal–layer RSA sanity check uses the same five verified subjects and
200-image ordering with the pinned DINOv2 block representations. Results,
nulls, reliability controls, figures, and hashes are under
`results/time2depth/phase0/`. The temporal-order and image-label nulls are
favorable, but late-window EEG reliability leaves an unresolved SNR confound;
the decision is **WEAK / INCONCLUSIVE**, so no training Phase 1 was started.

The protocol and complete code were immutably snapshotted before new-subject
access in Git commit `ac462a65b49158cf63311c6c2dfb596f7cc86af2`, at
`refs/neurocurate/phase16-protocol`; the working branch was preserved.
Exact hashes, failure policy, frozen decisions, runtime versions and snapshot
archive are under `results/real_phase0/prospective_03_05/`. This is prospective
replication of an exploratory-derived candidate, not original preregistration.

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
# Snapshot/acquisition/export already completed; do not rerun those one-time steps.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib \
  python scripts/prospective_analysis.py
MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib \
  python scripts/report_prospective.py
# Preserve the explicit post-result interpretation of borderline/unfavorable findings.
cat results/real_phase0/prospective_03_05/scientific_interpretation.md \
  >> results/real_phase0/prospective_03_05/scientific_summary.md
python - <<'PY'
import json
from pathlib import Path
from neurocurate.lowdisk import sha256
root = Path('results/real_phase0/prospective_03_05')
path = root / 'verified_outputs.json'
manifest = json.loads(path.read_text())
summary = root / 'scientific_summary.md'
manifest['outputs'][summary.name] = dict(bytes=summary.stat().st_size, sha256=sha256(summary))
path.write_text(json.dumps(manifest, indent=2) + '\n')
PY
python -m pytest -q
NEUROCURATE_PHASE16_RESULTS=results/real_phase0/prospective_03_05 \
  python -m pytest tests/test_real_prospective_outputs.py -q
```

The hash-refresh command preserves the post-result interpretation;
the retained `verified_outputs.json` describes this completed run. No source/config
change is allowed without a new protocol version. All EEG inference remains
conditional on official shared whitening, not session-independent or evidence of
improved data efficiency. Exact source URLs, archive/export hashes, capacity checks,
commands and test results are in `VALIDATION.md`.

## Prior brain-specific controls (Phase 1.5)

Phase 1.5 brain-specific controls are complete. The
[scientific report](results/real_phase0/brain_specific_controls/scientific_summary.md)
and [budget curves](results/real_phase0/brain_specific_controls/selection_budget_curves.png)
show mean-model row-rank cross-subject correlation 0.7435, versus brain-label null
medians −0.0054 (independent) and 0.5113 (shared). Partial rank association remains
0.6927 after controlling for five model-only geometry scores. Budgets 10–60 do
not clearly exceed the shared-brain shortlist null; budget 100 shares 81/100.
**GO for prospective subjects 03–05 replication; row-rank and shortlist policy
remain exploratory.** No new data or model weights were downloaded.

Reproduce Phase 1.5 using the retained frozen protocol and verified local numeric
exports/caches. These commands do not deserialize object arrays or access a dataset
endpoint:

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib \
  python scripts/brain_specific_controls.py
python scripts/report_brain_specific.py
python -m pytest -q
NEUROCURATE_PHASE15_RESULTS=results/real_phase0/brain_specific_controls \
  python -m pytest tests/test_real_brain_specific_outputs.py -q
```

`configs/brain_specific_controls.yaml` fixes the seeds, 1,000 permutations per
control, model-only covariates, all five scoring methods and selection budgets
10/20/40/60/100. Independent and shared controls permute both Brain RDM axes and
recompute scores against fixed model geometry; score-label nulls are separate.
Every result remains conditional on official shared whitening. See `VALIDATION.md`
for exact evidence, null definitions, tests, and limits.

## Prior cross-subject milestone (Phase 1.4)

Phase 1.4 real subject-01/02 replication is complete. See the
[scientific report](results/real_phase0/cross_subject_01_02/scientific_summary.md),
[scoring ablation](results/real_phase0/cross_subject_01_02/scoring_ablation.csv),
and [ranking figure](results/real_phase0/cross_subject_01_02/ranking_stability.png).
Median conditional split-half Brain RDM Spearman is 0.7065/0.6466;
cross-subject Brain RDM Spearman is 0.5495. K=10 mean-model disagreement ranking
correlates at 0.4185 across subjects, with 6/20 shared shortlist images.
**GO for limited frozen replication on more subjects; current shortlists are
exploratory and do not justify training or OOD evaluation.** All EEG inference
is conditional on the official shared whitening, not session-independent.

Use the existing environment and checkout for Phase 1.4:

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
# These two one-time steps already succeeded. The exporter refuses overwrites.
# Acquisition is narrowly pinned to the standard subject-02 archive only.
python scripts/acquire_subject02.py
python scripts/export_isolated_real.py --subject sub02
# Reproduce analyses from the verified pickle-free exports; no downloads.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib \
  python scripts/cross_subject_analysis.py
python scripts/report_cross_subject.py
python -m pytest -q
NEUROCURATE_PHASE14_RESULTS=results/real_phase0/cross_subject_01_02 \
NEUROCURATE_PHASE13_RESULTS=results/real_phase0/sub01 \
  python -m pytest tests/test_real_cross_subject_outputs.py \
    tests/test_real_scientific_outputs.py -q
```

The protocol was frozen in `configs/cross_subject_01_02.yaml` before acquisition.
All five scoring methods, both models, and their equal-weight mean are reported;
no metric was selected from subject-02 outcomes. Both subjects reuse the original
200-image embeddings and exactly the original EEG parameters. The source archive,
test-only NPY, trusted OSF metadata, isolated export, original subject-01 output
hashes, and analysis provenance are preserved. Optional real-data checks access
local numeric exports only; standard tests remain network independent.
Results and local data are git-ignored and retained in this workspace.
Exact acquisition sizes, hashes, storage checks, test results, tie-aware controls,
and inference limitations are documented in `VALIDATION.md`.

## Prior subject-01 milestone (Phase 1.3)

Phase 1.3 scientific results are available in
[`results/real_phase0/sub01/scientific_summary.md`](results/real_phase0/sub01/scientific_summary.md):
valid 200-image Brain RDM; conditional repetition-split Spearman 0.7065,
permutation p=0.001; moderate disagreement-rank stability. Subject-02 replication
is recommended before relying on a shortlist. All results use real subject-01
EEG and real images. No new subject was downloaded.

The current workflow confines the approved pickle exception to a descriptor-only,
unprivileged seccomp child, then uses pickle-free arrays exclusively:

```bash
# One-time export of ONLY the two explicitly approved, hash-pinned NPY files.
# Refuses an existing pickle_free directory; this export is already present.
python scripts/export_isolated_real.py
# Reuses the verified numeric export and cached 200-image vision embeddings.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib \
  python scripts/scientific_subject01.py
NEUROCURATE_PHASE13_RESULTS=results/real_phase0/sub01 \
  python -m pytest tests/test_real_scientific_outputs.py -q
```

The analysis script replaces only its named generated scientific outputs; it
preserves the acquired inputs and numeric export. Namespace isolation is unavailable
on this host, so the export uses a kernel syscall allowlist that denies all new
filesystem opens, network sockets and process creation. The child holds only the
two readonly inputs, designated output descriptors and stdio, with 4 GiB memory,
120-second CPU and 512 MiB/file limits. See `pickle_free/isolation_report.json` and
`VALIDATION.md`. For Phase 1.3, use this workflow rather than the earlier direct
pickle-loading smoke command shown below as historical reproduction.

Phase 1.1: official subject-01 ZIP Range inspection found the test EEG member at
217,601,432 uncompressed bytes (208,634,917 compressed). Metadata-only inspection
and storage estimates are in [VALIDATION.md](VALIDATION.md#phase-11--metadata-only-remote-zip-inspection).
No real EEG payload was acquired or processed during Phase 1.1. Reproduce the bounded inspection:

```bash
source /workspace/setup/neurocurate/venv/bin/activate
cd /workspace/neurocurate/neurocurate
python scripts/inspect_remote_eeg_zip.py --output outputs/range-audit/recheck.json
```

The inspector reads ZIP metadata only, refuses an existing output, and stops on
unsupported ZIP layouts or ignored Range requests. It does not extract EEG or
verify the complete archive SHA256. The existing low-disk processor still requires
a separately verified test-only export; the inspector does not unblock acquisition.

Phase 1.2 acquisition (run only within the authorized one-subject acquisition
scope) uses exact official sizes/SHA256, preserves archives, and extracts only
test EEG and 200 original test images:

```bash
python scripts/acquire_real_phase0.py --dry-run
python scripts/acquire_real_phase0.py
```

Defaults place retained inputs under `data/local/real_phase0/` and reports under
`results/real_phase0/sub01/`. The script refuses to overwrite an acquisition
report. For a new report reusing existing verified inputs, select a new
`--output-dir`. Failed partials are retained for inspection. NPY headers are
inspected without pickle; object-format EEG/metadata requires explicit approval
before deserialization. Acquisition alone does not establish image alignment,
EEG reliability or a valid Brain RDM. See the Phase 1.2 entry in `VALIDATION.md`.
The Phase 1.2 run verified all three official downloads and extracted test EEG
plus 200 images. After explicit approval for the two verified NPY files, Phase 1
validated all 200 image mappings, real Brain RDMs and matching DINOv2/CLIP subset
embeddings. Evidence is in `results/phase1/real_data_audit.json` and `VALIDATION.md`.

The real smoke workflow reuses these inputs and pinned local model weights:

```bash
# Header-only integrity audit: writes reports; expected exit 2 when pickle is blocked.
python scripts/audit_real_data.py
# Only after explicit approval for the verified official NPY files:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/smoke_real_data.py --config configs/real_subset.yaml --trust-official-pickle
# Optional real-file tests, with no network access:
NEUROCURATE_REAL_ACQUISITION_MANIFEST=results/real_phase0/sub01/acquisition_manifest.json \
NEUROCURATE_REAL_PICKLE_APPROVED=1 python -m pytest tests/test_real_local_inputs.py -q
```

The flags record the required trust choice; use them only for explicitly approved,
checksum-verified inputs. Reports and smoke directories refuse accidental overwrite.
For a new run, create a fresh audit with `--output NEW_AUDIT.json`, pass it to the
smoke script with `--audit NEW_AUDIT.json`, and use a config with a new `output_dir`.
The default smoke config selects original EEG indices 0–15, computes a full
200-image Brain RDM plus the 16-image subset, and extracts vision features only
for that subset. The repeated-image `test` split remains `sanity`, never training.

Initial research pipeline for image-level disagreement between EEG representations
and frozen DINOv2/CLIP representations. This milestone produces subject brain RDMs,
vision RDMs, Top-K neighborhoods, cross-subject consistency, scores, and figures.
It does not train a model or assume THINGS-EEG2 concepts are ImageNet classes.

## Installation and validation

Use the existing checkout and Python 3.12 venv. Conda is not supported at this
milestone; `environment.yml` documents that choice. Do not create a worktree.

```bash
cd /workspace/neurocurate/neurocurate
source /workspace/setup/neurocurate/venv/bin/activate
export MPLCONFIGDIR=/workspace/setup/neurocurate/matplotlib
export UV_CACHE_DIR=/workspace/setup/uv-cache
# Keep torch and torchaudio on matching CPU builds in this cloud environment.
uv pip install --python "$VIRTUAL_ENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.11.0+cpu' 'torchaudio==2.11.0+cpu'
uv pip install --python "$VIRTUAL_ENV/bin/python" -e '.[dev,data]'
uv pip check --python "$VIRTUAL_ENV/bin/python"
python -m pytest
```

Project dependencies and supported ranges are in `pyproject.toml`. CPU is the
default. A separate GPU environment needs compatible PyTorch builds. Dependency
ranges are not an exact lockfile: each run records library versions and model
provenance. Pin model revisions and retain an environment freeze for publication.
The optional `data` extra enables the EEGDash metadata inspection script.

## Offline execution

```bash
python scripts/make_synthetic.py outputs/synthetic-demo
neurocurate outputs/synthetic-demo/experiment.yaml
# Equivalent: python scripts/run_experiment.py CONFIG.yaml
```

The generator creates 12 images, three subjects, four trials per image, and
**random synthetic vision features**. It downloads nothing. Use a new directory
for every run: the generator and pipeline refuse to overwrite previous output.
The `COMPLETE` marker is written only after all analysis and figures succeed.

Outputs include `scores.csv`, ordered `image_ids.npy`, subject and model feature
caches, one RDM per subject/model, `neighbors.npz`, `rdms.png`,
`top_disagreement.png`, `config.yaml`, and `provenance.json`. RDM row order follows
`image_ids.npy`; the score CSV is independently ranked. Neighbor indices refer to
that RDM order. Figures display each subject separately; no pooled brain RDM
conceals cross-subject variability. Random-feature runs do not validate pretrained
vision representations or real EEG science.

## THINGS-EEG2 metadata and EEG inputs

All experimental inputs must already be local. The analysis never downloads EEG.
See `data/README.md` for metadata-only remote inspection. Inspect selected files
and required companion files before deliberately obtaining a subset. The full
THINGS-EEG2 dataset must not be downloaded for this milestone. EEGDash can cache
whole recordings; a cache directory does not impose a storage cap.

The stimulus manifest is a CSV, with one row per unique image:

```csv
image_id,image_path,split,eeg_index
test_images/example/image-a.jpg,test_images/example/image-a.jpg,sanity,0
test_images/example/image-b.jpg,test_images/example/image-b.jpg,sanity,1
training_images/example/image-c.jpg,training_images/example/image-c.jpg,train,0
```

This excerpt documents the schema, not a runnable experiment (at least three
images and two subjects are required). `image_id` is an opaque string identifying
the exact image, not a class label. Paths resolve relative to the manifest, or
relative to the explicitly configured `image_root`. YAML paths resolve relative
to the YAML file. Missing metadata, duplicate IDs, shared resolved image paths,
and unknown split names are errors. Keep one complete manifest of all splits so
cross-split aliases can be detected. Content identity must be audited when images
are copied under distinct paths; path validation cannot discover identical copies.

Supported EEG formats:

1. **Canonical safe NPZ**: `eeg` has shape `(trials, channels, time)`;
   `image_ids` is a Unicode string per trial (repetitions repeat IDs);
   `times` is a strictly increasing vector in seconds relative to onset;
   `channel_names` is a Unicode channel vector; `split` is a scalar string.
   Arrays load with `allow_pickle=False`. Trial IDs must exactly match all image
   IDs in the selected manifest split, before any `max_images` selection.
2. **Official preprocessed THINGS-EEG2 NPY**: a dictionary containing
   `preprocessed_eeg_data` shaped `(images, repetitions, channels, time)`, `times`,
   and `ch_names`. An explicit per-split `eeg_index` mapping is mandatory. Indices
   must be contiguous from zero and match the image axis. These files use pickle;
   set `trusted_pickle: true` only for a trusted, integrity-checked local source.
   File names alone do not prove which split an array contains: the person
   importing must verify the source file and mapping.
3. **MNE epochs FIF**: already epoched `.fif`/`.fif.gz` plus `trial_metadata`
   pointing to a CSV with `image_id,split`, exactly one row per epoch in file
   order. Mixed splits are rejected. Continuous NEMAR BrainVision recordings
   require a separate event-to-image mapping and epoch preparation; they cannot
   be passed directly as preprocessed arrays. This milestone does not invent
   mappings from annotations or numeric event codes.

Convert trusted official `image_metadata.npy` to an explicit manifest:

```bash
python scripts/import_things_metadata.py \
  data/local/image_metadata.npy data/local/images data/local/images.csv \
  --trust-pickle
```

The converter uses `test_img_files`/`train_img_files` and, for bare filenames,
one corresponding `test_img_concepts`/`train_img_concepts` folder per image. It
keeps the original array index, maps official **test to sanity** and official
**train to train**, and emits no class labels. Metadata with another layout must
be converted explicitly; no guessed ordering is accepted. Verify paths against
the actual stimulus layout before extraction.

## Configuration and real execution

Copy `configs/things_sanity.yaml` and set the local manifest, EEG paths, valid
channel names, time coverage, and output directory. At least two independent
subject files are required. Inputs must include the same selected image identities
for every subject; this implementation rejects missing IDs instead of silently
intersecting subject image sets.

Preprocessing is configurable: per-trial baseline subtraction, selected channels,
poststimulus time window, repetition count, temporal bin count, and optional
within-image feature standardization. Windows use `[start, stop)` seconds.
The full requested interval must lie inside the recorded epoch. All repetitions
are averaged independently per image/subject; channel/time-bin means are flattened.
Unequal repetition counts are recorded in provenance, not concealed.

RDM metrics are `correlation`, `cosine`, or `euclidean`. Correlation requires
nonconstant vectors; cosine requires nonzero vectors. Brain defaults to
correlation, vision to cosine. Invalid or nonfinite features are rejected.

For pretrained extraction, provide stimulus files and local model checkpoints,
or intentionally allow model downloads by setting `local_files_only: false`.
Network access to `huggingface.co` and the actual resolved artifact hosts is
required for downloads. The example stays offline by default. Pin `revision` to
a full Hub commit for reproducibility; the example pins the verified revisions.

```bash
neurocurate configs/things_sanity.yaml
```

DINOv2 uses the last-layer CLS embedding from `facebook/dinov2-base`; CLIP uses
the projected image embedding from `openai/clip-vit-base-patch32`. Each uses its
official image processor, RGB conversion, frozen parameters, eval mode, inference
mode, and optional L2 normalization. There are no text prompts or class labels.
Extraction batches selected images and records checkpoint, revision, resolved
commit where available, pooling, normalization, and SHA256 of stimulus bytes.

`configs/cached_features.yaml` instead loads NPZ features written by
`neurocurate.vision.save_feature_cache`. A cache must contain string `image_ids`,
finite `(images, features)` data, scalar `split`, scalar `model_name`, and JSON
`provenance`. Split/model mismatches fail; rows are aligned by image ID. A cache
is user-supplied research input, so verify its provenance and image identity;
unlabeled arbitrary arrays are not accepted. Synthetic sources are explicitly
marked. Cached mode downloads no model weights. When stimulus hashes are
available, local image files are checked and stale caches are rejected. Missing
files or absent hashes are recorded in provenance as unverified.

`max_images` selects the first IDs in lexical order, deterministically. It bounds
vision extraction and the analyzed subset, but loading a local EEG array still
reads that array. RDMs require quadratic memory in the number of images; start
with the 200-image sanity set or a smaller deliberate selection. Top-K results
depend on the selected stimulus set and K and should be compared under the same
configuration.

## Score definition and split safety

Each image excludes itself from its K nearest neighbors; ties break by lexical
image ID. For neighbor sets A and B, disagreement is `1 - |A∩B|/|A∪B|`.
Per-image consistency is the mean pairwise Jaccard similarity between subjects.
Model disagreement averages brain/model disagreement across subjects, then
averages models using configured nonnegative weights. The initial NeuroCurate
score is `consistency × model_disagreement`, in `[0,1]`. Higher scores identify
consistent brain neighborhoods that disagree with the configured models. The
CSV preserves consistency and each model's disagreement alongside the score.
This is an explicit heuristic, not a calibrated probability or validated quality
metric; no noise ceiling correction or crossvalidated distance is claimed.

Allowed splits are `sanity`, `train`, and `evaluation`. Each execution uses exactly
one split; EEG and feature-cache split metadata must match. The repeated official
test images are reserved for sanity analysis. Do not reuse sanity images for
training, evaluation, model choice, or hyperparameter selection in a later study.
No normalization/statistics are fitted across stimuli, no model is trained, and
there is no cross-split ranking. Any future fitting code must call
`assert_disjoint` and establish an independent evaluation manifest. The software
cannot prevent a user from falsifying split declarations, creating duplicate
image copies, or manually tuning settings on held-out outcomes.

## Verification boundaries

Pytest exercises averaging and feature shapes, exact-ID reordering, official
synthetic NPY mapping, safe NPZ loading, synthetic MNE FIF loading, RDM symmetry
and diagonals, ties and self-exclusion, Jaccard values, subject consistency,
weighted ranking, missing metadata, invalid inputs, split leakage, and end-to-end
figure generation. Vision tests cover frozen/inference behavior with a fake
backend and tiny **randomly initialized** local DINOv2 and CLIP checkpoints using
the actual Transformers architectures. They never fetch pretrained weights.

During initial implementation, the real nm000232 test-record metadata request
succeeded, and both actual pretrained vision checkpoints produced embeddings
for synthetic image files. A complete synthetic EEG experiment also succeeded
using those pretrained embeddings. The example config pins the verified model
revisions and uses their retained cloud cache.

Real EEG validation still requires trusted local EEG epochs/preprocessed arrays,
matching stimulus metadata, and real stimulus files. Metadata endpoint
reachability does not validate EEG loading or scientific results. No real EEG
recording bytes were downloaded. See `VALIDATION.md` for exact evidence and
remaining blockers. Full-scale training, OOD benchmarking, diffusion, and online
active learning are outside this milestone.

## Low-disk Phase 0 workflow

The source audit is in `data/lowdisk_source_audit.json`; exact commands and
outcomes are recorded in `VALIDATION.md`. The complete official OSF preprocessed
inventory exposes subject ZIPs, not separately listed test arrays. This task
therefore stopped acquisition under the requested constraint: no EEG or stimulus
payloads were fetched. The 200-image Brain RDM workflow below is verified on
synthetic inputs; real EEG remains unverified.

Inspect the blocked real-data plan without making any network requests:

```bash
python scripts/process_subjects_lowdisk.py configs/lowdisk_osf_audit.yaml --dry-run
```

Exit code **2** means the plan is blocked. It prints available disk bytes,
required direct files, known official archive alternatives, unknown sizes, output
budgets, reserve, and blockers. An unknown EEG size gives a null peak estimate,
never a speculative “fits” decision. This command creates no directories/files.
Archive inventory entries are documentation, not permission to acquire archives.

Verify the sequential workflow offline in a new directory:

```bash
python scripts/make_synthetic.py outputs/lowdisk-synthetic --images 200
python scripts/process_subjects_lowdisk.py outputs/lowdisk-synthetic/lowdisk.yaml --dry-run
python scripts/process_subjects_lowdisk.py outputs/lowdisk-synthetic/lowdisk.yaml --cleanup
```

This uses 200 generated images, three synthetic subject files, and random vision
caches. Every subject produces `features.npz`, `brain_rdm.npy` (200×200),
`image_ids.npy`, `config.yaml`, `provenance.json`, and `VERIFIED`. At the run root,
`manifest.csv`, `stimulus_hashes.json`, aligned vision caches, the storage plan,
and `COMPLETE` are retained. Features and RDMs are reloaded without pickle; hashes,
image order, finite values, symmetry/diagonal, and a recomputed RDM are checked
before cleanup. These generated inputs already existed before processing, so
`--cleanup` correctly preserves all of them.

For actual data, first stage one trusted **test-only** preprocessed array, a
complete exact-ID manifest with the 200 sanity images and `eeg_index`, and those
200 local stimulus files. Copy `configs/lowdisk_local.yaml`, set paths and actual
epoch coverage/channels, then run dry-run followed by processing. One subject is
allowed for a brain-only smoke test; cross-subject scoring still requires at least
two independent subjects. No training/evaluation split is processed by this tool.

Compute real-stimulus vision features once using the existing cached checkpoints:

```bash
python scripts/extract_sanity_vision.py data/local/sanity_images.csv data/local/vision
```

This uses pinned pretrained DINOv2/CLIP revisions with `local_files_only=True`;
it never downloads model weights. Set `vision_features` in the sequential config
to the two saved NPZ paths. Those features are checked/aligned once and referenced
by each subject; no model inference occurs inside the subject loop.

For an individually accessible test-only file, the EEG artifact declaration is:

```yaml
eeg:
  path: ../data/local/owned_downloads/sub-01/preprocessed_eeg_test.npy
  url: https://YOUR-AUDITED-HOST/preprocessed_eeg_test.npy
  bytes: 123456  # Replace with the EXACT independently audited size.
  sha256: REPLACE_WITH_THE_PUBLISHED_OR_EXPORT_VERIFIED_SHA256
  temporary: true
```

The example above is schema documentation, not a usable download declaration.
Missing files require an exact size, SHA256, HTTPS source, and destination inside
`work_dir`. Required direct stimulus files may be declared in `shared_files` with
the same fields. A local manifest must already be available to audit identity and
stimulus requirements. Archive acquisition/extraction is unsupported and blocked.
Files stream to a unique partial path with strict byte/checksum limits and HTTPS
redirect validation; TLS verification remains enabled. Failed partials stay for
inspection, and unknown prerequisites prevent **all** payload downloads.

Storage planning includes shared downloads, all retained subject downloads, the
largest explicitly temporary subject download when cleanup is enabled, a bounded
output budget for every subject plus shared outputs, and a default 1 GiB reserve.
Available space is rechecked before each subject download. Existing files are
already reflected in filesystem free space. This is a disk plan, not a RAM cap:
one local subject is loaded at a time and EEG preprocessing can allocate copies.

Cleanup is opt-in and only removes an EEG file created by **this execution**, with
`temporary: true`, a reproducible HTTPS source and matching SHA256, after verified
outputs. Local/pre-existing inputs, shared stimuli, feature caches, checkpoints,
failed partials, and failed-verification inputs are preserved. No glob or recursive
cache deletion occurs. New run directories prevent accidental output overwrite.

The recommended next step is a test-only export from an existing verified
single-subject archive on a data-holding machine. See `VALIDATION.md` for the
concrete transfer requirements and a bounded alternative to full-dataset fetching.
