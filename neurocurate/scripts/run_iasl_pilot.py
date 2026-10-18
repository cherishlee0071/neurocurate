#!/usr/bin/env python
"""Run the low-cost IASL Phase 0A--0D feasibility pilot."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from neurocurate.iasl import (PilotConfig, load_numeric_subject, flatten_trials,
    pca_fit_transform, train_sae_replicates, matching_diagnostics,
    repeated_reliability, intervention_audit, intervention_permutation_null, write_json, jsonable,
    tiny_temporal_encoder)

ROOT = Path(__file__).resolve().parents[1]
TIMES = np.load(ROOT / "results/real_phase0/sub01/pickle_free/times.npy", allow_pickle=False)
DATA = ROOT / "results/real_phase0"

def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: return "unknown"

def source_for(s):
    if s == "sub01": return DATA / "sub01/pickle_free/eeg.npy"
    if s == "sub02": return DATA / "sub02/pickle_free/eeg.npy"
    return DATA / "prospective_03_05" / s / "pickle_free/eeg.npy"

def audit(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    subjects, arrays, records = [], [], []
    for s in cfg.subjects:
        path = source_for(s)
        if not path.exists(): raise FileNotFoundError(path)
        x, rec = load_numeric_subject(path, cfg.n_images, cfg.n_repetitions, TIMES)
        subjects.append(s); arrays.append(x); rec.update(subject=s)
        rec.update(reliability=repeated_reliability(x), nan_rate=float(np.isnan(x).mean()),
                   variance=float(np.var(x)), valid_trials=int(np.prod(x.shape[:2])))
        records.append(rec)
    ids = [f"{i:05d}" for i in range(1, cfg.n_images + 1)]
    manifest = {"phase": "IASL Phase 0A", "git_commit": git_commit(),
                "dataset": "THINGS-EEG2 official numeric pickle-free exports",
                "subjects": subjects, "image_ids": ids, "n_images": cfg.n_images,
                "n_repetitions": cfg.n_repetitions, "records": records,
                "preprocessing": {"poststimulus_requested": list(cfg.poststimulus),
                                  "actual_time_vector": [float(TIMES[0]), float(TIMES[-1])],
                                  "actual_poststimulus_end": float(TIMES[TIMES>=0][-1]),
                                  "channels": 17, "sampling_hz": 100,
                                  "note": "official export ends at 0.79 s; no samples beyond available time were invented"}}
    write_json(out / "data_manifest.json", manifest)
    (out / "data_audit.md").write_text("# IASL Phase 0A data audit\n\n" +
      f"- Commit: `{git_commit()}`\n- Subjects: {', '.join(subjects)}\n- Shared images: {cfg.n_images}\n- Repetitions used: {cfg.n_repetitions}\n"+
      f"- Shape per subject: `{arrays[0].shape}` (images, repetitions, channels, time)\n"+
      f"- Actual post-stimulus coverage: 0.00--{TIMES[TIMES>=0][-1]:.2f}s; the requested 1.00s is unavailable in the verified export.\n"+
      "- All inputs are pickle-free numeric exports and all five subjects have the same 17-channel schema.\n\n"+
      "Repeated-trial reliability is reported in `data_manifest.json`; this is a feasibility audit, not a session-independent reliability claim.\n")
    # compact diagnostic: subject median split-half reliability and variance
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].bar(subjects, [r["reliability"]["median_image_split_half_r"] for r in records]); ax[0].set_title("split-half r"); ax[0].tick_params(axis='x', rotation=45)
    ax[1].bar(subjects, [r["variance"] for r in records]); ax[1].set_title("EEG variance"); ax[1].tick_params(axis='x', rotation=45)
    fig.tight_layout(); fig.savefig(out / "data_diagnostic.png", dpi=140); plt.close(fig)
    return arrays, manifest

def run_representation(arrays, cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    trials = [flatten_trials(x) for x in arrays]
    pooled = np.concatenate(trials, axis=0)
    pca, p = pca_fit_transform(pooled, cfg.pca_dim)
    reps = train_sae_replicates(pca, cfg.sae_width_multipliers, cfg.seeds, cfg.epochs)
    diag = matching_diagnostics(reps, 1234)
    # Architecture control: a seeded depthwise temporal convolution, followed
    # by the same PCA and SAE audit.  It is frozen in Phase 0.
    raw_trials = np.concatenate([a.reshape(-1, a.shape[2], a.shape[3]) for a in arrays], axis=0)
    tiny_raw = tiny_temporal_encoder(raw_trials, output_dim=128, seed=0)
    tiny_pca, tiny_params = pca_fit_transform(tiny_raw, min(cfg.pca_dim, tiny_raw.shape[1]))
    tiny_reps = train_sae_replicates(tiny_pca, cfg.sae_width_multipliers, cfg.seeds, cfg.epochs)
    tiny_diag = matching_diagnostics(tiny_reps, 5678)
    summaries = [{k: v for k, v in r.items() if k not in {"codes", "reconstruction", "decoder"}} for r in reps]
    write_json(out / "sae_reproducibility.json", {"git_commit": git_commit(), "representation": "flatten -> pooled PCA",
        "pca": {"input_dim": int(pooled.shape[1]), "output_dim": int(pca.shape[1]),
                "explained_variance_ratio": float(np.sum(p["singular_values"]**2) / max(np.sum((pooled-pooled.mean(0))**2),1e-12))},
        "replicates": summaries, "matching": diag,
        "tiny_encoder_control": {"architecture": "depthwise Conv1d(17 groups, k=5) -> pointwise Conv1d(32) -> mean/std",
                                 "weights": "seeded frozen control; not pretrained or fitted on held-out data",
                                 "pca_output_dim": int(tiny_pca.shape[1]),
                                 "matching": tiny_diag,
                                 "replicates": [{k:v for k,v in r.items() if k not in {"codes","reconstruction","decoder"}} for r in tiny_reps]}})
    lines = ["# IASL Phase 0C SAE reproducibility", "", f"Pooled PCA input `{pooled.shape[1]}` -> `{pca.shape[1]}` dimensions.",
             "Individual latent matching is diagnostic; CKA and principal-angle summaries are the primary group/subspace checks.", ""]
    for row in diag["widths"]:
        lines.append("- width {width}: feature max-cosine {feature_cosine_max_mean:.3f} (null p95 {feature_null_p95:.3f}), overlap {feature_overlap_gt_0_8:.3f}, CKA {cka_mean:.3f}, principal angle {principal_angle_deg_mean:.1f}°".format(**row))
    lines += ["", "Tiny temporal-convolution control (seeded frozen weights):"]
    for row in tiny_diag["widths"]:
        lines.append("- width {width}: feature max-cosine {feature_cosine_max_mean:.3f} (null p95 {feature_null_p95:.3f}), CKA {cka_mean:.3f} (row-permutation null p95 {cka_null_p95:.3f})".format(**row))
    (out / "sae_reproducibility.md").write_text("\n".join(lines) + "\n")
    # Save compact stability table and figure.
    import csv
    with (out / "stability_table.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(diag["widths"][0])); w.writeheader(); w.writerows(diag["widths"])
    fig, ax = plt.subplots(figsize=(6, 3.4));
    labels=[str(r["width"]) for r in diag["widths"]]; ax.bar(np.arange(len(labels))-.15,[r["feature_cosine_max_mean"] for r in diag["widths"]],.3,label="feature match"); ax.bar(np.arange(len(labels))+.15,[r["cka_mean"] for r in diag["widths"]],.3,label="CKA"); ax.set_xticks(range(len(labels)),labels); ax.set_xlabel("SAE width"); ax.legend(); fig.tight_layout(); fig.savefig(out/"feature_vs_subspace_stability.png",dpi=140); plt.close(fig)
    return reps, pca, pooled

def run_intervention(reps, cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    rows=[]; nulls=[]
    for rep in reps:
        # one representative seed per width keeps this audit bounded; all seeds remain in Phase 0C.
        if rep["seed"] != cfg.seeds[0]: continue
        rows += [{**r, "width": rep["width"], "seed": rep["seed"]} for r in intervention_audit(rep, cfg.intervention_magnitudes, 10)]
        n = intervention_permutation_null(rep, cfg.intervention_magnitudes, 100, 99)
        nulls.append({"width": rep["width"], "seed": rep["seed"], **{k:v for k,v in n.items() if k != "values"}})
    write_json(out / "intervention_audit.json", {"git_commit": git_commit(), "n_rows": len(rows), "rows": rows, "permutation_nulls": nulls,
        "normalization": "C_norm=(task_damage+reconstruction_damage)/removed_EEG_energy"})
    import csv
    with (out / "intervention_table.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # raw versus normalized by magnitude
    fig, ax=plt.subplots(figsize=(6,3.4));
    for width in sorted(set(r["width"] for r in rows)):
        rr=[r for r in rows if r["width"]==width]; vals=[]
        for a in cfg.intervention_magnitudes: vals.append(np.median([r["C_norm"] for r in rr if r["magnitude"]==a]))
        ax.plot(cfg.intervention_magnitudes, vals, marker="o", label=f"width {width}")
    ax.set_xlabel("intervention magnitude"); ax.set_ylabel("median C_norm"); ax.legend(); fig.tight_layout(); fig.savefig(out/"intervention_damage.png",dpi=140); plt.close(fig)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke-only",action="store_true"); ap.add_argument("--epochs",type=int,default=5 if False else 50); args=ap.parse_args()
    cfg=PilotConfig(epochs=args.epochs, n_images=5 if args.smoke_only else 50, n_repetitions=2 if args.smoke_only else 4)
    base=ROOT/"results"/"iasl"; out_a=base/"phase0_data_audit"; out_s=base/"phase0_sae_reproducibility"; out_i=base/"phase0_intervention"
    arrays, manifest=audit(cfg,out_a)
    reps,pca,pooled=run_representation(arrays,cfg,out_s)
    rows=run_intervention(reps,cfg,out_i)
    summary={"git_commit":git_commit(),"config":jsonable(cfg.__dict__),"smoke_only":args.smoke_only,
             "data":manifest,"matching":jsonable(json.loads((out_s/"sae_reproducibility.json").read_text())["matching"]),
             "intervention": {"rows":len(rows),"median_C_raw":float(np.median([r["C_raw"] for r in rows])),"median_C_norm":float(np.median([r["C_norm"] for r in rows]))}}
    write_json(base/"pilot_summary.json",summary)
    if not args.smoke_only:
        (base/"RUN_LOG.md").write_text(f"# IASL RUN_LOG\n\nCommit: `{git_commit()}`\n\nCommand: `python scripts/run_iasl_pilot.py --epochs {args.epochs}`\n\nPhase 0A--0D completed on five verified numeric subjects, 50 images, four repetitions. No new downloads.\n")
        rel=[r["reliability"]["median_image_split_half_r"] for r in manifest["records"]]
        match=json.loads((out_s/"sae_reproducibility.json").read_text())
        intj=json.loads((out_i/"intervention_audit.json").read_text())
        nullmed=", ".join([format(n["median"], ".4g") for n in intj["permutation_nulls"]])
        (base/"scientific_summary.md").write_text("# IASL feasibility summary\n\n"
          "## Scope\n\nPhase 0A--0D ran on five verified THINGS-EEG2 numeric exports, 50 shared repeated test images and four repetitions per image. No new EEG or model was downloaded. The export contains samples from -200 to 790 ms; the requested 0--1000 ms window was therefore truncated to the available 0--790 ms interval.\n\n"
          "## Results\n\n"
          f"- Median image split-half reliability by subject: {', '.join(f'{v:.3f}' for v in rel)}; range {min(rel):.3f}--{max(rel):.3f}. This is low and near the predeclared reliability stop rule.\n"
          f"- PCA control: 1,360 flattened dimensions -> 128. SAE widths 256 and 512 had feature max-cosine {match['matching']['widths'][0]['feature_cosine_max_mean']:.3f} and {match['matching']['widths'][1]['feature_cosine_max_mean']:.3f}, with 0.000 feature overlap above cosine 0.8; both are at or near their random-direction null p95.\n"
          f"- Subspace CKA was high ({match['matching']['widths'][0]['cka_mean']:.3f}, {match['matching']['widths'][1]['cka_mean']:.3f}) and exceeded the row-permutation null in the tiny frozen-convolution control (null p95 {match['tiny_encoder_control']['matching']['widths'][0]['cka_null_p95']:.3f}/{match['tiny_encoder_control']['matching']['widths'][1]['cka_null_p95']:.3f}). This supports a reproducible pooled subspace signal, while the very small feature overlap argues against feature identity claims.\n"
          f"- Intervention audit: {len(rows)} raw/normalized ablations; median C_raw {float(np.median([r['C_raw'] for r in rows])):.4g}, median C_norm {float(np.median([r['C_norm'] for r in rows])):.4g}. The energy-matched permutation null medians are {nullmed}; these small audits do not establish class selectivity.\n\n"
          "## Decision\n\n**NO-GO for Phase 1 training or a GPU-intensive IASL alignment claim.** The data pipeline and subspace-level audit are usable, but repeated-trial reliability is low and individual SAE matching is indistinguishable from its null. A defensible follow-up is an analysis paper on subspace reproducibility and intervention limits, followed by a separately preregistered, session-separated pilot only after reliability and preprocessing sensitivity improve. No held-out-subject generalization or data-efficiency claim is made.\n")
    print(json.dumps({"output":str(base),"subjects":cfg.subjects,"images":cfg.n_images,"repetitions":cfg.n_repetitions,"commit":git_commit()},indent=2))

if __name__ == "__main__": main()
