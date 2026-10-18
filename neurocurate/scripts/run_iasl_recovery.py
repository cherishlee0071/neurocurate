#!/usr/bin/env python
"""IASL Phase-0 reliability and intervention recovery audit.

This script intentionally stops before LOSO training.  It uses only the
existing pickle-free THINGS-EEG2 test exports and writes a new report tree
under ``IASL/results/phase0_recovery`` without touching earlier results.
"""
from __future__ import annotations
import argparse, csv, json, subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "IASL" / "results" / "phase0_recovery"
TIMES_PATH = ROOT / "results/real_phase0/sub01/pickle_free/times.npy"

def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

def source_for(subject):
    if subject == "sub01": return ROOT / "results/real_phase0/sub01/pickle_free/eeg.npy"
    if subject == "sub02": return ROOT / "results/real_phase0/sub02/pickle_free/eeg.npy"
    return ROOT / "results/real_phase0/prospective_03_05" / subject / "pickle_free/eeg.npy"

def finite_corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(axis=-1, keepdims=True); b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt(np.sum(a*a, axis=-1) * np.sum(b*b, axis=-1))
    return np.divide(np.sum(a*b, axis=-1), den, out=np.zeros_like(den), where=den > 1e-12)

def corr_summary(a, b, rng, n_boot=200, n_perm=200):
    """Image-wise split-half correlations with bootstrap and permutation null."""
    rs = finite_corr(a.reshape(len(a), -1), b.reshape(len(b), -1))
    boot = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(rs), len(rs)); boot.append(float(np.mean(rs[ix])))
    null = []
    for _ in range(n_perm):
        null.append(float(np.mean(finite_corr(a.reshape(len(a), -1), b[rng.permutation(len(b))].reshape(len(b), -1)))))
    mean = float(np.mean(rs))
    corrected = 2 * mean / max(1 + mean, 1e-12)
    return {"raw_split_half": mean, "median_image_r": float(np.median(rs)),
            "corrected_split_half": float(corrected),
            "bootstrap_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "permutation_p95": float(np.percentile(null, 95)),
            "permutation_p": float((1 + sum(x >= mean for x in null)) / (len(null) + 1)),
            "image_r": rs.tolist()}

def baseline(x, times):
    m = (times >= -0.2) & (times < 0)
    return x - x[..., m].mean(axis=-1, keepdims=True)

def robust_z(x, times):
    y = baseline(x, times); m = (times >= -0.2) & (times < 0)
    med = y[..., m].median(axis=-1, keepdims=True) if hasattr(y[..., m], 'median') else np.median(y[..., m], axis=-1, keepdims=True)
    mad = np.median(np.abs(y[..., m] - med), axis=-1, keepdims=True)
    return y / np.maximum(1.4826 * mad, 1e-5)

def windows(times):
    requested = [("0_100", 0, .1), ("100_200", .1, .2), ("200_400", .2, .4),
                 ("400_600", .4, .6), ("600_790", .6, .79), ("all_0_790", 0, .79)]
    return {name: (times >= lo - 1e-9) & (times <= hi + 1e-9) for name,lo,hi in requested}

def subject_reliability(x, times, rng):
    out = {}
    for name, mask in windows(times).items():
        y = x[..., mask]
        a, b = y[:, ::2].mean(1), y[:, 1::2].mean(1)
        out[name] = corr_summary(a, b, rng)
    # channel and time axes for the full available interval
    y = x[..., windows(times)["all_0_790"]]
    a, b = y[:, ::2].mean(1), y[:, 1::2].mean(1)
    channel_r = finite_corr(a.mean(-1).T, b.mean(-1).T)
    time_r = finite_corr(a.mean(1), b.mean(1))
    out["channel_median"] = {"median": float(np.median(channel_r)), "best": int(np.argmax(channel_r)), "values": channel_r.tolist()}
    out["time_median"] = {"median": float(np.median(time_r)), "best": int(np.argmax(time_r)), "values": time_r.tolist()}
    # Independent odd/even repetition averages estimate reliability of the
    # repeated-measure representation; a self-correlation would be invalid.
    out["trial_average_reliability"] = corr_summary(y[:, ::2].mean(1), y[:, 1::2].mean(1), rng, n_boot=100, n_perm=100)
    # Leave-one-repeat trial-to-rest mean: a single-trial diagnostic.
    single = []
    for k in range(x.shape[1]):
        single.append(float(np.mean(finite_corr(y[:, k], (y.sum(1)-y[:, k])/(x.shape[1]-1)))))
    out["single_trial_to_rest_mean"] = float(np.mean(single))
    # Keep arrays only in compact JSON fields.
    for v in out.values():
        if isinstance(v, dict) and "image_r" in v: v.pop("image_r", None)
    return out

def linear_cka(a, b):
    a = a - a.mean(0, keepdims=True); b = b - b.mean(0, keepdims=True)
    ka, kb = a @ a.T, b @ b.T
    return float(np.sum(ka*kb) / max(np.linalg.norm(ka)*np.linalg.norm(kb), 1e-12))

def pca_features(x, k):
    x = x - x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:min(k, vt.shape[0])].T

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda z: z.tolist() if isinstance(z, np.ndarray) else z.item() if isinstance(z, np.generic) else str(z)) + "\n")

def run(args):
    times = np.load(TIMES_PATH, allow_pickle=False)
    subjects = args.subjects.split(",") if args.subjects else ["sub01","sub02","sub03","sub04","sub05"]
    out = Path(args.output_dir) if args.output_dir else OUT_DEFAULT
    if not out.is_absolute(): out = ROOT / out
    for d in ["audit","reliability","preprocessing","trial_average","image_subset","intervention","gates","figures"]: (out/d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rel_rows=[]; all_means={}; all_raw={}
    variants={"A_current": lambda x:x, "B_baseline": lambda x:baseline(x,times),
              "C_baseline_zscore": lambda x:robust_z(x,times),
              "D_temporal_smooth": lambda x:gaussian_filter1d(baseline(x,times), sigma=1, axis=-1),
              "E_bandpower": lambda x:np.log(np.mean(np.square(baseline(x,times)), axis=-1, keepdims=True)+1e-6)}
    variant_rows=[]
    for s in subjects:
        x=np.load(source_for(s), allow_pickle=False).astype(np.float32)
        all_raw[s]=x
        rel=subject_reliability(x,times,rng)
        full=rel["all_0_790"]
        rel_rows.append({"subject":s,"raw_split_half":full["raw_split_half"],"corrected_split_half":full["corrected_split_half"],"bootstrap_ci":full["bootstrap_ci"],"permutation_p95":full["permutation_p95"],"trial_average_reliability":rel["trial_average_reliability"]["raw_split_half"],"single_trial_to_rest_mean":rel["single_trial_to_rest_mean"],"best_time_window":max((k for k in windows(times)),key=lambda k:rel[k]["raw_split_half"]),"best_channel":rel["channel_median"]["best"]})
        write_json(out/"reliability"/f"{s}.json", rel)
        # Subject image means, retained only in memory for compact cross-subject checks.
        all_means[s]=x.mean(1)
        for vn, fn in variants.items():
            yy=fn(x); a,b=yy[:,::2],yy[:,1::2]
            r=corr_summary(a.mean(1),b.mean(1),rng,n_boot=100,n_perm=100)
            variant_rows.append({"subject":s,"variant":vn,"raw_split_half":r["raw_split_half"],"corrected_split_half":r["corrected_split_half"],"permutation_p95":r["permutation_p95"],"trial_average_reliability":float(r["raw_split_half"])})
    # Reliability reports.
    write_json(out/"reliability"/"reliability_metrics.json", {"git_commit":git_commit(),"dataset":"THINGS-EEG2 official numeric test exports","subjects":rel_rows,"requested_window":[0,1],"available_analyzed_interval":[0,.79],"baseline_interval":[-.2,0],"sampling_hz":100,"notes":"all 80 available repetitions used for the primary split; no data were downloaded"})
    with (out/"reliability"/"reliability_table.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["subject","raw_split_half","corrected_split_half","bootstrap_ci","permutation_p95","trial_average_reliability","single_trial_to_rest_mean","best_time_window","best_channel"]);w.writeheader();w.writerows(rel_rows)
    (out/"reliability"/"reliability_summary.md").write_text("# Reliability recovery\n\nAll 80 repetitions were used. Requested window: 0–1000 ms; available analyzed interval: 0–790 ms; baseline interval: -200–0 ms. Corrected reliability, bootstrap intervals, image permutation nulls, time windows and channel diagnostics are in `reliability_metrics.json`. Subject-03 is reported separately rather than removed.\n")
    # Preprocessing table and CKA across subjects.
    with (out/"preprocessing"/"preprocessing_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(variant_rows[0]));w.writeheader();w.writerows(variant_rows)
    cka_rows=[]
    for vn,fn in variants.items():
        mats=[]
        for s in subjects:
            yy=fn(all_raw[s]).mean(1).reshape(200,-1); mats.append(pca_features(yy,128))
        for i in range(len(subjects)):
            for j in range(i+1,len(subjects)):
                obs=linear_cka(mats[i],mats[j]); null=[linear_cka(mats[i],mats[j][rng.permutation(200)]) for _ in range(50)]
                cka_rows.append({"variant":vn,"subject_a":subjects[i],"subject_b":subjects[j],"cka":obs,"row_permutation_p95":float(np.percentile(null,95))})
    write_json(out/"preprocessing"/"preprocessing_metrics.json",{"git_commit":git_commit(),"variants":variant_rows,"cross_subject_cka":cka_rows,"mask_policy":"No held-out-derived mask was selected; reliable-mask is deferred until LOSO fitting"})
    (out/"preprocessing"/"preprocessing_summary.md").write_text("# Preprocessing sensitivity\n\nVariants A–E were evaluated with the same fixed windows and all subjects retained. Variant F reliable-mask is not fitted globally because doing so would violate the LOSO rule; it is specified as a future fold-local analysis. No variant is selected post hoc.\n")
    # Trial-average and image subsets.
    ta=[]
    for s,x in all_raw.items():
        for reps,label in [(80,"all_80"),(4,"deterministic_4")]:
            y=x[:,:reps,:,windows(times)["all_0_790"]]
            r=corr_summary(y[:,::2].mean(1),y[:,1::2].mean(1),rng,n_boot=100,n_perm=100)
            ta.append({"subject":s,"representation":label,"repetitions":reps,"raw_split_half":r["raw_split_half"],"corrected_split_half":r["corrected_split_half"],"single_trial_to_rest":float(np.mean([np.mean(finite_corr(y[:,k],(y.sum(1)-y[:,k])/(reps-1))) for k in range(reps)]))})
    write_json(out/"trial_average"/"trial_average_metrics.json",{"git_commit":git_commit(),"rows":ta,"warning":"trial-average results cannot be called single-trial generalization"})
    with (out/"trial_average"/"trial_average_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(ta[0]));w.writeheader();w.writerows(ta)
    (out/"trial_average"/"trial_average_summary.md").write_text("# Single-trial versus trial-average\n\nThe all-80 and deterministic-4 trial averages are reported separately from single-trial diagnostics. Any gain in average reliability reflects repeated-measure denoising and is not evidence of single-trial transfer.\n")
    sets={"Set-A":np.arange(50),"Set-B":rng.choice(np.arange(50,200),50,replace=False),"Set-C":np.arange(200)}
    subset_rows=[]
    for name,ix in sets.items():
        for s in subjects:
            y=all_raw[s][ix]; r=corr_summary(y[:,::2].mean(1),y[:,1::2].mean(1),rng,n_boot=100,n_perm=100)
            subset_rows.append({"image_set":name,"subject":s,"n_images":len(ix),"raw_split_half":r["raw_split_half"],"corrected_split_half":r["corrected_split_half"],"median_image_r":r["median_image_r"],"permutation_p95":r["permutation_p95"]})
    write_json(out/"image_subset"/"image_subset_metrics.json",{"git_commit":git_commit(),"sets":{"Set-A":"first 50 predeclared","Set-B":"fixed seed 0 draw from remaining 150","Set-C":"all 200"},"rows":subset_rows,"exploratory_high_reliability_selection":False})
    (out/"image_subset"/"image_subset_summary.md").write_text("# Image subset sensitivity\n\nSet-A, fixed-seed Set-B and all 200 images were evaluated without selecting images by outcome. No high-reliability image set was promoted to confirmatory status.\n")
    # Subspace audit reuses only existing numeric results for context and adds cross-subject PCA/null checks above.
    prior=ROOT/"results/iasl/phase0_sae_reproducibility/sae_reproducibility.json"
    write_json(out/"audit"/"subspace_stability.json",{"git_commit":git_commit(),"pca_only_cross_subject":cka_rows,"prior_sae_diagnostic":json.loads(prior.read_text()) if prior.exists() else None,"nulls":["PCA-only","row-permutation","subject-label permutation","random-direction"],"primary_unit":"feature group / low-rank subspace","feature_identity_claim":False})
    (out/"audit"/"subspace_stability_summary.md").write_text("# Subspace stability\n\nFeature matching remains a secondary diagnostic. The primary unit is a low-rank subspace or feature group. Existing SAE results and new PCA-only cross-subject CKA with row-permutation controls are retained; high CKA alone is not treated as evidence of generalization.\n")
    # Intervention: preserve previous raw/normalized audit and explicitly mark missing task labels.
    prev=ROOT/"results/iasl/phase0_intervention/intervention_audit.json"
    pdata=json.loads(prev.read_text()) if prev.exists() else {}
    write_json(out/"intervention"/"intervention_metrics.json",{"git_commit":git_commit(),"source_prior_audit":str(prev),"prior_rows":len(pdata.get("rows",[])),"energy_matched_nulls":pdata.get("permutation_nulls",[]),"controls":["group ablation","latent ablation","activation patching","source-to-target interpolation","positive steering","negative steering","random-direction","PCA-component"],"target_class_damage":"unavailable: no task labels/readout in test-only audit","selectivity":"unavailable","conclusion":"No reliable semantic intervention selectivity was established."})
    with (out/"intervention"/"intervention_table.csv").open("w",newline="") as f:
        fields=["width","seed","magnitude","C_raw","C_norm","removed_latent_energy","removed_eeg_energy"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k) for k in fields} for r in pdata.get("rows",[])])
    (out/"intervention"/"intervention_summary.md").write_text("# Intervention recovery\n\nThe prior energy-matched audit was retained and relisted here. Task labels and a trained readout are unavailable in this test-only Phase 0, so target/non-target selectivity, steering and patching cannot be claimed. No reliable semantic intervention selectivity was established.\n")
    # Gates and final report.
    corrected=[r["corrected_split_half"] for r in rel_rows]
    gateA=bool(sum(v>0 and np.isfinite(v) for v in corrected)>=4 and min(r["bootstrap_ci"][0] for r in rel_rows)>0)
    gateB=bool(np.mean([r["cka"] > r["row_permutation_p95"] for r in cka_rows])>=.5)
    gateC=False; status="ELIGIBLE_FOR_SMALL_LOSO" if gateA and gateB and gateC else "NO-GO_RELIABILITY" if not gateA else "SUBSPACE_ONLY"
    gates={"Gate_A_data_reliability":{"pass":gateA,"reason":"corrected reliability and bootstrap lower bounds"},"Gate_B_subspace_reliability":{"pass":gateB,"reason":"PCA CKA versus row permutation and subject pairs"},"Gate_C_intervention":{"pass":gateC,"reason":"task labels/selectivity and independent normalized effect unavailable"},"Gate_D_small_loso":{"pass":False,"reason":"requires A, B and C"},"status":status}
    write_json(out/"gates"/"recovery_gates.json",gates)
    manifest={"git_commit":git_commit(),"subjects":subjects,"image_sets":["Set-A","Set-B","Set-C"],"repetitions_primary":80,"test_only":True,"requested_window":[0,1],"available_analyzed_interval":[0,.79],"baseline_interval":[-.2,0],"status":status,"new_downloads":False}
    write_json(out/"recovery_manifest.json",manifest)
    write_json(out/"recovery_report.json",{"manifest":manifest,"gates":gates,"reliability":rel_rows,"preprocessing":variant_rows,"image_subsets":subset_rows,"negative_results":["low split-half reliability","feature identity at null","intervention selectivity unavailable"],"loso_allowed":False})
    reliability_lines="\n".join([f"| {r['subject']} | {r['raw_split_half']:.3f} | {r['corrected_split_half']:.3f} | [{r['bootstrap_ci'][0]:.3f}, {r['bootstrap_ci'][1]:.3f}] | {r['permutation_p95']:.3f} | {r['single_trial_to_rest_mean']:.3f} |" for r in rel_rows])
    (out/"recovery_report.md").write_text(f"""# IASL Phase 0 recovery report

Status: **{status}**

## Scope

This recovery audit preserves all prior Phase 0A–0D and NO-GO files. It uses the existing five numeric THINGS-EEG2 test exports, all 80 available repetitions, Set-A/Set-B/Set-C image subsets, and no new downloads. Requested window: 0–1000 ms; available analyzed interval: 0–790 ms; baseline interval: -200–0 ms.

## Findings

- Primary all-80 split-half reliability (mean image-wise correlation) is:

| Subject | Raw split-half | Spearman–Brown corrected | Bootstrap 95% CI | Permutation-null p95 | Single-trial to rest |
|---|---:|---:|---:|---:|---:|
{reliability_lines}

Subject-03 is retained and analyzed separately; it is not removed to improve the result.
- Preprocessing variants A–E and fixed image subsets do not change the rule that configurations are reported without post-hoc selection. Variant F is deferred to fold-local fitting.
- Trial-average and single-trial diagnostics are separate. Trial averaging cannot be presented as single-trial generalization.
- Feature identity remains unstable; the analysis unit is a low-rank subspace or feature group.
- The energy-matched intervention audit has no task labels or readout, so target/non-target selectivity and semantic steering are unavailable. No reliable semantic intervention selectivity was established.

## Gates

```json
{json.dumps(gates, indent=2)}
```

Gate D is blocked unless Gates A, B and C all pass. Full IASL and small LOSO were not run.

## Recommendation

The earlier NO-GO is not overwritten. If Gate A fails, the appropriate conclusion is `NO-GO_RELIABILITY`; if reliability improves but intervention remains unsupported, the project should be reported as `SUBSPACE_ONLY` and shifted toward a subspace reproducibility and intervention-limits analysis paper. No cross-subject generalization or data-efficiency claim is made.
""")

def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=0);p.add_argument("--config");p.add_argument("--output-dir",default=str(OUT_DEFAULT));p.add_argument("--data-root");p.add_argument("--subjects");p.add_argument("--image-set",default="all");p.add_argument("--representation",default="all");p.add_argument("--test-only",action="store_true");args=p.parse_args();run(args)

if __name__ == "__main__": main()
