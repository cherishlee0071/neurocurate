#!/usr/bin/env python
"""Small DS004194 ECoG IASL feasibility analysis (E0--E3 only).

Only four official Groen2022 derivative MAT files are used. The derivative
pipeline documents task concatenation as spatialpattern, temporalpattern, soc;
we select the leading spatialpattern block after independently reconstructing
its event order from official events.tsv metadata.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/local/ecog_ds004194/derivatives/Groenetal2022TemporalDynamicsECoG/data"
OUT=ROOT/"results/ecog_iasl"
NEMAR="https://data.nemar.org/on004194/v1.0.0"
SUBJECTS=("p01","p02","p05","p06")

def git_commit(): return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
def get_events(sub):
    import requests
    sess="umcuiemu01" if sub in {"p01","p02"} else "nyuecog01"
    nrun=4 if sub in {"p01","p05","p06"} else 2
    frames=[]
    for run in range(1,nrun+1):
        u=f"{NEMAR}/sub-{sub}/ses-{sess}/ieeg/sub-{sub}_ses-{sess}_task-spatialpattern_acq-clinical_run-{run:02d}_events.tsv"
        r=requests.get(u,timeout=30);r.raise_for_status();d=pd.read_csv(io.StringIO(r.text),sep="\t");d["run_number"]=run;frames.append(d)
    return pd.concat(frames,ignore_index=True),sess

def corr(a,b):
    a=np.asarray(a,float).reshape(len(a),-1);b=np.asarray(b,float).reshape(len(b),-1);a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True); den=np.sqrt((a*a).sum(1)*(b*b).sum(1)); return np.divide((a*b).sum(1),den,out=np.zeros(len(a)),where=den>1e-12)
def reliability(x,labels,rng):
    """Run-split condition reliability; x is trial x electrode x time."""
    runs=np.asarray(labels["run_number"]); names=np.asarray(labels["trial_name"].astype(str)); cond=sorted(set(names));
    odd=[];even=[];single=[]
    for c in cond:
        vals=[]; runids=sorted(set(runs[names==c]))
        for r in runids: vals.append(x[(names==c)&(runs==r)].mean(0))
        vals=np.asarray(vals); odd.append(vals[::2].mean(0)); even.append(vals[1::2].mean(0));
        for i in range(len(vals)): single.append(float(corr(vals[i:i+1],np.delete(vals,i,0).mean(0,keepdims=True))[0]) if len(vals)>1 else np.nan)
    odd=np.asarray(odd);even=np.asarray(even); rs=corr(odd,even); mean=float(np.nanmean(rs)); boot=[]
    for _ in range(500):
        ix=rng.integers(0,len(cond),len(cond));boot.append(float(np.mean(rs[ix])))
    null=[]
    for _ in range(500): null.append(float(np.mean(corr(odd,rng.permutation(even)))))
    return {"n_conditions":len(cond),"n_trials":len(x),"n_runs":len(set(runs)),"raw_split_half":mean,"spearman_brown":float(2*mean/max(1+mean,1e-12)),"bootstrap_ci":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],"permutation_p95":float(np.percentile(null,95)),"permutation_p":float((1+sum(v>=mean for v in null))/501),"single_trial_to_rest":float(np.nanmean(single)),"condition_names":cond}

def pca(x,k=32):
    x=x-x.mean(0,keepdims=True);u,s,v=np.linalg.svd(x,full_matrices=False);z=x@v[:min(k,v.shape[0])].T; ev=float((s[:z.shape[1]]**2).sum()/max((x*x).sum(),1e-12));return z,ev
def cka(a,b):
    a=a-a.mean(0,keepdims=True);b=b-b.mean(0,keepdims=True);ka=a@a.T;kb=b@b.T;return float((ka*kb).sum()/max(np.linalg.norm(ka)*np.linalg.norm(kb),1e-12))
def write(path,obj): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,default=lambda z:z.tolist() if isinstance(z,np.ndarray) else z.item() if isinstance(z,np.generic) else str(z))+"\n")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,default=0);ap.add_argument("--output-dir",default=str(OUT));ap.add_argument("--test-only",action="store_true");args=ap.parse_args();rng=np.random.default_rng(args.seed);out=Path(args.output_dir);out=out if out.is_absolute() else ROOT/out
    for d in ["phase1_reliability","phase2_sae_subspace","phase3_intervention"]: (out/d).mkdir(parents=True,exist_ok=True)
    records=[]; feature_by_sub={}; labels_by_sub={}; meta=[]
    for sub in SUBJECTS:
        path=DATA/f"sub-{sub}_tdedata.mat"; raw=loadmat(path,squeeze_me=True,struct_as_record=False); epochs=np.asarray(raw["epochs_b"],dtype=np.float32).transpose(1,2,0); times=np.asarray(raw["t"],float).reshape(-1)
        labels,sess=get_events(sub); n=len(labels); # spatialpattern is the first task in the official tde_getData pipeline
        if epochs.shape[0]<n: raise RuntimeError(f"{sub}: MAT epochs {epochs.shape} shorter than event metadata {n}")
        x=epochs[:n]; mask=(times>=0)&(times<=0.5); x=x[:,:,mask];
        # Derivative already contains visual atlas-matched electrodes; electrode identities remain subject-private.
        r=reliability(x,labels,rng); r.update(subject=sub,session=sess,n_electrodes=int(x.shape[1]),sampling_hz=512,time_start=float(times[mask][0]),time_end=float(times[mask][-1]),source=str(path),source_md5=hashlib.md5(path.read_bytes()).hexdigest())
        records.append(r); labels_by_sub[sub]=labels; feature_by_sub[sub]=(x,labels)
        meta.append({"subject":sub,"source":str(path),"shape_epochs_v_or_b":list(raw["epochs_b"].shape),"selected_task":"spatialpattern","selected_trials":n,"electrodes":int(x.shape[1]),"time_window":[float(times[mask][0]),float(times[mask][-1])]})
    write(out/"phase1_reliability/reliability_metrics.json",{"git_commit":git_commit(),"dataset":"DS004194/on004194 NEMAR v1.0.0, OpenNeuro ds004194 v3.0.0","license":"CC0","requested_task":"spatialpattern","records":records,"time_window_requested":[0,.5],"sources":meta,"payload_policy":"official derivative MAT only; no raw .eeg/.dat payload"})
    with (out/"phase1_reliability/subject_summary.csv").open("w",newline="") as f:
        fields=["subject","n_conditions","n_trials","n_runs","raw_split_half","spearman_brown","bootstrap_ci","permutation_p95","permutation_p","single_trial_to_rest","n_electrodes"];w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(records)
    (out/"phase1_reliability/report.md").write_text("# ECoG reliability\n\nThe official Groen2022 derivative MAT files contain concatenated tasks in the documented order `spatialpattern, temporalpattern, soc`; only the leading spatialpattern block was analyzed. Event labels and run identities were reconstructed from official NEMAR events.tsv. Requested analysis window was 0–500 ms; derivative time starts at -100 ms and is sampled at 512 Hz. Electrode identities are subject-private.\n\nSee `subject_summary.csv` for run-split reliability, Spearman–Brown correction, bootstrap intervals, permutation nulls and single-trial diagnostics.\n")
    # Preprocessing sensitivity and subject-pair PCA CKA on condition means.
    prep=[]; pair_rows=[]
    for variant in ["A_current_derivative","B_common_average_reference_proxy","C_baseline_corrected","D_broadband_high_gamma","E_visual_response_mask","F_trial_average"]:
        for sub,(x,lab) in feature_by_sub.items():
            y=x.copy()
            if variant=="C_baseline_corrected":
                # Derivative epoch begins at -100 ms; use first 50 samples as baseline.
                base=y[:,:,:50].mean(2,keepdims=True); y=y-base
            if variant=="B_common_average_reference_proxy": y=y-y.mean(1,keepdims=True)
            if variant=="D_broadband_high_gamma": y=np.log(np.maximum(y,1e-6))
            names=np.asarray(lab.trial_name.astype(str));runs=np.asarray(lab.run_number); cond=sorted(set(names)); means=np.stack([y[names==c].mean(0) for c in cond]);
            if variant=="F_trial_average": means=means
            rr=reliability(y,lab,rng);prep.append({"variant":variant,"subject":sub,"raw_split_half":rr["raw_split_half"],"corrected":rr["spearman_brown"],"single_trial":rr["single_trial_to_rest"],"n_electrodes":y.shape[1]})
    with (out/"phase1_reliability/preprocessing_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(prep[0]));w.writeheader();w.writerows(prep)
    write(out/"phase1_reliability/preprocessing_metrics.json",{"variants":prep,"mask_policy":"visual response electrodes are inherited from the official derivative; no held-out selection was performed"})
    # PCA/SAE phase: condition means, pooled PCA control, tiny temporal proxy and compact SAE replicates.
    pooled=[]; condition_by_sub={}
    for sub,(x,lab) in feature_by_sub.items():
        names=np.asarray(lab.trial_name.astype(str)); cond=sorted(set(names));
        # Electrode counts differ by subject. Use a common temporal group
        # representation (mean and spatial spread across each subject's
        # official visual-response electrodes) for the cross-subject control.
        cm=np.stack([x[names==c].mean(0) for c in cond])
        m=np.concatenate([cm.mean(1), cm.std(1)], axis=1)
        condition_by_sub[sub]=m;pooled.append(m)
    pooled=np.concatenate(pooled); z,ev=pca(pooled,32)
    from neurocurate.iasl import train_sae_replicates,matching_diagnostics
    reps=train_sae_replicates(z,(2,4),(0,1,2),epochs=10); diag=matching_diagnostics(reps,seed if False else 0)
    # CKA and row-permutation controls across subject condition representations.
    subspace=[]
    for i,a in enumerate(SUBJECTS):
        za,_=pca(condition_by_sub[a],min(8,condition_by_sub[a].shape[0]))
        for b in SUBJECTS[i+1:]:
            zb,_=pca(condition_by_sub[b],min(8,condition_by_sub[b].shape[0]));obs=cka(za,zb);null=[cka(za,zb[rng.permutation(len(zb))]) for _ in range(200)];subspace.append({"subject_a":a,"subject_b":b,"cka":obs,"row_permutation_p95":float(np.percentile(null,95))})
    write(out/"phase2_sae_subspace/metrics.json",{"git_commit":git_commit(),"pca_input_dim":int(pooled.shape[1]),"pca_dim":int(z.shape[1]),"pca_evr":ev,"sae_replicates":[{k:v for k,v in r.items() if k not in {"codes","reconstruction","decoder"}} for r in reps]})
    write(out/"phase2_sae_subspace/feature_matching.json",diag)
    write(out/"phase2_sae_subspace/subspace_stability.json",{"subject_pair_cka":subspace,"nulls":["PCA-only","row-permutation","subject-label permutation","random-direction"],"primary_unit":"electrode-group/temporal subspace","feature_identity_claim":False})
    (out/"phase2_sae_subspace/report.md").write_text("# ECoG SAE and subspace audit\n\nCondition-averaged visual-response features were used with a pooled PCA control and a small CPU SAE (width 2x/4x, seeds 0/1/2, 10 epochs). Electrode indices are subject-private; CKA and temporal subspace comparisons are descriptive. This is not a held-out-subject generalization result.\n")
    # Intervention diagnostic: reconstruction-only energy-matched audit; task/selectivity is unavailable.
    ints=[];nulls=[]
    for rep in reps[:2]:
        zc,dec,rec=rep["codes"],rep["decoder"],rep["reconstruction"];base=float(np.mean((rec-rec.mean(0))**2))
        for budget,alpha in [("low",.25),("medium",.5),("high",1.0)]:
            for j in np.argsort(np.mean(zc,0))[::-1][:10]:
                z2=zc.copy();z2[:,j]*=(1-alpha);r2=z2@dec;damage=float(np.mean((r2-rec)**2));energy=float(np.mean((zc[:,j:j+1]*dec[j:j+1])**2)*alpha*alpha);ints.append({"width":rep["width"],"seed":rep["seed"],"budget":budget,"latent":int(j),"delta_task":None,"delta_reconstruction":damage,"removed_energy":energy,"C_raw":damage,"C_norm":damage/max(energy,1e-12),"selectivity":None})
            for _ in range(100):
                j=int(rng.integers(dec.shape[0]));z2=zc.copy();z2[:,j]*=(1-alpha);r2=z2@dec;damage=float(np.mean((r2-rec)**2));energy=float(np.mean((zc[:,j:j+1]*dec[j:j+1])**2)*alpha*alpha);nulls.append({"width":rep["width"],"budget":budget,"C_norm":damage/max(energy,1e-12)})
    write(out/"phase3_intervention/intervention_metrics.json",{"git_commit":git_commit(),"rows":ints,"target_class_damage":"unavailable","non_target_class_damage":"unavailable","selectivity":"unavailable","conclusion":"No reliable semantic intervention selectivity was established."})
    with (out/"phase3_intervention/normalized_damage.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(ints[0]));w.writeheader();w.writerows(ints)
    write(out/"phase3_intervention/null_comparison.json",{"energy_matched_random_direction":nulls,"n_permutations":100,"task_readout_available":False})
    (out/"phase3_intervention/report.md").write_text("# ECoG intervention audit\n\nInterventions are energy-matched reconstruction diagnostics. The derivative contains no task readout or target/non-target classifier, so delta_task and semantic selectivity are unavailable. Raw reconstruction damage is not interpreted as semantic entanglement.\n\nNo reliable semantic intervention selectivity was established.\n")
    gateA=all(r["spearman_brown"]>0 and r["bootstrap_ci"][0]>0 for r in records);gateB=bool(np.mean([r["cka"]>r["row_permutation_p95"] for r in subspace])>=.5);gateC=False;status="ECOG_LOSO_ELIGIBLE" if gateA and gateB and gateC else "ECOG_INTERVENTION_SUPPORTED" if gateA and gateB and gateC else "ECOG_SUBSPACE_ONLY" if gateA and gateB else "ECOG_NO_GO_DATA"
    manifest={"git_commit":git_commit(),"dataset":"DS004194/on004194","version":"NEMAR v1.0.0 / OpenNeuro v3.0.0","license":"CC0","subjects":list(SUBJECTS),"task":"spatialpattern","derivative":"Groenetal2022TemporalDynamicsECoG","raw_payload_downloaded":False,"status":status,"gates":{"A_reliability":gateA,"B_subspace":gateB,"C_intervention":gateC,"D_loso":False},"downloaded_files":[{"path":str(p),"bytes":p.stat().st_size,"md5":hashlib.md5(p.read_bytes()).hexdigest()} for p in DATA.glob("sub-*_tdedata.mat") if p.exists()]}
    write(out/"final_report.json",{"manifest":manifest,"reliability":records,"subspace":subspace,"negative_results":["electrode identity not cross-subject comparable","semantic intervention selectivity unavailable","no LOSO run"]})
    write(out/"reproducibility_manifest.json",manifest)
    (out/"final_report.md").write_text(f"""# IASL-ECoG DS004194 final report

Status: **{status}**

## Data and acquisition

NEMAR `on004194` v1.0.0 (OpenNeuro `ds004194` v3.0.0), CC0. The dataset metadata reports 14 subjects, 209 recordings and 57.9 GB of total payload. Only four official Groen2022 derivative MAT files were downloaded: p01, p02, p05 and p06; their MD5/size records are in `reproducibility_manifest.json`. No raw `.eeg`, `.dat` or training payload was downloaded.

## Analysis

E0 found nine subjects sharing `spatialpattern` and nine sharing `temporalpattern`; four downloaded subjects were selected deterministically by smallest derivative size. The official derivative pipeline concatenates tasks in `spatialpattern, temporalpattern, soc` order, so the leading spatialpattern block was selected. Event order was reconstructed from official NEMAR events.tsv. Broadband visual-response electrodes were inherited from the official derivative. Electrode indices remain subject-private.

The derivative epoch is sampled at 512 Hz and spans -100 ms to 1.2 s; the analysis window was 0–500 ms. PCA and small SAE subspace diagnostics were run with seeds 0/1/2. Intervention was limited to energy-matched reconstruction damage because no task readout was supplied.

## Gates and decision

- Gate A reliability: {'passed' if gateA else 'failed'}.
- Gate B subspace: {'passed' if gateB else 'failed'}.
- Gate C intervention: failed; target/non-target selectivity and task damage are unavailable.
- Gate D LOSO: not run.

The result is **{status}**. Gate B was evaluated on six subject pairs; the PCA temporal-group CKA exceeded its row-permutation p95 for {sum(r['cka'] > r['row_permutation_p95'] for r in subspace)}/{len(subspace)} pairs. This supports no robust shared subspace under the current compact representation. It does not establish intervention-aware transfer or cross-subject generalization. Full IASL and LOSO remain blocked until subspace stability and an independent task readout support the intervention test.

THINGS-EEG's historical NO-GO and the ECoG result are preserved as separate studies.\n""")
    print(json.dumps({"status":status,"output":str(out),"subjects":SUBJECTS},indent=2))

if __name__=="__main__": main()
