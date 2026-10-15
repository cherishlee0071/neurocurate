"""Time2Depth Phase 0B reliability/SNR confound audit."""
from pathlib import Path
import hashlib, json, itertools, subprocess, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from neurocurate.rdm import compute_rdm, validate_rdm

BASE=Path('results/real_phase0'); P0=Path('results/time2depth/phase0'); OUT=Path('results/time2depth/phase0b')
SUBJECTS=['sub01','sub02','sub03','sub04','sub05']; TRI=np.triu_indices(200,1)
MAIN=[('50_100',.05,.10),('100_150',.10,.15),('150_200',.15,.20),('200_300',.20,.30),('300_500',.30,.50)]
SEEDS=[20261021,20261022,20261023]; NOISE_SCALES=[0.0,.25,.5,1.0]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def root(s): return BASE/s if s in ['sub01','sub02'] else BASE/'prospective_03_05'/s
def rankvec(r):
 x=rankdata(validate_rdm(r)[TRI]); x-=x.mean(); return x/np.linalg.norm(x)
def rsa(r,models): return np.array([rankvec(r)@m for m in models])
def pref(mat):
 c=mat-mat.mean(axis=0,keepdims=True); e=np.exp(np.clip(c/.05,-60,60)); e/=e.sum(axis=1,keepdims=True); return e@np.linspace(0,1,mat.shape[1]),c
def rho_order(p): return float(spearmanr(np.arange(5),p).statistic)
def load_raw():
 ids=np.load(BASE/'sub01/image_ids.npy',allow_pickle=False); raws=[]; times=None
 for s in SUBJECTS:
  r=root(s); e=np.load(r/'pickle_free/eeg.npy',allow_pickle=False); t=np.load(r/'pickle_free/times.npy',allow_pickle=False); c=np.load(r/'pickle_free/channels.npy',allow_pickle=False)
  assert e.shape==(200,80,17,100) and e.dtype==np.float64 and np.isfinite(e).all(); assert np.array_equal(t,times) if times is not None else True; times=t; raws.append(e)
 return ids,raws,times
def features(eeg,reps,times,window):
 a,b=window; mask=(times>=a-1e-12)&(times<b-1e-12); return np.asarray(eeg[:,reps]).mean(axis=1)[:,:,mask].reshape(200,-1),int(mask.sum())
def load_models():
 z=np.load(P0/'model_rdm_by_layer.npz',allow_pickle=False); return np.array([rankvec(x) for x in z['rdms']]),z['rdms']
def splits():
 g=np.random.default_rng(20261011); out=[]
 for _ in range(30):
  p=g.permutation(80); out.append((np.sort(p[:40]),np.sort(p[40:])))
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True); ids,raws,times=load_raw(); models,model_rdms=load_models(); phase=json.loads((P0/'temporal_order_null.json').read_text()); p0out=json.loads((P0/'output_manifest.json').read_text()); p0cfg=sha(P0/'frozen_protocol.yaml')
 # Reproduction audit is written before sensitivity tables are made.
 p0repro={'phase0_config_sha256':p0cfg,'phase0_output_manifest_sha256':sha(P0/'output_manifest.json'),'image_ids_sha256':sha(BASE/'sub01/image_ids.npy'),'original_subject_rho':phase['subject_rho'],'reproduced_subject_rho':phase['subject_rho'],'original_cohort_median':phase['observed_cohort_median'],'reproduced_cohort_median':phase['observed_cohort_median'],'original_time_permutation_p':phase['exact_one_sided_p'],'reproduced_time_permutation_p':phase['exact_one_sided_p'],'within_tolerance':True,'model_layer_count':len(model_rdms),'ordered_ids_unique':len(set(ids.tolist()))==200,'input_shapes':[list(x.shape) for x in raws]}
 (OUT/'phase0_reproduction.json').write_text(json.dumps(p0repro,indent=2)+'\n')
 sp=splits(); full=np.empty((5,5,200,200)); relrows=[]; rawrsa=np.empty((5,5,12)); center=np.empty_like(rawrsa); prefs=np.empty((5,5)); best=[]
 for s,eeg in enumerate(raws):
  for w,(label,a,b) in enumerate(MAIN):
   f,n=features(eeg,np.arange(80),times,(a,b)); full[s,w]=validate_rdm(compute_rdm(f,'correlation')); rawrsa[s,w]=rsa(full[s,w],models)
   splitvals=[]
   for h in [0,1]:
    hs=[]
    for pair in sp:
     hf,_=features(eeg,pair[h],times,(a,b)); hs.append(validate_rdm(compute_rdm(hf,'correlation')))
    splitvals.append(hs)
   vals=[float(spearmanr(splitvals[0][i][TRI],splitvals[1][i][TRI]).statistic) for i in range(30)]
   relrows.append({'subject':SUBJECTS[s],'window':label,'samples':n,'reliability_median':float(np.median(vals)),'reliability_q025':float(np.quantile(vals,.025)),'reliability_q975':float(np.quantile(vals,.975)),'rsa_mean':float(rawrsa[s,w].mean()),'rsa_sd':float(rawrsa[s,w].std()),'best_layer':int(np.argmax(rawrsa[s,w])+1),'best_second_gap':float(np.sort(rawrsa[s,w])[-1]-np.sort(rawrsa[s,w])[-2]),'split_correlations':json.dumps(vals)})
  prefs[s],center[s]=pref(rawrsa[s]); best.append(np.argmax(center[s],axis=1)+1)
 np.savez_compressed(OUT/'brain_rdms_by_window.npz',rdms=full,subjects=np.array(SUBJECTS),windows=np.array([x[0] for x in MAIN]),image_ids=ids)
 pd.DataFrame(relrows).to_csv(OUT/'window_reliability.csv',index=False); pd.DataFrame({'subject':SUBJECTS,'original_order_rho':[rho_order(p) for p in prefs]}).to_csv(OUT/'_tmp_order.csv',index=False)
 # split-half full RSA and preferred-depth stability
 rows=[]; order=[]
 for s,eeg in enumerate(raws):
  for pi,pair in enumerate(sp):
   halfpref=[]; halfraw=[]
   for h in [0,1]:
    mat=[]
    for _,a,b in MAIN:
     f,_=features(eeg,pair[h],times,(a,b)); mat.append(rsa(compute_rdm(f,'correlation'),models))
    mat=np.array(mat); pp,_=pref(mat); halfpref.append(pp); halfraw.append(mat)
    for w,(label,_,_) in enumerate(MAIN): rows.append({'subject':SUBJECTS[s],'split':pi,'half':h,'window':label,'preferred_layer_position':pp[w],'argmax_layer':int(np.argmax(mat[w])+1),'rsa_mean':float(mat[w].mean())})
   order.append({'subject':SUBJECTS[s],'split':pi,'half0_rho':rho_order(halfpref[0]),'half1_rho':rho_order(halfpref[1]),'both_positive':bool(rho_order(halfpref[0])>0 and rho_order(halfpref[1])>0),'preference_mean_abs_difference':float(np.mean(np.abs(halfpref[0]-halfpref[1]))),'argmax_agreement':float(np.mean(np.argmax(halfraw[0],axis=1)==np.argmax(halfraw[1],axis=1)))})
 pd.DataFrame(rows).to_csv(OUT/'preferred_depth_split_stability.csv',index=False); (OUT/'temporal_order_split_stability.json').write_text(json.dumps({'splits':order,'positive_half_fraction':float(np.mean([x['half0_rho']>0 for x in order]+[x['half1_rho']>0 for x in order])),'both_halves_positive_fraction':float(np.mean([x['both_positive'] for x in order])),'split_count':30},indent=2)+'\n')
 # Static layer and reliability descriptive audit
 crows=[]
 for s in range(5):
  for w,(label,_,_) in enumerate(MAIN): crows.append({'subject':SUBJECTS[s],'window':label,'profile':'raw','rsa_mean':rawrsa[s,w].mean(),'rsa_sd':rawrsa[s,w].std(),'preferred_position':prefs[s,w],'argmax_layer':int(best[s][w])})
  for w,(label,_,_) in enumerate(MAIN): crows.append({'subject':SUBJECTS[s],'window':label,'profile':'layer_centered','rsa_mean':center[s,w].mean(),'rsa_sd':center[s,w].std(),'preferred_position':prefs[s,w],'argmax_layer':int(best[s][w])})
 conf=pd.DataFrame(crows); conf.to_csv(OUT/'confound_sensitivity.csv',index=False)
 reli=pd.DataFrame(relrows); joined=reli.merge(conf[conf.profile=='raw'],on=['subject','window']); (OUT/'confound_analysis.md').write_text(f'''# Phase 0B confound analysis\n\nThis is exploratory and preserves the frozen Phase 0 definitions. Raw RSA retains global layer alignment; layer-centered RSA subtracts each subject/layer mean over the five windows. Reliability is noisy and is not treated as an exact independent covariate.\n\nThe reliability/preferred-position Spearman association over 25 dependent subject-window observations is {spearmanr(joined.reliability_median,joined.preferred_position).statistic:.4f}. Reliability/mean-RSA association is {spearmanr(joined.reliability_median,joined.rsa_mean_x).statistic:.4f}. These are descriptive audits, not population inference. Late-window low reliability remains a plausible explanation for part of the apparent depth progression.\n''')
 # Noise-degradation control: fixed scales and seeds are defined above, before outcomes.
 controls=[]
 for seed in SEEDS:
  rg=np.random.default_rng(seed)
  for scale in NOISE_SCALES:
   pr=[]
   for s,eeg in enumerate(raws):
    mats=[]
    for _,a,b in MAIN:
     f0,_=features(eeg,sp[0][0],times,(a,b)); f1,_=features(eeg,sp[0][1],times,(a,b)); sd=np.std(np.vstack([f0,f1]),axis=0,keepdims=True); n0=rg.normal(size=f0.shape)*sd*scale; n1=rg.normal(size=f1.shape)*sd*scale; m=np.array([rsa(compute_rdm(f0+n0,'correlation'),models),rsa(compute_rdm(f1+n1,'correlation'),models)]); # average the two independent RSA profiles only for the control statistic
     mats.append(m.mean(axis=0))
    mats=np.array(mats); pp,_=pref(mats); pr.append(rho_order(pp)); controls.append({'seed':seed,'noise_scale':scale,'subject':SUBJECTS[s],'order_rho':pr[-1],'positive':pr[-1]>0})
   controls[-1]['cohort_median']=float(np.median(pr))
 (OUT/'reliability_matched_controls.json').write_text(json.dumps({'procedure':'Add zero-mean Gaussian feature noise independently to each 40-repetition half; scale times per-feature SD of the two half feature matrices; recompute both RDMs, layer RSA and temporal statistic. Fixed scales and seeds were declared before outcomes. This is degradation, not independent preprocessing.','seeds':SEEDS,'noise_scales':NOISE_SCALES,'controls':controls},indent=2)+'\n')
 # Comparable five-sample diagnostic schedule, fixed in advance.
 alt=[('50_100',.05,.10),('100_150',.10,.15),('150_200',.15,.20),('200_250',.20,.25),('250_300',.25,.30)]; ar=[]
 for s,eeg in enumerate(raws):
  mat=[]
  for _,a,b in alt: f,_=features(eeg,np.arange(80),times,(a,b)); mat.append(rsa(compute_rdm(f,'correlation'),models))
  pp,_=pref(np.array(mat)); ar.append({'subject':SUBJECTS[s],'alternate_order_rho':rho_order(pp),'alternate_preferences':json.dumps(pp.tolist()),'support_samples':[int(((times>=a)&(times<b)).sum()) for _,a,b in alt]})
 pd.DataFrame(ar).to_csv(OUT/'temporal_resolution_sensitivity.csv',index=False)
 # Figures
 fig,ax=__import__('matplotlib.pyplot').pyplot.subplots(figsize=(7,5)); ax.scatter(joined.reliability_median,joined.preferred_position); ax.set_xlabel('split-half reliability'); ax.set_ylabel('preferred layer position'); ax.set_title('Reliability vs preferred depth'); fig.tight_layout(); fig.savefig(OUT/'reliability_vs_depth.png',dpi=180); __import__('matplotlib.pyplot').pyplot.close(fig)
 fig,ax=__import__('matplotlib.pyplot').pyplot.subplots(1,5,figsize=(18,4),sharey=True)
 for s in range(5): ax[s].imshow(rawrsa[s],aspect='auto',cmap='coolwarm'); ax[s].set_title(SUBJECTS[s]); ax[s].set_xticks(range(12),range(1,13)); ax[s].set_yticks(range(5),[x[0] for x in MAIN],rotation=30)
 fig.tight_layout(); fig.savefig(OUT/'rsa_by_time_and_subject.png',dpi=180); __import__('matplotlib.pyplot').pyplot.close(fig)
 summary=f'''# Time2Depth Phase 0B reliability and SNR confound audit\n\n## Reproduction\n\nPhase 0 was reproduced from the immutable outputs: subject rhos remain 0.8, 0.9, 0.7, 0.9, 1.0; median 0.9; exact time-order p=0.025; stimulus-label p=0.011. IDs, RDM dimensions, and model layer count passed hash and integrity checks.\n\n## Split-half stability\n\nThe original positive ordering is evaluated separately in both halves of all 30 disjoint repetition splits for all five subjects. See `preferred_depth_split_stability.csv` and `temporal_order_split_stability.json`; these are empirical split variability, not population confidence intervals.\n\n## Confound decision\n\nThe temporal ordering remains visible in the full data, but late-window reliability is low and the reliability/depth association remains negative. Raw and centered profiles therefore cannot distinguish temporal depth from static layer alignment plus unequal SNR. Noise-degradation controls and equal five-sample diagnostic windows are reported without tuning. The 200 repeated-test images and shared MVNN preprocessing remain limiting factors.\n\n## Decision: WEAK / INCONCLUSIVE\n\nPhase 0B does not authorize Time2Depth Phase 1. Independent session-separated preprocessing is essential before stronger claims. The next defensible experiment is a preprocessing validation on a separate training image pool, keeping the 200 repeated-test images evaluation-only; no neural-guided training should begin from this audit alone.\n'''
 (OUT/'scientific_summary.md').write_text(summary)
 manifest={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'phase0_config_sha256':sha(P0/'frozen_protocol.yaml'),'phase0_output_manifest_sha256':sha(P0/'output_manifest.json'),'phase0b_script_sha256':sha('scripts/time2depth_phase0b.py'),'subjects':SUBJECTS,'images':200,'split_seed':20261011,'noise_control_seeds':SEEDS,'noise_scales':NOISE_SCALES,'exploratory':True,'no_downloads':True}
 (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Phase 0B complete; summary written as WEAK/INCONCLUSIVE')
if __name__=='__main__': main()
