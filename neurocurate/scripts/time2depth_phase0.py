"""Frozen Time2Depth Phase 0 analysis using verified real EEG only."""
from pathlib import Path
import argparse, hashlib, itertools, json, platform, shutil, subprocess, sys
import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax
from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt
from neurocurate.rdm import compute_rdm, validate_rdm

BASE=Path('results/real_phase0'); P16=BASE/'prospective_03_05'; OUT=Path('results/time2depth/phase0')
CFG=Path('configs/time2depth_phase0.yaml'); SUBJECTS=['sub01','sub02','sub03','sub04','sub05']
WINDOWS=[('50_100',.05,.10),('100_150',.10,.15),('150_200',.15,.20),('200_300',.20,.30),('300_500',.30,.50),('baseline_-200_0',-.20,0.)]
TRI=np.triu_indices(200,1)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def root(sub): return BASE/sub if sub in ['sub01','sub02'] else P16/sub
def rv(rdm):
 x=rankdata(validate_rdm(rdm)[TRI]); x-=x.mean(); return x/np.linalg.norm(x)
def pref(matrix,tau=.05):
 centered=matrix-matrix.mean(axis=0,keepdims=True); pos=np.linspace(0,1,matrix.shape[1]); return centered,softmax(centered/tau,axis=1)@pos
def freeze(protocol,task):
 if OUT.exists(): raise SystemExit('output directory already exists; refusing to overwrite')
 OUT.mkdir(parents=True); shutil.copy2(CFG,OUT/'frozen_protocol.yaml')
 (OUT/'git_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()+'\n')
 (OUT/'environment.txt').write_text(f'python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\npandas={pd.__version__}\n')
 ids=np.load(BASE/'sub01/image_ids.npy',allow_pickle=False); sources={'config':{'path':str(CFG),'sha256':sha(CFG)},'protocol':{'path':str(protocol),'sha256':sha(protocol)},'task':{'path':str(task),'sha256':sha(task)},'image_ids':{'path':str(BASE/'sub01/image_ids.npy'),'sha256':sha(BASE/'sub01/image_ids.npy')},'inputs':{}}
 for sub in SUBJECTS:
  r=root(sub); sources['inputs'][sub]={n:{'path':str(r/'pickle_free'/n),'sha256':sha(r/'pickle_free'/n)} for n in ['eeg.npy','times.npy','channels.npy','export_manifest.json']}
 cache_root=Path('/workspace/setup/neurocurate/model_cache/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415')
 sources['dinov2_cache']={'revision':'f9e44c814b77203eaa57a6bdbbd535f21ede1415','path':str(cache_root),'local_only':True,'files':{p.name:{'path':str(p),'sha256':sha(p)} for p in cache_root.iterdir() if p.is_file()}}
 (OUT/'source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n'); print('FROZEN',sha(OUT/'frozen_protocol.yaml'))
def model_rdms(ids,paths):
 import torch
 from PIL import Image
 from transformers import AutoImageProcessor,Dinov2Model
 rev='f9e44c814b77203eaa57a6bdbbd535f21ede1415'; cache='/workspace/setup/neurocurate/model_cache'; name='facebook/dinov2-base'
 processor=AutoImageProcessor.from_pretrained(name,revision=rev,cache_dir=cache,local_files_only=True,use_fast=False); model=Dinov2Model.from_pretrained(name,revision=rev,cache_dir=cache,local_files_only=True); model.eval(); model.requires_grad_(False)
 layers=[]
 with torch.inference_mode():
  for start in range(0,200,8):
   ims=[]
   for p in paths[start:start+8]:
    with Image.open(p) as im: ims.append(im.convert('RGB'))
   inputs=processor(images=ims,return_tensors='pt'); out=model(**inputs,output_hidden_states=True)
   hidden=out.hidden_states[1:]; assert len(hidden)==12
   block=torch.stack([model.layernorm(h)[:,0] for h in hidden],dim=1); block=block/block.norm(dim=2,keepdim=True); layers.append(block.cpu().numpy())
 feats=np.concatenate(layers); frozen=np.load(BASE/'v21_frozen/vision_features/dinov2.npy',allow_pickle=False); smoke=float(np.max(np.abs(feats[:5,-1]-frozen[:5])))
 rdms=np.array([validate_rdm(compute_rdm(feats[:,i],'cosine')) for i in range(12)])
 schema={'checkpoint':name,'revision':rev,'processor_class':type(processor).__name__,'hook':'Dinov2Model output_hidden_states[1:]; model.layernorm(block_output)[:,0]','layers':12,'feature_dimension':768,'l2_normalized':True,'smoke_images':5,'final_layer_vs_frozen_max_abs':smoke,'final_layer_matches_frozen':bool(np.allclose(feats[:5,-1],frozen[:5],atol=1e-6))}
 return rdms,schema
def run():
 frozen=yaml.safe_load((OUT/'frozen_protocol.yaml').read_text()); sources=json.loads((OUT/'source_manifest.json').read_text()); assert sha(CFG)==sources['config']['sha256']==sha(OUT/'frozen_protocol.yaml')
 ids=np.load(BASE/'sub01/image_ids.npy',allow_pickle=False); assert len(ids)==200 and len(set(ids.tolist()))==200; paths=[Path('data/local/real_phase0/stimuli')/x for x in ids]; assert all(p.is_file() for p in paths)
 mrdm,mschema=model_rdms(ids,paths); np.savez_compressed(OUT/'model_rdm_by_layer.npz',rdms=mrdm,layers=np.arange(1,13),image_ids=ids); (OUT/'model_layer_schema.json').write_text(json.dumps(mschema,indent=2)+'\n')
 rng=np.random.default_rng(20261011); splits=[]
 for _ in range(30): p=rng.permutation(80); splits.append((np.sort(p[:40]),np.sort(p[40:])))
 brains=np.empty((5,6,200,200)); reli=[]; eschema={'subjects':{},'windows':{}}
 tref=None; cref=None
 for si,sub in enumerate(SUBJECTS):
  r=root(sub); eeg=np.load(r/'pickle_free/eeg.npy',allow_pickle=False,mmap_mode='r'); times=np.load(r/'pickle_free/times.npy',allow_pickle=False); channels=np.load(r/'pickle_free/channels.npy',allow_pickle=False)
  assert eeg.shape==(200,80,17,100) and eeg.dtype==np.float64 and np.isfinite(eeg).all(); assert tref is None or np.array_equal(times,tref); assert cref is None or np.array_equal(channels,cref); tref=times; cref=channels
  eschema['subjects'][sub]={'shape':list(eeg.shape),'dtype':str(eeg.dtype),'sampling_rate_hz':float(1/np.median(np.diff(times))),'times':[float(times[0]),float(times[-1])],'channels':channels.tolist()}
  for wi,(label,start,end) in enumerate(WINDOWS):
   mask=(times>=start-1e-12)&(times<end-1e-12); eschema['windows'][label]={'seconds':[start,end],'samples':int(mask.sum()),'sample_times':times[mask].tolist()}; assert mask.any()
   def rdm(rep): return validate_rdm(compute_rdm(np.asarray(eeg[:,rep]).mean(axis=1)[:,:,mask].reshape(200,-1),'correlation'))
   brains[si,wi]=rdm(np.arange(80)); vals=[]
   for a,b in splits: vals.append(float(spearmanr(rdm(a)[TRI],rdm(b)[TRI]).statistic))
   reli.append({'subject':sub,'window':label,'samples':int(mask.sum()),'median':float(np.median(vals)),'q025':float(np.quantile(vals,.025)),'q975':float(np.quantile(vals,.975)),'split_correlations':json.dumps(vals)})
  print('EEG',sub,'complete',flush=True)
 np.savez_compressed(OUT/'brain_rdm_by_window.npz',rdms=brains,subjects=np.array(SUBJECTS),windows=np.array([x[0] for x in WINDOWS]),image_ids=ids)
 pd.DataFrame(reli).to_csv(OUT/'window_reliability.csv',index=False); (OUT/'eeg_window_schema.json').write_text(json.dumps(eschema,indent=2)+'\n')
 mr=np.array([rv(x) for x in mrdm]); raw=np.empty((5,6,12))
 for s in range(5):
  for w in range(6): raw[s,w]=np.array([rv(brains[s,w])@x for x in mr])
 main=raw[:,:5]; centered=np.empty_like(main); prefs=np.empty((5,5)); subject_rho=[]; argmax=[]
 for s in range(5): centered[s],prefs[s]=pref(main[s]); subject_rho.append(float(spearmanr(np.arange(5),prefs[s]).statistic)); argmax.append(np.argmax(centered[s],axis=1)+1)
 np.savez_compressed(OUT/'rsa_matrix_by_subject.npz',raw=raw,centered=centered,preferences=prefs,subject_rho=np.array(subject_rho),subjects=np.array(SUBJECTS),windows=np.array([x[0] for x in WINDOWS]),layers=np.arange(1,13))
 rows=[]
 for s,sub in enumerate(SUBJECTS):
  for w,(label,_,_) in enumerate(WINDOWS):
   for l in range(12): rows.append({'subject':sub,'window':label,'layer':l+1,'rsa':raw[s,w,l],'centered_rsa':centered[s,w,l] if w<5 else np.nan})
 pd.DataFrame(rows).to_csv(OUT/'rsa_long.csv',index=False)
 perms=list(itertools.permutations(range(5))); null=[]
 for p in perms: null.append(float(np.median([spearmanr(np.arange(5),prefs[s,list(p)]).statistic for s in range(5)])))
 observed=float(np.median(subject_rho)); time_report={'observed_cohort_median':observed,'subject_rho':dict(zip(SUBJECTS,subject_rho)),'subject_argmax_layers':{SUBJECTS[i]:argmax[i].tolist() for i in range(5)},'exact_one_sided_p':float(np.mean(np.asarray(null)>=observed)),'null_min':float(np.min(null)),'null_max':float(np.max(null)),'permutations':[{'order':list(p),'statistic':v} for p,v in zip(perms,null)],'coarse_exact_null':True}
 (OUT/'temporal_order_null.json').write_text(json.dumps(time_report,indent=2)+'\n')
 # Common whole-RDM image permutations preserve all subject/window dependencies.
 brank=np.zeros((5,5,200,200),dtype=np.float32)
 for s in range(5):
  for w in range(5):
   x=rv(brains[s,w]).astype(np.float32); brank[s,w][TRI]=x; brank[s,w][(TRI[1],TRI[0])]=x
 srng=np.random.default_rng(2026101501); snull=[]
 for draw in range(1000):
  p=srng.permutation(200); x=brank[:,:,p][:,:,:,p][:,:,TRI[0],TRI[1]].reshape(25,-1); rsa=(x@mr.T).reshape(5,5,12); rr=[]
  for s in range(5): rr.append(float(spearmanr(np.arange(5),pref(rsa[s])[1]).statistic))
  snull.append(float(np.median(rr)))
  if (draw+1)%100==0: print('stimulus null',draw+1,flush=True)
 snull=np.asarray(snull); stim={'draws':1000,'seed':2026101501,'observed':observed,'null_median':float(np.median(snull)),'q025':float(np.quantile(snull,.025)),'q975':float(np.quantile(snull,.975)),'one_sided_p':float((1+np.sum(snull>=observed))/1001),'null_values':snull.tolist(),'permutation':'same image permutation applied to both EEG RDM axes, common to all subjects/windows'}
 (OUT/'stimulus_label_null.json').write_text(json.dumps(stim,indent=2)+'\n')
 rel=pd.DataFrame(reli); snr=[]
 for s,sub in enumerate(SUBJECTS):
  for w,(label,_,_) in enumerate(WINDOWS[:5]): snr.append({'subject':sub,'window':label,'reliability_median':float(rel[(rel.subject==sub)&(rel.window==label)].median.iloc[0]),'mean_raw_rsa':float(main[s,w].mean()),'max_raw_rsa':float(main[s,w].max()),'preferred_layer':float(prefs[s,w]),'argmax_layer':int(argmax[s][w])})
 snrdf=pd.DataFrame(snr); snrdf.to_csv(OUT/'snr_sensitivity.csv',index=False)
 integrity={'brain_shape':list(brains.shape),'model_shape':list(mrdm.shape),'rsa_shape':list(raw.shape),'brain_all_finite':bool(np.isfinite(brains).all()),'model_all_finite':bool(np.isfinite(mrdm).all()),'brain_symmetric':bool(np.allclose(brains,brains.swapaxes(-1,-2),atol=1e-10)),'model_symmetric':bool(np.allclose(mrdm,mrdm.swapaxes(-1,-2),atol=1e-10)),'all_diagonals_zero':bool(np.allclose(np.diagonal(brains,axis1=-2,axis2=-1),0) and np.allclose(np.diagonal(mrdm,axis1=-2,axis2=-1),0)),'ordered_ids_unique':True}
 (OUT/'rdm_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
 med=np.median(main,axis=0); cm=np.median(centered,axis=0)
 for data,name,title in [(med,'rsa_raw_heatmap.png','Median raw RSA'),(cm,'rsa_layer_centered_heatmap.png','Median layer-centered RSA')]:
  fig,ax=plt.subplots(figsize=(9,4)); im=ax.imshow(data,aspect='auto',cmap='coolwarm'); ax.set_xticks(range(12),range(1,13)); ax.set_yticks(range(5),[x[0] for x in WINDOWS[:5]]); ax.set_xlabel('DINOv2 block'); ax.set_ylabel('EEG window'); ax.set_title(title); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(OUT/name,dpi=180); plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5));
 for s,sub in enumerate(SUBJECTS): ax.plot(range(5),prefs[s],marker='o',label=f'{sub} rho={subject_rho[s]:.2f}')
 ax.set_xticks(range(5),[x[0] for x in WINDOWS[:5]],rotation=25); ax.set_ylabel('soft preferred layer position'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'subject_layer_preferences.png',dpi=180); plt.close(fig)
 positive=sum(x>0 for x in subject_rho); timep=time_report['exact_one_sided_p']; snr_r=float(spearmanr(snrdf.reliability_median,snrdf.preferred_layer).statistic); baseline=np.median(raw[:,5],axis=0)
 decision='GO' if positive>=4 and observed>0 and timep<=.05 and stim['one_sided_p']<=.05 and min(rel[rel.window!='baseline_-200_0']['median'])>0 else ('WEAK' if observed>0 and positive>=3 else 'NO-GO')
 summary=f'''# Time2Depth Phase 0 scientific summary\n\n## Result\n\nSubject temporal-order rhos: {dict(zip(SUBJECTS,[round(x,4) for x in subject_rho]))}. Cohort median={observed:.4f}; exact common time-order p={timep:.4f}. Positive subjects={positive}/5. Whole-RDM stimulus-label null p={stim['one_sided_p']:.4f}, median={stim['null_median']:.4f}, 95% interval=[{stim['q025']:.4f}, {stim['q975']:.4f}].\n\nMain-window split-half reliability medians range from {rel[rel.window!='baseline_-200_0']['median'].min():.4f} to {rel[rel.window!='baseline_-200_0']['median'].max():.4f}. Reliability versus soft preferred layer across subject/windows has descriptive Spearman rho={snr_r:.4f}. The shortest windows contain five 100-Hz samples. Baseline median RSA across layers ranges from {baseline.min():.4f} to {baseline.max():.4f}. Split intervals are repetition-split variability, not population confidence intervals or session-independent reliability.\n\nRaw RSA includes global layer alignment; layer-centered RSA removes a layer's across-window mean before preference estimation. Any visual difference between the heatmaps therefore diagnoses static layer bias rather than proving a biological hierarchy. The official shared MVNN whitening used training/test partitions and repetitions, so every EEG result is conditional on supplied preprocessing.\n\n## Decision: {decision}\n\nGO criteria were frozen before result inspection: at least 4/5 positive subject rhos, positive cohort median, exact time-order p<=0.05, and controls inconsistent with an SNR/static-layer explanation. A successful RSA result would only justify testing temporal supervision; it would not show improved performance. Phase 1 is not started automatically.\n\nA later Phase 1, only if GO, would compare fixed/random/unconstrained/monotonic time-to-layer regularizers on a separate training image pool. The 200 repeated-test images would remain evaluation-only and would not be training targets or hyperparameter-selection data.\n'''
 (OUT/'scientific_summary.md').write_text(summary)
 (OUT/'validation.md').write_text(f'''# Validation\n\n- Frozen protocol SHA256: `{sha(OUT/'frozen_protocol.yaml')}`\n- Real EEG: five verified float64 `(200,80,17,100)` arrays loaded with `allow_pickle=False`.\n- Images: 200 unique ordered IDs; all local stimulus files present.\n- DINOv2: pinned local revision; 5-image final-block smoke match={mschema['final_layer_matches_frozen']}, max absolute difference={mschema['final_layer_vs_frozen_max_abs']:.3g}.\n- RDM integrity: {integrity}.\n- No EEG/checkpoint downloads, no training, and no earlier result modifications.\n''')
 files={}
 for p in OUT.iterdir():
  if p.is_file() and p.name!='output_manifest.json': files[p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
 (OUT/'output_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
 print('DECISION',decision,'observed',observed,'time_p',timep,'stim_p',stim['one_sided_p'])
def main():
 p=argparse.ArgumentParser(); p.add_argument('--freeze-only',action='store_true'); p.add_argument('--protocol-source',type=Path); p.add_argument('--task-source',type=Path); a=p.parse_args()
 if a.freeze_only: freeze(a.protocol_source,a.task_source)
 else: run()
if __name__=='__main__': main()
