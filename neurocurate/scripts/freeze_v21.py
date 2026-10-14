"""Create the immutable v2.1 P0 disagreement measurement artifacts."""
from pathlib import Path
import hashlib, json, subprocess
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from neurocurate.rdm import compute_rdm, validate_rdm
from neurocurate.ablation import DisagreementMetrics

BASE=Path('results/real_phase0'); OUT=BASE/'v21_frozen'; P16=BASE/'prospective_03_05'
SUBJECTS=['sub01','sub02','sub03','sub04','sub05']
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 for d in ['brain_features','brain_rdms','vision_features','model_rdms','disagreement']: (OUT/d).mkdir(exist_ok=True)
 ids=np.load(BASE/'sub01/image_ids.npy',allow_pickle=False)
 cfg={
  'protocol':'NeuroCurate v2.1 P0 disagreement freeze','status':'immutable measurement definition; exploratory downstream use only',
  'dataset':{'identifier':'THINGS-EEG2 official standard preprocessed test split','subjects':SUBJECTS,'images':200,'image_ids_sha256':sha(BASE/'sub01/image_ids.npy'),'split':'repeated 200-image sanity/test split'},
  'eeg':{'input':'verified numeric EEG exports only','channels':17,'channel_policy':'all supplied posterior/occipital channels in verified order','time_window_seconds':[0.1,0.5],'preprocessing':'official shared MVNN whitening; inherited and not refit in P0','trial_averaging':'mean over all 80 repetitions','feature_extraction':'flatten 17 channels x 40 samples after averaging','pca':None,'seeds':{'repetition_split':20261011}},
  'vision_models':{'clip':{'name':'openai/clip-vit-base-patch32','revision':'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268','layer':'projected image embedding','pooling':'projected_image_embedding','dimension':512,'preprocessing':'cached pinned CLIPImageProcessor','normalize':True},'dinov2':{'name':'facebook/dinov2-base','revision':'f9e44c814b77203eaa57a6bdbbd535f21ede1415','layer':'last hidden CLS','pooling':'cls','dimension':768,'preprocessing':'cached pinned BitImageProcessor','normalize':True}},
  'rdm':{'metric':'correlation distance = 1 - Pearson correlation','shape':[200,200],'diagonal':0,'symmetry_tolerance':1e-10},
  'disagreement':{'primary':'row-wise representational disagreement','formula':'(1 - Spearman(brain_RDM_row_without_self, model_RDM_row_without_self))/2','models':['clip','dinov2'],'aggregation':'equal-weight mean retained only as consensus descriptive column','tie_policy':'average ranks for Spearman; image IDs for ordering','forbidden_in_freeze':['Jaccard','neighborhood overlap','reciprocal rank','learned scoring']},
  'provenance':{'created_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'phase16_protocol_sha256':sha(P16/'frozen_protocol.json'),'source_note':'P0 reads existing pickle-free numeric exports and pinned cached vision embeddings; no downloads.'}}
 (Path('configs')/'v21_disagreement_frozen.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 model_features={}
 for model in ['clip','dinov2']:
  z=np.load(BASE/'sub01'/f'{model}_features.npz',allow_pickle=False); feats=z['features']; assert np.array_equal(z['image_ids'],ids)
  np.save(OUT/'vision_features'/f'{model}.npy',feats,allow_pickle=False); model_features[model]=feats
  model_rdm=validate_rdm(compute_rdm(feats,'cosine')); np.save(OUT/'model_rdms'/f'{model}.npy',model_rdm,allow_pickle=False)
 brain_rdms={}; score_rows=[]
 for sub in SUBJECTS:
  source_root = BASE/sub if sub in ['sub01','sub02'] else P16/sub
  feats=np.load(source_root/'eeg_features.npy',allow_pickle=False); assert feats.shape==(200,680) and np.isfinite(feats).all()
  np.save(OUT/'brain_features'/f'{sub}.npy',feats,allow_pickle=False)
  brain=validate_rdm(compute_rdm(feats,'correlation')); np.save(OUT/'brain_rdms'/f'{sub}.npy',brain,allow_pickle=False); brain_rdms[sub]=brain
  vals=DisagreementMetrics({m:np.load(OUT/'model_rdms'/f'{m}.npy',allow_pickle=False) for m in ['clip','dinov2']},ids,ks=(5,))(brain)
  clip=vals[('row_spearman','clip')]; dino=vals[('row_spearman','dinov2')]; mean=(clip+dino)/2
  pd.DataFrame({'image_id':ids,'clip_row_spearman':clip,'dinov2_row_spearman':dino,'mean_models_row_spearman':mean}).to_csv(OUT/'disagreement'/f'{sub}_scores.csv',index=False)
  order=np.lexsort((ids,-mean)); pd.DataFrame({'rank':np.arange(1,201),'image_id':ids[order],'clip_score':clip[order],'dinov2_score':dino[order],'mean_score':mean[order]}).to_csv(OUT/'disagreement'/f'{sub}_ranking.csv',index=False)
  score_rows.append({'subject':sub,'clip_mean':float(clip.mean()),'dinov2_mean':float(dino.mean()),'mean_models_mean':float(mean.mean())})
 corr=[]
 for i,a in enumerate(SUBJECTS):
  va=pd.read_csv(OUT/'disagreement'/f'{a}_scores.csv').mean_models_row_spearman.to_numpy()
  for b in SUBJECTS[i+1:]:
   vb=pd.read_csv(OUT/'disagreement'/f'{b}_scores.csv').mean_models_row_spearman.to_numpy(); corr.append({'subject_a':a,'subject_b':b,'spearman_mean_disagreement':float(spearmanr(va,vb).statistic)})
 pd.DataFrame(score_rows).to_csv(OUT/'disagreement/subject_summary.csv',index=False); pd.DataFrame(corr).to_csv(OUT/'disagreement/subject_correlation.csv',index=False)
 checks={'config_sha256':sha('configs/v21_disagreement_frozen.yaml'),'image_ids_sha256':sha(BASE/'sub01/image_ids.npy'),'subjects':{}}
 for sub in SUBJECTS:
  b=np.load(OUT/'brain_rdms'/f'{sub}.npy',allow_pickle=False); checks['subjects'][sub]={'feature_shape':list(np.load(OUT/'brain_features'/f'{sub}.npy',allow_pickle=False).shape),'brain_rdm_valid':bool(np.isfinite(b).all() and np.allclose(b,b.T,atol=1e-10) and np.allclose(np.diag(b),0))}
 for model in ['clip','dinov2']:
  m=np.load(OUT/'model_rdms'/f'{model}.npy',allow_pickle=False); checks[model+'_rdm_valid']=bool(np.isfinite(m).all() and np.allclose(m,m.T,atol=1e-10) and np.allclose(np.diag(m),0))
 (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
if __name__=='__main__': main()
