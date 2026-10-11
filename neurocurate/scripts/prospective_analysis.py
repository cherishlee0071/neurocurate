"""Frozen five-subject analyses with prospective, joint new-subject null tests."""
import json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml
from neurocurate.ablation import DisagreementMetrics
from neurocurate.brain_controls import (normalized_score_ranks,RankControls,model_only_scores,
    selection_statistics,overlap_budgets,chance_adjusted)
from neurocurate.lowdisk import sha256
from neurocurate.rdm import compute_rdm,validate_rdm
from neurocurate.realdata import validate_test_order
from neurocurate.vision import load_feature_cache
from cross_subject_analysis import rdm_comparison

BASE=Path('results/real_phase0');OUT=BASE/'prospective_03_05'
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def interval(x):return dict(median=float(np.median(x)),q025=float(np.quantile(x,.025)),q975=float(np.quantile(x,.975)))
def null_summary(value,null):
 return dict(observed=float(value),**{f'null_{k}':v for k,v in interval(null).items()},p=float((1+np.sum(null>=value))/(len(null)+1)))
def holm(values):
 values=np.asarray(values);order=np.argsort(values);adjusted=np.zeros(len(values));running=0
 for rank,i in enumerate(order):running=max(running,min(1,float(values[i])*(len(values)-rank)));adjusted[i]=running
 return adjusted.tolist()

def main():
 frozen=json.loads((OUT/'frozen_protocol.json').read_text());cfg=frozen['config']
 for path,digest in frozen['files_sha256'].items():assert sha256(path)==digest,path
 assert sha256(OUT/'protocol_snapshot.tar')==frozen['snapshot_sha256']
 assert sha256('configs/brain_specific_controls.yaml')==cfg['phase15_config_sha256']
 oldcfg=yaml.safe_load(Path('configs/brain_specific_controls.yaml').read_text())
 reference=yaml.safe_load(Path('configs/cross_subject_01_02.yaml').read_text())
 assert cfg['budgets']==oldcfg['selection_budgets']
 for key in ['permutations','brain_independent_seed','brain_shared_seed','score_label_seed','repetition_split_seed','repetition_splits']:assert cfg[key]==oldcfg[key]
 for folder in ['cross_subject_01_02','brain_specific_controls']:
  manifest=json.loads((BASE/folder/'verified_outputs.json').read_text())
  for name,pin in manifest['outputs'].items():assert sha256(BASE/folder/name)==pin['sha256']
 statuses={};subjects=[];brains=[];half_rdms=[];reliability=[];input_hashes={}
 ids=np.load(BASE/'sub01/image_ids.npy',allow_pickle=False);times_ref=np.load(BASE/'sub01/pickle_free/times.npy',allow_pickle=False)
 channels_ref=np.load(BASE/'sub01/pickle_free/channels.npy',allow_pickle=False)
 prng=np.random.default_rng(cfg['brain_shared_seed']);shared=np.array([prng.permutation(200) for _ in range(1000)])
 rng=np.random.default_rng(cfg['repetition_split_seed']);halves=[]
 for _ in range(30):
  p=rng.permutation(80);halves.append([np.sort(p[:40]),np.sort(p[40:])])
 mask=(times_ref>=.1-1e-9)&(times_ref<.5-1e-9)
 assert reference['preprocessing']==dict(time_window=[.1,.5],baseline=None,channels=None,temporal_bins=None,min_repetitions=40,standardize=False)
 for subject in cfg['discovery_subjects']+cfg['replication_subjects']:
  new=subject in cfg['replication_subjects'];root=OUT/subject if new else BASE/subject
  status=json.loads((root/'subject_status.json').read_text()) if new and (root/'subject_status.json').exists() else dict(subject=subject,status='missing' if new else 'discovery_verified')
  statuses[subject]=status
  try:
   if new:assert status['status']=='verified_numeric_export',status.get('error','not acquired/verified')
   exports=root/'pickle_free';assert (exports/'VERIFIED').exists()
   manifest=json.loads((exports/'export_manifest.json').read_text())
   for name,digest in manifest['exports'].items():assert sha256(exports/name)==digest
   assert np.array_equal(np.load(exports/'times.npy',allow_pickle=False),times_ref)
   assert np.array_equal(np.load(exports/'channels.npy',allow_pickle=False),channels_ref)
   metadata=json.loads((exports/'metadata.json').read_text());alignment=validate_test_order(metadata,Path('data/local/real_phase0/stimuli'))
   mapped=np.array([f'test_images/{c}/{f}' for c,f in zip(metadata['test_img_concepts'],metadata['test_img_files'])]);assert np.array_equal(ids,mapped)
   raw=np.load(exports/'eeg.npy',allow_pickle=False,mmap_mode='r');assert raw.shape==(200,80,17,100) and raw.dtype==np.float64 and np.isfinite(raw).all()
   def features(indices):return np.asarray(raw[:,indices]).mean(axis=1)[:,:,mask].reshape(200,-1)
   features_full=features(np.arange(80));brain=validate_rdm(compute_rdm(features_full,reference['brain_metric']))
   if new:
    np.save(root/'eeg_features.npy',features_full,allow_pickle=False);np.save(root/'brain_rdm.npy',brain,allow_pickle=False);np.save(root/'image_ids.npy',ids,allow_pickle=False)
    write(root/'eeg_shape_report.json',dict(shape=list(raw.shape),dtype=str(raw.dtype),channels=channels_ref.tolist(),sampling_rate_hz=float(1/np.median(np.diff(times_ref))),times_seconds=times_ref.tolist(),repetitions=80,nonfinite_values=0,identical_subject01_schema=True))
    write(root/'image_alignment_report.json',alignment)
    matrices=[];rhos=[];nulls=[]
    for i,pair in enumerate(halves):
     rdms=[validate_rdm(compute_rdm(features(indices))) for indices in pair]
     rho,null=rdm_comparison(*rdms,shared);matrices.append(rdms);rhos.append(rho);nulls.append(null)
     if (i+1)%10==0:print(subject,'reliability splits',i+1,flush=True)
    matrices=np.array(matrices);reliability_report=dict(statistics=interval(rhos),permutation=null_summary(np.median(rhos),np.median(nulls,axis=0)),split_spearman=rhos,conditional_on_shared_preprocessing=True,session_independent=False)
    write(root/'brain_reliability.json',reliability_report)
    np.savez_compressed(root/'half_rdms.npz',rdms=matrices,image_ids=ids,repetition_indices=np.array(halves))
   else:
    assert np.allclose(brain,np.load(root/'brain_rdm.npy',allow_pickle=False),atol=1e-12)
    with np.load(BASE/'cross_subject_01_02'/f'{subject}_half_rdms.npz',allow_pickle=False) as cache:
     matrices=cache['rdms'];assert np.array_equal(cache['repetition_indices'],halves)
    previous=json.loads((BASE/'cross_subject_01_02/cross_subject_rdm.json').read_text())['within_subject_repetition_reliability'][subject]
    st=previous['statistics'];lo,hi=st['random_split_95_percentile_interval'];pp=previous['permutation']
    reliability_report=dict(statistics=dict(median=st['median'],q025=lo,q975=hi),permutation=dict(p=pp['one_sided_monte_carlo_p']),split_spearman=previous['split_spearman'])
   subjects.append(subject);brains.append(brain);half_rdms.append(matrices)
   stats=reliability_report['statistics'];reliability.append(dict(subject=subject,cohort='replication' if new else 'discovery',status='verified',shape='200,80,17,100',repetitions=80,brain_spearman_median=stats['median'],brain_q025=stats['q025'],brain_q975=stats['q975'],brain_permutation_p=reliability_report['permutation']['p']))
   input_hashes[str(exports/'export_manifest.json')]=sha256(exports/'export_manifest.json')
   status['analysis_status']='verified';del raw
  except Exception as error:
   if not new:raise
   status.update(analysis_status='failed',analysis_error_type=type(error).__name__,analysis_error=str(error))
   reliability.append(dict(subject=subject,cohort='replication',status='failed_or_missing',error=str(error)))
 write(OUT/'subject_status.json',statuses)
 models={}
 for name,revision in reference['vision_revisions'].items():
  path=BASE/'sub01'/f'{name}_features.npz';cached,embedding,provenance=load_feature_cache(path,split='sanity',model_name=name)
  assert np.array_equal(ids,cached) and provenance['resolved_revision']==revision
  for image_id in ids:assert sha256(Path('data/local/real_phase0/stimuli')/image_id)==provenance['image_sha256'][image_id]
  models[name]=validate_rdm(compute_rdm(embedding,'cosine'));assert np.allclose(models[name],np.load(BASE/'sub01'/f'{name}_rdm.npy',allow_pickle=False))
  input_hashes[str(path)]=sha256(path)
 keys=[(m,v) for m in oldcfg['methods'] for v in oldcfg['models']];names=['/'.join(key) for key in keys];primary=names.index(cfg['primary'])
 metric=DisagreementMetrics(models,ids,[5,10,20])
 def score(matrix):
  values=metric(matrix);return np.column_stack([values[key] for key in keys])
 scores=np.array([score(b) for b in brains]);ranks=[normalized_score_ranks(s) for s in scores]
 modelrow=DisagreementMetrics({'clip':models['clip']},ids,[20])(models['dinov2'])[('row_spearman','clip')]
 controls=model_only_scores(models,ids,modelrow);covariates=np.column_stack([controls[k] for k in oldcfg['model_only_controls']]);projection=RankControls(covariates)
 residual=[];r_squared=[]
 for values in scores:
  r,r2=projection.residuals(values);residual.append(r);r_squared.append(r2)
 pairs=list(itertools.combinations(range(len(subjects)),2));new_pairs=[p for p,(a,b) in enumerate(pairs) if subjects[a] in cfg['replication_subjects'] and subjects[b] in cfg['replication_subjects']]
 def group(a,b):
  if subjects[a] in cfg['discovery_subjects'] and subjects[b] in cfg['discovery_subjects']:return 'discovery_pair'
  if subjects[a] in cfg['replication_subjects'] and subjects[b] in cfg['replication_subjects']:return 'new_new'
  return 'discovery_new'
 observed=np.array([np.sum(ranks[a]*ranks[b],axis=0) for a,b in pairs]);partial=np.array([np.sum(residual[a]*residual[b],axis=0) for a,b in pairs])
 triangle=np.triu_indices(200,1)
 rdm_rows=[dict(subject_a=subjects[a],subject_b=subjects[b],group=group(a,b),brain_spearman=float(spearmanr(brains[a][triangle],brains[b][triangle]).statistic)) for a,b in pairs]
 pd.DataFrame(rdm_rows).to_csv(OUT/'cross_subject_rdm.csv',index=False)
 count=1000;budgets=cfg['budgets'];nscore=len(keys)
 irng=np.random.default_rng(cfg['brain_independent_seed']);all_subjects=cfg['discovery_subjects']+cfg['replication_subjects']
 # Generate all five assignments even if a subject fails; missingness cannot
 # change other subjects' RNG streams or redefine the prospective cohort.
 independent=np.array([[irng.permutation(200) for _ in all_subjects] for _ in range(count)])
 null_rho=np.zeros((2,count,len(pairs),nscore));null_partial=np.zeros_like(null_rho);null_overlap=np.zeros((2,count,len(pairs),nscore,len(budgets)))
 for mode in [0,1]:
  for i in range(count):
   assigned=[independent[i,all_subjects.index(s)] if mode==0 else shared[i] for s in subjects]
   values=[score(b[np.ix_(p,p)]) for b,p in zip(brains,assigned)]
   rr=[normalized_score_ranks(v) for v in values];res=[projection.residuals(v)[0] for v in values]
   for p,(a,b) in enumerate(pairs):
    null_rho[mode,i,p]=np.sum(rr[a]*rr[b],axis=0);null_partial[mode,i,p]=np.sum(res[a]*res[b],axis=0)
    for j in range(nscore):null_overlap[mode,i,p,j]=overlap_budgets(values[a][:,j],values[b][:,j],ids,budgets)
   if (i+1)%100==0:print(['independent','shared'][mode],'joint brain permutations',i+1,flush=True)
 half_scores=np.array([[[score(matrix) for matrix in pair] for pair in matrices] for matrices in half_rdms])
 lrng=np.random.default_rng(cfg['score_label_seed']);label_permutations=np.array([lrng.permutation(200) for _ in range(count)])
 budget_rows=[];within_rows=[]
 for s,subject in enumerate(subjects):
  for j,(method,model) in enumerate(keys):
   split_rho=[float(spearmanr(pair[0,:,j],pair[1,:,j]).statistic) for pair in half_scores[s]]
   within_rows.append(dict(subject=subject,method=method,model=model,**interval(split_rho)))
   if j==primary:next(r for r in reliability if r['subject']==subject)['within_primary_rank_median']=float(np.median(split_rho))
   overlaps=[];nulls=[]
   for pair in half_scores[s]:
    o,n=selection_statistics(pair[0,:,j],pair[1,:,j],ids,label_permutations,budgets);overlaps.append(o);nulls.append(n)
   overlaps=np.array(overlaps);aggregate=np.median(nulls,axis=0)
   for k,budget in enumerate(budgets):
    obs=float(np.median(overlaps[:,k]));stats=null_summary(obs,aggregate[:,k]);q=np.quantile(overlaps[:,k],[.025,.975])
    budget_rows.append(dict(subject_a=subject,subject_b=subject,group='within_subject',method=method,model=model,budget=budget,
     overlap_fraction=obs,split_q025=q[0],split_q975=q[1],null_type='score_labels_tie_aware',chance_adjusted=float(chance_adjusted(obs,stats['null_median'])),**stats))
  print(subject,'within-subject budgets complete',flush=True)
 cross_rows=[];report_pairs={};observed_overlap=np.zeros((len(pairs),nscore,len(budgets)))
 for p,(a,b) in enumerate(pairs):
  label=f'{subjects[a]}_{subjects[b]}';report_pairs[label]={}
  for j,(method,model) in enumerate(keys):
   ind=null_summary(observed[p,j],null_rho[0,:,p,j]);sh=null_summary(observed[p,j],null_rho[1,:,p,j])
   pi=null_summary(partial[p,j],null_partial[0,:,p,j]);ps=null_summary(partial[p,j],null_partial[1,:,p,j])
   cross_rows.append(dict(subject_a=subjects[a],subject_b=subjects[b],group=group(a,b),method=method,model=model,
    rank_spearman=observed[p,j],partial_rank_correlation=partial[p,j],independent_null_median=ind['null_median'],independent_p=ind['p'],
    shared_null_median=sh['null_median'],shared_p=sh['p'],partial_shared_null_median=ps['null_median'],partial_shared_p=ps['p']))
   report_pairs[label][names[j]]=dict(independent=ind,shared=sh,partial_independent=pi,partial_shared=ps)
   obs,score_null=selection_statistics(scores[a,:,j],scores[b,:,j],ids,label_permutations,budgets);observed_overlap[p,j]=obs
   halves_overlap=np.array([overlap_budgets(half_scores[a,i,h,:,j],half_scores[b,i,h,:,j],ids,budgets) for i in range(30) for h in range(2)])
   for k,budget in enumerate(budgets):
    for mode,nulltype in [(-1,'score_labels_tie_aware'),(0,'brain_independent'),(1,'brain_shared')]:
     null=score_null[:,k] if mode==-1 else null_overlap[mode,:,p,j,k];stats=null_summary(obs[k],null)
     budget_rows.append(dict(subject_a=subjects[a],subject_b=subjects[b],group=group(a,b),method=method,model=model,budget=budget,
      overlap_fraction=obs[k],half_median=np.median(halves_overlap[:,k]),half_q025=np.quantile(halves_overlap[:,k],.025),half_q975=np.quantile(halves_overlap[:,k],.975),
      null_type=nulltype,chance_adjusted=float(chance_adjusted(obs[k],stats['null_median'])),**stats))
 pd.DataFrame(reliability).to_csv(OUT/'subject_reliability.csv',index=False);pd.DataFrame(within_rows).to_csv(OUT/'within_rank_stability.csv',index=False)
 pd.DataFrame(cross_rows).to_csv(OUT/'cross_subject_disagreement.csv',index=False);pd.DataFrame(budget_rows).to_csv(OUT/'selection_budget_replication.csv',index=False)
 complete=len(new_pairs)==3 and all(statuses[s].get('analysis_status')=='verified' for s in cfg['replication_subjects'])
 cohort=dict(complete=complete,statistic=cfg['cohort_statistic'])
 if complete:
  o=float(np.median(observed[new_pairs,primary]));op=float(np.median(partial[new_pairs,primary]))
  joint=np.median(null_rho[:,:,new_pairs,primary],axis=2);jp=np.median(null_partial[:,:,new_pairs,primary],axis=2)
  cohort.update(primary_independent=null_summary(o,joint[0]),primary_shared=null_summary(o,joint[1]),
   partial_independent=null_summary(op,jp[0]),partial_shared=null_summary(op,jp[1]))
  cohort['primary_holm_p']=holm([cohort['primary_shared']['p'],cohort['partial_shared']['p']])
  cohort['budgets']={};budgetp=[]
  for k,budget in enumerate(budgets):
   ov=float(np.median(observed_overlap[new_pairs,primary,k]));null=np.median(null_overlap[:,:,new_pairs,primary,k],axis=2)
   cohort['budgets'][str(budget)]=dict(independent=null_summary(ov,null[0]),shared=null_summary(ov,null[1]));budgetp.append(cohort['budgets'][str(budget)]['shared']['p'])
  cohort['budget_holm_p']=holm(budgetp)
 verified=complete and all(next(r for r in reliability if r['subject']==s)['brain_permutation_p']<=.05 for s in cfg['replication_subjects'])
 go_a=verified and all(observed[p,primary]>0 for p in new_pairs) and max(cohort.get('primary_holm_p',[1]))<=.05
 go_c=go_a and min(cohort.get('budget_holm_p',[1]))<=.05
 decisions=dict(A='GO' if go_a else ('NO-GO' if complete else 'INCOMPLETE / NO-GO'),B='GO, separate exploratory investigation only' if verified else 'NO-GO',C='GO, design only with independent pool/preprocessing' if go_c else 'NO-GO')
 write(OUT/'brain_specific_nulls.json',dict(frozen_protocol_sha256=sha256(OUT/'frozen_protocol.json'),pairs=report_pairs,replication_cohort=cohort,decisions=decisions,
  model_controls=dict(names=oldcfg['model_only_controls'],design_rank=projection.rank,condition_number=projection.condition_number,
   in_sample_rank_r_squared={s:dict(zip(names,r_squared[i].tolist())) for i,s in enumerate(subjects)}),
  limitations='Shared MVNN; not session-independent. Pairwise correlations dependent. Quantiles are split variability or null envelopes, not population CI. Secondary p-values unadjusted.'))
 np.savez_compressed(OUT/'null_statistics.npz',subjects=np.array(subjects),pairs=np.array(pairs),score_names=np.array(names),budgets=np.array(budgets),
  independent_permutations=independent,shared_permutations=shared,rank_null=null_rho,partial_null=null_partial,overlap_null=null_overlap)
 np.savez_compressed(OUT/'observed_scores.npz',subjects=np.array(subjects),image_ids=ids,score_names=np.array(names),scores=scores,half_scores=half_scores,covariates=covariates,
  cross_rank=observed,cross_partial=partial,cross_overlap=observed_overlap)
 write(OUT/'analysis_provenance.json',dict(input_sha256=input_hashes,frozen_protocol_sha256=sha256(OUT/'frozen_protocol.json'),
  no_training=True,pickle_free_downstream=True,complete=complete,verified_subjects=subjects))
 print('COHORT',json.dumps(cohort,indent=2),flush=True);print('DECISIONS',decisions,flush=True)
if __name__=='__main__':main()
