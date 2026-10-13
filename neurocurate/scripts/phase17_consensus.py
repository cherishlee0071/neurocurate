"""Exploratory Phase 1.7 consensus and shortlist diagnostics."""
from pathlib import Path
import json, hashlib, subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE=Path('results/real_phase0'); P16=BASE/'prospective_03_05'; OUT=BASE/'phase17_consensus'
SUBJECTS=['sub01','sub02','sub03','sub04','sub05']; BUDGETS=[10,20,40,60,100]; PRIMARY='row_spearman/mean_models'; SEED=20261017; N_PERM=1000
def ranks(x): return np.argsort(np.argsort(-x,axis=-1),axis=-1)
def rho(a,b): return float(spearmanr(a,b).statistic)
def quant(x): return {'median':float(np.median(x)),'q025':float(np.quantile(x,.025)),'q975':float(np.quantile(x,.975))}
def pval(v,n): return float((1+np.sum(n>=v))/(len(n)+1))
def pick(x,k): return set(np.argsort(-x,kind='stable')[:k].tolist())
def main():
 OUT.mkdir(parents=True,exist_ok=True); z=np.load(P16/'observed_scores.npz',allow_pickle=False)
 subjects=z['subjects'].tolist(); names=z['score_names'].tolist(); scores=z['scores']; halves=z['half_scores']; primary=names.index(PRIMARY)
 manifest={'phase':'1.7','exploratory':True,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'phase16_protocol_sha256':hashlib.sha256((P16/'frozen_protocol.json').read_bytes()).hexdigest(),'seed':SEED,'permutations':N_PERM,'primary':PRIMARY,'subjects':subjects,'budgets':BUDGETS,'no_training':True}
 (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 rr=ranks(scores[:,:,primary]); observed=[]
 for held in range(5): observed.append(rho(-rr[held],-np.mean(rr[[i for i in range(5) if i!=held]],axis=0)))
 rng=np.random.default_rng(SEED); indep=np.array([[rng.permutation(200) for _ in range(5)] for _ in range(N_PERM)]); shared=np.array([rng.permutation(200) for _ in range(N_PERM)])
 nullrows=[]
 for mode,perms in [('independent',indep),('shared',np.repeat(shared[:,None,:],5,axis=1))]:
  for held in range(5):
   vals=[]
   for p in range(N_PERM):
    ps=np.array([scores[s,perms[p,s],primary] for s in range(5)]); pr=ranks(ps); vals.append(rho(-pr[held],-np.mean(pr[[i for i in range(5) if i!=held]],axis=0)))
   nullrows.append({'held_out':subjects[held],'null':mode,'observed':observed[held],**quant(vals),'p':pval(observed[held],np.asarray(vals))})
 loso=[]
 for s in range(5):
  row={'held_out':subjects[s],'primary_observed':observed[s]}
  for j,key in enumerate(names):
   r=ranks(scores[:,:,j]); row[key]=rho(-r[s],-np.mean(r[[i for i in range(5) if i!=s]],axis=0))
  loso.append(row)
 pd.DataFrame(loso).to_csv(OUT/'loso_consensus.csv',index=False); (OUT/'consensus_nulls.json').write_text(json.dumps({'manifest':manifest,'primary_loso':nullrows,'folds_dependent':True},indent=2)+'\n')
 freq=[]; margins=[]; rel=[]
 for s,sub in enumerate(subjects):
  full=scores[s,:,primary]; sr=ranks(scores[s,:,primary]); split=halves[s,:,:, :,primary]
  for k in BUDGETS:
   cut=np.sort(full)[-k]; gap=float(cut-np.sort(full)[-k-1]) if k<200 else float('nan'); margins.append({'subject':sub,'budget':k,'cutoff_score':float(cut),'boundary_gap':gap,'ties_at_cutoff':int(np.sum(full==cut)),'near_cutoff_1pct':int(np.sum(np.abs(full-cut)<=.01*max(np.ptp(full),1e-12)))})
   for i in range(200): freq.append({'subject':sub,'image_index':i,'budget':k,'full_selected':int(i in pick(full,k)),'within_split_frequency':float(np.mean([i in pick(split[t,h],k) for t in range(30) for h in range(2)])),'between_subject_frequency':float(np.mean([i in pick(scores[u,:,primary],k) for u in range(5)]))})
  instability=np.mean(np.abs(split[:,0]-split[:,1]),axis=0); var=np.var(split.mean(axis=1),axis=0)
  for i in range(200): rel.append({'subject':sub,'image_index':i,'primary_score':float(full[i]),'score_split_instability':float(instability[i]),'score_split_variance':float(var[i]),'subject_full_score_rank':int(sr[i])})
 pd.DataFrame(freq).to_csv(OUT/'selection_frequency.csv',index=False); pd.DataFrame(margins).to_csv(OUT/'cutoff_margin_analysis.csv',index=False); pd.DataFrame(rel).to_csv(OUT/'reliability_relationship.csv',index=False)
 plan='''# Preprocessing sensitivity plan (Phase 1.7)\n\nNo raw EEG is downloaded in Phase 1.7. Existing results use official shared MVNN preprocessing, so reliability is conditional.\n\nAcquire one subject/session pilot with raw EEG, event triggers, montage, sampling rate, bad-channel annotations, and the verified 200-image manifest. Fit bad-channel detection, line-noise, baseline, whitening/MVNN, and dimensionality reduction on training repetitions or sessions only; apply transforms to held-out repetitions.\n\nUse subjects 01–02, two sessions if available, 40 repetitions per image, 30 fixed splits, 200x200 Brain RDMs and frozen row-rank disagreement. Require event-to-image validation before epochs enter the analysis. Reserve 2–3x the raw input size and at least 16 GB RAM per pilot; keep raw and temporary files outside Git.\n\nAny ranking change is preprocessing sensitivity, not evidence of improved data efficiency. An independent-pool experiment requires a separately fitted pipeline and new stimulus pool.\n'''
 (OUT/'preprocessing_sensitivity_plan.md').write_text(plan)
 summary={'exploratory':True,'primary':PRIMARY,'loso_primary_median':float(np.median(observed)),'loso_primary_values':dict(zip(subjects,observed)),'conclusion':'LOSO and shortlist diagnostics do not establish population generalization or data efficiency.'}
 (OUT/'scientific_summary.md').write_text('# NeuroCurate Phase 1.7\n\n'+json.dumps(summary,indent=2)+'\n\nFolds are dependent; interpret margins, ties, and subject heterogeneity jointly.\n')
if __name__=='__main__': main()
