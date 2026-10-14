"""Finalize P1-A statistics from already generated drift CSVs."""
from pathlib import Path
import json, hashlib, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
BASE=Path('results/real_phase0'); OUT=BASE/'v21_p1a'; SEED=20261018; N_PERM=10000; N_RANDOM=1000
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 results=pd.read_csv(OUT/'drift_results.csv'); rng=np.random.default_rng(SEED); stats=[]; permd={}
 for (model,kind), group in results.groupby(['model','perturbation']): group.to_csv(OUT/'drift'/f'{model}_{kind}.csv',index=False)
 for (model,kind,level),g in results.groupby(['model','perturbation','level'],sort=True):
  top=np.argsort(-g.disagreement.to_numpy(),kind='stable')[:40]; low=np.argsort(g.disagreement.to_numpy(),kind='stable')[:40]
  d=g.drift.to_numpy(); delta=float(d[top].mean()-d[low].mean()); sr=spearmanr(g.disagreement,d); perm=[]; random=[]
  for _ in range(N_PERM):
   p=rng.permutation(200); perm.append(float(d[p[:40]].mean()-d[p[40:80]].mean()))
  for _ in range(N_RANDOM):
   p=rng.permutation(200); random.append(float(d[p[:40]].mean()-d[p[40:80]].mean()))
  perm=np.asarray(perm); random=np.asarray(random); key=f'{model}_{kind}_{level:g}'
  stats.append({'model':model,'perturbation':kind,'level':level,'spearman_rho':float(sr.statistic),'spearman_p':float(sr.pvalue),'high_low_mean_difference':delta,'permutation_p_two_sided':float((1+np.sum(np.abs(perm)>=abs(delta)))/(N_PERM+1)),'permutation_q025':float(np.quantile(perm,.025)),'permutation_q975':float(np.quantile(perm,.975)),'random_baseline_median':float(np.median(random)),'random_baseline_q025':float(np.quantile(random,.025)),'random_baseline_q975':float(np.quantile(random,.975)),'random_baseline_exceedance':float(np.mean(np.abs(random)>=abs(delta)))})
  permd[key]={'observed_delta':delta,'permutation':perm.tolist(),'random_baseline':random.tolist()}
 report=pd.DataFrame(stats); report.to_csv(OUT/'group_statistics.csv',index=False); (OUT/'permutation_tests.json').write_text(json.dumps(permd,indent=2)+'\n')
 fig,axs=plt.subplots(2,5,figsize=(18,7)); conditions=[('blur',1),('blur',2),('blur',4),('crop',.9),('crop',.75)]
 for j,(model,kind,lev) in enumerate([(m,k,l) for m in ['clip','dinov2'] for k,l in conditions]):
  row=j//5; col=j%5; g=results[(results.model==model)&(results.perturbation==kind)&(results.level==lev)]; top=np.argsort(-g.disagreement.to_numpy(),kind='stable')[:40]; low=np.argsort(g.disagreement.to_numpy(),kind='stable')[:40]; axs[row,col].scatter(g.disagreement,g.drift,s=8,alpha=.5); axs[row,col].set_title(f'{model} {kind} {lev:g}'); axs[row,col].set_xlabel('frozen disagreement'); axs[row,col].set_ylabel('cosine drift')
  axs[row,col].text(.03,.96,'high/low Δ %.4f' % (g.drift.to_numpy()[top].mean()-g.drift.to_numpy()[low].mean()),transform=axs[row,col].transAxes,va='top',fontsize=8)
 fig.tight_layout(); fig.savefig(OUT/'figures/disagreement_vs_drift.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,5)); ax.bar(np.arange(len(report)),report.high_low_mean_difference); ax.axhline(0,color='k',lw=.8); ax.set_xticks(np.arange(len(report)),[f"{r.model}\n{r.perturbation}{r.level:g}" for _,r in report.iterrows()],rotation=45,ha='right'); ax.set_ylabel('high-minus-low drift'); fig.tight_layout(); fig.savefig(OUT/'figures/group_comparison.png',dpi=160); plt.close(fig)
 manifest={'phase':'P1-A','exploratory':True,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'seed':SEED,'permutations':N_PERM,'random_baselines':N_RANDOM,'images':200,'image_ids_sha256':sha(BASE/'sub01/image_ids.npy'),'p0_config_sha256':sha('configs/v21_disagreement_frozen.yaml'),'p0_artifact_policy':'read-only; disagreement was not recomputed','models':{'clip':{'revision':'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'},'dinov2':{'revision':'f9e44c814b77203eaa57a6bdbbd535f21ede1415'}},'conditions':[{'family':k,'level':l} for k,l in conditions],'crop_resize':'PIL BICUBIC','group_definition':'top/bottom 20% population consensus disagreement','excluded_controls':['uncertainty','behavior','additional models']}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 lines=['# NeuroCurate v2.1 P1-A Robustness Drift Pilot','', '## Objective', '', 'This exploratory sanity pilot tests whether the frozen population-consensus neural disagreement score is associated with cosine representation drift under fixed blur and crop perturbations. It does not claim model failure, robustness improvement, prediction, or causality.','', '## Frozen inputs', '', 'Only the read-only P0 artifacts under `results/real_phase0/v21_frozen/` were used. Disagreement was not recomputed. The five-subject, 200-image population consensus was formed from the saved subject-level mean-model row-rank scores. No uncertainty, behavior, confidence, or additional model signal entered the score.','', '## Results','']
 for _,r in report.iterrows(): lines.append(f"- {r.model} {r.perturbation} {r.level:g}: rho={r.spearman_rho:.3f}, p={r.spearman_p:.3g}; high-minus-low drift={r.high_low_mean_difference:.5f}; two-sided permutation p={r.permutation_p_two_sided:.3g}; random-baseline exceedance={r.random_baseline_exceedance:.3f}.")
 lines += ['', '## Decision', '', 'NO-GO for the claim that frozen neural disagreement predicts perturbation drift in this pilot. The only nominally strong condition is CLIP crop 0.75, with a negative association (rho=-0.286), opposite to the proposed direction; the remaining conditions are weak, null, or inconsistent. This is a weak/inconclusive diagnostic result, not evidence of predictive validity.', '', 'P1-B uncertainty matching is NO-GO as a follow-on confirmatory experiment. It may be revisited only as a separately frozen exploratory control after preregistering a broader perturbation design and independently auditing preprocessing. Do not claim model vulnerability, causal effects, or generalization.']
 (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
