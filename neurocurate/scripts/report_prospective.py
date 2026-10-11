"""Report every prospective subject and the frozen expansion decisions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from neurocurate.lowdisk import sha256

root=Path('results/real_phase0/prospective_03_05')
frozen=json.loads((root/'frozen_protocol.json').read_text());report=json.loads((root/'brain_specific_nulls.json').read_text())
reliability=pd.read_csv(root/'subject_reliability.csv');rdm=pd.read_csv(root/'cross_subject_rdm.csv');scores=pd.read_csv(root/'cross_subject_disagreement.csv')
primary=scores[(scores.method=='row_spearman')&(scores.model=='mean_models')]
cohort=report['replication_cohort']
lines=['# Phase 1.6 — Prospective replication on subjects 03–05','',
 'This is a prospective replication of mean-model row-rank disagreement discovered using subjects 01–02, not an original preregistration. All secondary metrics/models and budgets are retained. No new scoring definition, preprocessing choice, checkpoint or budget was introduced.',
 '',f"Protocol snapshot Git commit: `{frozen['snapshot_git_commit']}`; parent checkout commit: `{frozen['base_git_commit']}`. Snapshot SHA256: `{frozen['snapshot_sha256']}`. Exact source/config hashes and timestamp are in `frozen_protocol.json`; the immutable Git object contains `protocol_snapshot.tar`. The branch and checkout were not replaced. This snapshot preceded new-subject network access and EEG inspection.",
 '', '## Acquisition, schema and reliability', '',
 'Only the official standard subject-03/04/05 archives were requested, with fresh bounded official OSF metadata checks against independently recorded trusted hash/size pins and full downloaded ZIP SHA256 verification. Standard ZIPs contain both test and training members; only the test member was extracted or processed. No separate training EEG or 63-channel archive was acquired. Verified archives, extracted object arrays and trusted metadata are preserved. Previously verified images and image metadata are reused.',
 '', 'Conversion uses the unchanged approved UID-1000, descriptor-only, default-deny seccomp child, with no filesystem opens, networking or process creation after confinement; all three denial probes run before pickle loading. Resource limits: 4 GiB address space, CPU 120 seconds, file size 512 MiB, parent timeout 150 seconds. All subsequent loads use allow_pickle=False.',
 '', '| Subject | Cohort | Status | Brain split-half median | 95% split interval | Reliability p | Within primary score-rank median |',
 '|---|---|---|---:|---|---:|---:|']
for _,r in reliability.iterrows():
 if r.status=='verified':lines.append(f"| {r.subject} | {r.cohort} | verified | {r.brain_spearman_median:.4f} | [{r.brain_q025:.4f}, {r.brain_q975:.4f}] | {r.brain_permutation_p:.6f} | {r.within_primary_rank_median:.4f} |")
 else:lines.append(f'| {r.subject} | replication | failed/missing; see subject_status.json | — | — | — | — |')
lines += ['', 'Successful new subjects have actual float64 EEG (200,80,17,100), 80 repetitions, identical 17 channel names/order and 100-Hz times −0.20 to 0.79 s. Every numeric condition maps to the exact existing image path; all 200 ordered IDs match discovery. Each subject has a shape report, mapping audit, 680-dimensional features, Brain RDM and split-half outputs. Failures are retained, never replaced or silently excluded; incomplete cohorts cannot receive a prospective GO decision.',
 '', 'Brain features use the unchanged [100,500) ms window and all 17 supplied posterior channels, no added baseline, bins or standardization. Brain RDM correlation, model RDM cosine and pinned DINOv2/CLIP embeddings are identical to discovery. Thirty 40/40 repetition splits use seed 20261011; split intervals describe variability, not independent sessions or population confidence intervals.',
 '', '## Pairwise Brain RDM and disagreement results', '',
 '| Pair | Group | Brain RDM ρ | Primary score ρ | Shared-brain null median | Shared p | Partial rank correlation | Partial shared p |',
 '|---|---|---:|---:|---:|---:|---:|---:|']
for _,r in primary.iterrows():
 b=rdm[(rdm.subject_a==r.subject_a)&(rdm.subject_b==r.subject_b)].iloc[0]
 lines.append(f'| {r.subject_a}–{r.subject_b} | {r.group} | {b.brain_spearman:.4f} | {r.rank_spearman:.4f} | {r.shared_null_median:.4f} | {r.shared_p:.6f} | {r.partial_rank_correlation:.4f} | {r.partial_shared_p:.6f} |')
lines += ['', 'Pairwise correlations share subjects and images and are dependent. No pair-based t-test, standard error or population confidence interval is calculated. Discovery pair 01–02, the three new-new pairs, discovery-new pairs, and all-five descriptive results are distinguished.', '']
for group in ['discovery_pair','new_new','discovery_new']:
 sub=primary[primary.group==group];b=rdm[rdm.group==group]
 if len(sub):lines.append(f"- {group}: {len(sub)} dependent pairs; descriptive Brain median {b.brain_spearman.median():.4f}, primary ranking median {sub.rank_spearman.median():.4f}, partial median {sub.partial_rank_correlation.median():.4f}.")
lines += [f'- All available five-subject pairs: descriptive Brain median {rdm.brain_spearman.median():.4f}, primary rank median {primary.rank_spearman.median():.4f}; this full-pool description includes discovery and is not a new-subject confirmatory test.',
 '', '## Frozen brain-specific and residual controls', '',
 'Independent-brain controls draw one independent assignment for each of all five subjects per permutation; each Brain RDM is relabeled on BOTH axes and every score is recomputed against fixed model RDMs. All five assignments are generated even if a subject fails. Shared-brain controls use one identical assignment for every subject, preserving between-brain geometry while disrupting model correspondence. These are not score-label shuffles. The shared-null excess is not a causal decomposition.',
 '', 'Seeds/counts: 1,000 joint independent draws at 2026101501; 1,000 shared permutations at 2026101502; 1,000 tie-aware score-label permutations at 2026101503. The shared draws match Phase 1.5; independent draws extend the same seed to coherent five-subject assignments, so discovery independent Monte Carlo samples differ from the earlier two-subject run. Definitions and counts remain frozen.',
 '', 'Model-only covariates and residualization are unchanged: DINOv2–CLIP row disagreement; each model nearest-20 mean cosine distance; each model all-nonself mean cosine distance; ranked/standardized controls plus intercept. Partial rank association is Pearson correlation of residualized score ranks, not a predictive model. The fixed model-control projection is reapplied to every null score. In-sample explained rank variance is descriptive only.']
if cohort['complete']:
 for key in ['primary_independent','primary_shared','partial_independent','partial_shared']:
  v=cohort[key];lines.append(f"- New-new joint cohort {key}: observed median {v['observed']:.4f}, null median {v['null_median']:.4f}, 95% null interval [{v['null_q025']:.4f}, {v['null_q975']:.4f}], p={v['p']:.6f}.")
 lines.append(f"- Prospective primary Holm-adjusted p-values (full ranking, partial ranking): {cohort['primary_holm_p']}. The null statistic takes the median of the three dependent new-new pairs for EACH joint permutation; it does not treat pairs as independent replicates.")
else:lines.append('The prospective cohort is incomplete; no complete-cohort confirmatory result or GO decision is claimed.')
lines += ['', '## Selection-budget replication', '',
 'All five budgets and every method/model/subject are reported in selection_budget_replication.csv. Within-subject medians and split quantiles use disjoint repetition halves. Cross-subject full-score values have separate half-RDM variability summaries; those are not confidence intervals for full 80-repetition scores. Nulls reapply the exact fixed-ID tie rule after relabeling/recomputation. Negative median-null-adjusted overlap is retained. This adjustment is (observed−median null)/(1−median null), not mean-chance kappa.', '',
 '| Budget | New-new median observed overlap | Shared joint-null median | Shared joint-null 95% interval | Shared p | Holm p across five budgets |',
 '|---|---:|---:|---|---:|---:|']
if cohort['complete']:
 for i,budget in enumerate(frozen['config']['budgets']):
  v=cohort['budgets'][str(budget)]['shared'];lines.append(f"| {budget} | {v['observed']:.4f} | {v['null_median']:.4f} | [{v['null_q025']:.4f}, {v['null_q975']:.4f}] | {v['p']:.6f} | {cohort['budget_holm_p'][i]:.6f} |")
lines += ['', 'No optimal budget is selected. High whole-ranking correlation does not imply a stable narrow shortlist. Inspect every new-subject pair, model variant, split interval and independent/shared null in the CSV/JSON, including unfavorable findings. Cohort budget statistics use a median per joint draw, preserving dependence among pairs.', '', '## Frozen decisions', '']
for key,label in [('A','Extend to all ten subjects'),('B','Investigate reliability-aware scoring separately'),('C','Design an independent-pool selection experiment')]:lines.append(f"- **{key}. {label}: {report['decisions'][key]}.**")
lines += ['', 'The decision rules were frozen before acquisition: A requires all three subjects verified, reliable Brain RDMs, all new-new primary correlations positive, and both primary Holm p≤0.05. B requires all three verified reliable datasets and authorizes only a separate exploratory investigation. C requires A and at least one Holm-adjusted cohort budget excess; it authorizes study design only, with an independent stimulus pool and independently fitted preprocessing. No new reliability-aware score was implemented, no next subject was downloaded, and no training or data-efficiency experiment was executed.',
 '', '## Limitations and reproducibility', '',
 'All EEG findings are conditional on the official shared MVNN whitening, which combined training/test partitions and repetitions during covariance fitting. Random disjoint halves are not independently preprocessed or session-independent. Raw event roundtrip/session labels and training-pixel alias audits are unavailable. Image-label permutations can disrupt image-specific confounds as well as genuine alignment; five model covariates do not exhaust visual/semantic confounds. Discovery-informed candidate selection is explicitly acknowledged; prospective replication does not remove preprocessing limitations.',
 '', 'Only the two prospective cohort primary tests and five cohort budget tests receive their predeclared Holm corrections. All other method/model/pair p-values are secondary, exploratory and unadjusted with Monte Carlo resolution 1/1001. Population generalization, causality, classifier usefulness and improved data efficiency are not established. The original 200-image test split must remain protected from future training decisions.',
 '', 'Exact full/half scores, joint assignments and raw null statistics are retained in pickle-free NPZs. frozen_protocol.json pins Git snapshot, exact sources/config, dependency versions and failure policy. Per-subject acquisition manifests pin official metadata, full ZIP/test-member hashes and confinement/export evidence. analysis_provenance.json and verified_outputs.json provide input/output checksums. See VALIDATION.md and test_results.json for executed commands and tests.', '']
(root/'scientific_summary.md').write_text('\n'.join(lines))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(14,11));subjects=['sub01','sub02','sub03','sub04','sub05'];mapping={s:i for i,s in enumerate(subjects)}
for ax,table,column,title in [(axes[0,0],rdm,'brain_spearman','Brain RDM Spearman'),(axes[0,1],primary,'rank_spearman','Primary row-rank score Spearman')]:
 matrix=np.full((5,5),np.nan);np.fill_diagonal(matrix,1)
 for _,r in table.iterrows():matrix[mapping[r.subject_a],mapping[r.subject_b]]=matrix[mapping[r.subject_b],mapping[r.subject_a]]=r[column]
 plot=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis');fig.colorbar(plot,ax=ax);ax.set(xticks=range(5),yticks=range(5),xticklabels=subjects,yticklabels=subjects,title=title)
 for i in range(5):
  for j in range(5):
   if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]<.7 else 'black')
valid=reliability[reliability.status=='verified'];axes[1,0].bar(valid.subject,valid.brain_spearman_median,color=['#888888' if s in ['sub01','sub02'] else '#2978a0' for s in valid.subject])
axes[1,0].errorbar(valid.subject,valid.brain_spearman_median,yerr=[valid.brain_spearman_median-valid.brain_q025,valid.brain_q975-valid.brain_spearman_median],fmt='none',color='black',capsize=4)
axes[1,0].set(title='Brain split-half reliability: empirical split variability',ylim=(0,1),ylabel='Median Spearman')
if cohort['complete']:
 budgets=frozen['config']['budgets'];observed=[cohort['budgets'][str(b)]['shared']['observed'] for b in budgets]
 axes[1,1].plot(budgets,observed,'o-',label='New-new observed median',color='black')
 for kind,color in [('shared','#a34c82'),('independent','#2978a0')]:
  records=[cohort['budgets'][str(b)][kind] for b in budgets]
  axes[1,1].plot(budgets,[v['null_median'] for v in records],'--',label=f'Joint {kind} brain null',color=color)
  axes[1,1].fill_between(budgets,[v['null_q025'] for v in records],[v['null_q975'] for v in records],color=color,alpha=.15)
 axes[1,1].legend();axes[1,1].set(xticks=budgets,ylim=(0,1),title='Primary new-subject selection budgets',xlabel='Budget / 200 images',ylabel='Median pairwise overlap fraction')
else:axes[1,1].text(.5,.5,'Incomplete prospective cohort',ha='center');axes[1,1].axis('off')
fig.suptitle('Prospective subjects 03–05 replication; discovery 01–02 shown separately\nConditional on shared official MVNN; pairwise observations are dependent')
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(root/'replication_figures.png',dpi=160);plt.close(fig)
paths=[p for p in root.iterdir() if p.is_file() and p.name not in ['verified_outputs.json','analysis.log']]
(root/'verified_outputs.json').write_text(json.dumps(dict(outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in paths}),indent=2)+'\n')
print(root/'scientific_summary.md')
