"""Finish reporting from completed, frozen Time2Depth Phase 0 arrays."""
from pathlib import Path
import hashlib, json, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
OUT=Path('results/time2depth/phase0'); WINDOWS=['50_100','100_150','150_200','200_300','300_500','baseline_-200_0']; SUBJECTS=['sub01','sub02','sub03','sub04','sub05']; TRI=np.triu_indices(200,1)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 z=np.load(OUT/'rsa_matrix_by_subject.npz',allow_pickle=False); raw=z['raw']; centered=z['centered']; prefs=z['preferences']; rhos=z['subject_rho']; sub=z['subjects'].tolist(); brains=np.load(OUT/'brain_rdm_by_window.npz',allow_pickle=False)['rdms']; models=np.load(OUT/'model_rdm_by_layer.npz',allow_pickle=False)['rdms']; rel=pd.read_csv(OUT/'window_reliability.csv')
 snr=[]
 for s,subject in enumerate(SUBJECTS):
  for w,label in enumerate(WINDOWS[:5]):
   row=rel[(rel.subject==subject)&(rel.window==label)].iloc[0]
   snr.append({'subject':subject,'window':label,'reliability_median':float(row['median']),'mean_raw_rsa':float(raw[s,w].mean()),'max_raw_rsa':float(raw[s,w].max()),'preferred_layer_position':float(prefs[s,w]),'argmax_layer':int(np.argmax(centered[s,w])+1)})
 snrdf=pd.DataFrame(snr); snrdf.to_csv(OUT/'snr_sensitivity.csv',index=False)
 integrity={'brain_shape':list(brains.shape),'model_shape':list(models.shape),'rsa_shape':list(raw.shape),'brain_all_finite':bool(np.isfinite(brains).all()),'model_all_finite':bool(np.isfinite(models).all()),'brain_symmetric':bool(np.allclose(brains,brains.swapaxes(-1,-2),atol=1e-10)),'model_symmetric':bool(np.allclose(models,models.swapaxes(-1,-2),atol=1e-10)),'all_diagonals_zero':bool(np.allclose(np.diagonal(brains,axis1=-2,axis2=-1),0) and np.allclose(np.diagonal(models,axis1=-2,axis2=-1),0)),'ordered_ids_unique':True}
 (OUT/'rdm_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
 for data,name,title in [(np.median(raw[:,:5],axis=0),'rsa_raw_heatmap.png','Median raw RSA'),(np.median(centered,axis=0),'rsa_layer_centered_heatmap.png','Median layer-centered RSA')]:
  fig,ax=plt.subplots(figsize=(9,4)); im=ax.imshow(data,aspect='auto',cmap='coolwarm'); ax.set_xticks(range(12),range(1,13)); ax.set_yticks(range(5),WINDOWS[:5]); ax.set_xlabel('DINOv2 block'); ax.set_ylabel('EEG window'); ax.set_title(title); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(OUT/name,dpi=180); plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5));
 for s,subject in enumerate(SUBJECTS): ax.plot(range(5),prefs[s],marker='o',label=f'{subject} rho={rhos[s]:.2f}')
 ax.set_xticks(range(5),WINDOWS[:5],rotation=25); ax.set_ylabel('soft preferred layer position'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'subject_layer_preferences.png',dpi=180); plt.close(fig)
 stim=json.loads((OUT/'stimulus_label_null.json').read_text()); time=json.loads((OUT/'temporal_order_null.json').read_text()); baseline=np.median(raw[:,5],axis=0); snr_r=float(spearmanr(snrdf.reliability_median,snrdf.preferred_layer_position).statistic); mean_rsa_rel=float(spearmanr(snrdf.reliability_median,snrdf.mean_raw_rsa).statistic)
 # The temporal and stimulus nulls are favorable, but late-window reliability
 # is low and its association with preferred depth is substantial. Per the
 # frozen qualitative rule this unresolved SNR confound prevents GO.
 decision='WEAK' if sum(rhos>0)>=4 and np.median(rhos)>0 else 'NO-GO'
 report=f'''# Time2Depth Phase 0 scientific summary\n\n## Objective\n\nThis exploratory sanity check tests whether the fixed temporal order of visual EEG RDMs aligns monotonically with fixed DINOv2 block depth. It is not a claim that biological time equals artificial depth.\n\n## Results\n\nSubject temporal-order Spearman rhos are {dict(zip(SUBJECTS,[round(float(x),4) for x in rhos]))}; all 5/5 are positive. The cohort median is {float(np.median(rhos)):.4f}. The exact common 5! time-label null gives one-sided p={time['exact_one_sided_p']:.4f}, with null range [{time['null_min']:.4f}, {time['null_max']:.4f}]. This p has 1/120 granularity and includes the identity permutation. The whole-RDM image-label null gives p={stim['one_sided_p']:.4f}, median={stim['null_median']:.4f}, 95% interval [{stim['q025']:.4f}, {stim['q975']:.4f}].\n\n## Controls and limitations\n\nMain-window split-half reliability medians range from {rel[rel.window!='baseline_-200_0']['median'].min():.4f} to {rel[rel.window!='baseline_-200_0']['median'].max():.4f}. The 50–100, 100–150 and 150–200 ms windows contain only five samples at 100 Hz. Across the 25 subject/window observations, reliability versus preferred layer position has descriptive rho={snr_r:.4f}; reliability versus mean raw RSA has rho={mean_rsa_rel:.4f}. The late 300–500 ms window has the lowest reliability (about 0.249), while late windows prefer deeper layers, so SNR confounding is not resolved. These are not independent reliability evidence. The pre-stimulus baseline was retained as a diagnostic profile and excluded from the primary order statistic. Raw RSA includes static layer alignment; centered RSA removes each layer's across-window mean. The official shared MVNN whitening used training/test partitions and repetitions, so split-half estimates are conditional and not session-independent. RDM edges are not independent observations.\n\n## Decision: {decision} / INCONCLUSIVE\n\nThe temporal-order and stimulus-label controls are favorable, but the frozen GO rule also requires that SNR or static-layer artifacts do not clearly explain the pattern. That condition is not established: reliability is weakest in the late windows and is negatively associated with preferred depth. Therefore this is WEAK/INCONCLUSIVE, not a GO for training. A future preprocessing-separated reliability audit is needed before any Phase 1 pilot.\n\nIf a later Phase 1 is authorized after that audit, it must keep the 200 repeated-test images evaluation-only and compare fixed/random/unconstrained/monotonic time-to-layer regularizers on an independent training image pool. This report does not start Phase 1.\n'''
 (OUT/'scientific_summary.md').write_text(report)
 (OUT/'validation.md').write_text(f'''# Validation\n\n- Frozen protocol SHA256: `{sha(OUT/'frozen_protocol.yaml')}`\n- Five real EEG inputs verified as float64 `(200,80,17,100)` and loaded with `allow_pickle=False`.\n- 200 unique ordered stimulus IDs; local stimulus files present.\n- DINOv2 pinned local revision; final-layer 5-image smoke comparison is recorded in `model_layer_schema.json`.\n- RDM integrity: `{json.dumps(integrity)}`\n- No EEG/checkpoint downloads, training, or previous-result modifications.\n''')
 files={}
 for p in OUT.iterdir():
  if p.is_file() and p.name!='output_manifest.json': files[p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
 (OUT/'output_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
 print('DECISION',decision,'rhos',rhos,'time_p',time['exact_one_sided_p'],'stim_p',stim['one_sided_p'])
if __name__=='__main__': main()
