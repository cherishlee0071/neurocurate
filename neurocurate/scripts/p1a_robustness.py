"""P1-A frozen disagreement versus vision representation drift pilot."""
from pathlib import Path
import hashlib, json, subprocess
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import torch
from transformers import AutoImageProcessor, Dinov2Model, CLIPImageProcessor, CLIPVisionModelWithProjection

BASE=Path('results/real_phase0'); P0=BASE/'v21_frozen'; OUT=BASE/'v21_p1a'; IMG=Path('data/local/real_phase0/stimuli')
SEED=20261018; N_PERM=10000; N_RANDOM=1000
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def files():
 ids=np.load(BASE/'sub01/image_ids.npy',allow_pickle=False); return ids,[IMG/i for i in ids]
def make_image(image,kind,level,seed):
 if kind=='blur': return image.filter(ImageFilter.GaussianBlur(radius=float(level)))
 rng=np.random.default_rng(seed); ratio=float(level); w,h=image.size; cw,ch=max(1,int(w*ratio)),max(1,int(h*ratio)); x=int(rng.integers(0,w-cw+1)); y=int(rng.integers(0,h-ch+1)); return image.crop((x,y,x+cw,y+ch)).resize((w,h),Image.Resampling.BICUBIC)
def extract(model,processor,images):
 out=[]
 with torch.inference_mode():
  for start in range(0,len(images),8):
   batch=processor(images=images[start:start+8],return_tensors='pt'); z=model(**batch); f=z.last_hidden_state[:,0] if isinstance(model,Dinov2Model) else z.image_embeds; f=f/f.norm(dim=1,keepdim=True); out.append(f.cpu().numpy())
 return np.concatenate(out)
def main():
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'drift').mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
 ids,paths=files(); assert len(ids)==200 and all(p.is_file() for p in paths)
 score=[]
 for p in sorted((P0/'disagreement').glob('sub*_scores.csv')): score.append(pd.read_csv(p).mean_models_row_spearman.to_numpy())
 assert len(score)==5 and all(len(x)==200 for x in score)
 consensus=np.mean(np.array([np.argsort(np.argsort(-x)) for x in score]),axis=0); image_order=np.array(ids)
 conditions=[('blur',1),('blur',2),('blur',4),('crop',.90),('crop',.75)]
 allrows=[]; rng=np.random.default_rng(SEED)
 for model_name,checkpoint,revision in [('clip','openai/clip-vit-base-patch32','3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'),('dinov2','facebook/dinov2-base','f9e44c814b77203eaa57a6bdbbd535f21ede1415')]:
  if model_name=='clip': processor=CLIPImageProcessor.from_pretrained(checkpoint,revision=revision,local_files_only=True,cache_dir='/workspace/setup/neurocurate/model_cache'); model=CLIPVisionModelWithProjection.from_pretrained(checkpoint,revision=revision,local_files_only=True,cache_dir='/workspace/setup/neurocurate/model_cache')
  else: processor=AutoImageProcessor.from_pretrained(checkpoint,revision=revision,local_files_only=True,cache_dir='/workspace/setup/neurocurate/model_cache',use_fast=False); model=Dinov2Model.from_pretrained(checkpoint,revision=revision,local_files_only=True,cache_dir='/workspace/setup/neurocurate/model_cache')
  model.eval(); model.requires_grad_(False); originals=[]
  for p in paths:
   with Image.open(p) as im: originals.append(im.convert('RGB'))
  original=np.load(P0/'vision_features'/f'{model_name}.npy',allow_pickle=False); assert original.shape[0]==200
  for kind,level in conditions:
   transformed=[make_image(im,kind,level,SEED+int(level*1000)+i) for i,im in enumerate(originals)]
   pert=extract(model,processor,transformed); drift=1-np.sum(original*pert,axis=1)/(np.linalg.norm(original,axis=1)*np.linalg.norm(pert,axis=1)); label=f'{kind}_{str(level).replace(".","")}'
   pd.DataFrame({'image_id':ids,'disagreement':consensus,'drift':drift,'perturbation':kind,'level':level,'seed':SEED+int(level*1000)+np.arange(200)}).to_csv(OUT/'drift'/f'{model_name}_{kind}_{str(level).replace(".","")}.csv',index=False)
   for i in range(200): allrows.append({'model':model_name,'perturbation':kind,'level':level,'image_id':ids[i],'disagreement':float(consensus[i]),'drift':float(drift[i])})
 results=pd.DataFrame(allrows); stats=[]; permout={}
 top=np.argsort(-consensus,kind='stable')[:40]; low=np.argsort(consensus,kind='stable')[:40];
 for (model,kind,level),g in results.groupby(['model','perturbation','level'],sort=True):
  d=g.drift.to_numpy(); high=d[top]; lo=d[low]; delta=float(high.mean()-lo.mean()); obs=float(spearmanr(g.disagreement,d).statistic)
  diffs=[]; random_diffs=[]
  for _ in range(N_PERM):
   pick=rng.permutation(200); diffs.append(d[pick[:40]].mean()-d[pick[40:80]].mean())
  for _ in range(N_RANDOM):
   pick=rng.permutation(200); random_diffs.append(d[pick[:40]].mean()-d[pick[40:80]].mean())
  diffs=np.asarray(diffs); random_diffs=np.asarray(random_diffs); key=f'{model}_{kind}_{level:g}'
  stats.append({'model':model,'perturbation':kind,'level':level,'spearman_rho':obs,'spearman_p':float(spearmanr(g.disagreement,d).pvalue),'high_low_mean_difference':delta,'permutation_p_two_sided':float((1+np.sum(np.abs(diffs)>=abs(delta)))/(N_PERM+1)),'permutation_q025':float(np.quantile(diffs,.025)),'permutation_q975':float(np.quantile(diffs,.975)),'random_baseline_median':float(np.median(random_diffs)),'random_baseline_q025':float(np.quantile(random_diffs,.025)),'random_baseline_q975':float(np.quantile(random_diffs,.975)),'random_baseline_exceedance':float(np.mean(np.abs(random_diffs)>=abs(delta)))})
  permout[key]={'observed_delta':delta,'permutation':diffs.tolist(),'random_baseline':random_diffs.tolist()}
 results.to_csv(OUT/'drift_results.csv',index=False); pd.DataFrame(stats).to_csv(OUT/'group_statistics.csv',index=False); (OUT/'permutation_tests.json').write_text(json.dumps(permout,indent=2)+'\n')
 fig,ax=plt.subplots(2,5,figsize=(18,7));
 for col,(model,kind,level) in enumerate([(m,k,l) for m in ['clip','dinov2'] for k,l in conditions]):
  g=results[(results.model==model)&(results.perturbation==kind)&(results.level==level)]; a=ax[0,col if col<5 else 4]; a.scatter(g.disagreement,g.drift,s=8,alpha=.55); a.set_title(f'{model} {kind} {level:g}'); a.set_xlabel('frozen disagreement'); a.set_ylabel('cosine drift')
 for j,(model,kind,level) in enumerate([('clip','blur',1),('clip','blur',2),('clip','blur',4),('clip','crop',.9),('clip','crop',.75)]):
  g=results[(results.model==model)&(results.perturbation==kind)&(results.level==level)]; ax[1,j].boxplot([g.drift.iloc[top],g.drift.iloc[low]],labels=['high','low']); ax[1,j].set_title(f'CLIP {kind} {level:g}')
 fig.tight_layout(); fig.savefig(OUT/'figures/disagreement_vs_drift.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5)); labels=[]; vals=[]
 for row in stats: labels.append(f"{row['model']}\n{row['perturbation']}{row['level']:g}"); vals.append(row['high_low_mean_difference'])
 ax.bar(np.arange(len(vals)),vals); ax.axhline(0,color='k',lw=.8); ax.set_xticks(np.arange(len(vals)),labels,rotation=45,ha='right'); ax.set_ylabel('high-minus-low drift'); fig.tight_layout(); fig.savefig(OUT/'figures/group_comparison.png',dpi=160); plt.close(fig)
 manifest={'phase':'P1-A','exploratory':True,'seed':SEED,'permutations':N_PERM,'random_baselines':N_RANDOM,'images':200,'image_ids_sha256':sha(BASE/'sub01/image_ids.npy'),'p0_config_sha256':sha('configs/v21_disagreement_frozen.yaml'),'inputs':'P0 frozen disagreement and vision features only','models':{'clip':revision,'dinov2':revision},'conditions':[{'family':k,'level':l} for k,l in conditions],'crop_resize':'PIL BICUBIC','crop_seed':'SEED + int(level*1000) + image_index','group_definition':'top/bottom 20% of population consensus disagreement','no_controls':['uncertainty','behavior','additional models']}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 report=pd.DataFrame(stats); lines=['# NeuroCurate v2.1 P1-A Robustness Drift Pilot','', 'This exploratory pilot tests association between the frozen population-consensus disagreement and cosine representation drift. It does not claim model failure, causal effects, or robustness improvement.','', '## Results','']
 for _,r in report.iterrows(): lines.append(f"- {r.model} {r.perturbation} {r.level:g}: Spearman rho={r.spearman_rho:.3f} (p={r.spearman_p:.3g}); high-minus-low drift={r.high_low_mean_difference:.5f}; two-sided permutation p={r.permutation_p_two_sided:.3g}; random-baseline median={r.random_baseline_median:.5f}.")
 lines += ['', '## Decision', '', 'The P1-A signal is a diagnostic sanity check only. Interpret consistency across all ten conditions, not the strongest condition. P1-B uncertainty matching is justified only as a separately preregistered exploratory control if the association is directionally consistent across both models and perturbation families; these results do not establish predictive validity or causality.']
 (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
