"""Phase 1.5: local-only brain-label, model-only and selection-budget controls."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from neurocurate.ablation import DisagreementMetrics
from neurocurate.brain_controls import (permute_brain,normalized_score_ranks,model_only_scores,
    RankControls,selection_statistics,overlap_budgets,chance_adjusted)
from neurocurate.rdm import compute_rdm,validate_rdm
from neurocurate.lowdisk import sha256
from neurocurate.vision import load_feature_cache

BASE=Path('results/real_phase0');OLD=BASE/'cross_subject_01_02';OUT=BASE/'brain_specific_controls'

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def null_summary(observed,values):
    values=np.asarray(values)
    return dict(observed=float(observed),null_median=float(np.median(values)),
        null_95_percentile_interval=np.quantile(values,[.025,.975]).tolist(),
        excess_above_null_median=float(observed-np.median(values)),
        exploratory_one_sided_p=float((1+np.sum(values>=observed))/(len(values)+1)))

def quantiles(values):
    return dict(median=float(np.median(values)),quantile_025=float(np.quantile(values,.025)),quantile_975=float(np.quantile(values,.975)))

def main():
    frozen=json.loads((OUT/'frozen_protocol.json').read_text())
    assert sha256('configs/brain_specific_controls.yaml')==frozen['config_sha256']
    cfg=yaml.safe_load(frozen['config_text'])
    assert sha256('configs/cross_subject_01_02.yaml')==cfg['phase14_config_sha256']
    reference=yaml.safe_load(Path('configs/cross_subject_01_02.yaml').read_text())
    old_manifest=json.loads((OLD/'verified_outputs.json').read_text())
    for name,pin in old_manifest['outputs'].items(): assert sha256(OLD/name)==pin['sha256']
    for name,pin in old_manifest['subject02_outputs'].items(): assert sha256(BASE/'sub02'/name)==pin['sha256']
    prior=json.loads((BASE/'sub01/scientific_provenance.json').read_text())
    for name,digest in prior['output_sha256'].items(): assert sha256(BASE/'sub01'/name)==digest
    old_provenance=json.loads((OLD/'provenance.json').read_text())
    ids=np.load(BASE/'sub01/image_ids.npy',allow_pickle=False)
    n=len(ids);assert n==200
    inputs={};brains=[];half_rdms=[]
    for subject in cfg['subjects']:
        root=BASE/subject;exports=root/'pickle_free'
        assert (exports/'VERIFIED').exists()
        manifest=json.loads((exports/'export_manifest.json').read_text())
        for name,digest in manifest['exports'].items(): assert sha256(exports/name)==digest
        assert np.array_equal(ids,np.load(root/'image_ids.npy',allow_pickle=False))
        times=np.load(exports/'times.npy',allow_pickle=False)
        raw=np.load(exports/'eeg.npy',allow_pickle=False,mmap_mode='r')
        start,stop=reference['preprocessing']['time_window']
        mask=(times>=start-1e-9)&(times<stop-1e-9)
        features=raw.mean(axis=1)[:,:,mask].reshape(n,-1)
        assert raw.shape==(200,80,17,100) and np.allclose(features,np.load(root/'eeg_features.npy',allow_pickle=False),atol=1e-12)
        brain=validate_rdm(np.load(root/'brain_rdm.npy',allow_pickle=False))
        assert np.allclose(brain,compute_rdm(features,reference['brain_metric']),atol=1e-12)
        brains.append(brain)
        with np.load(OLD/f'{subject}_half_rdms.npz',allow_pickle=False) as saved:
            assert np.array_equal(saved['image_ids'],ids)
            halves=saved['repetition_indices'];matrices=saved['rdms']
        rng=np.random.default_rng(cfg['repetition_split_seed'])
        expected=[]
        for _ in range(cfg['repetition_splits']):
            order=rng.permutation(80);expected.append([np.sort(order[:40]),np.sort(order[40:])])
        assert np.array_equal(halves,expected) and matrices.shape==(30,2,200,200)
        for pair in matrices:
            for matrix in pair: validate_rdm(matrix)
        half_rdms.append(matrices)
        for p in [root/'brain_rdm.npy',root/'image_ids.npy',exports/'export_manifest.json',OLD/f'{subject}_half_rdms.npz']:
            inputs[str(p)]=sha256(p)
        del raw
    models={}
    for name,revision in reference['vision_revisions'].items():
        cache=BASE/'sub01'/f'{name}_features.npz'
        assert sha256(cache)==old_provenance['vision'][name]['sha256']
        cached,features,provenance=load_feature_cache(cache,split='sanity',model_name=name)
        assert np.array_equal(cached,ids) and provenance['resolved_revision']==revision
        matrix=validate_rdm(np.load(BASE/'sub01'/f'{name}_rdm.npy',allow_pickle=False))
        assert np.allclose(matrix,compute_rdm(features,reference['vision_metric']))
        models[name]=matrix;inputs[str(cache)]=sha256(cache)
    metrics=DisagreementMetrics(models,ids,reference['k_values'])
    keys=[(method,model) for method in cfg['methods'] for model in cfg['models']]
    names=['/'.join(key) for key in keys]
    def scores(matrix):
        result=metrics(matrix)
        return np.column_stack([result[key] for key in keys])
    observed=[scores(brain) for brain in brains]
    previous=json.loads((OLD/'cross_subject_disagreement.json').read_text())['results']
    observed_ranks=[normalized_score_ranks(s) for s in observed]
    observed_rho=np.sum(observed_ranks[0]*observed_ranks[1],axis=0)
    for j,name in enumerate(names): assert np.isclose(observed_rho[j],previous[name]['cross_subject_spearman'])
    model_row=DisagreementMetrics({'clip':models['clip']},ids,(20,))(models['dinov2'])[('row_spearman','clip')]
    control_dict=model_only_scores(models,ids,model_row)
    controls=np.column_stack([control_dict[key] for key in cfg['model_only_controls']])
    control_ranks=normalized_score_ranks(controls)
    projection=RankControls(controls)
    observed_residuals=[];explained=[]
    for s in observed:
        residual,r_squared=projection.residuals(s)
        observed_residuals.append(residual);explained.append(r_squared)
    observed_partial=np.sum(observed_residuals[0]*observed_residuals[1],axis=0)
    count=cfg['permutations'];budgets=cfg['selection_budgets']
    irng=np.random.default_rng(cfg['brain_independent_seed'])
    independent=np.array([[irng.permutation(n),irng.permutation(n)] for _ in range(count)])
    srng=np.random.default_rng(cfg['brain_shared_seed'])
    shared=np.array([srng.permutation(n) for _ in range(count)])
    lrng=np.random.default_rng(cfg['score_label_seed'])
    score_permutations=np.array([lrng.permutation(n) for _ in range(count)])
    null_rho=np.zeros((2,count,len(keys)));null_partial=np.zeros_like(null_rho)
    null_overlap=np.zeros((2,count,len(keys),len(budgets)))
    null_aligned=np.zeros((2,count,2,len(keys)));null_means=np.zeros_like(null_aligned)
    null_model_association=np.zeros((2,count,2,len(keys),controls.shape[1]))
    for mode,label in enumerate(['independent','shared']):
        for i in range(count):
            ps=independent[i] if mode==0 else [shared[i],shared[i]]
            permuted=[scores(permute_brain(brain,p)) for brain,p in zip(brains,ps)]
            ranked=[normalized_score_ranks(s) for s in permuted]
            residuals=[projection.residuals(s)[0] for s in permuted]
            null_rho[mode,i]=np.sum(ranked[0]*ranked[1],axis=0)
            null_partial[mode,i]=np.sum(residuals[0]*residuals[1],axis=0)
            for subject in range(2):
                null_aligned[mode,i,subject]=np.sum(observed_ranks[subject]*ranked[subject],axis=0)
                null_means[mode,i,subject]=permuted[subject].mean(axis=0)
                null_model_association[mode,i,subject]=ranked[subject].T@control_ranks
            for j in range(len(keys)):
                null_overlap[mode,i,j]=overlap_budgets(permuted[0][:,j],permuted[1][:,j],ids,budgets)
            if (i+1)%100==0: print(label,'brain permutations',i+1,flush=True)
    half_scores=np.empty((2,30,2,n,len(keys)))
    for subject in range(2):
        for split,pair in enumerate(half_rdms[subject]):
            for half,matrix in enumerate(pair): half_scores[subject,split,half]=scores(matrix)
    rows=[];observed_overlaps=[]
    selection_raw={}
    for j,(method,model) in enumerate(keys):
        observed_cross,cross_score_null=selection_statistics(observed[0][:,j],observed[1][:,j],ids,score_permutations,budgets)
        observed_overlaps.append(observed_cross)
        cross_halves=np.array([overlap_budgets(half_scores[0,split,half,:,j],half_scores[1,split,half,:,j],ids,budgets)
            for split in range(30) for half in range(2)])
        within=[]
        for subject in range(2):
            overlaps=[];nulls=[]
            for split in range(30):
                observed_half,half_null=selection_statistics(*half_scores[subject,split,:,:,j],ids,score_permutations,budgets)
                overlaps.append(observed_half);nulls.append(half_null)
            within.append((np.array(overlaps),np.median(nulls,axis=0)))
        selection_raw[names[j]]=dict(cross_half_overlaps=cross_halves.tolist(),
            sub01_within_overlaps=within[0][0].tolist(),sub02_within_overlaps=within[1][0].tolist())
        for b,budget in enumerate(budgets):
            for s,(overlaps,median_null) in enumerate(within):
                med=float(np.median(overlaps[:,b]));null=median_null[:,b];reference_null=float(np.median(null))
                q=np.quantile(overlaps[:,b],[.025,.975]);adjusted=chance_adjusted(q,reference_null)
                rows.append(dict(method=method,model=model,comparison=f'within_sub0{s+1}',budget=budget,
                    overlap_fraction=med,expected_shared_images=med*budget,
                    split_q025=q[0],split_q975=q[1],uncertainty='empirical 30 repetition-split variability; not population CI',
                    null_type='score_labels_tie_aware_median_across_splits',null_median=reference_null,
                    null_q025=np.quantile(null,.025),null_q975=np.quantile(null,.975),
                    chance_adjusted_overlap=float(chance_adjusted(med,reference_null)),
                    chance_adjusted_split_q025=adjusted[0],chance_adjusted_split_q975=adjusted[1],
                    exploratory_p=(1+np.sum(null>=med))/(count+1)))
            for mode,label in [(-1,'score_labels_tie_aware'),(0,'brain_independent'),(1,'brain_shared')]:
                null=cross_score_null[:,b] if mode==-1 else null_overlap[mode,:,j,b]
                observed_value=observed_cross[b];ref=float(np.median(null))
                rows.append(dict(method=method,model=model,comparison='cross_subject_full',budget=budget,
                    overlap_fraction=observed_value,expected_shared_images=observed_value*budget,
                    split_q025=None,split_q975=None,uncertainty='full statistic has no population CI; half-RDM variability reported separately',
                    cross_half_median=np.median(cross_halves[:,b]),cross_half_q025=np.quantile(cross_halves[:,b],.025),
                    cross_half_q975=np.quantile(cross_halves[:,b],.975),null_type=label,
                    null_median=ref,null_q025=np.quantile(null,.025),null_q975=np.quantile(null,.975),
                    chance_adjusted_overlap=float(chance_adjusted(observed_value,ref)),
                    exploratory_p=(1+np.sum(null>=observed_value))/(count+1)))
        print('selection budgets complete',names[j],flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'selection_budget_curves.csv',index=False)
    write(OUT/'selection_split_statistics.json',selection_raw)
    model_frame=pd.DataFrame({'image_id':ids,**control_dict})
    for s in range(2):
        for j,name in enumerate(names): model_frame[f'sub0{s+1}_{name}']=observed[s][:,j]
    model_frame.to_csv(OUT/'model_only_controls.csv',index=False)
    model_associations=[]
    for s in range(2):
        associations=observed_ranks[s].T@control_ranks
        for j,name in enumerate(names):
            for c,control in enumerate(cfg['model_only_controls']):
                model_associations.append(dict(subject=cfg['subjects'][s],method_model=name,control=control,
                    score_spearman=float(associations[j,c]),in_sample_rank_r_squared=float(explained[s][j]),
                    independent_brain_null=null_summary(associations[j,c],null_model_association[0,:,s,j,c]),
                    shared_brain_null=null_summary(associations[j,c],null_model_association[1,:,s,j,c])))
    report=dict(config=frozen,counts=dict(images=200,subjects=2,brain_permutations_each=1000,repetition_splits=30),
        null_definitions=dict(independent=cfg['brain_null_independent'],shared=cfg['brain_null_shared'],score=cfg['score_null']),
        conditional_on_shared_preprocessing=True,session_independent=False,
        model_control_design=dict(rank=projection.rank,columns=projection.design.shape[1],condition_number=projection.condition_number,
            controls=cfg['model_only_controls'],residual_degrees_of_freedom=n-projection.rank,
            definition=cfg['partial_association'],in_sample_only=True),results={})
    for j,name in enumerate(names):
        subjects={}
        for s in range(2):
            subjects[cfg['subjects'][s]]=dict(in_sample_model_only_rank_r_squared=float(explained[s][j]),
                observed_mean_disagreement=float(observed[s][:,j].mean()),
                independent_brain_null_mean_disagreement=quantiles(null_means[0,:,s,j]),
                independent_brain_null_correlation_with_aligned_ranking=quantiles(null_aligned[0,:,s,j]))
        report['results'][name]=dict(observed_cross_subject_spearman=float(observed_rho[j]),
            independent_brain_null=null_summary(observed_rho[j],null_rho[0,:,j]),
            shared_brain_null=null_summary(observed_rho[j],null_rho[1,:,j]),
            observed_partial_rank_correlation=float(observed_partial[j]),
            partial_independent_brain_null=null_summary(observed_partial[j],null_partial[0,:,j]),
            partial_shared_brain_null=null_summary(observed_partial[j],null_partial[1,:,j]),subjects=subjects)
    write(OUT/'brain_specific_nulls.json',report)
    write(OUT/'model_only_associations.json',dict(associations=model_associations,
        interpretation='Descriptive in-sample rank associations and exploratory brain-label controls; not causal or predictive.',
        p_values='Upper-tail, unadjusted; negative associations require inspecting direction and null intervals.'))
    np.savez_compressed(OUT/'null_statistics.npz',score_names=np.array(names),budgets=np.array(budgets),
        independent_permutations=independent,shared_permutations=shared,score_label_permutations=score_permutations,
        cross_score_rho=null_rho,partial_cross_score_rho=null_partial,selection_overlap=null_overlap,
        correlation_with_aligned_ranking=null_aligned,mean_disagreement=null_means,model_association=null_model_association)
    np.savez_compressed(OUT/'observed_scores.npz',image_ids=ids,score_names=np.array(names),
        scores=np.asarray(observed),half_scores=half_scores,model_controls=controls,
        control_names=np.array(cfg['model_only_controls']))
    write(OUT/'provenance.json',dict(input_sha256=inputs,phase14_manifest_sha256=sha256(OLD/'verified_outputs.json'),
        phase14_provenance_sha256=sha256(OLD/'provenance.json'),frozen_config_sha256=frozen['config_sha256'],
        code_sha256={p:sha256(p) for p in ['scripts/brain_specific_controls.py','src/neurocurate/brain_controls.py','src/neurocurate/ablation.py']},
        all_numpy_loads_allow_pickle_false=True,no_downloads=True,no_classifier_training=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),versions=dict(numpy=np.__version__,pandas=pd.__version__),
        checks=['all Phase 1.4 output/input hashes','original subject-01 scientific output hashes','exact 200 image order',
                'Brain RDM recomputation from numeric EEG','cached model RDM recomputation','identical frozen split assignments',
                'observed cross-subject scores reproduce Phase 1.4']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(17,10))
    for c,model in enumerate(cfg['models']):
        selected=frame[(frame.method=='row_spearman')&(frame.model==model)]
        for s,color in [('within_sub01','#2978a0'),('within_sub02','#db7836')]:
            sub=selected[selected.comparison==s]
            axes[0,c].plot(sub.budget,sub.overlap_fraction,'o-',label=s,color=color)
            axes[0,c].fill_between(sub.budget,sub.split_q025,sub.split_q975,alpha=.15,color=color)
        cross=selected[(selected.comparison=='cross_subject_full')&(selected.null_type=='score_labels_tie_aware')]
        axes[0,c].plot(cross.budget,cross.cross_half_median,'s--',label='Cross-subject half RDM median',color='#477f57')
        axes[0,c].fill_between(cross.budget,cross.cross_half_q025,cross.cross_half_q975,alpha=.12,color='#477f57')
        axes[0,c].plot(cross.budget,cross.null_median,':',label='Full score-label null median',color='gray')
        axes[0,c].set(title=model,ylabel='Repetition-split shortlist overlap fraction',ylim=(0,1),xticks=budgets)
        axes[1,c].plot(cross.budget,cross.overlap_fraction,'o-',color='black',label='Observed full cross-subject')
        for label,color in [('brain_independent','#2978a0'),('brain_shared','#a34c82'),('score_labels_tie_aware','gray')]:
            sub=selected[(selected.comparison=='cross_subject_full')&(selected.null_type==label)]
            axes[1,c].plot(sub.budget,sub.null_median,'--',label=label,color=color)
            axes[1,c].fill_between(sub.budget,sub.null_q025,sub.null_q975,alpha=.12,color=color)
        axes[1,c].set(ylabel='Full cross-subject shortlist overlap fraction',xlabel='Selection budget / 200 images',ylim=(0,1),xticks=budgets)
        for ax in axes[:,c]: ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Exploratory row-rank disagreement: selection budgets and brain-specific controls\nTop bands: repetition-split variability. Bottom bands: permutation quantiles. Neither is a population CI.')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'selection_budget_curves.png',dpi=160);plt.close(fig)
    print('PRIMARY',json.dumps(report['results'][cfg['primary_exploratory_candidate']],indent=2),flush=True)

if __name__=='__main__': main()
