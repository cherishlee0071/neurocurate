"""Frozen real subject-01/02 replication, using verified pickle-free exports."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml
from neurocurate.ablation import DisagreementMetrics, unit_ranks, score_nulls, distribution
from neurocurate.eeg import EEGConfig, EEGTrials, extract_eeg_features
from neurocurate.metadata import align_features
from neurocurate.realdata import validate_test_order
from neurocurate.rdm import compute_rdm, validate_rdm
from neurocurate.lowdisk import sha256
from neurocurate.vision import load_feature_cache
from neurocurate.visualization import plot_rdms

ROOT = Path('results/real_phase0')
OUT = ROOT/'cross_subject_01_02'

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def interval(values):
    return dict(median=float(np.median(values)),
                random_split_95_percentile_interval=np.quantile(values,[.025,.975]).tolist())

def null_report(observed, null):
    return dict(observed=float(observed), null_median=float(np.median(null)),
                null_95_percentile_interval=np.quantile(null,[.025,.975]).tolist(),
                one_sided_monte_carlo_p=float((1+np.sum(null>=observed))/(len(null)+1)))

def rdm_comparison(a, b, permutations):
    n=len(a); triangle=np.triu_indices(n,1)
    ra=unit_ranks(a[triangle]); original=rankdata(b[triangle])
    rb=unit_ranks(b[triangle]); center=original.mean(); norm=np.linalg.norm(original-center)
    matrix=np.zeros((n,n));matrix[triangle]=original;matrix+=matrix.T
    null=np.array([ra@((matrix[np.ix_(p,p)][triangle]-center)/norm) for p in permutations])
    return float(ra@rb),null

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((OUT/'frozen_protocol.json').read_text())
    assert sha256('configs/cross_subject_01_02.yaml')==frozen['config_sha256']
    cfg=yaml.safe_load(frozen['config_text'])
    reference=yaml.safe_load((ROOT/'sub01/scientific_config.yaml').read_text())
    assert sha256(ROOT/'sub01/scientific_config.yaml')==frozen['subject01_config_sha256']
    for key in ['preprocessing','brain_metric','vision_metric','k_values','split_seed','permutation_seed','repetition_splits','permutations']:
        assert cfg[key]==reference[key], f'Frozen setting mismatch: {key}'
    config=EEGConfig(**{**cfg['preprocessing'],'time_window':tuple(cfg['preprocessing']['time_window'])})
    ids=np.load(ROOT/'sub01/image_ids.npy',allow_pickle=False)
    image_root=Path('data/local/real_phase0/stimuli')
    model_rdms={}
    vision_provenance={}
    for name,revision in cfg['vision_revisions'].items():
        cache=ROOT/'sub01'/f'{name}_features.npz'
        cached_ids,features,provenance=load_feature_cache(cache,split='sanity',model_name=name)
        assert np.array_equal(cached_ids,ids) and provenance['resolved_revision']==revision
        for image_id in ids:
            assert sha256(image_root/image_id)==provenance['image_sha256'][image_id]
        matrix=validate_rdm(np.load(ROOT/'sub01'/f'{name}_rdm.npy',allow_pickle=False))
        assert np.allclose(matrix,compute_rdm(features,cfg['vision_metric']))
        model_rdms[name]=matrix
        vision_provenance[name]=dict(cache=str(cache),sha256=sha256(cache),revision=revision,reused_for_both_subjects=True)
    metrics=DisagreementMetrics(model_rdms,ids,cfg['k_values'])
    rng=np.random.default_rng(cfg['split_seed'])
    halves=[]
    for _ in range(cfg['repetition_splits']):
        order=rng.permutation(80)
        halves.append([np.sort(order[:40]),np.sort(order[40:])])
    prng=np.random.default_rng(cfg['permutation_seed'])
    permutations=np.array([prng.permutation(len(ids)) for _ in range(cfg['permutations'])])
    summaries={};brains={};full_scores={};export_provenance={}
    for subject in cfg['subjects']:
        root=ROOT/subject;exports=root/'pickle_free'
        assert (exports/'VERIFIED').is_file()
        manifest=json.loads((exports/'export_manifest.json').read_text())
        for name,digest in manifest['exports'].items():
            assert sha256(exports/name)==digest
        isolation=manifest['isolation']
        assert isolation['all_exports_pickle_free'] and all(v=='denied' for v in isolation['probes'].values())
        export_provenance[subject]=dict(manifest=str(exports/'export_manifest.json'),sha256=sha256(exports/'export_manifest.json'))
        eeg=np.load(exports/'eeg.npy',allow_pickle=False,mmap_mode='r')
        times=np.load(exports/'times.npy',allow_pickle=False)
        channels=np.load(exports/'channels.npy',allow_pickle=False)
        assert eeg.shape==(200,80,17,100) and eeg.dtype==np.float64 and np.isfinite(eeg).all()
        assert np.array_equal(times,np.load(ROOT/'sub01/pickle_free/times.npy',allow_pickle=False))
        assert np.array_equal(channels,np.load(ROOT/'sub01/pickle_free/channels.npy',allow_pickle=False))
        metadata=json.loads((exports/'metadata.json').read_text())
        alignment=validate_test_order(metadata,image_root)
        mapped=np.array([f'test_images/{c}/{f}' for c,f in zip(metadata['test_img_concepts'],metadata['test_img_files'])])
        assert np.array_equal(mapped,ids)
        assert len({sha256(image_root/i) for i in ids})==200
        alignment.update(subject=subject,exact_subject01_order_match=True,distinct_image_hashes=200,
                         shared_metadata_sha256=manifest['original_input_sha256'][1],raw_trial_roundtrip_available=False)
        # Keep the hash-pinned subject-01 Phase 1.3 reports unchanged.
        report_root = root if subject == 'sub02' else OUT
        prefix = '' if subject == 'sub02' else 'sub01_'
        write(report_root/(prefix+'image_alignment_report.json'),alignment)
        write(report_root/(prefix+'eeg_shape_report.json'),dict(shape=list(eeg.shape),dtype=str(eeg.dtype),
            image_count=200,repetition_count=80,channel_count=17,channel_names=channels.tolist(),
            sampling_rate_hz=float(1/np.median(np.diff(times))),times_seconds=times.tolist(),nonfinite_values=0,
            image_axis_semantics='condition i+1 at image-axis index i; official numeric event sorting and metadata verified',
            exact_subject01_schema_match=True,original_split='test',repository_split='sanity',session_labels_available=False))
        def features(indices):
            trials=EEGTrials(np.asarray(eeg[:,indices,:,:]).reshape(-1,17,100),
                np.repeat(ids,len(indices)),times,channels,'sanity')
            extracted_ids,values,_=extract_eeg_features(trials,config)
            return align_features(ids,extracted_ids,values)
        values=features(np.arange(80));brain=validate_rdm(compute_rdm(values,cfg['brain_metric']))
        if subject=='sub01':
            assert np.allclose(brain,np.load(root/'brain_rdm.npy',allow_pickle=False),atol=1e-12)
        else:
            np.save(root/'eeg_features.npy',values,allow_pickle=False)
            np.save(root/'brain_rdm.npy',brain,allow_pickle=False)
            np.save(root/'image_ids.npy',ids,allow_pickle=False)
            (root/'scientific_config.yaml').write_text(yaml.safe_dump({**cfg,'subject':subject}))
            plot_rdms({'Subject-02 Brain (100–500 ms)':brain},ids,root/'brain_rdm.png')
        brains[subject]=brain;full_scores[subject]=metrics(brain)
        records={key:dict(spearman=[],top20_overlap_fraction=[],rank_null=[],top_null=[]) for key in full_scores[subject]}
        reliability=[];rdm_nulls=[];half_rdms=[]
        for number,(ha,hb) in enumerate(halves):
            assert len(set(ha)&set(hb))==0
            matrices=[validate_rdm(compute_rdm(features(indices),cfg['brain_metric'])) for indices in (ha,hb)]
            half_rdms.append(matrices)
            rho,null=rdm_comparison(*matrices,permutations)
            reliability.append(rho);rdm_nulls.append(null)
            scored=[metrics(matrix) for matrix in matrices]
            for key,record in records.items():
                sr,overlap,rnull,tnull=score_nulls(scored[0][key],scored[1][key],ids,permutations,cfg['top_n'])
                record['spearman'].append(sr);record['top20_overlap_fraction'].append(overlap)
                record['rank_null'].append(rnull);record['top_null'].append(tnull)
            print(subject,'split',number+1,'RDM rho',round(rho,4),flush=True)
        report=dict(statistics=interval(reliability),permutation=null_report(np.median(reliability),np.median(rdm_nulls,axis=0)),
            repetition_splits=30,repetitions_per_half=40,split_spearman=reliability,
            conditional_on_shared_preprocessing=True,session_independent=False,
            uncertainty='empirical random-split variability interval; not a population confidence interval')
        if subject=='sub01':
            previous=json.loads((root/'brain_reliability.json').read_text())
            assert np.allclose(reliability,[s['spearman'] for s in previous['splits']],atol=1e-12)
        else:
            write(root/'brain_reliability.json',report)
        summary={}
        for (method,model),record in records.items():
            summary[f'{method}/{model}']=dict(spearman=interval(record['spearman']),
                top20_overlap_fraction=interval(record['top20_overlap_fraction']),
                rank_null=null_report(np.median(record['spearman']),np.median(record['rank_null'],axis=0)),
                tie_aware_top20_null=null_report(np.median(record['top20_overlap_fraction']),np.median(record['top_null'],axis=0)),
                raw_splits=dict(spearman=record['spearman'],top20_overlap_fraction=record['top20_overlap_fraction']),
                full_score_distribution=distribution(full_scores[subject][(method,model)],ids))
        summaries[subject]=dict(brain_reliability=report,scoring=summary)
        np.savez_compressed(OUT/f'{subject}_half_rdms.npz',rdms=np.asarray(half_rdms),image_ids=ids,
                            repetition_indices=np.array(halves))
        if subject=='sub02':
            write(root/'disagreement_stability.json',summary)
        del eeg
    cross_rho,cross_null=rdm_comparison(brains['sub01'],brains['sub02'],permutations)
    cross_rdm=dict(subjects=cfg['subjects'],image_count=200,image_order_sha256=sha256(ROOT/'sub01/image_ids.npy'),
        cross_subject_spearman=cross_rho,image_label_permutation=null_report(cross_rho,cross_null),
        within_subject_repetition_reliability={s:v['brain_reliability'] for s,v in summaries.items()},
        interpretation='Cross-subject correlation is distinct from split-half reliability; no session-independent inference.',
        conditional_on_shared_preprocessing=True)
    write(OUT/'cross_subject_rdm.json',cross_rdm)
    comparisons={};csv_rows=[];score_rows=[]
    for method,model in full_scores['sub01']:
        key=f'{method}/{model}'
        a,b=[full_scores[s][(method,model)] for s in cfg['subjects']]
        rho,overlap,rnull,tnull=score_nulls(a,b,ids,permutations,cfg['top_n'])
        comparisons[key]=dict(cross_subject_spearman=rho,top20_overlap_fraction=overlap,
            top20_shared_images=int(round(overlap*20)),rank_permutation=null_report(rho,rnull),
            tie_aware_top20_permutation=null_report(overlap,tnull),
            subjects={s:summaries[s]['scoring'][key] for s in cfg['subjects']})
        row=dict(method=method,model=model,cross_subject_spearman=rho,cross_subject_rank_p=comparisons[key]['rank_permutation']['one_sided_monte_carlo_p'],
            cross_subject_top20_shared=int(round(overlap*20)),cross_subject_top20_null_median=float(np.median(tnull)),
            cross_subject_top20_p=comparisons[key]['tie_aware_top20_permutation']['one_sided_monte_carlo_p'])
        for s in cfg['subjects']:
            info=summaries[s]['scoring'][key];dist=info['full_score_distribution']
            row.update({f'{s}_within_rank_median':info['spearman']['median'],
                f'{s}_within_rank_q025':info['spearman']['random_split_95_percentile_interval'][0],
                f'{s}_within_rank_q975':info['spearman']['random_split_95_percentile_interval'][1],
                f'{s}_within_top20_median':info['top20_overlap_fraction']['median'],
                f'{s}_within_top20_null_median':info['tie_aware_top20_null']['null_median'],
                f'{s}_within_top20_p':info['tie_aware_top20_null']['one_sided_monte_carlo_p'],
                f'{s}_score_mean':dist['mean'],f'{s}_score_std':dist['standard_deviation'],
                f'{s}_score_unique_values':dist['unique_values'],f'{s}_tied_images':dist['tied_images'],
                f'{s}_top20_boundary_ties':dist['top20_boundary_tie_size']})
            for i,value in enumerate(full_scores[s][(method,model)]):
                score_rows.append(dict(subject=s,method=method,model=model,image_id=ids[i],score=value))
        csv_rows.append(row)
    frame=pd.DataFrame(csv_rows)
    frame.to_csv(OUT/'scoring_ablation.csv',index=False)
    pd.DataFrame(score_rows).to_csv(OUT/'image_level_scores.csv',index=False)
    write(OUT/'cross_subject_disagreement.json',dict(results=comparisons,
        # YAML treats the predeclared bare key `null` as None. Preserve the
        # frozen protocol bytes and read that key explicitly.
        tie_policy=cfg['tie_policy'],null=cfg[None],permutations=1000,
        intervals='Random-split variability; splits overlap across reruns.',
        p_values='Exploratory one-sided Monte Carlo, unadjusted; all predeclared methods reported.',
        conditional_on_shared_preprocessing=True,metric_selection='No final scoring method selected from subject-02.'))
    provenance=dict(frozen_protocol=frozen,vision=vision_provenance,exports=export_provenance,
        all_downstream_numpy_allow_pickle=False,same_repetition_indices_both_subjects=True,
        split_and_null_seeds={k:cfg[k] for k in ['split_seed','permutation_seed']},
        code_sha256={p:sha256(p) for p in ['scripts/cross_subject_analysis.py','src/neurocurate/ablation.py']})
    write(OUT/'provenance.json',provenance)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(15,12))
    for index,model in enumerate(['dinov2','clip','mean_models']):
        subset=frame[frame.model==model];x=np.arange(len(subset));labels=['K5','K10','K20','Row rank','RR weighted']
        for offset,s,color in [(-.17,'sub01','#2978a0'),(.17,'sub02','#db7836')]:
            med=subset[f'{s}_within_rank_median'].to_numpy()
            low=subset[f'{s}_within_rank_q025'].to_numpy();high=subset[f'{s}_within_rank_q975'].to_numpy()
            axes[index,0].bar(x+offset,med,.32,label=s,color=color,yerr=[med-low,high-med],capsize=3)
        axes[index,0].set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel='Median split-half score Spearman',title=model)
        axes[index,0].legend();axes[index,0].grid(axis='y',alpha=.2)
        axes[index,1].bar(x,subset.cross_subject_spearman,color='#477f57',label='Cross-subject score Spearman')
        axes[index,1].plot(x,subset.cross_subject_top20_shared/20,'o-',color='#793c89',label='Cross-subject Top-20 overlap')
        axes[index,1].plot(x,subset.cross_subject_top20_null_median,'x--',color='gray',label='Tie-aware overlap null median')
        axes[index,1].set(xticks=x,xticklabels=labels,ylim=(-.1,1),title=model)
        axes[index,1].legend(fontsize=8);axes[index,1].grid(axis='y',alpha=.2)
    fig.suptitle('Frozen subject-01/02 replication: shared-preprocessing conditional results\nWhiskers: empirical repetition-split variability, not population confidence intervals')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'ranking_stability.png',dpi=160);plt.close(fig)
    print('Cross-subject Brain RDM Spearman',cross_rho,flush=True)
    print(frame[['method','model','sub01_within_rank_median','sub02_within_rank_median','cross_subject_spearman','cross_subject_top20_shared']].to_string(index=False),flush=True)

if __name__=='__main__':
    main()
