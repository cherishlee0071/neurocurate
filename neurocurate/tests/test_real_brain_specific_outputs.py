"""Opt-in local Phase 1.5 tests, with no network or object-array loading."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from neurocurate.ablation import DisagreementMetrics
from neurocurate.brain_controls import RankControls
from neurocurate.lowdisk import sha256


@pytest.fixture
def outputs():
    location=os.environ.get('NEUROCURATE_PHASE15_RESULTS')
    if not location:
        pytest.skip('set NEUROCURATE_PHASE15_RESULTS to local brain_specific_controls outputs')
    root=Path(location)
    with np.load(root/'observed_scores.npz',allow_pickle=False) as saved:
        observed={name:saved[name] for name in saved.files}
    with np.load(root/'null_statistics.npz',allow_pickle=False) as saved:
        nulls={name:saved[name] for name in saved.files}
    return root,observed,nulls


def test_saved_nulls_recompute_from_brain_geometry(outputs):
    root,observed,nulls=outputs
    ids=observed['image_ids'];names=observed['score_names'].tolist()
    brains=[np.load(root.parent/s/'brain_rdm.npy',allow_pickle=False) for s in ['sub01','sub02']]
    models={name:np.load(root.parent/'sub01'/f'{name}_rdm.npy',allow_pickle=False) for name in ['dinov2','clip']}
    metric=DisagreementMetrics(models,ids,[5,10,20]);projection=RankControls(observed['model_controls'])
    triangle=np.triu_indices(200,1)
    original_brain_correlation=spearmanr(brains[0][triangle],brains[1][triangle]).statistic
    for mode in [0,1]:
        for index in [0,1]:
            p=nulls['independent_permutations'][index] if mode==0 else [nulls['shared_permutations'][index]]*2
            permuted=[brain[np.ix_(order,order)] for brain,order in zip(brains,p)]
            if mode==1:
                assert spearmanr(permuted[0][triangle],permuted[1][triangle]).statistic==pytest.approx(original_brain_correlation)
            scores=[]
            for matrix in permuted:
                result=metric(matrix)
                scores.append(np.column_stack([result[tuple(name.split('/'))] for name in names]))
            correlations=[spearmanr(scores[0][:,j],scores[1][:,j]).statistic for j in range(15)]
            assert np.allclose(correlations,nulls['cross_score_rho'][mode,index])
            residuals=[projection.residuals(values)[0] for values in scores]
            assert np.allclose(np.sum(residuals[0]*residuals[1],axis=0),nulls['partial_cross_score_rho'][mode,index])
            for j in range(15):
                tops=[np.lexsort((ids,-s[:,j])) for s in scores]
                expected=[len(set(tops[0][:k])&set(tops[1][:k]))/k for k in nulls['budgets']]
                assert np.allclose(expected,nulls['selection_overlap'][mode,index,j])


def test_null_reports_and_seeds_match_retained_samples(outputs):
    root,observed,nulls=outputs
    report=json.loads((root/'brain_specific_nulls.json').read_text())
    for j,name in enumerate(nulls['score_names']):
        actual=spearmanr(observed['scores'][0,:,j],observed['scores'][1,:,j]).statistic
        assert actual==pytest.approx(report['results'][name]['observed_cross_subject_spearman'])
        for mode,label in [(0,'independent_brain_null'),(1,'shared_brain_null')]:
            raw=nulls['cross_score_rho'][mode,:,j];stats=report['results'][name][label]
            assert len(raw)==1000 and stats['null_median']==pytest.approx(np.median(raw))
            assert np.allclose(stats['null_95_percentile_interval'],np.quantile(raw,[.025,.975]))
            assert stats['exploratory_one_sided_p']==pytest.approx((1+sum(raw>=actual))/1001)
    rng=np.random.default_rng(2026101501)
    assert np.array_equal(nulls['independent_permutations'][0],[rng.permutation(200),rng.permutation(200)])
    rng=np.random.default_rng(2026101502)
    assert np.array_equal(nulls['shared_permutations'][0],rng.permutation(200))
    rng=np.random.default_rng(2026101503)
    assert np.array_equal(nulls['score_label_permutations'][0],rng.permutation(200))


def test_selection_budgets_and_chance_adjustments(outputs):
    root,saved,nulls=outputs
    frame=pd.read_csv(root/'selection_budget_curves.csv')
    assert len(frame)==375 and set(frame.budget)=={10,20,40,60,100}
    names=saved['score_names'].tolist();ids=saved['image_ids']
    for _,row in frame.iterrows():
        j=names.index(f'{row.method}/{row.model}');k=int(row.budget)
        if row.comparison=='cross_subject_full':
            tops=[np.lexsort((ids,-s[:,j]))[:k] for s in saved['scores']]
            actual=len(set(tops[0])&set(tops[1]))/k
        else:
            s=0 if row.comparison=='within_sub01' else 1
            actual=[]
            for pair in saved['half_scores'][s]:
                tops=[np.lexsort((ids,-half[:,j]))[:k] for half in pair]
                actual.append(len(set(tops[0])&set(tops[1]))/k)
            actual=np.median(actual)
        assert row.overlap_fraction==pytest.approx(actual)
        assert row.chance_adjusted_overlap==pytest.approx((actual-row.null_median)/(1-row.null_median))


def test_input_and_code_provenance_preserved(outputs):
    root,_,_=outputs
    provenance=json.loads((root/'provenance.json').read_text())
    assert provenance['all_numpy_loads_allow_pickle_false'] and provenance['no_downloads']
    for path,digest in {**provenance['input_sha256'],**provenance['code_sha256']}.items():
        assert sha256(path)==digest
    old=root.parent/'cross_subject_01_02'
    assert sha256(old/'verified_outputs.json')==provenance['phase14_manifest_sha256']
    assert sha256(old/'provenance.json')==provenance['phase14_provenance_sha256']
