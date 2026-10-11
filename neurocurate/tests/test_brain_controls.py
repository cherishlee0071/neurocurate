import numpy as np
import pytest
from scipy.stats import rankdata
from neurocurate.rdm import compute_rdm
from neurocurate.ablation import DisagreementMetrics,top_indices
from neurocurate.brain_controls import (permute_brain,normalized_score_ranks,RankControls,
    selection_statistics,chance_adjusted,model_only_scores)


def test_brain_permutation_preserves_geometry_not_score_labels():
    rng=np.random.default_rng(42);n=11
    brain,model=[compute_rdm(rng.normal(size=(n,7))) for _ in range(2)]
    p=rng.permutation(n);ids=np.array([str(i) for i in range(n)])
    permuted=permute_brain(brain,p);triangle=np.triu_indices(n,1)
    assert np.array_equal(np.sort(brain[triangle]),np.sort(permuted[triangle]))
    metric=DisagreementMetrics({'model':model},ids,(3,))
    original=metric(brain)[('row_spearman','model')]
    recomputed=metric(permuted)[('row_spearman','model')]
    assert not np.allclose(recomputed,original[p])
    assert np.allclose(metric(permute_brain(brain,np.arange(n)))[('row_spearman','model')],original)
    with pytest.raises(ValueError,match='every image'):
        permute_brain(brain,np.zeros(n,dtype=int))


def test_rank_controls_match_direct_ols_and_remove_covariate_association():
    rng=np.random.default_rng(6);covariates=rng.normal(size=(40,3));scores=rng.normal(size=(40,2))
    controls=RankControls(covariates);residual,r_squared=controls.residuals(scores)
    ranked=normalized_score_ranks(scores)
    direct=ranked-controls.design@np.linalg.lstsq(controls.design,ranked,rcond=None)[0]
    norms=np.linalg.norm(direct,axis=0)
    assert np.allclose(residual,direct/norms)
    assert np.allclose(r_squared,1-norms**2)
    assert np.allclose(controls.design.T@residual,0,atol=1e-12)
    with pytest.raises(ValueError,match='collinear'):
        RankControls(np.column_stack([covariates[:,0],covariates[:,0]]))


def test_selection_nulls_reapply_ties_at_every_budget():
    ids=np.array(['z','a','y','b','x','c']);a=np.array([0,1,0,1,0,1.]);b=np.array([0,1,1,1,1,1.])
    ps=np.array([np.arange(6),np.arange(6)[::-1]])
    budgets=[1,2,4]
    observed,null=selection_statistics(a,b,ids,ps,budgets)
    for i,k in enumerate(budgets):
        topa=set(top_indices(a,ids,k))
        expected=[len(topa&set(top_indices(b[p],ids,k)))/k for p in ps]
        assert np.allclose(null[:,i],expected) and observed[i]==expected[0]
    # An all-tied score is handled for selection; it need not have a Spearman.
    obs,null=selection_statistics(np.ones(6),np.ones(6),ids,ps,[2])
    assert obs[0]==1 and np.all(null==1)
    with pytest.raises(ValueError,match='undefined'):
        chance_adjusted(1,1)
    assert chance_adjusted(.1,.2)==pytest.approx(-.125)


def test_model_only_atypicality_excludes_self():
    ids=np.array(['a','b','c','d'])
    matrix=compute_rdm(np.array([[0],[1],[3],[7]]),'euclidean')
    controls=model_only_scores({'dinov2':matrix,'clip':matrix},ids,np.zeros(4),k=2)
    assert np.allclose(controls['dinov2_mean_nearest2_cosine_distance'],[2,1.5,2.5,5])
    assert np.allclose(controls['clip_mean_nonself_cosine_distance'],matrix.sum(axis=1)/3)


def test_invalid_rank_scores_rejected():
    with pytest.raises(ValueError,match='constant'):
        normalized_score_ranks(np.ones(10))
    with pytest.raises(ValueError,match='finite'):
        normalized_score_ranks([1,np.nan])
