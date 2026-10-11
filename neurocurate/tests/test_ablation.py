import numpy as np
import pytest
from scipy.stats import spearmanr
from neurocurate.ablation import DisagreementMetrics, row_ranks, score_nulls, unit_ranks, top_indices
from neurocurate.rdm import compute_rdm


def test_identical_geometry_has_zero_disagreement():
    matrix=compute_rdm(np.random.default_rng(3).normal(size=(12,8)))
    scores=DisagreementMetrics({'model':matrix},np.array([str(i) for i in range(12)]),ks=(2,5))(matrix)
    for values in scores.values():
        assert np.allclose(values,0,atol=1e-12)


def test_row_spearman_excludes_self_and_matches_scipy():
    rng=np.random.default_rng(4)
    a,b=[compute_rdm(rng.normal(size=(9,6))) for _ in range(2)]
    values=DisagreementMetrics({'model':b},np.array([str(i) for i in range(9)]),ks=(2,))(a)
    expected=[(1-spearmanr(np.delete(a[i],i),np.delete(b[i],i)).statistic)/2 for i in range(9)]
    assert np.allclose(values[('row_spearman','model')],expected)
    assert row_ranks(a).shape==(9,8)


def test_weighted_metric_uses_all_reciprocal_ranks():
    from neurocurate.ablation import reciprocal_weights
    neighbors=np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
    weights=reciprocal_weights(neighbors)
    assert np.allclose(weights.sum(axis=1),1)
    assert np.allclose(np.diag(weights),0)
    assert weights[0,1]/weights[0,3]==pytest.approx(3)
    with pytest.raises(ValueError,match='all nonself'):
        reciprocal_weights(neighbors[:,:2])


def test_tie_aware_null_reapplies_image_id_tie_break():
    ids=np.array(['z','a','y','b','x','c'])
    a=np.array([0,1,0,1,0,1.]);b=np.array([0,1,1,1,1,1.])
    permutations=np.array([np.arange(6),np.arange(6)[::-1],np.roll(np.arange(6),1)])
    rho,overlap,rnull,tnull=score_nulls(a,b,ids,permutations,top_n=2)
    top=set(top_indices(a,ids,2))
    expected=[len(top&set(top_indices(b[p],ids,2)))/2 for p in permutations]
    assert np.allclose(tnull,expected)
    assert rho==pytest.approx(spearmanr(a,b).statistic)
    assert np.allclose(rnull,[spearmanr(a,b[p]).statistic for p in permutations])
    assert overlap==expected[0]


def test_invalid_inputs_and_constant_spearman_are_rejected():
    with pytest.raises(ValueError,match='constant'):
        unit_ranks(np.ones(10))
    with pytest.raises(ValueError,match='finite'):
        row_ranks(np.array([[0,np.nan],[np.nan,0]]))
    with pytest.raises(ValueError,match='invalid'):
        top_indices([1,np.nan],['a','b'],1)
    with pytest.raises(ValueError,match='every image'):
        score_nulls([1,2,3],[3,1,2],np.array(['a','b','c']),[[0,0,2]],top_n=1)
