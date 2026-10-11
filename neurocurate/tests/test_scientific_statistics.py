import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from neurocurate.rdm import compute_rdm


spec = importlib.util.spec_from_file_location("scientific_subject01", Path(__file__).resolve().parents[1] / "scripts/scientific_subject01.py")
scientific = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scientific)


def test_rank_matrix_permutation_matches_scipy_with_pair_dependence():
    rng = np.random.default_rng(42)
    left = compute_rdm(rng.normal(size=(12,8)))
    right = compute_rdm(rng.normal(size=(12,8)))
    tri = np.triu_indices(12,1)
    rank_left,_ = scientific.rank_vector(left,tri)
    rank_right,ranks = scientific.rank_vector(right,tri)
    assert np.isclose(rank_left @ rank_right,spearmanr(left[tri],right[tri]).statistic)
    rank_matrix = np.zeros((12,12));rank_matrix[tri]=ranks;rank_matrix+=rank_matrix.T
    for _ in range(5):
        p=rng.permutation(12)
        permuted=(rank_matrix[np.ix_(p,p)][tri]-ranks.mean())/np.linalg.norm(ranks-ranks.mean())
        assert np.isclose(rank_left @ permuted,spearmanr(left[tri],right[np.ix_(p,p)][tri]).statistic)


def test_rank_normalization_handles_ties():
    matrix=np.array([[0,1,1,2],[1,0,2,3],[1,2,0,3],[2,3,3,0]],dtype=float)
    tri=np.triu_indices(4,1)
    normalized,ranks=scientific.rank_vector(matrix,tri)
    assert ranks[0] == ranks[1]
    assert np.isclose(normalized @ normalized,1)
    assert np.isclose(normalized.sum(),0)


def test_uncertainty_is_split_variability_interval():
    result=scientific.interval([.2,.4,.6,.8])
    assert np.isclose(result['median'],.5)
    assert np.allclose(result['random_split_95_percentile_interval'],[.215,.785])
