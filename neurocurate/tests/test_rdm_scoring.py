import numpy as np
import pytest
from neurocurate.rdm import compute_rdm, nearest_neighbors
from neurocurate.scoring import jaccard_disagreement, cross_subject_consistency, neurocurate_scores


@pytest.mark.parametrize("metric", ["correlation", "cosine", "euclidean"])
def test_rdm_symmetry_diagonal(metric):
    features = [[1, 2, 3], [3, 2, 1], [2, 1, 3]]
    rdm = compute_rdm(features, metric)
    np.testing.assert_allclose(rdm, rdm.T)
    np.testing.assert_array_equal(np.diag(rdm), 0)
    assert np.isfinite(rdm).all()


def test_known_correlation_distances():
    rdm = compute_rdm([[1, 2, 3], [3, 2, 1], [2, 4, 6]])
    np.testing.assert_allclose(rdm, [[0, 2, 0], [2, 0, 2], [0, 2, 0]], atol=1e-12)


@pytest.mark.parametrize("features,metric", [([[1, 1], [2, 3]], "correlation"),
    ([[0, 0], [1, 2]], "cosine"), ([[1, np.nan], [2, 3]], "euclidean"),
    ([[1, 2]], "cosine"), ([[1, 2], [2, 3]], "unsupported")])
def test_invalid_rdm_inputs(features, metric):
    with pytest.raises(ValueError): compute_rdm(features, metric)


def test_nearest_neighbors_distances_and_ties():
    rdm = compute_rdm([[0], [1], [2], [3]], "euclidean")
    neighbors = nearest_neighbors(rdm, 2, ["d", "c", "b", "a"])
    np.testing.assert_array_equal(neighbors, [[1, 2], [2, 0], [3, 1], [2, 1]])
    assert all(i not in row for i, row in enumerate(neighbors))


@pytest.mark.parametrize("k", [0, 3, -1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError): nearest_neighbors(np.zeros((3, 3)), k)


@pytest.mark.parametrize("rdm", [np.ones((3, 3)), np.array([[0, 1], [2, 0]]),
                                  np.array([[0, -1], [-1, 0]]), np.array([[0, np.nan], [np.nan, 0]])])
def test_invalid_neighbor_rdm(rdm):
    with pytest.raises(ValueError): nearest_neighbors(rdm, 1)


def test_jaccard_expected_value():
    a = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    b = np.array([[1, 3], [0, 3], [0, 3], [0, 2]])
    np.testing.assert_allclose(jaccard_disagreement(a, b), 2 / 3)
    np.testing.assert_allclose(jaccard_disagreement(a, a), 0)


def test_cross_subject_consistency():
    a = np.array([[1], [0], [0]])
    b = np.array([[2], [2], [1]])
    np.testing.assert_allclose(cross_subject_consistency([a, a]), 1)
    np.testing.assert_allclose(cross_subject_consistency([a, b]), 0)
    np.testing.assert_allclose(cross_subject_consistency([a, a, b]), 1 / 3)
    with pytest.raises(ValueError, match="two subjects"):
        cross_subject_consistency([a])


def test_score_ranking_and_consistency_gate():
    a = np.array([[1], [0], [0], [0]])
    b = np.array([[1], [0], [1], [0]])
    model = np.array([[2], [0], [3], [0]])
    scores = neurocurate_scores(["a", "b", "c", "d"], [a, b], {"clip": model})
    assert scores.image_id.tolist() == ["a", "b", "c", "d"]
    np.testing.assert_allclose(scores.score, [1, 0, 0, 0])
    assert scores.loc[scores.image_id == "c", "disagreement"].item() == 1
    assert scores.loc[scores.image_id == "c", "consistency"].item() == 0


def test_model_weights():
    brain = np.array([[1], [0], [0]])
    other = np.array([[2], [2], [1]])
    scores = neurocurate_scores(["a", "b", "c"], [brain, brain],
                               {"clip": brain, "dinov2": other}, {"clip": 1, "dinov2": 3})
    np.testing.assert_allclose(scores.score, 0.75)
    for weights in [{"clip": 1}, {"clip": -1, "dinov2": 2}, {"clip": 0, "dinov2": 0}]:
        with pytest.raises(ValueError):
            neurocurate_scores(["a", "b", "c"], [brain, brain], {"clip": brain, "dinov2": other}, weights)


@pytest.mark.parametrize("bad", [[[0], [0], [0]], [[1, 1], [0, 2], [0, 1]],
                                  [[4], [0], [0]], [[1.0], [0.0], [0.0]]])
def test_invalid_neighbors(bad):
    with pytest.raises(ValueError): jaccard_disagreement(bad, bad)
