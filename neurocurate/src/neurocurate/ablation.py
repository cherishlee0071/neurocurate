"""Predeclared, unfitted disagreement metrics and image-label null controls."""
import numpy as np
from scipy.stats import rankdata
from .metadata import validate_ids
from .rdm import nearest_neighbors, validate_rdm
from .scoring import jaccard_disagreement


def unit_ranks(values):
    ranks = rankdata(values).astype(float)
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    if norm == 0:
        raise ValueError("Spearman correlation undefined for constant scores")
    return ranks / norm


def row_ranks(rdm):
    values = validate_rdm(rdm)
    n = len(values)
    rows = values[~np.eye(n, dtype=bool)].reshape(n, n-1)
    ranks = rankdata(rows, axis=1)
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("row Spearman undefined for constant off-diagonal rows")
    return ranks / norms


def reciprocal_weights(neighbors):
    n, count = neighbors.shape
    if count != n-1:
        raise ValueError("reciprocal weighting requires all nonself neighbors")
    weights = 1 / np.arange(1, n, dtype=float)
    weights /= weights.sum()
    matrix = np.zeros((n, n))
    matrix[np.arange(n)[:, None], neighbors] = weights
    return matrix


class DisagreementMetrics:
    """Cache the fixed model geometry; compute each brain geometry once."""
    def __init__(self, models, image_ids, ks=(5, 10, 20)):
        self.ids = validate_ids(image_ids)
        self.ks = tuple(ks)
        self.models = {}
        for name, matrix in models.items():
            neighbors = nearest_neighbors(matrix, len(self.ids)-1, self.ids)
            self.models[name] = (neighbors, row_ranks(matrix), reciprocal_weights(neighbors))
        if not self.models:
            raise ValueError("at least one fixed vision model required")

    def __call__(self, brain):
        neighbors = nearest_neighbors(brain, len(self.ids)-1, self.ids)
        ranks, weights = row_ranks(brain), reciprocal_weights(neighbors)
        scores = {}
        for model, (mn, mr, mw) in self.models.items():
            for k in self.ks:
                if not 1 <= k < len(self.ids):
                    raise ValueError("invalid neighborhood K")
                scores[(f"jaccard_k{k}", model)] = jaccard_disagreement(neighbors[:, :k], mn[:, :k])
            scores[("row_spearman", model)] = np.clip((1 - np.sum(ranks*mr, axis=1))/2, 0, 1)
            scores[("reciprocal_rank_weighted_jaccard", model)] = 1 - np.minimum(weights, mw).sum(axis=1)/np.maximum(weights, mw).sum(axis=1)
        for method in dict.fromkeys(key[0] for key in scores):
            scores[(method, "mean_models")] = np.mean([scores[(method, m)] for m in self.models], axis=0)
        return scores


def top_indices(scores, image_ids, top_n=20):
    ids = validate_ids(image_ids)
    values = np.asarray(scores, dtype=float)
    if values.shape != ids.shape or not np.isfinite(values).all() or not 1 <= top_n <= len(ids):
        raise ValueError("invalid scores, IDs, or top_n")
    return np.lexsort((ids, -values))[:top_n]


def score_nulls(a, b, ids, permutations, top_n=20):
    """Relabel B, then reapply score ordering and the same ID tie rule."""
    top_a = top_indices(a, ids, top_n)
    top_b = top_indices(b, ids, top_n)
    ra, rb = unit_ranks(a), unit_ranks(b)
    permutations = np.asarray(permutations)
    n = len(ids)
    if permutations.ndim != 2 or permutations.shape[1] != n or not np.all(np.sort(permutations, axis=1) == np.arange(n)):
        raise ValueError("each permutation must contain every image exactly once")
    id_order = np.argsort(ids, kind="stable")
    relabeled = np.asarray(b)[permutations][:, id_order]
    permuted_top = id_order[np.argsort(-relabeled, axis=1, kind="stable")[:, :top_n]]
    in_a = np.zeros(n, dtype=bool)
    in_a[top_a] = True
    overlap = len(set(top_a) & set(top_b)) / top_n
    return (float(ra @ rb), overlap,
            rb[permutations] @ ra, in_a[permuted_top].mean(axis=1))


def distribution(scores, ids, top_n=20):
    scores = np.asarray(scores)
    top = top_indices(scores, ids, top_n)
    boundary = scores[top[-1]]
    unique, counts = np.unique(scores, return_counts=True)
    return dict(min=float(scores.min()), max=float(scores.max()), mean=float(scores.mean()),
                standard_deviation=float(scores.std()), quantiles=np.quantile(scores, [0, .25, .5, .75, 1]).tolist(),
                unique_values=len(unique), tied_images=int(counts[counts > 1].sum()),
                tie_groups=int((counts > 1).sum()), top20_boundary_score=float(boundary),
                top20_boundary_tie_size=int((scores == boundary).sum()),
                images_strictly_above_boundary=int((scores > boundary).sum()),
                top20_ids=np.asarray(ids)[top].tolist())
