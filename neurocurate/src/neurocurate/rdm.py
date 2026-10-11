"""Representational distances and deterministic nearest neighbors."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from .metadata import validate_ids


def compute_rdm(features, metric="correlation"):
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or len(values) < 2:
        raise ValueError("features must be (at least two images, nonempty features)")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite features")
    if metric not in {"correlation", "cosine", "euclidean"}:
        raise ValueError("metric must be correlation, cosine, or euclidean")
    if metric == "correlation" and np.any(values.std(axis=1) <= 1e-12):
        raise ValueError("correlation distance undefined for constant feature vectors")
    if metric == "cosine" and np.any(np.linalg.norm(values, axis=1) <= 1e-12):
        raise ValueError("cosine distance undefined for zero feature vectors")
    rdm = squareform(pdist(values, metric=metric))
    if not np.isfinite(rdm).all():
        raise ValueError("distance computation produced nonfinite values")
    rdm = np.maximum(rdm, 0.0)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def validate_rdm(rdm):
    values = np.asarray(rdm, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or len(values) < 2:
        raise ValueError("RDM must be square with at least two images")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("RDM must contain finite nonnegative distances")
    if not np.allclose(values, values.T, atol=1e-10, rtol=0):
        raise ValueError("RDM must be symmetric")
    if not np.allclose(np.diag(values), 0, atol=1e-10, rtol=0):
        raise ValueError("RDM diagonal must be zero")
    return values


def nearest_neighbors(rdm, k, image_ids=None):
    values = validate_rdm(rdm)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k < len(values):
        raise ValueError("k must be an integer in [1, number of images - 1]")
    ids = validate_ids(image_ids) if image_ids is not None else np.array([f"{i:012d}" for i in range(len(values))])
    if len(ids) != len(values):
        raise ValueError("image IDs do not match RDM")
    distances = values.copy()
    np.fill_diagonal(distances, np.inf)
    return np.stack([np.lexsort((ids, row))[:k] for row in distances])
