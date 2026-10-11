"""An explicit research score, not a learned or calibrated quality estimate."""

from itertools import combinations
import numpy as np
from .metadata import validate_ids


def _neighbors(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] < 2 or not 1 <= values.shape[1] < values.shape[0]:
        raise ValueError("neighbors must be (images, k), 1 <= k < images")
    if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= len(values)):
        raise ValueError("neighbor indices are out of range or not integers")
    if any(len(set(row)) != len(row) or i in row for i, row in enumerate(values)):
        raise ValueError("duplicate neighbors or self-neighbor")
    return values


def jaccard_disagreement(left, right):
    left, right = _neighbors(left), _neighbors(right)
    if left.shape != right.shape:
        raise ValueError("neighborhood shapes must match")
    return np.array([1.0 - len(set(a) & set(b)) / len(set(a) | set(b))
                     for a, b in zip(left, right)])


def cross_subject_consistency(subject_neighbors):
    """Per-image mean pairwise neighborhood Jaccard similarity."""
    subjects = list(subject_neighbors)
    if len(subjects) < 2:
        raise ValueError("cross-subject consistency requires at least two subjects")
    return np.mean([1.0 - jaccard_disagreement(a, b)
                    for a, b in combinations(subjects, 2)], axis=0)


def neurocurate_scores(image_ids, subject_neighbors, model_neighbors, model_weights=None):
    """Score_i = consistency_i * weighted mean_model mean_subject disagreement_i."""
    import pandas as pd
    ids = validate_ids(image_ids)
    if not model_neighbors:
        raise ValueError("at least one vision model is required")
    subjects = [_neighbors(s) for s in subject_neighbors]
    consistency = cross_subject_consistency(subjects)
    if len(consistency) != len(ids):
        raise ValueError("image IDs do not match neighborhood rows")
    names = sorted(model_neighbors)
    if model_weights is not None and set(model_weights) != set(names):
        raise ValueError("model_weights must cover exactly the configured models")
    weights = np.array([1 if model_weights is None else model_weights[n] for n in names], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("model weights must be finite, nonnegative, with a positive sum")
    columns = {"image_id": ids, "consistency": consistency}
    disagreements = []
    for name in names:
        model = _neighbors(model_neighbors[name])
        disagreement = np.mean([jaccard_disagreement(s, model) for s in subjects], axis=0)
        columns[f"disagreement_{name}"] = disagreement
        disagreements.append(disagreement)
    columns["disagreement"] = np.average(disagreements, axis=0, weights=weights)
    columns["score"] = consistency * columns["disagreement"]
    table = pd.DataFrame(columns).sort_values(["score", "image_id"], ascending=[False, True]).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table
