"""Geometry-preserving brain nulls and descriptive selection controls."""
import numpy as np
from scipy.stats import rankdata
from .ablation import top_indices
from .rdm import validate_rdm, nearest_neighbors


def permute_brain(rdm, permutation):
    matrix=validate_rdm(rdm)
    p=np.asarray(permutation)
    if p.dtype.kind not in 'iu' or p.shape!=(len(matrix),) or not np.array_equal(np.sort(p),np.arange(len(matrix))):
        raise ValueError('permutation must contain every image exactly once')
    return matrix[np.ix_(p,p)]


def normalized_score_ranks(scores):
    values=np.asarray(scores,dtype=float)
    if values.ndim not in (1,2) or not np.isfinite(values).all():
        raise ValueError('scores must be finite vectors or column matrices')
    ranked=rankdata(values,axis=0)
    ranked-=ranked.mean(axis=0)
    norm=np.linalg.norm(ranked,axis=0)
    if np.any(norm==0):
        raise ValueError('constant score ranking is undefined')
    return ranked/norm


def model_only_scores(models, ids, row_disagreement, k=20):
    controls={'dinov2_clip_row_disagreement':np.asarray(row_disagreement)}
    for name,rdm in models.items():
        matrix=validate_rdm(rdm)
        neighbors=nearest_neighbors(matrix,k,ids)
        controls[f'{name}_mean_nearest{k}_cosine_distance']=matrix[np.arange(len(ids))[:,None],neighbors].mean(axis=1)
        controls[f'{name}_mean_nonself_cosine_distance']=matrix.sum(axis=1)/(len(ids)-1)
    return controls


class RankControls:
    """OLS projection of score ranks on fixed model-only covariates, no prediction."""
    def __init__(self, covariates):
        values=np.asarray(covariates,dtype=float)
        if values.ndim!=2 or values.shape[0]<=values.shape[1]+2:
            raise ValueError('insufficient images for rank controls')
        ranks=normalized_score_ranks(values)*np.sqrt(len(values))
        self.design=np.column_stack([np.ones(len(values)),ranks])
        self.rank=int(np.linalg.matrix_rank(self.design))
        if self.rank!=self.design.shape[1]:
            raise ValueError('rank-control design is collinear')
        self.condition_number=float(np.linalg.cond(self.design))
        self.q=np.linalg.qr(self.design,mode='reduced')[0]

    def residuals(self, scores):
        ranked=normalized_score_ranks(scores)
        residual=ranked-self.q@(self.q.T@ranked)
        norm=np.linalg.norm(residual,axis=0)
        if np.any(norm<1e-12):
            raise ValueError('model controls explain the entire score rank')
        return residual/norm,1-norm**2


def selection_statistics(a,b,ids,permutations,budgets):
    """Score-label null only; this does not recompute brain–model disagreement."""
    n=len(ids);p=np.asarray(permutations)
    if p.dtype.kind not in 'iu' or p.ndim!=2 or p.shape[1]!=n or not np.all(np.sort(p,axis=1)==np.arange(n)):
        raise ValueError('invalid score-label permutations')
    order_a=top_indices(a,ids,n);order_b=top_indices(b,ids,n)
    id_order=np.argsort(ids,kind='stable')
    null_order=id_order[np.argsort(-np.asarray(b)[p][:,id_order],axis=1,kind='stable')]
    observed=[];null=[]
    for budget in budgets:
        if isinstance(budget,bool) or not isinstance(budget,(int,np.integer)) or not 1<=budget<=n:
            raise ValueError('invalid selection budget')
        mask=np.zeros(n,dtype=bool);mask[order_a[:budget]]=True
        observed.append(float(mask[order_b[:budget]].mean()))
        null.append(mask[null_order[:,:budget]].mean(axis=1))
    return np.asarray(observed),np.asarray(null).T


def overlap_budgets(a,b,ids,budgets):
    first=top_indices(a,ids,len(ids));second=top_indices(b,ids,len(ids))
    return np.array([len(set(first[:k])&set(second[:k]))/k for k in budgets])


def chance_adjusted(observed,null_reference):
    """Normalize excess overlap above an explicit null, without clipping negatives."""
    observed=np.asarray(observed);reference=np.asarray(null_reference)
    if np.any(reference>=1) or np.any(reference<0):
        raise ValueError('chance adjustment undefined for null overlap of one')
    return (observed-reference)/(1-reference)
