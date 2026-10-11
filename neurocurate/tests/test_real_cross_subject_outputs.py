"""Opt-in local real-data checks; no network and no pickle deserialization."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from neurocurate.rdm import compute_rdm, validate_rdm
from neurocurate.ablation import top_indices
from neurocurate.lowdisk import sha256


@pytest.fixture
def root():
    value=os.environ.get('NEUROCURATE_PHASE14_RESULTS')
    if not value:
        pytest.skip('set NEUROCURATE_PHASE14_RESULTS to local cross_subject_01_02 outputs')
    return Path(value)


def test_subject02_numeric_feature_rdm_and_isolation(root):
    subject=root.parent/'sub02'
    report=json.loads((subject/'pickle_free/isolation_report.json').read_text())
    assert report['uid']!=0 and report['network_access'] is False
    assert report['probes']==dict(filesystem_open='denied',network_socket='denied',process_creation='denied')
    raw=np.load(subject/'pickle_free/eeg.npy',allow_pickle=False,mmap_mode='r')
    times=np.load(subject/'pickle_free/times.npy',allow_pickle=False)
    mask=(times>=.1-1e-9)&(times<.5-1e-9)
    expected=raw.mean(axis=1)[:,:,mask].reshape(200,-1)
    features=np.load(subject/'eeg_features.npy',allow_pickle=False)
    assert raw.shape==(200,80,17,100) and features.shape==(200,680)
    assert np.allclose(expected,features,atol=1e-12)
    rdm=validate_rdm(np.load(subject/'brain_rdm.npy',allow_pickle=False))
    assert np.allclose(rdm,compute_rdm(expected),atol=1e-12)
    for s in ['sub01','sub02']:
        manifest=json.loads((root.parent/s/'pickle_free/export_manifest.json').read_text())
        for name,digest in manifest['exports'].items():
            assert sha256(root.parent/s/'pickle_free'/name)==digest


def test_real_cross_subject_rdm_and_frozen_order(root):
    ids=[np.load(root.parent/s/'image_ids.npy',allow_pickle=False) for s in ['sub01','sub02']]
    assert np.array_equal(*ids)
    triangle=np.triu_indices(200,1)
    matrices=[np.load(root.parent/s/'brain_rdm.npy',allow_pickle=False) for s in ['sub01','sub02']]
    actual=spearmanr(matrices[0][triangle],matrices[1][triangle]).statistic
    report=json.loads((root/'cross_subject_rdm.json').read_text())
    assert actual==pytest.approx(report['cross_subject_spearman'])
    assert report['conditional_on_shared_preprocessing']
    prior=json.loads((root.parent/'sub01/scientific_provenance.json').read_text())
    for name,digest in prior['output_sha256'].items():
        assert sha256(root.parent/'sub01'/name)==digest


def test_repetition_splits_preserve_image_correspondence(root):
    for subject in ['sub01','sub02']:
        with np.load(root/f'{subject}_half_rdms.npz',allow_pickle=False) as saved:
            indices=saved['repetition_indices'];matrices=saved['rdms']
        assert indices.shape==(30,2,40) and matrices.shape==(30,2,200,200)
        for halves in indices:
            assert not set(halves[0])&set(halves[1])
            assert set(halves.ravel())==set(range(80))
        report=json.loads((root/'cross_subject_rdm.json').read_text())['within_subject_repetition_reliability'][subject]
        triangle=np.triu_indices(200,1)
        computed=[spearmanr(pair[0][triangle],pair[1][triangle]).statistic for pair in matrices]
        assert np.allclose(computed,report['split_spearman'])
        raw=np.load(root.parent/subject/'pickle_free/eeg.npy',allow_pickle=False,mmap_mode='r')
        times=np.load(root.parent/subject/'pickle_free/times.npy',allow_pickle=False)
        mask=(times>=.1-1e-9)&(times<.5-1e-9)
        # Independently recompute one pair directly from numeric repetitions.
        for h in range(2):
            features=raw[:,indices[0,h]].mean(axis=1)[:,:,mask].reshape(200,-1)
            assert np.allclose(matrices[0,h],compute_rdm(features),atol=1e-12)


def test_disagreement_cross_correlations_and_top20(root):
    frame=pd.read_csv(root/'image_level_scores.csv')
    report=json.loads((root/'cross_subject_disagreement.json').read_text())['results']
    assert len(report)==15
    for key,result in report.items():
        method,model=key.split('/')
        subset=frame[(frame.method==method)&(frame.model==model)]
        a,b=[subset[subset.subject==s].sort_values('image_id') for s in ['sub01','sub02']]
        assert np.array_equal(a.image_id,b.image_id) and len(a)==200
        rho=spearmanr(a.score,b.score).statistic
        assert rho==pytest.approx(result['cross_subject_spearman'])
        ids=a.image_id.to_numpy();topa=top_indices(a.score.to_numpy(),ids);topb=top_indices(b.score.to_numpy(),ids)
        assert len(set(topa)&set(topb))==result['top20_shared_images']
        for s in ['sub01','sub02']:
            statistics=result['subjects'][s]
            assert len(statistics['raw_splits']['spearman'])==30
            assert 1/1001<=statistics['tie_aware_top20_null']['one_sided_monte_carlo_p']<=1
