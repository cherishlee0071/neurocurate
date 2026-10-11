"""Optional Phase 1.3 output verification; never deserializes source pickle."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import spearmanr
from neurocurate.lowdisk import sha256
from neurocurate.rdm import compute_rdm, validate_rdm
from neurocurate.vision import load_feature_cache


@pytest.fixture(scope="module")
def science_root():
    value=os.environ.get("NEUROCURATE_PHASE13_RESULTS")
    if not value:
        pytest.skip("optional real scientific outputs not selected")
    root=Path(value)
    assert (root/"scientific_provenance.json").is_file()
    return root


def test_isolated_export_evidence_and_hashes(science_root):
    export=science_root/"pickle_free"
    manifest=json.loads((export/"export_manifest.json").read_text())
    for name,digest in manifest['exports'].items():
        assert sha256(export/name)==digest
    evidence=manifest['isolation']
    assert not evidence['root'] and not evidence['network_access']
    assert all(v=='denied' for v in evidence['probes'].values())
    assert evidence['address_space_limit_bytes']==4*1024**3
    assert evidence['cpu_limit_seconds']==120
    assert not set(evidence['allowed_syscalls']) & {'socket','open','openat','execve','clone','fork','ptrace','seccomp'}


def test_real_saved_features_rdms_and_provenance(science_root):
    provenance=json.loads((science_root/'scientific_provenance.json').read_text())
    for name,digest in provenance['output_sha256'].items():
        assert sha256(science_root/name)==digest
    ids=np.load(science_root/'image_ids.npy',allow_pickle=False)
    assert ids.shape==(200,) and len(set(ids))==200
    raw=np.load(science_root/'pickle_free/eeg.npy',allow_pickle=False,mmap_mode='r')
    times=np.load(science_root/'pickle_free/times.npy',allow_pickle=False)
    mask=(times>=.1-1e-9)&(times<.5-1e-9)
    reference=raw.mean(axis=1)[:,:,mask].reshape(200,-1)
    features=np.load(science_root/'eeg_features.npy',allow_pickle=False)
    assert features.shape==(200,680) and np.allclose(reference,features,atol=1e-12)
    brain=validate_rdm(np.load(science_root/'brain_rdm.npy',allow_pickle=False))
    assert np.allclose(brain,compute_rdm(reference),atol=1e-12)
    for kind in ['dinov2','clip']:
        cached,features,provenance=load_feature_cache(science_root/f'{kind}_features.npz',split='sanity',model_name=kind)
        assert np.array_equal(cached,ids) and provenance['frozen']
        rdm=validate_rdm(np.load(science_root/f'{kind}_rdm.npy',allow_pickle=False))
        assert np.allclose(rdm,compute_rdm(features,'cosine'),atol=1e-12)


def test_real_split_correspondence_and_permutation_p(science_root):
    report=json.loads((science_root/'brain_reliability.json').read_text())
    assert report['conditional_on_shared_preprocessing'] and not report['session_independent']
    raw=np.load(science_root/'pickle_free/eeg.npy',allow_pickle=False,mmap_mode='r')
    times=np.load(science_root/'pickle_free/times.npy',allow_pickle=False)
    mask=(times>=.1-1e-9)&(times<.5-1e-9)
    triangle=np.triu_indices(200,1)
    assert len(report['splits'])==30
    for split in report['splits']:
        a,b=split['half_a'],split['half_b']
        assert len(set(a))==len(set(b))==40 and not set(a)&set(b)
        assert set(a)|set(b)==set(range(80))
    for split in report['splits'][:2]:
        rdms=[]
        for indices in [split['half_a'],split['half_b']]:
            features=raw[:,indices].mean(axis=1)[:,:,mask].reshape(200,-1)
            rdms.append(compute_rdm(features))
        assert np.isclose(spearmanr(rdms[0][triangle],rdms[1][triangle]).statistic,split['spearman'])
    with np.load(science_root/'permutation_nulls.npz',allow_pickle=False) as arrays:
        null=arrays['rdm_median_null']
    assert null.shape==(1000,)
    observed=np.median([s['spearman'] for s in report['splits']])
    assert np.isclose((1+np.sum(null>=observed))/1001,report['permutation']['one_sided_monte_carlo_p'])
