"""Local artifact invariants for the exploratory Phase 0B audit."""
from pathlib import Path
import hashlib, json
import numpy as np

ROOT=Path('results/time2depth/phase0b')
def test_phase0_reproduction_and_required_outputs():
    r=json.loads((ROOT/'phase0_reproduction.json').read_text())
    assert r['within_tolerance'] is True and r['ordered_ids_unique'] is True
    for name in ['window_reliability.csv','preferred_depth_split_stability.csv','temporal_order_split_stability.json','confound_sensitivity.csv','confound_analysis.md','reliability_matched_controls.json','temporal_resolution_sensitivity.csv','scientific_summary.md','provenance.json']:
        assert (ROOT/name).is_file()
def test_split_integrity_and_real_shapes():
    z=np.load('results/time2depth/phase0b/brain_rdms_by_window.npz',allow_pickle=False)
    assert z['rdms'].shape==(5,5,200,200)
    assert np.isfinite(z['rdms']).all()
    assert np.allclose(z['rdms'],z['rdms'].swapaxes(-1,-2))
    s=json.loads((ROOT/'temporal_order_split_stability.json').read_text())
    assert len(s['splits'])==150
    assert all(0 <= x['argmax_agreement'] <= 1 for x in s['splits'])
def test_noise_control_is_predeclared_and_complete():
    z=json.loads((ROOT/'reliability_matched_controls.json').read_text())
    assert z['seeds']==[20261021,20261022,20261023]
    assert z['noise_scales']==[0.0,0.25,0.5,1.0]
    assert len(z['controls'])==60
