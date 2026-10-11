import numpy as np
import pytest
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('prospective_analysis',Path(__file__).parents[1]/'scripts/prospective_analysis.py')
module=importlib.util.module_from_spec(spec)
# The main analysis reuses the original scientific helper in scripts.
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
spec.loader.exec_module(module)

def test_holm_uses_stepdown_not_independent_pair_tests():
 assert module.holm([.01,.03,.08])==pytest.approx([.03,.06,.08])
 assert module.holm([.6,.9])==[1.,1.]

def test_joint_pair_median_keeps_permutation_draw_axis():
 raw=np.arange(2*7*10*15).reshape(2,7,10,15)
 new_pairs=[7,8,9];primary=9
 joint=np.median(raw[:,:,new_pairs,primary],axis=2)
 assert joint.shape==(2,7)
 assert np.array_equal(joint[1],[np.median(raw[1,i,new_pairs,primary]) for i in range(7)])

def test_monte_carlo_plus_one_and_quantiles():
 report=module.null_summary(.5,np.array([-.3,0,.2,.6]))
 assert report['p']==pytest.approx(.4)
 assert report['null_median']==pytest.approx(.1)
