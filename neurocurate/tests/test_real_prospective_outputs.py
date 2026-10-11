import json,os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from neurocurate.rdm import compute_rdm,validate_rdm
from neurocurate.ablation import DisagreementMetrics
from neurocurate.lowdisk import sha256

@pytest.fixture
def root():
 path=os.environ.get('NEUROCURATE_PHASE16_RESULTS')
 if not path:pytest.skip('optional local prospective results not selected')
 return Path(path)

def test_snapshot_and_all_three_numeric_exports(root):
 frozen=json.loads((root/'frozen_protocol.json').read_text())
 assert sha256(root/'protocol_snapshot.tar')==frozen['snapshot_sha256']
 for path,digest in frozen['files_sha256'].items():assert sha256(path)==digest
 for subject in ['sub03','sub04','sub05']:
  folder=root/subject;exports=folder/'pickle_free';report=json.loads((exports/'isolation_report.json').read_text())
  assert report['uid']!=0 and all(v=='denied' for v in report['probes'].values())
  manifest=json.loads((exports/'export_manifest.json').read_text())
  for name,digest in manifest['exports'].items():assert sha256(exports/name)==digest
  raw=np.load(exports/'eeg.npy',allow_pickle=False,mmap_mode='r');times=np.load(exports/'times.npy',allow_pickle=False)
  mask=(times>=.1-1e-9)&(times<.5-1e-9);features=raw.mean(axis=1)[:,:,mask].reshape(200,-1)
  assert raw.shape==(200,80,17,100) and np.allclose(features,np.load(folder/'eeg_features.npy',allow_pickle=False))
  assert np.allclose(validate_rdm(np.load(folder/'brain_rdm.npy',allow_pickle=False)),compute_rdm(features))

def test_pair_correlations_and_joint_cohort_null(root):
 with np.load(root/'observed_scores.npz',allow_pickle=False) as saved:subjects=saved['subjects'].tolist();scores=saved['scores'];names=saved['score_names'].tolist()
 with np.load(root/'null_statistics.npz',allow_pickle=False) as saved:pairs=saved['pairs'];null=saved['rank_null']
 table=pd.read_csv(root/'cross_subject_disagreement.csv');primary=names.index('row_spearman/mean_models')
 new=[];values=[]
 for i,(a,b) in enumerate(pairs):
  rho=spearmanr(scores[a,:,primary],scores[b,:,primary]).statistic
  r=table[(table.subject_a==subjects[a])&(table.subject_b==subjects[b])&(table.method=='row_spearman')&(table.model=='mean_models')].iloc[0]
  assert rho==pytest.approx(r.rank_spearman)
  if subjects[a] in ['sub03','sub04','sub05'] and subjects[b] in ['sub03','sub04','sub05']:new.append(i);values.append(rho)
 assert len(pairs)==10 and len(new)==3
 report=json.loads((root/'brain_specific_nulls.json').read_text())['replication_cohort']['primary_shared']
 joint=np.median(null[1,:,new,primary],axis=0)
 assert joint.shape==(1000,)
 assert report['observed']==pytest.approx(np.median(values))
 assert report['null_median']==pytest.approx(np.median(joint))
 assert report['p']==pytest.approx((1+sum(joint>=np.median(values)))/1001)

def test_joint_brain_null_recomputes_disagreement(root):
 with np.load(root/'null_statistics.npz',allow_pickle=False) as saved:subjects=saved['subjects'].tolist();pairs=saved['pairs'];permutations=saved['shared_permutations'];null=saved['rank_null']
 base=root.parent;ids=np.load(base/'sub01/image_ids.npy',allow_pickle=False)
 models={k:np.load(base/'sub01'/f'{k}_rdm.npy',allow_pickle=False) for k in ['dinov2','clip']};metric=DisagreementMetrics(models,ids,[5,10,20])
 vectors=[]
 for subject in subjects:
  folder=base/subject if subject in ['sub01','sub02'] else root/subject
  brain=np.load(folder/'brain_rdm.npy',allow_pickle=False);p=permutations[0]
  vectors.append(metric(brain[np.ix_(p,p)])[('row_spearman','mean_models')])
 for i,(a,b) in enumerate(pairs):assert spearmanr(vectors[a],vectors[b]).statistic==pytest.approx(null[1,0,i,11])
