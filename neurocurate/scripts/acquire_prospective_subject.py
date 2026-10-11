"""Only standard subjects 03-05 archives, test-only extraction and confined export."""
import argparse,json,os,shutil,subprocess,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
import requests
from neurocurate.acquisition import acquire,extract_member,verify_artifact
from neurocurate.lowdisk import sha256

parser=argparse.ArgumentParser();parser.add_argument('subject',choices=['sub03','sub04','sub05']);subject=parser.parse_args().subject
out=Path('results/real_phase0/prospective_03_05');frozen=json.loads((out/'frozen_protocol.json').read_text())
for path,digest in frozen['files_sha256'].items(): assert sha256(path)==digest,path
pins={'sub03':('62f6b7060beb5f05deb0c554','fy4n6',1071567672,'ec12b31410b8161f5b274b4c61890a615e5fdbb5e58e6e9325a0188272daa936'),
 'sub04':('62f6b70721d7ec0d04c71205','py8jz',1071421033,'eec31590d21248d9cc8caf3b93f5711b2c24540f080eab4f83f152155f5a6eb5'),
 'sub05':('62f6b7085775130b6bf252d5','2cn7v',1072655693,'d353e154e840ff9c07b2de19db2105626df4bc556313365cff392dd56d8df40e')}
root=out/subject;root.mkdir(parents=True,exist_ok=True);work=Path('data/local/real_phase0')/subject;work.mkdir(parents=True,exist_ok=True)
status=dict(subject=subject,status='pending',started_at_utc=datetime.now(timezone.utc).isoformat())
def save(): (root/'subject_status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
try:
 file_id,guid,size,digest=pins[subject];name=f'sub-{subject[-2:]}.zip';api=f'https://api.osf.io/v2/files/{file_id}/'
 with requests.get(api,headers={'Accept-Encoding':'identity'},stream=True,allow_redirects=False,timeout=(10,30)) as response:
  assert response.status_code==200 and response.headers.get('Content-Encoding','identity')=='identity'
  data=response.raw.read(65537);assert len(data)<=65536
 document=json.loads(data);attrs=document['data']['attributes']
 assert attrs['name']==name and attrs['size']==size and attrs['extra']['hashes']['sha256']==digest
 (root/'official_source_metadata.json').write_text(json.dumps(document,indent=2)+'\n')
 free=shutil.disk_usage(work).free
 # Budget ALL three retained archives, test and numeric exports, outputs and reserve.
 peak=sum(v[2] for v in pins.values())+3*1024**3+512*1024**2+1024**3
 assert peak<=free,'conservative whole-cohort storage estimate exceeds available disk'
 artifact=dict(path=str((work/name).resolve()),url=f'https://osf.io/download/{guid}/',bytes=size,sha256=digest)
 manifest=dict(subject=subject,source_api=api,source_metadata_sha256=sha256(root/'official_source_metadata.json'),
  available_disk_before_bytes=free,conservative_cohort_peak_additional_bytes=peak,files=[],extracted=[])
 print(subject,'official metadata verified; downloading',size,'bytes',flush=True)
 manifest['files'].append(acquire(artifact,work.resolve(),1024**3))
 with zipfile.ZipFile(artifact['path']) as archive:
  members=[m for m in archive.infolist() if m.filename==f'sub-{subject[-2:]}/preprocessed_eeg_test.npy']
  assert len(members)==1 and 0<members[0].file_size<=512*1024**2
  member=members[0];path=extract_member(archive,member,work/'eeg')
  manifest['extracted'].append(dict(path=str(path.resolve()),archive=name,member=member.filename,bytes=path.stat().st_size,
   compressed_bytes=member.compress_size,sha256=sha256(path),crc32_verified=True))
 (root/'acquisition_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 eeg=manifest['extracted'][0]
 original=json.loads(Path('results/real_phase0/sub01/acquisition_manifest.json').read_text())
 metadata=next(a for a in original['files'] if Path(a['path']).name=='image_metadata.npy')
 assert metadata['sha256']=='681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c'
 for a in [artifact,eeg,metadata]: verify_artifact(a)
 export=root/'pickle_free';export.mkdir()
 names=['eeg.npy','times.npy','channels.npy','metadata.json','isolation_report.json']
 handles=[open(a['path'],'rb') for a in [eeg,metadata]]+[open(export/n,'xb') for n in names]
 try:
  fds=[h.fileno() for h in handles]
  result=subprocess.run([sys.executable,'-I','-u','scripts/_isolated_export.py',*map(str,fds),eeg['sha256'],metadata['sha256']],
   pass_fds=fds,close_fds=True,stdin=subprocess.DEVNULL,timeout=150,
   env={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','LC_ALL':'C.UTF-8'})
  assert result.returncode==0,'isolation/conversion failed; no downstream analysis'
 finally:
  for h in handles:h.close()
 report=json.loads((export/'isolation_report.json').read_text())
 assert report['all_exports_pickle_free'] and all(v=='denied' for v in report['probes'].values())
 (export/'export_manifest.json').write_text(json.dumps(dict(source_acquisition_manifest=str(root/'acquisition_manifest.json'),
  original_input_sha256=[eeg['sha256'],metadata['sha256']],isolation=report,exports={n:sha256(export/n) for n in names}),indent=2)+'\n')
 (export/'VERIFIED').write_text('isolated numeric export verified; downstream pickle forbidden\n')
 status.update(status='verified_numeric_export',available_disk_after_bytes=shutil.disk_usage(work).free)
 print(subject,'verified confined numeric export',flush=True)
except Exception as error:
 status.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
finally:save()
