"""One official subject-02 ZIP, fresh trusted metadata, test-only extraction."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile
import requests
from neurocurate.acquisition import acquire, extract_member
from neurocurate.lowdisk import sha256

root=Path('results/real_phase0/sub02');root.mkdir(parents=True,exist_ok=True)
work=Path('data/local/real_phase0/sub02').resolve();work.mkdir(parents=True,exist_ok=True)
cross=Path('results/real_phase0/cross_subject_01_02');cross.mkdir(parents=True,exist_ok=True)
protocol=Path('configs/cross_subject_01_02.yaml')
frozen=cross/'frozen_protocol.json'
if not frozen.exists():
 frozen.write_text(json.dumps(dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
  config_sha256=sha256(protocol),config_text=protocol.read_text(),
  subject01_config_sha256=sha256('results/real_phase0/sub01/scientific_config.yaml'),
  status='predeclared before subject-02 acquisition'),indent=2)+'\n')
else:
 assert json.loads(frozen.read_text())['config_sha256']==sha256(protocol)
api='https://api.osf.io/v2/files/62f6b6eb170d520c73c27e35/'
with requests.get(api,headers={'Accept-Encoding':'identity'},stream=True,allow_redirects=False,timeout=(10,20)) as response:
 assert response.status_code==200 and response.headers.get('Content-Encoding','identity')=='identity'
 data=response.raw.read(65537);assert len(data)<=65536
metadata=json.loads(data);attributes=metadata['data']['attributes']
assert attributes['name']=='sub-02.zip' and attributes['size']==1071516660
assert attributes['extra']['hashes']['sha256']=='e1eba2d74009c6dd41becf53d84b8d47532103b41110a5855e7e286bcca6fe41'
(root/'official_source_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
free=shutil.disk_usage(work).free
# Retained archive + conservative 512 MiB EEG export + 512 MiB numeric conversion
# + 256 MiB outputs + 1 GiB reserve. No new stimuli or vision weights.
estimate=attributes['size']+2*512*1024**2+256*1024**2+1024**3
assert estimate<=free,'insufficient disk; no payload download'
artifact=dict(path=str(work/'sub-02.zip'),url='https://osf.io/download/u43p5/',
 bytes=attributes['size'],sha256=attributes['extra']['hashes']['sha256'])
manifest=dict(subject='sub-02',started_at_utc=datetime.now(timezone.utc).isoformat(),
 source_api=api,source_metadata_sha256=sha256(root/'official_source_metadata.json'),
 available_disk_before_bytes=free,estimated_peak_additional_bytes=estimate,
 frozen_protocol_sha256=sha256(protocol),files=[],extracted=[])
def save():
 (root/'acquisition_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
save();print(json.dumps(manifest,indent=2),flush=True)
manifest['files'].append(acquire(artifact,work,1024**3));save()
print('Verified complete subject-02 ZIP SHA256',flush=True)
with zipfile.ZipFile(work/'sub-02.zip') as archive:
 members=[m for m in archive.infolist() if m.filename.split('/')[-1]=='preprocessed_eeg_test.npy']
 assert len(members)==1 and 0<members[0].file_size<=512*1024**2
 member=members[0];path=extract_member(archive,member,work/'eeg')
 manifest['extracted'].append(dict(path=str(path),archive='sub-02.zip',member=member.filename,
  bytes=path.stat().st_size,compressed_bytes=member.compress_size,sha256=sha256(path),crc32_verified=True))
manifest['status']='verified test-only acquisition; source ZIP preserved'
manifest['available_disk_after_bytes']=shutil.disk_usage(work).free
save();print(json.dumps(manifest['extracted'],indent=2),flush=True)
