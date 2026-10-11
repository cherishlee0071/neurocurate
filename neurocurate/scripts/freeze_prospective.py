"""Create a content-addressed Git protocol snapshot without changing the checkout."""
import io,json,subprocess,tarfile,sys
from datetime import datetime,timezone
from pathlib import Path
from importlib.metadata import distributions
from neurocurate.lowdisk import sha256

root=Path('results/real_phase0/prospective_03_05');root.mkdir(parents=True,exist_ok=True)
assert not (root/'frozen_protocol.json').exists(),'Never replace a frozen protocol'
files=sorted({p for folder in ['src','scripts','configs','tests'] for p in Path(folder).rglob('*') if p.suffix in ['.py','.yaml']})
versions=dict(python=sys.version,packages={d.metadata['Name']:d.version for d in distributions() if d.metadata['Name']})
(root/'environment_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
files+=[root/'environment_versions.json',Path('pyproject.toml')]
snapshot=root/'protocol_snapshot.tar'
with tarfile.open(snapshot,'w') as archive:
 for path in files:
  data=path.read_bytes();info=tarfile.TarInfo(str(path));info.size=len(data);info.mode=0o444;info.mtime=0
  archive.addfile(info,io.BytesIO(data))
def git(*args,**kwargs): return subprocess.check_output(['git',*args],**kwargs).decode().strip()
base=git('rev-parse','HEAD')
blob=git('hash-object','-w',str(snapshot))
tree=git('mktree',input=f'100644 blob {blob}\tprotocol_snapshot.tar\n'.encode())
commit=git('-c','user.name=NeuroCurate','-c','user.email=neurocurate-protocol@localhost','commit-tree',tree,'-p',base,
 input=b'Freeze prospective subjects 03-05 protocol before new-subject access\n')
git('update-ref','refs/neurocurate/phase16-protocol',commit,'0'*40)
snapshot.chmod(0o444)
record=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),designation='Prospective replication of an exploratory-derived candidate; not original preregistration',
 base_git_commit=base,snapshot_git_commit=commit,snapshot_git_blob=blob,snapshot_sha256=sha256(snapshot),
 files_sha256={str(p):sha256(p) for p in files},config=json.loads(json.dumps(__import__('yaml').safe_load(Path('configs/prospective_03_05.yaml').read_text()))))
(root/'frozen_protocol.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
