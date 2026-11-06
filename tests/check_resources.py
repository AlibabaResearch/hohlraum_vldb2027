"""Resource selection, integrity, traversal and local-data preservation checks."""
import io,json,sys,tarfile,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hohlraum.experiments.resources import digest,install_archive,ensure_resources

def archive(path,name,content=b'data'):
    with tarfile.open(path,'w:gz') as t:
        info=tarfile.TarInfo(name);info.size=len(content);t.addfile(info,io.BytesIO(content))

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);a=root/'source.tar.gz';archive(a,'processed/rows.json')
    manifest=root/'resources.json';manifest.write_text(json.dumps({'serving':dict(filename=a.name,url=a.as_uri(),sha256=digest(a),required_paths=['processed/rows.json']),'hisim':dict(url=None,required_paths=['predictor']),'policy':dict(url=None,required_paths=['policy'])}))
    ensure_resources(root/'data','sglang',manifest=manifest)
    assert (root/'data/processed/rows.json').read_bytes()==b'data'
    # Native fetch must not require a simulator profile or policy bundle.
    ensure_resources(root/'data','sglang',check_only=True,manifest=manifest)
    try:ensure_resources(root/'data','hisim',check_only=True,manifest=manifest)
    except ValueError:pass
    else:raise AssertionError('missing simulator resources accepted')
    try:install_archive(a,root/'bad','0'*64)
    except ValueError:pass
    else:raise AssertionError('checksum mismatch accepted')
    evil=root/'evil.tar.gz';archive(evil,'../escaped')
    try:install_archive(evil,root/'unsafe',digest(evil))
    except ValueError:pass
    else:raise AssertionError('path traversal accepted')
    assert not (root/'escaped').exists()
    (root/'data/processed/rows.json').write_bytes(b'custom')
    try:install_archive(a,root/'data',digest(a))
    except ValueError:pass
    else:raise AssertionError('custom data overwritten')
    assert (root/'data/processed/rows.json').read_bytes()==b'custom'
print('PASS backend-specific downloads, checksum validation, traversal rejection, existing data preserved')
