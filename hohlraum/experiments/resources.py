"""Fetch only the data resources required by the selected experiment backend."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
import urllib.error

MANIFEST = Path(__file__).with_name('resources.json')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def install_archive(archive, destination, expected_sha256):
    """Verify before extraction; never replace a user's different local data."""
    if digest(archive) != expected_sha256:
        raise ValueError('Downloaded resource checksum mismatch: '+str(archive))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.resource-', dir=destination.parent) as tmp:
        unpacked = Path(tmp)
        with tarfile.open(archive) as package:
            members = package.getmembers()
            for member in members:
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe resource archive member: '+member.name)
            package.extractall(unpacked, members=members)
        files = [p for p in unpacked.rglob('*') if p.is_file()]
        for source in files:
            target = destination/source.relative_to(unpacked)
            if target.exists() and (not target.is_file() or digest(target) != digest(source)):
                raise ValueError('Resource conflicts with existing data: '+str(target))
            if any(parent.is_symlink() for parent in [target, *target.parents]):
                raise ValueError('Refusing resource extraction through a symbolic link: '+str(target))
        for source in files:
            target = destination/source.relative_to(unpacked)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)


def ensure_resources(data, backend, *, check_only=False, manifest=MANIFEST):
    resources = json.loads(Path(manifest).read_text())
    required = ['policy'] if backend == 'policy' else ['serving']
    if backend == 'hisim':
        required.append('hisim')
    for name in required:
        resource = resources[name]
        if all((data/p).exists() for p in resource['required_paths']):
            continue
        url = resource.get('url')
        if check_only:
            raise ValueError('Missing '+name+' data under '+str(data)+'. Validation does not download resources.')
        if not url:
            raise ValueError('The '+name+' resource has no published download URL yet. Supply its local data directory with --data.')
        print('Downloading '+name+' resources...', flush=True)
        with tempfile.TemporaryDirectory(prefix='hohlraum-download-') as tmp:
            archive = Path(tmp)/resource['filename']
            try:
                with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as out:
                    shutil.copyfileobj(response, out)
            except urllib.error.URLError as error:
                raise ValueError('Cannot download '+name+' resources from '+url+
                                 '. Check that the data release is published, or supply an existing --data directory. '+str(error)) from error
            install_archive(archive, data, resource['sha256'])
        if not all((data/p).exists() for p in resource['required_paths']):
            raise ValueError('Resource archive lacks required '+name+' files')
