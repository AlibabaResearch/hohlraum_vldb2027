"""Install bundled processed inputs required by the selected experiment backend."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

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
        raise ValueError('Resource checksum mismatch: '+str(archive))
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


def ensure_resources(data, backend, *, manifest=MANIFEST):
    resources = json.loads(Path(manifest).read_text())
    resource = resources['serving']
    simulator_files = resources['hisim']['files'] if backend == 'hisim' else {}
    required = (['runtime/priors/agentx', *simulator_files] if simulator_files
                else resource['required_paths'])
    if not all((data/path).exists() for path in required):
        archive = (Path(manifest).resolve().parent / resource['archive']).resolve()
        if not archive.is_file():
            raise ValueError('Missing bundled resource: '+str(archive))
        print('Extracting bundled experiment inputs...', flush=True)
        install_archive(archive, data, resource['sha256'])
    for path in required:
        if not (data/path).exists():
            raise ValueError('Missing required experiment input: '+str(data/path))
    for path, expected in simulator_files.items():
        if digest(data/path) != expected:
            raise ValueError('HiSim asset checksum mismatch: '+str(data/path))


def lifecycle_model_count(directory):
    """Use the exported schema; surplus numbered models are an input error."""
    directory = Path(directory)
    metadata = json.loads((directory/'training.json').read_text())
    count = metadata.get('model_count') if isinstance(metadata, dict) else None
    if type(count) is not int or count not in (2, 6):
        raise ValueError('Current lifecycle priors require model_count 2 or 6 in training.json: '+str(directory))
    found = {p.name for p in directory.glob('[0-9].txt')}
    expected = {f'{i}.txt' for i in range(count)}
    if found != expected:
        raise ValueError('Lifecycle prior files disagree with training.json: '
                         f'{directory}; missing={sorted(expected-found)}, '
                         f'outdated={sorted(found-expected)}')
    if count == 2 and (directory/'model-schema.txt').read_text().strip() != 'session-qg-only-v1':
        raise ValueError('AgentX prior schema is not session-qg-only-v1')
    return count


def lifecycle_environment(priors):
    priors = Path(priors).resolve()
    return dict(
        HOHLRAUM_JOINT_MODELS=str(priors),
        HOHLRAUM_MODEL_COUNT=str(lifecycle_model_count(priors)),
        HOHLRAUM_BRANCH_PRIOR=str(priors/'branch-prior.txt'),
        HOHLRAUM_LENGTH_CURVES=str(priors/'length_curves.txt'),
        HOHLRAUM_ONLINE_UPDATES='1',
        HOHLRAUM_STATISTICS_PERIOD_S=os.environ.get('HOHLRAUM_STATISTICS_PERIOD_S','3600'),
    )
