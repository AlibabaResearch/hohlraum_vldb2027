"""Data validation, reuse and rebuild orchestration; Playing reconstruction lives in rebuild_playing_trace.py."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('HOHLRAUM_DATA_DIR', str(HERE.parents[2]/'data'))).expanduser().resolve()


"""Validate, reuse or rebuild prepared inputs before either serving backend."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from hohlraum.experiments.common.workload import normalize


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4<<20),b''):value.update(chunk)
    return value.hexdigest()


def validate(dataset, selection):
    rows_path=Path(selection['rows'])
    digest=sha(rows_path)
    if selection.get('rows_sha256') and digest!=selection['rows_sha256']:
        raise ValueError('request rows checksum mismatch')
    rows=json.loads(rows_path.read_text())
    if not isinstance(rows,list) or not rows:raise ValueError('empty or invalid request rows')
    fields=('root','chain','index','owner','reference_fold','admission_fold','root_offer','first_delay','think_time','raw_input_tokens','blocks')
    import math
    for row in rows:
        for key in fields:
            if key not in row:raise ValueError('missing request field '+key)
        n=row['raw_input_tokens']
        if not isinstance(n,int) or n<1 or len(row['blocks'])!=n//1024:
            raise ValueError('request length/block count mismatch')
        if not all(isinstance(b,int) and 0<=b<2**64 for b in row['blocks']):
            raise ValueError('invalid block ID')
        for key in ['root_offer','first_delay','think_time']:
            if not isinstance(row[key],(int,float)) or not math.isfinite(row[key]) or row[key]<0:
                raise ValueError('invalid timing field '+key)
    deps=None;dep_hash=None
    if dataset=='agentx':
        if not selection.get('dependencies'):raise ValueError('AgentX dependency DAG missing')
        dep_hash=sha(selection['dependencies'])
        if selection.get('dependencies_sha256') and dep_hash!=selection['dependencies_sha256']:
            raise ValueError('dependency checksum mismatch')
        deps=json.loads(Path(selection['dependencies']).read_text())
    normalize(rows,deps)
    return dict(requests=len(rows),roots=len({r['root'] for r in rows}),rows_sha256=digest,
                dependencies_sha256=dep_hash,checks=['schema','page_lengths','timing','dependency_graph'],
                expected_checksum_checked=bool(selection.get('rows_sha256')))


def ensure_inputs(dataset, selection, seed):
    """Rebuild only from an explicit recipe or immutable prepared source bundle."""
    selection=dict(selection)
    recipe=selection.get('rebuild',{})
    if recipe.get('kind')=='prepared-source':
        for key in ['rows_sha256','dependencies_sha256']:
            if key not in selection and recipe.get(key):selection[key]=recipe[key]
    if selection.get('seed',seed)!=seed:raise ValueError('selection seed mismatch')
    try:
        validation=validate(dataset,selection)
        return selection,dict(action='reused',validation=validation)
    except (OSError,ValueError,KeyError,TypeError) as error:
        reason=str(error)
    recipe=selection.get('rebuild')
    if not recipe:
        raise ValueError('Input validation failed: '+reason+'. No reproducible source recipe is supplied for this selection; refusing to substitute another workload.')
    rows=Path(selection['rows']).resolve();rows.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-',dir=rows.parent) as tmp:
        temp=Path(tmp);candidate=dict(selection)
        if recipe['kind']=='prepared-source':
            # Both datasets may ship an immutable prepared source bundle.
            # This restores the specified selection, not a new sample.
            source=Path(recipe['rows'])
            if not recipe.get('rows_sha256') or sha(source)!=recipe['rows_sha256']:
                raise ValueError('Prepared source rows missing or checksum mismatch')
            generated=temp/'selected_rows.json';shutil.copyfile(source,generated)
            candidate.update(rows=str(generated),rows_sha256=recipe['rows_sha256'])
            if dataset=='agentx':
                source_deps=Path(recipe['dependencies'])
                if not recipe.get('dependencies_sha256') or sha(source_deps)!=recipe['dependencies_sha256']:
                    raise ValueError('Prepared source DAG missing or checksum mismatch')
                shutil.copyfile(source_deps,temp/'dependencies.json')
                candidate.update(dependencies=str(temp/'dependencies.json'),dependencies_sha256=recipe['dependencies_sha256'])
        elif recipe['kind'] in ('playing-v7','playing-balanced') and dataset=='playing':
            # Explicit settings are necessary; never silently replace a balanced
            # sampled pool with the reconstruction tool's uniform-copy default.
            required=['raw','tokenizer','seed','mean_arrival']
            if recipe['kind']=='playing-v7':required.append('copies')
            for key in required:
                if key not in recipe:raise ValueError('Incomplete Playing reconstruction recipe: '+key)
            if recipe['seed']!=seed:raise ValueError('Reconstruction seed mismatch')
            generated_dir=temp/'rebuilt'
            cmd=[sys.executable,'-m','hohlraum.experiments.preprocessing.run','build','--dataset','playing','--output',str(generated_dir)]
            for key in required:cmd+=['--'+key.replace('_','-'),str(recipe[key])]
            cmd+=['--layout','balanced' if recipe['kind']=='playing-balanced' else 'uniform']
            subprocess.run(cmd,check=True)
            candidate['rows']=str(generated_dir/f's{seed}'/'calibration_rows.json' if recipe['kind']=='playing-balanced' else generated_dir/'selected_rows.json')
        else:raise ValueError('Unsupported reconstruction recipe for '+dataset)
        validation=validate(dataset,candidate)
        # Expected hashes from the selection must also agree with restored source.
        if selection.get('rows_sha256') and validation['rows_sha256']!=selection['rows_sha256']:
            raise ValueError('Rebuilt rows differ from the requested selection')
        if selection.get('dependencies_sha256') and validation['dependencies_sha256']!=selection['dependencies_sha256']:
            raise ValueError('Rebuilt DAG differs from the requested selection')
        # Publish only after all validation succeeds. Preserve old files on failure.
        Path(candidate['rows']).replace(rows)
        if dataset=='agentx':
            destination=Path(selection['dependencies']);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(candidate['dependencies'],destination.with_suffix(destination.suffix+'.pending'))
            destination.with_suffix(destination.suffix+'.pending').replace(destination)
    return selection,dict(action='rebuilt',reason=reason,validation=validation)


def validate_checksums(directory):
    inventory=json.loads(directory.with_suffix('.checksums.json').read_text())
    for row in inventory['files']:
        path=directory/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Checksum mismatch: '+str(path))
    print(f"Verified {len(inventory['files'])} files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    build = actions.add_parser('build', help='Reconstruct Playing requests from raw logs')
    build.add_argument('--dataset', choices=['playing'], default='playing')
    build.add_argument('--raw', type=Path, default=DATA/'raw/playing')
    build.add_argument('--tokenizer', type=Path, default=DATA/'tokenizer')
    build.add_argument('--output', type=Path, default=DATA/'processed/playing')
    build.add_argument('--workers', type=int, default=4)
    build.add_argument('--layout', choices=['balanced', 'uniform'], default='balanced')
    build.add_argument('--seeds', type=int, nargs='+')
    build.add_argument('--copies', type=int, default=10)
    build.add_argument('--seed', type=int, default=272015)
    build.add_argument('--mean-arrival', type=float, default=15)
    build.add_argument('--resume', action='store_true')
    verify = actions.add_parser('validate', help='Verify a converted snapshot against its saved checksums')
    verify.add_argument('--data', type=Path, default=DATA/'processed/playing')
    args = parser.parse_args()
    if args.action == 'validate':
        validate_checksums(args.data)
        return
    providers = ['claude_13agents', 'gpt_5agents', 'kimi_5agents']
    if any(not (args.raw/p).is_dir() for p in providers):
        parser.error('Missing Playing raw provider directories under '+str(args.raw))
    if not (args.tokenizer/'tokenizer.json').is_file():
        parser.error('Missing tokenizer.json under '+str(args.tokenizer))
    if args.workers < 1 or args.copies < 1 or args.mean_arrival < 0:
        parser.error('workers/copies must be positive; mean-arrival must be nonnegative')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(HERE/'rebuild_playing_trace.py'),
               '--raw', str(args.raw), '--tokenizer', str(args.tokenizer),
               '--output', str(args.output), '--workers', str(args.workers),
               '--copies', str(args.copies), '--seed', str(args.seed),
               '--mean-arrival', str(args.mean_arrival)]
    command += ['--layout', args.layout]
    if args.seeds: command += ['--seeds', *map(str,args.seeds)]
    if args.resume: command.append('--resume')
    subprocess.run(command, check=True)
    # Integrity inventory for later verification, not proof of reconstruction assumptions.
    files = []
    for path in sorted(args.output.rglob('*')):
        if not path.is_file(): continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4<<20), b''): digest.update(chunk)
        files.append(dict(path=str(path.relative_to(args.output)), bytes=path.stat().st_size, sha256=digest.hexdigest()))
    args.output.with_suffix('.checksums.json').write_text(json.dumps(dict(files=files), indent=2)+'\n')
    print('Prepared data:', args.output)
    print('Saved checksum inventory; this checks file integrity, not the validity of missing-text assumptions.')


if __name__ == '__main__':
    main()
