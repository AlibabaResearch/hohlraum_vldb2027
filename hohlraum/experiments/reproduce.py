"""One-command HiSim reproduction from the source and sibling data bundle."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT/'third_party/sglang'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def runtime_environment():
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(map(str, [ROOT, VENDOR, VENDOR/'python',
                                   VENDOR/'tools/sglang-simulator/src']))
    env.update(SGLANG_USE_CPU_ENGINE='1', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return env


def prepare_python(env):
    print('Checking the available HiSim runtime...', flush=True)
    probe = ['-c', 'import torch, transformers, sklearn, joblib; '
             'assert transformers.__version__ == "5.12.1"; '
             'assert sklearn.__version__ == "1.7.2"; '
             'from benchmark.simulator.bench_runner import SGLangBenchmarkRunner']
    if subprocess.run([sys.executable, *probe], env=env, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode == 0:
        print('Using compatible runtime: '+sys.executable, flush=True)
        return sys.executable
    venv = ROOT/'.venv-reproduce'
    python = venv/'bin/python'
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    marker = venv/'hohlraum-ready'
    if not marker.exists():
        print('Preparing the isolated SGLang/HiSim environment (first run only).', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '-e', str(ROOT),
                        '-e', str(VENDOR/'python'), '-e', str(VENDOR/'tools/sglang-simulator'),
                        'scikit-learn==1.7.2', 'joblib==1.5.3'], check=True)
        subprocess.run([str(python), *probe], env=env, check=True)
        marker.write_text('installed from bundled source\n')
    else:
        subprocess.run([str(python), *probe], env=env, check=True)
    return str(python)


def prepare_assets(data, directory, datasets, seeds):
    if not data.is_dir():
        raise ValueError('Missing data bundle: '+str(data)+'. Extract the complete reproduction '
                         'package (hohlraum-paper/ and hohlraum-data/ side by side), '
                         'or specify --data /path/to/hohlraum-data. A source-only Git clone has no dataset payloads.')
    for item in json.loads((data/'runtime/checksums.json').read_text())['files']:
        path=data/'runtime'/item['path']
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Runtime asset checksum mismatch: '+str(path))
    template = json.loads((ROOT/'assets.example.json').read_text())
    profile = json.loads((data/'runtime/hisim-profile.json').read_text())
    profile['server_args']['model_path'] = str(data/'runtime'/profile['server_args']['model_path'])
    profile['sim_config']['predictor']['database_path'] = str(data/'runtime'/profile['sim_config']['predictor']['database_path'])
    # Reserve a currently unused port for this serial suite. Each phase exits before the next.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    profile['server_args']['nccl_port'] = port
    for path in [Path(profile['server_args']['model_path'])/'config.json',
                 Path(profile['sim_config']['predictor']['database_path'])]:
        if not path.is_file(): raise ValueError('Missing bundled runtime asset: '+str(path))
    profile_path = directory/'hisim-profile.json'; dump(profile_path, profile)
    paths = {}
    for dataset in datasets:
        selections = {}
        for seed in seeds:
            folder = data/'processed'/dataset/f's{seed}'
            rows = folder/('calibration_rows.json' if dataset=='playing' else 'selected_rows.json')
            # Published seeds are checked against code-pinned hashes. Other seeds use their generated manifest.
            reference = template['datasets'].get(dataset, {}).get(str(seed), {})
            if not rows.is_file(): raise ValueError('Missing prepared selection: '+str(rows))
            digest = sha(rows)
            if reference.get('rows_sha256') and digest != reference['rows_sha256']:
                raise ValueError('Published selection checksum mismatch: '+str(rows))
            selected = dict(seed=seed, rows=str(rows), rows_sha256=digest)
            if dataset=='agentx':
                deps=folder/'dependencies.json'; dh=sha(deps)
                if reference.get('dependencies_sha256') and dh!=reference['dependencies_sha256']:
                    raise ValueError('Dependency checksum mismatch: '+str(deps))
                selected.update(dependencies=str(deps), dependencies_sha256=dh)
            selections[str(seed)] = selected
        priors = data/'runtime/priors'/dataset
        models=sorted(priors.glob('[0-9].txt'))
        if not models or {int(p.stem) for p in models} != set(range(len(models))):
            raise ValueError('Missing or noncontiguous prior model files: '+str(priors))
        for name in [*[p.name for p in models], 'branch-prior.txt', 'length_curves.txt', 'training.json']:
            if not (priors/name).is_file(): raise ValueError('Missing bundled prior: '+str(priors/name))
        assets = dict(datasets={dataset:selections}, hisim_profile=str(profile_path), priors=str(priors))
        paths[dataset] = directory/f'{dataset}.json'; dump(paths[dataset], assets)
    return paths


def summarize(output):
    rows=[]
    for file in sorted(output.glob('*/*/result.json')):
        if file.parent.name=='warmup': continue
        result=json.loads(file.read_text())
        rows.append(dict(case=file.parent.parent.name, policy=file.parent.name,
                         throughput=result.get('throughput'), goodput=result.get('goodput'),
                         ttft_p90_s=result.get('ttft_p90_s'), write_bytes=result.get('write_bytes'),
                         budget_exhausted_at_s=result.get('budget_exhausted_at_s')))
    dump(output/'summary.json', rows)
    if rows:
        with (output/'summary.csv').open('w', newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Results: '+str(output/('summary.csv' if rows else 'summary.json')), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(os.environ.get('HOHLRAUM_DATA_DIR',str(ROOT.parent/'hohlraum-data'))))
    parser.add_argument('--datasets', nargs='+', choices=['playing','agentx'], default=['playing','agentx'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--loads', nargs='+', choices=['low','medium','high'], default=['low','medium','high'])
    parser.add_argument('--policies', nargs='+', choices=['unlimited','budget640','admission-dynamic','staging'],
                        default=['unlimited','budget640','admission-dynamic','staging'])
    parser.add_argument('--staging-gb', type=float, default=30)
    parser.add_argument('--dynamic-capacity', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    a=parser.parse_args()
    if a.staging_gb<0: parser.error('--staging-gb must be nonnegative')
    if not shutil.which('c++'): parser.error('A C++17 compiler is required (install g++ or clang).')
    protocol=json.loads((ROOT/'hohlraum/experiments/protocol.json').read_text())
    seeds=a.seeds or protocol['seeds'];data=a.data.expanduser().resolve()
    output=(a.output or ROOT.parent/'hohlraum-results'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    if output.exists(): parser.error('Output already exists; choose a fresh --output directory: '+str(output))
    if not data.is_dir(): parser.error('Missing '+str(data)+'. Use the complete reproduction package or --data; no internal configuration file is required.')
    output.mkdir(parents=True)
    try: assets=prepare_assets(data,output/'assets',a.datasets,seeds)
    except (OSError,ValueError,KeyError) as e: parser.error(str(e))
    env=runtime_environment()
    python=sys.executable if a.check_only else prepare_python(env)
    dump(output/'suite.json',dict(datasets=a.datasets,seeds=seeds,loads=a.loads,policies=a.policies,
                                 data=str(data),staging_GB=a.staging_gb,dynamic_capacity=a.dynamic_capacity,
                                 protocol=protocol,python=python,check_only=a.check_only))
    print(f'Suite: {len(a.datasets)*len(seeds)*len(a.loads)} load points; policies={a.policies}; serial execution.',flush=True)
    try:
        for dataset in a.datasets:
            for seed in seeds:
                for load in a.loads:
                    case=output/f'{dataset}-s{seed}-{load}'
                    cmd=[python,'-m','hohlraum.experiments.run','--backend','hisim','--assets',str(assets[dataset]),
                         '--dataset',dataset,'--seed',str(seed),'--load',load,'--output',str(case),
                         '--policies',*a.policies,'--staging-gb',str(a.staging_gb)]
                    if a.dynamic_capacity:cmd.append('--dynamic-capacity')
                    if a.check_only:cmd.append('--check-only')
                    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
                    if not a.check_only:summarize(output)
    except subprocess.CalledProcessError as error:
        raise SystemExit('Experiment failed (exit '+str(error.returncode)+'). Inspect per-case run.log under '+str(output)) from None
    finally:
        if not a.check_only:summarize(output)
    print('Configuration validation passed.' if a.check_only else 'Suite completed.',flush=True)


if __name__=='__main__': main()
