"""Run Hohlraum experiments on SGLang (default), policy replay, or HiSim."""
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


def runtime_environment(backend="sglang"):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(map(str, [ROOT, VENDOR, VENDOR/'python',
                                   VENDOR/'tools/sglang-simulator/src']))
    if backend == 'hisim':
        env.update(SGLANG_USE_CPU_ENGINE='1', CUDA_VISIBLE_DEVICES='',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    else:
        for key in list(env):
            if key.startswith(('SGLANG_SIMULATOR_', 'HISIM_')) or key == 'SGLANG_USE_CPU_ENGINE':
                env.pop(key)
    return env


def prepare_python(env, backend="sglang"):
    print('Checking the '+backend+' runtime...', flush=True)
    probe = ['-c', 'import torch, transformers, sklearn, joblib; '
             'assert transformers.__version__ == "5.12.1"; '
             'assert sklearn.__version__ == "1.7.2"; '
             'from benchmark.simulator.bench_runner import SGLangBenchmarkRunner']
    if backend == 'sglang':
        probe = ['-c', 'import torch; from sglang.srt.server_args import ServerArgs; '
                 'from sglang.launch_server import run_server']
    if subprocess.run([sys.executable, *probe], env=env, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode == 0:
        print('Using compatible runtime: '+sys.executable, flush=True)
        return sys.executable
    venv = ROOT/('.venv-'+backend)
    python = venv/'bin/python'
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    marker = venv/'hohlraum-ready'
    if not marker.exists():
        print('Preparing the isolated SGLang/HiSim environment (first run only).', flush=True)
        command = [str(python), '-m', 'pip', 'install', '-e', str(ROOT), '-e', str(VENDOR/'python')]
        if backend == 'hisim':
            command += ['-e', str(VENDOR/'tools/sglang-simulator'), 'scikit-learn==1.7.2', 'joblib==1.5.3']
        subprocess.run(command, check=True)
        subprocess.run([str(python), *probe], env=env, check=True)
        marker.write_text('installed from bundled source\n')
    else:
        subprocess.run([str(python), *probe], env=env, check=True)
    return str(python)


def prepare_assets(data, directory, datasets, seeds, *, backend="hisim", model=None, ssd=None, server_args=None):
    if not data.is_dir():
        raise ValueError('Missing experiment data: '+str(data)+'. Set --data to the downloaded data directory.')
    template = json.loads((ROOT/'hohlraum/experiments/inputs.json').read_text())
    profile_path = directory/('hisim-profile.json' if backend=='hisim' else 'server-args.json')
    if backend=='hisim':
        for item in json.loads((data/'runtime/checksums.json').read_text())['files']:
            path=data/'runtime'/item['path']
            if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
                raise ValueError('Runtime asset checksum mismatch: '+str(path))
        profile = json.loads((data/'runtime/hisim-profile.json').read_text())
        profile['server_args']['model_path'] = str(data/'runtime'/profile['server_args']['model_path'])
        profile['sim_config']['predictor']['database_path'] = str(data/'runtime'/profile['sim_config']['predictor']['database_path'])
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); profile['server_args']['nccl_port'] = sock.getsockname()[1]
    else:
        profile=json.loads((ROOT/'hohlraum/experiments/server_defaults.json').read_text())
        if server_args: profile.update(json.loads(Path(server_args).read_text()))
        profile['model_path']=str(model)
        if profile.get('load_format')=='dummy' or profile.get('device')=='cpu':
            raise ValueError('Native runs require real model weights and a GPU; dummy/CPU profiles are not native experiments.')
        for name in ('port','nccl_port'):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));profile[name]=sock.getsockname()[1]
    dump(profile_path, profile)
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
        assets = dict(datasets={dataset:selections}, priors=str(priors))
        if backend=='hisim':assets['hisim_profile']=str(profile_path)
        else:assets.update(native_server_args=str(profile_path),ssd_directory=str(ssd))
        paths[dataset] = directory/f'{dataset}.json'; dump(paths[dataset], assets)
    return paths


def summarize(output):
    rows=[]
    for file in sorted(list(output.glob('*/*/result.json'))+list(output.glob('*/*/client/result.json'))):
        phase=file.parent.parent if file.parent.name=='client' else file.parent
        if phase.name=='warmup': continue
        result=json.loads(file.read_text())
        rows.append(dict(case=phase.parent.name, policy=phase.name,
                         throughput=result.get('throughput', result.get('total_input_tokens_s')),
                         goodput=result.get('goodput', result.get('slo10_input_tokens_s')),
                         ttft_p90_s=result.get('ttft_p90_s'), write_bytes=result.get('write_bytes'),
                         budget_exhausted_at_s=result.get('budget_exhausted_at_s')))
    dump(output/'summary.json', rows)
    if rows:
        with (output/'summary.csv').open('w', newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Results: '+str(output/('summary.csv' if rows else 'summary.json')), flush=True)


def resolve_model(model, python, env, check_only):
    path=Path(model).expanduser()
    if path.is_dir():
        if not (path/'config.json').is_file():raise ValueError('Model directory lacks config.json: '+str(path))
        if not any(path.glob('*.safetensors')) and not any(path.glob('pytorch_model*.bin')):
            raise ValueError('Model directory has no inference weights: '+str(path))
        return path.resolve()
    if check_only:raise ValueError('For --check-only supply --model /path/to/downloaded/model; model downloads are disabled during validation.')
    print('Downloading model weights: '+model,flush=True)
    command=[python,'-c',"from huggingface_hub import snapshot_download; import sys; print(snapshot_download(sys.argv[1],ignore_patterns=['*.gguf','*.onnx']))",model]
    result=subprocess.check_output(command,env=env,text=True)
    return Path(result.strip().splitlines()[-1]).resolve()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['sglang','policy','hisim'],default='sglang')
    parser.add_argument('--data',type=Path,default=Path(os.environ.get('HOHLRAUM_DATA_DIR',str(ROOT/'data'))),help='Experiment data directory (default: ./data)')
    parser.add_argument('--model',default=os.environ.get('HOHLRAUM_MODEL','Qwen/Qwen3-8B'),help='Native model path or Hugging Face model ID')
    parser.add_argument('--ssd',type=Path,default=Path(os.environ.get('HOHLRAUM_SSD_DIR',str(ROOT/'ssd-cache'))),help='Directory on the SSD used by native experiments')
    parser.add_argument('--server-args',type=Path,help='Optional JSON overrides for native SGLang settings')
    parser.add_argument('--datasets',nargs='+',choices=['playing','agentx'])
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--loads',nargs='+',choices=['low','medium','high'])
    parser.add_argument('--concurrency',type=int,help='Override session concurrency for serving experiments')
    parser.add_argument('--policies',nargs='+',choices=['unlimited','budget640','admission-dynamic','staging'],default=['unlimited','budget640','admission-dynamic','staging'])
    parser.add_argument('--staging-gb',type=float,default=30)
    parser.add_argument('--dynamic-capacity',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true')
    a=parser.parse_args()
    if a.staging_gb<0:parser.error('--staging-gb must be nonnegative')
    if a.concurrency is not None and a.concurrency<1:parser.error('--concurrency must be positive')
    if not shutil.which('c++'):parser.error('A C++17 compiler is required.')
    protocol=json.loads((ROOT/'hohlraum/experiments/protocol.json').read_text())
    data=a.data.expanduser().resolve()
    output=(a.output or ROOT/'results'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    if output.exists():parser.error('Output already exists: '+str(output))
    from hohlraum.experiments.resources import ensure_resources
    try:ensure_resources(data,a.backend,check_only=a.check_only)
    except (OSError,ValueError) as error:parser.error(str(error))
    env=runtime_environment(a.backend)
    if a.backend=='policy':
        if a.concurrency is not None or a.loads or a.seeds or a.datasets:
            parser.error('Policy replay uses its frozen stream/configuration, not serving concurrency or seed overrides.')
        cmd=['bash',str(ROOT/'hohlraum/experiments/backends/policy_replay/reproduce.sh'),'--data',str(data/'policy'),'--output',str(output)]
        if a.check_only:cmd.append('--check-only')
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
        return
    seeds=a.seeds or protocol['seeds'];datasets=a.datasets or ['playing','agentx'];loads=a.loads or ['low','medium','high']
    python=sys.executable if a.check_only else prepare_python(env,a.backend)
    if a.backend=='sglang' and not a.check_only:
        probe=subprocess.run([python,'-c','import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'],env=env)
        if probe.returncode:parser.error('Native experiments require a CUDA-capable PyTorch installation and visible GPUs. Use --backend policy for CPU-only replay.')
    try:model=resolve_model(a.model,python,env,a.check_only) if a.backend=='sglang' else None
    except (ValueError,subprocess.CalledProcessError) as e:parser.error(str(e))
    ssd=a.ssd.expanduser().resolve()
    if a.backend=='sglang':ssd.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True)
    try:assets=prepare_assets(data,output/'assets',datasets,seeds,backend=a.backend,model=model,ssd=ssd,server_args=a.server_args)
    except (OSError,ValueError,KeyError) as e:parser.error(str(e))
    dump(output/'suite.json',dict(backend=a.backend,datasets=datasets,seeds=seeds,loads=loads,policies=a.policies,data=str(data),concurrency=a.concurrency,staging_GB=a.staging_gb,dynamic_capacity=a.dynamic_capacity,protocol=protocol,python=python,check_only=a.check_only))
    print(f'{a.backend}: {len(datasets)*len(seeds)*len(loads)} load points; policies={a.policies}; serial execution.',flush=True)
    try:
        for dataset in datasets:
            for seed in seeds:
                for load in loads:
                    case=output/f'{dataset}-s{seed}-{load}'
                    cmd=[python,'-m','hohlraum.experiments.run','--backend',a.backend,'--assets',str(assets[dataset]),'--dataset',dataset,'--seed',str(seed),'--load',load,'--output',str(case),'--policies',*a.policies,'--staging-gb',str(a.staging_gb)]
                    if a.concurrency is not None:cmd+=['--concurrency',str(a.concurrency)]
                    if a.dynamic_capacity:cmd.append('--dynamic-capacity')
                    if a.check_only:cmd.append('--check-only')
                    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
                    if not a.check_only:summarize(output)
    except subprocess.CalledProcessError as error:
        raise SystemExit('Experiment failed (exit '+str(error.returncode)+'). Inspect logs under '+str(output)) from None
    finally:
        if not a.check_only:summarize(output)
    print('Configuration validation passed.' if a.check_only else 'Suite completed.',flush=True)


if __name__=='__main__':main()
