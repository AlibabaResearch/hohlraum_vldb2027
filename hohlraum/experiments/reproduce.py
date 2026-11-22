"""Run Hohlraum reference workloads, verify the HiSim path, or replay cache policies."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from hohlraum.experiments.resources import digest, ensure_resources, lifecycle_model_count

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT/'third_party/sglang'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def runtime_environment(backend="sglang", *, native_installed=False):
    env = os.environ.copy()
    paths = [ROOT]
    if backend != 'policy' and not native_installed:
        paths += [VENDOR, VENDOR/'python', VENDOR/'tools/sglang-simulator/src']
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths))
    if backend in ('hisim', 'policy'):
        env = {key: value for key, value in env.items()
               if not key.startswith(('HOHLRAUM_', 'HISIM_', 'SGLANG_SIMULATOR_'))}
    if backend == 'hisim':
        env.update(SGLANG_USE_CPU_ENGINE='1', CUDA_VISIBLE_DEVICES='',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    elif backend == 'sglang':
        for key in list(env):
            if key.startswith(('SGLANG_SIMULATOR_', 'HISIM_')) or key == 'SGLANG_USE_CPU_ENGINE':
                env.pop(key)
    return env


def prepare_python(env, backend="sglang", *, native_installed=False):
    print('Checking the '+backend+' runtime...', flush=True)
    probe = ['-c', 'import torch, transformers, sklearn, joblib; '
             'assert transformers.__version__ == "5.12.1"; '
             'assert sklearn.__version__ == "1.7.2"; '
             'from benchmark.simulator.bench_runner import SGLangBenchmarkRunner']
    if backend == 'sglang':
        probe = ['-c', 'import torch, sglang; from pathlib import Path; '
                 f'bundled = Path(sglang.__file__).resolve().is_relative_to(Path({str(VENDOR / "python")!r}).resolve()); '
                 f'assert bundled != {native_installed!r}, "Unexpected SGLang source: " + sglang.__file__; '
                 'from sglang.srt.server_args import ServerArgs; '
                 'from sglang.launch_server import run_server']
    elif backend == 'policy':
        probe = ['-c', 'import matplotlib, numpy']
    if subprocess.run([sys.executable, *probe], env=env, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode == 0:
        print('Using compatible runtime: '+sys.executable, flush=True)
        return sys.executable
    if native_installed:
        subprocess.run([sys.executable, *probe], env=env, check=True)
        return sys.executable
    venv = ROOT/('.venv-'+backend)
    python = venv/'bin/python'
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    marker = venv/'hohlraum-ready'
    if not marker.exists():
        print('Preparing the isolated '+backend+' environment (first run only).', flush=True)
        command = [str(python), '-m', 'pip', 'install']
        if backend == 'policy':
            command += ['numpy>=1.24,<3', 'matplotlib>=3.7,<4']
        else:
            command += ['-e', str(ROOT), '-e', str(VENDOR/'python')]
            if backend == 'hisim':
                command += ['-e', str(VENDOR/'tools/sglang-simulator') + '[aic]', 'scikit-learn==1.7.2', 'joblib==1.5.3']
        install_env = os.environ.copy()
        if backend != 'policy':
            install_env.setdefault('SGLANG_BUILD_RUST_EXTS', 'none')
        subprocess.run(command, env=install_env, check=True)
        if backend == 'hisim':
            # aiconfigurator==0.10.0 pins numpy~=1.26.4, while the bundled predictor
            # pickle and pyarrow>=26 require NumPy 2.x.
            subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', 'numpy>=2.2,<2.4'],
                           env=install_env, check=True)
        subprocess.run([str(python), *probe], env=env, check=True)
        marker.write_text('installed from bundled source\n')
    else:
        subprocess.run([str(python), *probe], env=env, check=True)
    return str(python)


def prepare_hisim_profile(data, path):
    runtime = data/'runtime'
    profile = json.loads((runtime/'hisim-profile.json').read_text())
    profile['server_args']['model_path'] = str(runtime/profile['server_args']['model_path'])
    predictor = profile['sim_config']['predictor']
    predictor['database_path'] = str(runtime/predictor['database_path'])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        profile['server_args']['nccl_port'] = sock.getsockname()[1]
    dump(path, profile)


def prepare_assets(data, directory, load_points, *, model, ssd, server_args=None):
    if not data.is_dir():
        raise ValueError('Missing experiment data: '+str(data)+'. Set --data to the prepared data directory.')
    template = json.loads((ROOT/'hohlraum/experiments/inputs.json').read_text())
    profile_path = directory/'server-args.json'
    profile=json.loads((ROOT/'hohlraum/experiments/server_defaults.json').read_text())
    if server_args: profile.update(json.loads(Path(server_args).read_text()))
    profile['model_path']=str(model)
    if profile.get('load_format')=='dummy' or profile.get('device')=='cpu':
        raise ValueError('Native runs require real model weights and a GPU.')
    for name in ('port','nccl_port'):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            profile[name]=sock.getsockname()[1]
    dump(profile_path, profile)
    paths = {}
    for dataset in dict.fromkeys(point['dataset'] for point in load_points):
        references = template['datasets'].get(dataset, {})
        points = [point for point in load_points if point['dataset']==dataset]
        priors = data/'runtime/priors'/dataset
        lifecycle_model_count(priors)
        if any(str(point['seed']) in references for point in points):
            for name, expected in template['priors'][dataset].items():
                if digest(priors/name)!=expected:
                    raise ValueError('Published prior checksum mismatch: '+str(priors/name))
        for name in ('branch-prior.txt','length_curves.txt'):
            if not (priors/name).is_file(): raise ValueError('Missing prior: '+str(priors/name))
        assets = dict(priors=str(priors),native_server_args=str(profile_path),ssd_directory=str(ssd))
        for point in points:
            seed, concurrency = point['seed'], point['concurrency']
            key = (dataset, seed, concurrency)
            if key in paths:
                continue
            folder = data/'processed'/dataset/f's{seed}'
            reference = references.get(str(seed))
            if reference:
                variants = reference['load_points']
                source = concurrency if str(concurrency) in variants else reference['default_concurrency']
                variant = variants[str(source)]
                rows = data/variant['rows']
                deps = data/reference['dependencies']
            else:
                variant = {}
                if dataset=='playing':
                    rows = folder/f'c{concurrency}'/'rows.json'
                else:
                    rows = folder/'selected_rows.json'
                deps = folder/'dependencies.json'
            if not rows.is_file(): raise ValueError('Missing prepared selection: '+str(rows))
            row_digest = digest(rows)
            if variant.get('rows_sha256', row_digest)!=row_digest:
                raise ValueError('Published selection checksum mismatch: '+str(rows))
            selected = dict(seed=seed, rows=str(rows), rows_sha256=row_digest)
            if reference:
                selected['arrival_plan_concurrency'] = source
            if reference or dataset=='agentx' or deps.exists():
                dh=digest(deps)
                if reference and dh!=reference['dependencies_sha256']:
                    raise ValueError('Dependency checksum mismatch: '+str(deps))
                selected.update(dependencies=str(deps), dependencies_sha256=dh)
            paths[key] = directory/f'{dataset}-s{seed}-c{concurrency}.json'
            dump(paths[key], dict(assets,datasets={dataset:{str(seed):selected}}))
    return paths


def summarize(output):
    from hohlraum.experiments.common.diagnostics import serving_latency_summary

    rows=[]
    for file in sorted(list(output.glob('*/*/result.json'))+list(output.glob('*/*/client/result.json'))):
        phase=file.parent.parent if file.parent.name=='client' else file.parent
        if phase.name=='warmup': continue
        result=json.loads(file.read_text())
        rows.append(dict(case=phase.parent.name, policy=phase.name,
                         throughput=result.get('throughput', result.get('total_input_tokens_s')),
                         **serving_latency_summary(result), write_bytes=result.get('write_bytes'),
                         budget_exhausted_at_s=result.get('budget_exhausted_at_s')))
    dump(output/'summary.json', rows)
    if rows:
        with (output/'summary.csv').open('w', newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Results: '+str(output/('summary.csv' if rows else 'summary.json')), flush=True)


def resolve_model(model, python, env):
    path=Path(model).expanduser()
    if path.is_dir():
        if not (path/'config.json').is_file():raise ValueError('Model directory lacks config.json: '+str(path))
        if not any(path.glob('*.safetensors')) and not any(path.glob('pytorch_model*.bin')):
            raise ValueError('Model directory has no inference weights: '+str(path))
        return path.resolve()
    print('Downloading model weights: '+model,flush=True)
    command=[python,'-c',"from huggingface_hub import snapshot_download; import sys; print(snapshot_download(sys.argv[1],ignore_patterns=['*.gguf','*.onnx']))",model]
    result=subprocess.check_output(command,env=env,text=True)
    return Path(result.strip().splitlines()[-1]).resolve()


def main():
    parser=argparse.ArgumentParser(prog='./run.sh',description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify',action='store_true',help='Run the fixed CPU HiSim end-to-end verification case')
    mode.add_argument('--policy-replay',action='store_true',help='Run the CPU policy replay and generate the evaluation plots')
    parser.add_argument('--data',type=Path,help='Prepared serving or policy input directory')
    parser.add_argument('--jobs',type=int,help='Parallel policy replays (default: 4)')
    parser.add_argument('--model',default=os.environ.get('HOHLRAUM_MODEL','Qwen/Qwen3-8B'),help='Native model path or Hugging Face model ID')
    parser.add_argument('--ssd',type=Path,default=Path(os.environ.get('HOHLRAUM_SSD_DIR',str(ROOT/'ssd-cache'))),help='Directory on the SSD used by native experiments')
    parser.add_argument('--server-args',type=Path,help='Optional JSON overrides for native SGLang settings')
    parser.add_argument('--native-installed-sglang',action='store_true',
                        help='Use the current environment’s SGLang build for native serving')
    parser.add_argument('--datasets',nargs='+',choices=['playing','agentx'])
    parser.add_argument('--seeds',nargs='+',type=int,help='Select prepared workloads by their reference seeds')
    parser.add_argument('--loads',nargs='+',choices=['low','medium','high'],help='Select reference load presets')
    parser.add_argument('--concurrency',type=int,help='Override session concurrency for serving experiments')
    parser.add_argument('--protocol',type=Path,help='Serving protocol JSON')
    parser.add_argument('--policies',nargs='+',choices=['unlimited','budget640','admission-dynamic','staging'],default=['unlimited','budget640','admission-dynamic','staging'])
    parser.add_argument('--staging-gb',type=float,default=30)
    parser.add_argument('--dynamic-capacity',action='store_true')
    parser.add_argument('--admission-threshold',type=float)
    parser.add_argument('--fixed-admission-threshold',action='store_true')
    parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    if (a.verify or a.policy_replay) and a.native_installed_sglang:
        parser.error('--native-installed-sglang applies to GPU serving runs only')
    if a.admission_threshold is not None and not 0<=a.admission_threshold<=1:parser.error('--admission-threshold must be in [0,1]')
    if a.staging_gb<0:parser.error('--staging-gb must be nonnegative')
    if a.concurrency is not None and a.concurrency<1:parser.error('--concurrency must be positive')
    if a.jobs is not None and (not a.policy_replay or a.jobs < 1):
        parser.error('--jobs applies to --policy-replay and must be positive')
    if not shutil.which('c++'):parser.error('A C++17 compiler is required.')
    backend = 'policy' if a.policy_replay else 'hisim' if a.verify else 'sglang'
    if a.policy_replay and not shutil.which('make'):parser.error('--policy-replay requires make.')
    default_data = ROOT/'data/serving-94266-94268'
    data=(a.data or Path(os.environ.get('HOHLRAUM_DATA_DIR',str(default_data)))).expanduser().resolve()
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = f'policy-{timestamp}' if a.policy_replay else f'verify-{timestamp}' if a.verify else timestamp
    output=(a.output or ROOT/'results'/run_name).resolve()
    if output.exists():parser.error('Output already exists: '+str(output))
    env=runtime_environment(backend,native_installed=a.native_installed_sglang)
    try:
        ensure_resources(data,backend)
        python=prepare_python(env,backend,native_installed=a.native_installed_sglang)
    except (OSError,ValueError,subprocess.CalledProcessError) as error:
        parser.error(str(error))
    if a.policy_replay:
        command = [python, '-m', 'hohlraum.experiments.backends.policy_replay.reproduce',
                   '--data', str(data), '--output', str(output), '--jobs', str(a.jobs or 4)]
        try:
            subprocess.run(command,cwd=ROOT,env=env,check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(f'Policy replay failed (exit {error.returncode}). See {output}') from None
        return
    if a.verify:
        from hohlraum.experiments.backends.hisim.validate import validate

        profile = output/'hisim-profile.json'
        try:
            prepare_hisim_profile(data,profile)
            validate(data,output,python,profile=profile,env=env)
        except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as error:
            raise SystemExit(f'HiSim validation failed: {error}. See {output}') from None
        return

    from hohlraum.experiments.run import LOADS, read_serving_protocol, session_concurrency

    try:
        protocol=read_serving_protocol(a.protocol)
        seeds=a.seeds or protocol['seeds']
        datasets=a.datasets or ['playing','agentx']
        loads=a.loads or list(LOADS)
        load_points=[
            dict(dataset=dataset,seed=seed,load=load,
                 concurrency=session_concurrency(protocol,dataset,seed,load,a.concurrency))
            for dataset in datasets for seed in seeds for load in loads
        ]
    except (OSError,ValueError,KeyError) as error:parser.error(str(error))
    probe=subprocess.run([python,'-c','import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'],env=env)
    if probe.returncode:
        parser.error('Native experiments require CUDA-capable PyTorch and visible GPUs. Use --verify for CPU HiSim verification.')
    try:model=resolve_model(a.model,python,env)
    except (ValueError,subprocess.CalledProcessError) as error:parser.error(str(error))
    ssd=a.ssd.expanduser().resolve()
    ssd.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True)
    try:assets=prepare_assets(data,output/'assets',load_points,model=model,ssd=ssd,server_args=a.server_args)
    except (OSError,ValueError,KeyError) as e:parser.error(str(e))
    resolved_protocol=output/'protocol.json'
    dump(resolved_protocol,protocol)
    dump(output/'suite.json',dict(backend=backend,datasets=datasets,seeds=seeds,loads=loads,policies=a.policies,data=str(data),concurrency=a.concurrency,load_points=load_points,staging_GB=a.staging_gb,dynamic_capacity=a.dynamic_capacity,admission_threshold_override=a.admission_threshold,fixed_admission_threshold=a.fixed_admission_threshold,protocol=protocol,python=python,native_installed_sglang=a.native_installed_sglang))
    print(f'{backend}: {len(load_points)} load points; policies={a.policies}; serial execution.',flush=True)
    try:
        for point in load_points:
            dataset,seed,load=(point[key] for key in ('dataset','seed','load'))
            case=output/f'{dataset}-s{seed}-{load}'
            cmd=[python,'-m','hohlraum.experiments.run',
                 '--assets',str(assets[dataset,seed,point['concurrency']]),'--protocol',str(resolved_protocol),
                 '--dataset',dataset,'--seed',str(seed),'--load',load,
                 '--concurrency',str(point['concurrency']),'--output',str(case),
                 '--policies',*a.policies,'--staging-gb',str(a.staging_gb)]
            if a.dynamic_capacity:cmd.append('--dynamic-capacity')
            if a.fixed_admission_threshold:cmd.append('--fixed-admission-threshold')
            if a.native_installed_sglang:cmd.append('--native-installed-sglang')
            if a.admission_threshold is not None:cmd+=['--admission-threshold',str(a.admission_threshold)]
            subprocess.run(cmd,cwd=ROOT,env=env,check=True)
            summarize(output)
    except subprocess.CalledProcessError as error:
        raise SystemExit('Experiment failed (exit '+str(error.returncode)+'). Inspect logs under '+str(output)) from None
    finally:
        summarize(output)
    print('Suite completed.',flush=True)


if __name__=='__main__':main()
