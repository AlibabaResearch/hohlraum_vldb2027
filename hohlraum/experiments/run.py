"""Internal single-point runner used by the public run.sh experiment suite."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'third_party/sglang'
def load(path):return json.loads(Path(path).read_text())
from hohlraum.experiments.common.diagnostics import report


def policy_config(policy, protocol, case, library=None, staging_gb=0, *, warmup=False, dynamic_capacity=False):
    """One policy configuration for native and simulated serving."""
    if policy not in ('unlimited', 'budget640', 'admission07', 'admission-dynamic', 'staging'):
        raise ValueError('Unknown policy: '+policy)
    enabled = policy in ('admission07', 'admission-dynamic', 'staging')
    cfg = dict(enabled=enabled, write_budget_bytes=None if policy=='unlimited' else protocol['write_budget_bytes'])
    if enabled:
        if library is None: raise ValueError('Admission requires a history library')
        cfg.update(staging_GB=staging_gb if policy=='staging' else 0,
                   threshold=protocol['admission_threshold'],
                   dynamic_admission_threshold=policy!='admission07',
                   threshold_period_s=protocol['admission_period_seconds'],
                   threshold_window_s=protocol['admission_window_seconds'],
                   history_library=str(Path(library).resolve()), reference_history=True,
                   admission_policy='agent', residency_policy='joint_hts',
                   duration_s=protocol['warmup_seconds' if warmup else 'measurement_seconds'],
                   stats_path=str(Path(case)/'policy-stats'))
    if enabled and policy=="staging":
        cfg.update(stage_hbm_leaves=True,stage_hbm_dram_leaves=True)
    if enabled and policy=="staging" and dynamic_capacity:
        cfg.update(dynamic_capacity=True,capacity_control=dict(protocol["capacity_control"]))
    return cfg


def call(command, env=None):
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL);process.wait()


def gpu_inventory():
    """Physical NVIDIA inventory; CUDA visibility and device choice remain native."""
    result = dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  scope='Physical inventory, not a count of CUDA-visible devices')
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                '--format=csv,noheader,nounits'], capture_output=True,
                               text=True, timeout=10, check=True)
        import csv
        result['devices'] = [dict(uuid=row[0].strip(), name=row[1].strip(),
                                  memory_mib=int(row[2].strip()))
                             for row in csv.reader(query.stdout.splitlines())]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        result['unavailable'] = str(error)
    return result


def main():
    paths = [str(ROOT), str(VENDOR), str(VENDOR/'python'), str(VENDOR/'tools/sglang-simulator/src')]
    os.environ['PYTHONPATH'] = os.pathsep.join(paths + ([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH') else []))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['policy','hisim','sglang'],default='sglang')
    p.add_argument('--analyze',action='store_true',help='Analyze an existing case without launching an experiment')
    p.add_argument('--assets',type=Path,default=ROOT/'assets.local.json',help='Local paths to prepared inputs/models; see README')
    p.add_argument('--dataset',choices=['playing','agentx'],default='playing')
    p.add_argument('--seed',type=int,default=272015)
    p.add_argument('--load',choices=['low','medium','high'],default='medium')
    p.add_argument('--concurrency',type=int)
    p.add_argument('--policies',nargs='+',choices=['unlimited','budget640','admission-dynamic','staging'],default=['unlimited','budget640','admission-dynamic'])
    p.add_argument('--staging-gb',type=float,default=30)
    p.add_argument('--dynamic-capacity',action='store_true',help='Enable capacity controller for staging policy; initial size is --staging-gb')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();a.output=a.output.resolve()
    if a.concurrency is not None and a.concurrency<1:p.error('concurrency must be positive')
    if a.staging_gb<0:p.error('staging-gb must be nonnegative')
    if a.analyze:report(a.output);return
    if not a.assets.is_file():p.error('Missing asset manifest '+str(a.assets)+'. Supply local input/model paths; no calibration work is required for native runs.')
    assets=load(a.assets)
    if a.backend=='policy':
        data=assets.get('policy_data')
        if not data or not Path(data).is_dir():p.error('Missing policy_data directory in asset manifest')
        cmd=['bash',str(ROOT/'hohlraum/experiments/backends/policy_replay/reproduce.sh'),'--data',data,'--output',str(a.output)]
        if a.check_only:cmd.append('--check-only')
        call(cmd);return
    protocol=load(ROOT/'hohlraum/experiments/protocol.json')
    concurrency=a.concurrency or protocol[a.dataset]['concurrency'][['low','medium','high'].index(a.load)]
    selection=assets.get('datasets',{}).get(a.dataset,{}).get(str(a.seed))
    if not selection:p.error('No prepared selection for this dataset/seed; seed does not resample rows')
    from hohlraum.experiments.preprocessing.run import ensure_inputs
    selection,data_preparation=ensure_inputs(a.dataset,selection,a.seed)
    print('Data preparation: '+json.dumps(data_preparation))
    rows=selection.get('rows');deps=selection.get('dependencies')
    required=[rows]+([deps] if a.dataset=='agentx' else [])
    profile_path=assets.get('hisim_profile' if a.backend=='hisim' else 'native_server_args')
    required.append(profile_path)
    admission=any(x in a.policies for x in ['admission-dynamic','staging'])
    if admission:required.append(assets.get('priors'))
    if any(not x or not Path(x).exists() for x in required):p.error('Missing prepared data, dependency DAG, server settings or priors in asset manifest')
    profile=load(profile_path)
    if a.backend=='hisim':
        from hohlraum.experiments.backends.hisim.case import validate_profile
        validate_profile(profile)
    else:
        if profile.get('hicache_storage_backend')!='file':p.error('Native paired runner currently requires file storage to guarantee separate cold SSD namespaces')
        if not assets.get('ssd_directory') or not Path(assets['ssd_directory']).is_dir():p.error('Provide ssd_directory on the intended SSD')
        if not Path(profile.get('model_path','')).is_dir():p.error('Missing model_path in native server settings')
    from hohlraum.experiments.common.workload import normalize
    normalize(load(rows),load(deps) if deps else None)
    print(f'{a.backend}: {a.dataset}, seed={a.seed}, concurrency={concurrency}, warmup={protocol["warmup_seconds"]}s, formal={protocol["measurement_seconds"]}s, queue={protocol["queue_delay_seconds"]}s/{protocol["waiting_queue_limit"]}')
    hardware = gpu_inventory() if a.backend=='sglang' else None
    if hardware is not None:print('GPU inventory: '+json.dumps(hardware))
    if a.check_only:return
    a.output.mkdir(parents=True,exist_ok=False)
    if hardware is not None:(a.output/'hardware.json').write_text(json.dumps(hardware,indent=2))
    (a.output/'data_preparation.json').write_text(json.dumps(data_preparation,indent=2))
    (a.output/'selection.json').write_text(json.dumps(dict(dataset=a.dataset,seed=a.seed,concurrency=concurrency,selection=selection,protocol=protocol),indent=2))
    library=a.output/'libhistory_bridge.so'
    if admission:
        call(['c++','-O2','-std=c++17','-shared','-fPIC','-I'+str(ROOT/'hohlraum/core/cpp'),str(ROOT/'hohlraum/core/cpp/sli_bridge.cpp'),'-o',str(library)])
    checkpoint=a.output/'warmup/client_checkpoint.json'
    for label in ['warmup']+a.policies:
        case=a.output/label;warm=label=='warmup';policy='unlimited' if warm else label
        if a.backend=='hisim':
            cmd=[sys.executable,'-m','hohlraum.experiments.backends.hisim.case','prepare','--dataset',a.dataset,'--rows',rows,'--profile',profile_path,'--seed',str(a.seed),'--concurrency',str(concurrency),'--policy',policy,'--staging-gb',str(a.staging_gb),'--output',str(case)]
            if a.dynamic_capacity:cmd+=['--dynamic-capacity']
            if deps:cmd+=['--dependencies',deps]
            if warm:cmd+=['--warmup']
            else:cmd+=['--checkpoint',str(checkpoint)]
            if policy in ['admission-dynamic','staging']:cmd+=['--priors',assets['priors'],'--history-library',str(library)]
            call(cmd)
            call([sys.executable,'-m','hohlraum.experiments.backends.hisim.case','run','--case',str(case)])
            continue
        case.mkdir();feedback=case/'feedback';feedback.mkdir()
        cfg=policy_config(policy,protocol,case,library,a.staging_gb,warmup=warm,dynamic_capacity=a.dynamic_capacity)
        (case/'hohlraum.json').write_text(json.dumps(cfg,indent=2));(case/'server_args.json').write_text(json.dumps(profile,indent=2))
        import tempfile
        ssd=Path(tempfile.mkdtemp(prefix='hohlraum-'+label+'-',dir=assets['ssd_directory']))
        env={k:v for k,v in os.environ.items() if not k.startswith(('SGLANG_SIMULATOR_', 'HISIM_')) and k!='SGLANG_USE_CPU_ENGINE'};env.update(HOHLRAUM_NATIVE_FEEDBACK=str(feedback),HOHLRAUM_NATIVE_SERVER_ARGS=str(case/'server_args.json'),HOHLRAUM_CONFIG=str(case/'hohlraum.json'),SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=str(ssd),SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND='python')
        if cfg['enabled']:
            priors=Path(assets['priors']);env.update(HOHLRAUM_JOINT_MODELS=str(priors),HOHLRAUM_MODEL_COUNT=str(len(list(priors.glob('[0-9].txt')))),HOHLRAUM_BRANCH_PRIOR=str(priors/'branch-prior.txt'),HOHLRAUM_LENGTH_CURVES=str(priors/'length_curves.txt'),HOHLRAUM_ONLINE_PRIOR='20')
        (case/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('HOHLRAUM_','SGLANG_HICACHE'))},indent=2))
        url='http://127.0.0.1:'+str(profile.get('port',30000))
        # A dedicated fresh process and storage directory per phase; no cache flush race.
        with (case/'server.log').open('w') as log:
            server=subprocess.Popen([sys.executable,'-m','hohlraum.experiments.backends.sglang.server'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+900
                while True:
                    if server.poll() is not None:raise RuntimeError('Native server exited; inspect '+str(case/'server.log'))
                    try:
                        with urllib.request.urlopen(url+'/health',timeout=1) as response:healthy=response.status==200
                    except Exception:healthy=False
                    if healthy and (feedback/'queue.json').exists():break
                    if time.monotonic()>deadline:raise RuntimeError('Native server startup timed out')
                    time.sleep(1)
                model=load(Path(profile['model_path'])/'config.json')
                cmd=[sys.executable,'-m','hohlraum.experiments.backends.sglang.client','--url',url,'--input',rows,'--output',str(case/'client'),'--dataset',a.dataset,'--concurrency',str(concurrency),'--feedback-dir',str(feedback),'--vocab-size',str(model['vocab_size']),'--seconds',str(protocol['warmup_seconds'] if warm else protocol['measurement_seconds']),'--idle-timeout',str(protocol['idle_timeout_seconds']),'--queue-delay',str(protocol['queue_delay_seconds']),'--waiting-limit',str(protocol['waiting_queue_limit'])]
                if deps:cmd+=['--dependencies',deps]
                if warm:cmd+=['--save-checkpoint']
                else:cmd+=['--checkpoint',str(checkpoint)]
                call(cmd)
                if warm:
                    import shutil
                    shutil.copy2(case/'client/client_checkpoint.json',checkpoint)
                report(case/'client')
            finally:stop(server)
        # Only remove the unique disposable cache directory created by this run.
        import shutil
        shutil.rmtree(ssd)

if __name__=='__main__':main()
