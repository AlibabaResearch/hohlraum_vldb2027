"""Internal single-point runner used by the public run.sh experiment suite."""
from hohlraum.experiments.resources import lifecycle_environment
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
DEFAULT_PROTOCOL=ROOT/'hohlraum/experiments/protocol.json'
LOADS=('low','medium','high')

def load(path):return json.loads(Path(path).read_text())

from hohlraum.experiments.common.diagnostics import report


def read_serving_protocol(path=None):
    from hohlraum.experiments.common.workload import validate_measurement_duration

    protocol=load(path.expanduser() if path is not None else DEFAULT_PROTOCOL)
    if not isinstance(protocol,dict):
        raise ValueError('Serving protocol must be a JSON object')
    validate_measurement_duration(protocol['measurement_seconds'])
    if not isinstance(protocol.get('uniform_warmup_arrivals',False),bool):
        raise ValueError('uniform_warmup_arrivals must be true or false')
    return protocol


def session_concurrency(protocol,dataset,seed,load_tier,override=None):
    if override is not None:
        return override
    levels=protocol[dataset]['concurrency']
    if isinstance(levels,dict):
        levels=levels.get(str(seed))
    if (not isinstance(levels,list) or len(levels)!=len(LOADS)
            or any(type(value) is not int or value<1 for value in levels)):
        raise ValueError(f'Expected three positive concurrency values for {dataset} seed {seed}, in low, medium, high order')
    return levels[LOADS.index(load_tier)]


def select_capacity_control(protocol, controller=None):
    if controller is None:
        return protocol
    current = protocol.get('capacity_control') or {}
    if controller == 'throughput':
        if current.get('mode') != 'throughput' or 'expansion_thresholds' in current:
            protocol['capacity_control'] = dict(
                mode='throughput', period_s=60, window_s=300,
                step_fraction=0.025, min_read_fraction=float(current.get('min_read_fraction', 0.25)),
            )
    elif controller == 'tiered':
        if 'expansion_thresholds' not in current:
            if 'tiered_capacity_control' not in protocol:
                raise ValueError('Protocol lacks tiered_capacity_control parameters for --capacity-controller tiered')
            protocol['capacity_control'] = dict(protocol['tiered_capacity_control'])
    else:
        raise ValueError('Unknown capacity controller: ' + str(controller))
    return protocol


def policy_config(policy, protocol, case, library=None, staging_gb=0, *, warmup=False, dynamic_capacity=False):
    """One policy configuration for native and simulated serving."""
    if policy not in ('unlimited', 'budget640', 'admission-dynamic', 'staging'):
        raise ValueError('Unknown policy: '+policy)
    enabled = policy in ('admission-dynamic', 'staging')
    cfg = dict(enabled=enabled, write_budget_bytes=None if policy=='unlimited' else protocol['write_budget_bytes'])
    if enabled:
        if library is None: raise ValueError('Admission requires a history library')
        cfg.update(staging_GB=staging_gb if policy=='staging' else 0,
                   threshold=protocol['admission_threshold'],
                   dynamic_admission_threshold=True,
                   threshold_period_s=protocol['admission_period_seconds'],
                   threshold_window_s=protocol['admission_window_seconds'],
                   history_library=str(Path(library).resolve()),
                   admission_policy='agent', residency_policy='joint_hts',
                   duration_s=protocol['warmup_seconds' if warmup else 'measurement_seconds'],
                   stats_path=str(Path(case)/'policy-stats'))
    if enabled and policy=="staging":
        # (Staging + SSD) assumes WT backing ownership, allowing read copies
        # to leave while HBM remains. These flags are not prefetch-failure tuning.
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['hisim','sglang'],default='sglang')
    p.add_argument('--assets',type=Path,default=ROOT/'assets.local.json',help='Local paths to prepared inputs/models; see README')
    p.add_argument('--dataset',choices=['playing','agentx'],default='playing')
    p.add_argument('--seed',type=int)
    p.add_argument('--load',choices=['low','medium','high'],default='medium')
    p.add_argument('--concurrency',type=int)
    p.add_argument('--protocol',type=Path,help='Serving protocol JSON')
    p.add_argument('--policies',nargs='+',choices=['unlimited','budget640','admission-dynamic','staging'],default=['unlimited','budget640','admission-dynamic'])
    p.add_argument('--staging-gb',type=float,default=30)
    p.add_argument('--dynamic-capacity',action='store_true',help='Enable capacity controller for staging policy; initial size is --staging-gb')
    p.add_argument('--capacity-controller',choices=['throughput','tiered'],help='Select capacity controller mode (default: protocol capacity_control, throughput)')
    p.add_argument('--admission-threshold',type=float,help='Explicit threshold override recorded in the protocol')
    p.add_argument('--fixed-admission-threshold',action='store_true',help='Keep the configured threshold constant for a controlled comparison')
    p.add_argument('--uniform-warmup-arrivals',action='store_true',help='Spread initial roots over the warmup hour without changing within-root delays')
    p.add_argument('--warmup-checkpoint',type=Path,help='Reuse an existing warmup client_checkpoint.json when cold_cache_after_warmup is true')
    p.add_argument('--native-installed-sglang',action='store_true',
                   help='Use the current environment’s SGLang build for native serving')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve()
    if a.native_installed_sglang and a.backend!='sglang':p.error('--native-installed-sglang requires --backend sglang')
    paths = [str(ROOT)]
    inherited = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    if a.native_installed_sglang:
        inherited = [path for path in inherited if path and not Path(path).resolve().is_relative_to(VENDOR)]
    else:
        paths += [str(VENDOR), str(VENDOR/'python'), str(VENDOR/'tools/sglang-simulator/src')]
    os.environ['PYTHONPATH'] = os.pathsep.join(paths + [path for path in inherited if path])
    if a.concurrency is not None and a.concurrency<1:p.error('concurrency must be positive')
    if a.staging_gb<0:p.error('staging-gb must be nonnegative')
    if not a.assets.is_file():p.error('Missing asset manifest '+str(a.assets)+'. Supply local input/model paths; no calibration work is required for native runs.')
    assets=load(a.assets)
    try:
        protocol=read_serving_protocol(a.protocol)
        select_capacity_control(protocol,a.capacity_controller)
        if a.seed is None:a.seed=protocol['seeds'][0]
        concurrency=session_concurrency(protocol,a.dataset,a.seed,a.load,a.concurrency)
    except (OSError,ValueError,KeyError,IndexError) as error:p.error(str(error))
    a.uniform_warmup_arrivals=a.uniform_warmup_arrivals or protocol.get('uniform_warmup_arrivals',False)
    protocol['uniform_warmup_arrivals']=a.uniform_warmup_arrivals
    if a.backend=='hisim' and protocol['cold_cache_after_warmup']:
        p.error('HiSim uses retained warmup KV in a single process')
    if a.admission_threshold is not None:
        if not 0<=a.admission_threshold<=1:p.error('admission threshold must be in [0,1]')
        protocol['admission_threshold']=a.admission_threshold
    selection=assets.get('datasets',{}).get(a.dataset,{}).get(str(a.seed))
    if not selection:p.error('No prepared selection for this dataset/seed; seed does not resample rows')
    from hohlraum.experiments.preprocessing.validation import ensure_inputs
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
    print(f'{a.backend}: {a.dataset}, seed={a.seed}, concurrency={concurrency}, warmup={protocol["warmup_seconds"]}s, formal={protocol["measurement_seconds"]}s, queue={protocol["queue_delay_seconds"]}s/{protocol["waiting_queue_limit"]}')
    hardware = gpu_inventory() if a.backend=='sglang' else None
    if hardware is not None:print('GPU inventory: '+json.dumps(hardware))
    a.output.mkdir(parents=True,exist_ok=False)
    if hardware is not None:(a.output/'hardware.json').write_text(json.dumps(hardware,indent=2))
    (a.output/'data_preparation.json').write_text(json.dumps(data_preparation,indent=2))
    (a.output/'selection.json').write_text(json.dumps(dict(dataset=a.dataset,seed=a.seed,concurrency=concurrency,selection=selection,protocol=protocol,native_installed_sglang=a.native_installed_sglang),indent=2))
    resolved_protocol=a.output/'protocol.json';resolved_protocol.write_text(json.dumps(protocol,indent=2))
    if a.uniform_warmup_arrivals and a.backend=='sglang':
        from hohlraum.experiments.common.workload import normalize,uniform_warmup_offers
        prepared_rows=load(rows);plans=normalize(prepared_rows,load(deps) if deps else None)
        prepared_rows=uniform_warmup_offers(prepared_rows,plans,concurrency,protocol['warmup_seconds'])
        rows=str(a.output/'uniform-arrival-rows.json');Path(rows).write_text(json.dumps(prepared_rows,separators=(',',':')))
    library=a.output/'libhistory_bridge.so'
    if admission:
        call(['c++','-O2','-std=c++17','-shared','-fPIC','-I'+str(ROOT/'hohlraum/core/cpp'),str(ROOT/'hohlraum/core/cpp/sli_bridge.cpp'),'-o',str(library)])
    checkpoint=a.output/'warmup/client_checkpoint.json'
    resident_warmup=not protocol['cold_cache_after_warmup']
    if a.warmup_checkpoint is not None:
        if resident_warmup or a.backend!='sglang':
            p.error('--warmup-checkpoint requires native sglang with cold_cache_after_warmup=true')
        if not a.warmup_checkpoint.is_file():
            p.error('Missing warmup checkpoint: '+str(a.warmup_checkpoint))
        import shutil
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(a.warmup_checkpoint,checkpoint)
    labels = a.policies if (resident_warmup or a.warmup_checkpoint is not None) else ['warmup']+a.policies
    for label in labels:
        case=a.output/label;warm=label=='warmup';policy='unlimited' if warm else label
        if a.backend=='hisim':
            cmd=[sys.executable,'-m','hohlraum.experiments.backends.hisim.case','prepare','--dataset',a.dataset,'--rows',rows,'--profile',profile_path,'--seed',str(a.seed),'--concurrency',str(concurrency),'--policy',policy,'--staging-gb',str(a.staging_gb),'--output',str(case)]
            if a.dynamic_capacity:cmd+=['--dynamic-capacity']
            if deps:cmd+=['--dependencies',deps]
            cmd+=['--protocol',str(resolved_protocol)]
            if a.fixed_admission_threshold:cmd+=['--fixed-admission-threshold']
            if a.uniform_warmup_arrivals:cmd+=['--uniform-warmup-arrivals']
            if policy in ['admission-dynamic','staging']:cmd+=['--priors',assets['priors'],'--history-library',str(library)]
            call(cmd)
            call([sys.executable,'-m','hohlraum.experiments.backends.hisim.case','run','--case',str(case)])
            continue
        case.mkdir();feedback=case/'feedback';feedback.mkdir()
        cfg=policy_config(policy,protocol,case,library,a.staging_gb,warmup=warm,dynamic_capacity=a.dynamic_capacity)
        if a.fixed_admission_threshold and cfg['enabled']:cfg['dynamic_admission_threshold']=False
        initial=dict(enabled=False,write_budget_bytes=None) if resident_warmup else cfg
        (case/'hohlraum.json').write_text(json.dumps(initial,indent=2));(case/'server_args.json').write_text(json.dumps(profile,indent=2))
        if resident_warmup:(case/'formal-policy.json').write_text(json.dumps(cfg,indent=2))
        import tempfile
        ssd=Path(tempfile.mkdtemp(prefix='hohlraum-'+label+'-',dir=assets['ssd_directory']))
        env={k:v for k,v in os.environ.items() if not k.startswith(('SGLANG_SIMULATOR_', 'HISIM_')) and k!='SGLANG_USE_CPU_ENGINE'};env.update(HOHLRAUM_NATIVE_FEEDBACK=str(feedback),HOHLRAUM_NATIVE_SERVER_ARGS=str(case/'server_args.json'),HOHLRAUM_CONFIG=str(case/'hohlraum.json'),SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=str(ssd),SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND='python')
        # Profiles explicitly request trace context lengths; synthetic replay
        # does not evaluate model quality beyond its trained context window.
        env.setdefault('SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN','1')
        if resident_warmup:env['HOHLRAUM_RESIDENT_WARMUP']='1'
        if cfg['enabled']:
            env.update(lifecycle_environment(assets['priors']))
        (case/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('HOHLRAUM_','SGLANG_HICACHE','SGLANG_ALLOW_'))},indent=2))
        url='http://127.0.0.1:'+str(profile.get('port',30000))
        # A fresh process and SSD namespace per policy; resident warmup retains KV into measurement.
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
                if resident_warmup:
                    cmd+=['--warmup-seconds',str(protocol['warmup_seconds']),'--formal-policy',str(case/'formal-policy.json')]
                elif warm:cmd+=['--save-checkpoint']
                else:cmd+=['--checkpoint',str(checkpoint)]
                cmd+=['--final-snapshot']
                call(cmd,env=env)
                if warm:
                    import shutil
                    shutil.copy2(case/'client/client_checkpoint.json',checkpoint)
                report(case/'client')
            finally:stop(server)
        # Only remove the unique disposable cache directory created by this run.
        import shutil
        shutil.rmtree(ssd)

if __name__=='__main__':main()
