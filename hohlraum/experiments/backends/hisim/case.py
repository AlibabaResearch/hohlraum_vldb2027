"""Prepare portable experiment cases; native HiSim remains responsible for timing."""
import argparse,json,os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[4]
VENDOR=ROOT/'third_party/sglang'
def load(p):return json.loads(pathlib.Path(p).read_text())
def validate_profile(profile):
 def walk(x,path='profile'):
  if x is None:raise ValueError('Missing calibrated value: '+path)
  if isinstance(x,dict):
   for k,v in x.items():walk(v,path+'.'+k)
 walk(profile)
 for key in ['sim_config','server_args']:
  if key not in profile:raise ValueError('Missing '+key)
 for k in ['disk_read_bandwidth_gb','disk_write_bandwidth_gb','memory_read_bandwidth_gb','memory_write_bandwidth_gb']:
  if profile['sim_config']['platform'].get(k,0)<=0:raise ValueError('Missing positive bandwidth: '+k)
 for p in [profile['server_args'].get('model_path',''),profile['sim_config']['predictor'].get('database_path','')]:
  if not p or not pathlib.Path(p).exists():raise ValueError('Missing local model/predictor asset: '+p)
from hohlraum.experiments.common.workload import normalize, input_contract
from hohlraum.experiments.run import policy_config

def main():
 a=argparse.ArgumentParser(description=__doc__);sub=a.add_subparsers(dest='command',required=True)
 c=sub.add_parser('prepare');c.add_argument('--dynamic-capacity',action='store_true');c.add_argument('--staging-gb',type=float,default=30);c.add_argument('--dataset',choices=['agentx','playing'],required=True);c.add_argument('--rows',type=pathlib.Path,required=True);c.add_argument('--dependencies',type=pathlib.Path);c.add_argument('--profile',type=pathlib.Path,required=True);c.add_argument('--output',type=pathlib.Path,required=True);c.add_argument('--concurrency',type=int,required=True);c.add_argument('--policy',choices=['unlimited','budget640','admission07','admission-dynamic','staging'],required=True);c.add_argument('--seed',type=int,required=True);c.add_argument('--warmup',action='store_true');c.add_argument('--checkpoint',type=pathlib.Path);c.add_argument('--priors',type=pathlib.Path);c.add_argument('--history-library',type=pathlib.Path)
 r=sub.add_parser('run');r.add_argument('--case',type=pathlib.Path,required=True)
 args=a.parse_args()
 if args.command=='run':
  case=args.case.resolve();env=os.environ.copy();env.update(load(case/'env.json'))
  with (case/'run.log').open('w') as log:
   p=subprocess.Popen([sys.executable,'-m','hohlraum.experiments.backends.hisim.worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);(case/'process.json').write_text(json.dumps({'pid':p.pid}));rc=p.wait()
  (case/'exit.json').write_text(json.dumps({'returncode':rc,'has_result':(case/'result.json').exists()}))
  from hohlraum.experiments.common.diagnostics import report
  report(case)
  raise SystemExit(rc)
 profile=load(args.profile)
 if profile.get("server_args",{}).get("context_length") is None:
  profile["server_args"]["context_length"]=990144
 validate_profile(profile)
 if args.warmup and args.checkpoint:a.error('--warmup and --checkpoint are mutually exclusive')
 if args.concurrency<1:a.error('concurrency must be positive')
 for p in [args.rows,args.dependencies,args.checkpoint]:
  if p is not None and not p.is_file():a.error('missing input: '+str(p))
 if args.dataset=='agentx' and not args.dependencies:a.error('AgentX requires dependency DAG')
 if args.policy in ('admission07','admission-dynamic','staging') and (not args.priors or not args.history_library):a.error('admission policies require --priors and --history-library')
 plans=normalize(load(args.rows),load(args.dependencies) if args.dependencies else None)
 case=args.output.resolve();case.mkdir(parents=True,exist_ok=False)
 (case/'dependencies.json').write_text(json.dumps(plans))
 (case/'input_contract.json').write_text(json.dumps(input_contract(args.dataset),indent=2))
 protocol=load(ROOT/'hohlraum/experiments/protocol.json');h=protocol['warmup_seconds'] if args.warmup else protocol['measurement_seconds'];budget=None if args.policy=='unlimited' else protocol['write_budget_bytes']
 cfg=policy_config(args.policy,protocol,case,args.history_library,args.staging_gb,warmup=args.warmup,dynamic_capacity=args.dynamic_capacity)
 env={'PYTHONPATH':os.pathsep.join(map(str,[ROOT,VENDOR,VENDOR/'python',VENDOR/'tools/sglang-simulator/src'])),'SGLANG_USE_CPU_ENGINE':'1','SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND':'python','SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN':'1','SGLANG_SIMULATOR_OUTPUT_MODE':'OFFLINE','SGLANG_SIMULATOR_OUTPUT_DIR':str(case/'output'),'SGLANG_SIMULATOR_CONFIG_PATH':str(case/'sim_config.json'),'SGLANG_SIMULATOR_HICACHE_STORAGE_KEYS_PATH':str(case/'keys.txt'),'HOHLRAUM_CONFIG':str(case/'hohlraum.json'),'HOHLRAUM_EXPERIMENT_CASE':str(case),'HISIM_AGENTX_COMPACT_ROWS':str(args.rows.resolve()),'HISIM_AGENTX_HORIZON_S':str(h),'HISIM_NEW_ROOT_IDLE_TIMEOUT_S':str(protocol['idle_timeout_seconds']),'HISIM_CLIENT_QUEUE_DELAY_S':str(protocol['queue_delay_seconds']),'CUDA_VISIBLE_DEVICES':'','MAX_JOBS':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
 env['HISIM_CLIENT_WAITING_LIMIT']=str(protocol['waiting_queue_limit'])
 env['HOHLRAUM_TRACE_DEPENDENCIES']=str(case/'dependencies.json')
 if args.warmup:env['HISIM_SAVE_CLIENT_CHECKPOINT']='1'
 if args.checkpoint:env['HISIM_RESTORE_CLIENT_CHECKPOINT']=str(args.checkpoint.resolve())
 if args.priors:
  m=args.priors.resolve();env.update(HOHLRAUM_JOINT_MODELS=str(m),HOHLRAUM_MODEL_COUNT=str(len(list(m.glob('[0-9].txt')))),HOHLRAUM_BRANCH_PRIOR=str(m/'branch-prior.txt'),HOHLRAUM_LENGTH_CURVES=str(m/'length_curves.txt'),HOHLRAUM_ONLINE_PRIOR='20')
 manifest={'workload':args.dataset,'seed':args.seed,'concurrency':args.concurrency,'horizon_s':h,'policy':args.policy,'budget_bytes':budget,'dynamic_capacity':cfg.get('dynamic_capacity',False),'capacity_control':cfg.get('capacity_control'),'staging_GB':cfg.get('staging_GB',0),'profile':str(args.profile.resolve()),'ttft_includes_external_wait':True,'scoring_clock':'client_ready_to_completion_excluded','dynamic_admission_threshold':cfg.get('dynamic_admission_threshold',False)}
 for n,d in [('server_args.json',profile['server_args']),('sim_config.json',profile['sim_config']),('env.json',env),('manifest.json',manifest),('hohlraum.json',cfg)]:
  (case/n).write_text(json.dumps(d,indent=2)+'\n')
 print(case)
if __name__=='__main__':main()
