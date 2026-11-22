"""External HTTP trace client; wall-clock arrivals, no simulator/predictor."""
import argparse,asyncio,json,time,hashlib,statistics,traceback,os
from pathlib import Path
from types import SimpleNamespace as NS
from hohlraum.experiments.common.workload import TraceSource, dump, restore, encode as encode_tokens
from hohlraum.experiments.common.backpressure import QueueDelayAdmission
from hohlraum.experiments.backends.sglang.feedback import NativeFeedback
from hohlraum.experiments.common.history_restore import load_checkpoint
from hohlraum.experiments.common.workload import normalize, input_contract, reference_model_fold, validate_measurement_duration

def save(p,data):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def encode(row, serial, dataset, vocab_size):
 return encode_tokens(row['root'], row['blocks'], row['raw_input_tokens'], serial,
                      vocab_size, input_contract(dataset)['token_tail'])
async def replay(args):
 validate_measurement_duration(args.seconds,warmup=args.save_checkpoint)
 if args.warmup_seconds and (args.save_checkpoint or args.checkpoint or not args.formal_policy):
  raise ValueError('Resident warmup requires a formal policy and one continuous client/server')
 import aiohttp
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 blob=Path(args.input).read_bytes();rows=json.loads(blob);queued=[]
 training=None
 if os.environ.get('HOHLRAUM_JOINT_MODELS'):
  training_path=Path(os.environ['HOHLRAUM_JOINT_MODELS'])/'training.json'
  if args.dataset=='playing' and training_path.exists():training=json.loads(training_path.read_text())
 for serial,row in enumerate(rows):
  meta={k:row[k] for k in ['root','chain','index','root_offer','first_delay','think_time','owner','reference_fold']}
  meta['prefix_blocks']=row['blocks'];meta['prefix_page_size']=1024
  meta['reference_fold']=reference_model_fold(row,args.dataset,training)
  meta['root_offer']*=args.arrival_scale;meta['serial']=serial
  queued.append((0,serial,NS(rid=f'trace-{serial}',sampling_params=NS(custom_params={'hohlraum':meta,'simulation':{'agentx':meta}}))))
 plans=normalize(rows,json.loads(Path(args.dependencies).read_text()) if args.dependencies else None)
 driver=TraceSource(queued,args.concurrency,plans)
 driver.idle_timeout=args.idle_timeout
 if args.checkpoint:
  save(out/'resume_checks.json',restore(driver,load_checkpoint(args.checkpoint)))
 transport=NativeFeedback(args.feedback_dir)
 feedback=(0.,0)
 gate=QueueDelayAdmission(lambda: feedback,args.queue_delay,args.waiting_limit)
 transport.gate=gate
 save(out/'protocol.json',dict(vars(args),input_sha256=hashlib.sha256(blob).hexdigest(),timing='real wall clock; client encoding/HTTP included; one generated token',trace_semantics='synthetic token IDs preserving block prefix sharing, trace prompt lengths and session timing'))
 results=[];pending={};lastsave=0
 if args.checkpoint:
  token=transport.notify_restore(args.checkpoint,args.input,args.dataset)
  save(out/'history-restore-acks.json',await transport.wait_restored(token))
 start=time.monotonic();activated=not args.warmup_seconds
 horizon=args.warmup_seconds+args.seconds
 save(out/'status.json',dict(state='running',start_unix=time.time()))
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),connector=aiohttp.TCPConnector(limit=256,force_close=True)) as client:
  async def send(req):
   meta=req.sampling_params.custom_params['hohlraum'];serial=meta['serial'];row=rows[serial]
   ids=req.input_ids
   agent={k:meta[k] for k in ['root','chain','index','owner','reference_fold','prefix_blocks','prefix_page_size']}
   body=dict(rid=req.rid,input_ids=ids,sampling_params=dict(max_new_tokens=1,temperature=0,ignore_eos=True,custom_params={'hohlraum':agent}))
   http_start=time.monotonic()-start
   async with client.post(args.url+'/generate',json=body) as response:
    raw=await response.text()
    if response.status!=200:raise RuntimeError(f'HTTP {response.status}: {raw[:1000]}')
    data=json.loads(raw)
   del req.input_ids
   end=time.monotonic()-start
   return req,dict(**gate.records[req.rid],serial=serial,rid=req.rid,arrival_s=driver.arrivals[req.rid],completion_s=end,ttft_s=end-driver.arrivals[req.rid],http_s=end-http_start,input_tokens=len(ids),meta_info=data.get('meta_info',{}),output_ids=data.get('output_ids'))
  with (out/'requests.jsonl').open('w',buffering=1) as log:
   try:
    while (now:=time.monotonic()-start)<horizon:
     if not activated and now>=args.warmup_seconds:
      save(out/'activation.json',await transport.activate(args.formal_policy,start,horizon))
      activated=True;now=time.monotonic()-start
     due=driver.pop_due(now)
     # Readiness precedes input encoding and external backpressure. Neither
     # belongs to the trace's idle interval used for lifecycle scoring.
     for req in due:
      serial=req.sampling_params.custom_params['hohlraum']['serial']
      req.input_length=rows[serial]['raw_input_tokens']
      transport.notify_ready(req)
     for req in due:
      meta=req.sampling_params.custom_params['hohlraum'];serial=meta['serial']
      req.input_ids=await asyncio.to_thread(encode,rows[serial],serial,args.dataset,args.vocab_size)
      gate.offer(req,driver.arrivals[req.rid])
     feedback=transport.refresh(now)
     for req in gate.release(now):pending[asyncio.create_task(send(req))]=req
     if now-lastsave>=30:
      save(out/'progress.json',dict(elapsed_s=now,phase='formal' if activated else 'warmup',completed=len(results),inflight=len(pending),live_roots=len(driver.live)));lastsave=now
     nxt=driver.next_time();delay=max(0,min(.05,horizon-now,nxt-now if nxt is not None else 1))
     if pending:
      done,_=await asyncio.wait(pending,timeout=delay,return_when=asyncio.FIRST_COMPLETED)
      for task in done:
       pending.pop(task);req,row=task.result()
       if row['completion_s']<=horizon:
        results.append(row);log.write(json.dumps(row)+'\n');gate.finish(req.rid);driver.finish(req,row['completion_s'])
     else:await asyncio.sleep(delay)
    if args.final_snapshot or args.warmup_seconds:
     save(out/'server-final.json',await transport.snapshot())
   finally:
    if args.save_checkpoint:
     save(out/'client_checkpoint.json',dump(driver,args.seconds))
    for task in pending:task.cancel()
    await asyncio.gather(*pending,return_exceptions=True)
    save(out/'admission-events.json',driver.admission_audit)
    save(out/'client-admission.json',gate.records)
    transport.close()
 results=[r for r in results if r['arrival_s']>=args.warmup_seconds]
 if not results:raise RuntimeError('no completed requests')
 tt=sorted(r['ttft_s'] for r in results)
 def quantile(q):
  x=(len(tt)-1)*q;i=int(x);return tt[i]+(x-i)*(tt[min(i+1,len(tt)-1)]-tt[i])
 summary=dict(completed=len(results),total_input_tokens_s=sum(r['input_tokens'] for r in results)/args.seconds,slo10_input_tokens_s=sum(r['input_tokens'] for r in results if r['ttft_s']<=10)/args.seconds,ttft_p90_s=quantile(.9),ttft_p50_s=quantile(.5))
 save(out/'result.json',summary);save(out/'status.json',dict(state='complete'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--dependencies',help='Optional normalized dependency DAG; omitted means sequential chains');p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,default=14400);p.add_argument('--concurrency',type=int,required=True);p.add_argument('--idle-timeout',type=float,default=1200);p.add_argument('--arrival-scale',type=float,default=1)
 p.add_argument('--dataset',choices=['playing','agentx'],default='playing')
 p.add_argument('--vocab-size',type=int,default=151936)
 p.add_argument('--feedback-dir',required=True)
 p.add_argument('--queue-delay',type=float,default=10)
 p.add_argument('--waiting-limit',type=int,default=8)
 p.add_argument('--checkpoint');p.add_argument('--save-checkpoint',action='store_true')
 p.add_argument('--warmup-seconds',type=float,default=0,help='Native full-admission warmup, retaining all KV and the same client/server')
 p.add_argument('--formal-policy',help='Policy activated after resident warmup; initial server policy must be unlimited')
 p.add_argument('--final-snapshot',action='store_true',help='Export per-rank budget and policy state at completion')
 args=p.parse_args()
 try:asyncio.run(replay(args))
 except Exception:
  out=Path(args.output)
  if out.exists():save(out/'status.json',dict(state='failed',error=traceback.format_exc()))
  raise
