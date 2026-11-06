"""Paper trace client; no simulator timing/model replacements."""
import os,json,asyncio,time
from pathlib import Path
from hohlraum.experiments.common.workload import reference_model_fold
CASE=Path(os.environ['HOHLRAUM_EXPERIMENT_CASE'])
from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
from sglang.srt.server_args import ServerArgs
from sglang_simulator.dataset import GenericRequest,SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from hohlraum.experiments.backends.hisim.dispatch import install, install_horizon
manifest=json.loads((CASE/'manifest.json').read_text())
contract=json.loads((CASE/'input_contract.json').read_text())
install(manifest['concurrency'], contract)
install_horizon(manifest['horizon_s'],CASE)

class Runner(SGLangBenchmarkRunner):
    def get_request(self,dataset,**kwargs):
        for req in dataset:
            yield req,dict(total_request=len(dataset),created_time=0,agentx=req.custom_params['agentx'])

async def run(runner,dataset):
    task=asyncio.create_task(runner.async_benchmark(BenchmarkConfig(request_rate=1,ignore_request_timestamp=False),dataset))
    try:
        while not task.done():
            if (CASE/'horizon.json').exists():return
            await asyncio.sleep(.1)
        await task
        if not (CASE/'horizon.json').exists():raise RuntimeError('input ended before requested horizon')
    finally:
        if not task.done():
            task.cancel()
            try:await task
            except asyncio.CancelledError:pass

def main():
    args=json.loads((CASE/'server_args.json').read_text())
    rows=json.loads(Path(os.environ['HISIM_AGENTX_COMPACT_ROWS']).read_text())
    vocab=json.loads((Path(args['model_path'])/'config.json').read_text())['vocab_size']
    training=None
    if manifest['workload']=='playing' and json.loads((CASE/'hohlraum.json').read_text()).get('enabled'):
        training_path=Path(os.environ['HOHLRAUM_JOINT_MODELS'])/'training.json'
        training=json.loads(training_path.read_text())
    reqs=[]
    for serial,row in enumerate(rows):
        meta={k:row[k] for k in ['root','chain','index','owner','reference_fold','admission_fold','root_offer','first_delay','think_time']}
        meta["reference_fold"]=reference_model_fold(row,manifest["workload"],training)
        meta.update(trace_serial=serial,vocab_size=vocab)
        reqs.append(GenericRequest(token_ids=[7],input_length=row['raw_input_tokens'],output_length=1,custom_params={'agentx':meta}))
    started=time.monotonic();runner=Runner(ServerArgs(**args))
    try:runner.engine.loop.run_until_complete(run(runner,SimpleDataset(reqs=reqs)))
    finally:runner.shutdown()
    snap=json.loads((CASE/'horizon.json').read_text());h=manifest['horizon_s']
    stats=[x for x in snap['requests'] if x['last_event_time']<=h]
    times=sorted(x['gen_token_latencies'][0] for x in stats)
    if not times:raise RuntimeError('no completed requests inside horizon')
    p=.9*(len(times)-1);i=int(p);q=times[i]+(p-i)*(times[min(i+1,len(times)-1)]-times[i])
    result=dict(manifest=manifest,completed=len(stats),throughput=sum(x['input_length'] for x in stats)/h,
        goodput=sum(x['input_length'] for x in stats if x['gen_token_latencies'][0]<=10)/h,
        ttft_p90_s=q,write_bytes=snap['progress']['write_bytes'],budget_exhausted_at_s=snap['progress']['budget_exhausted_at_s'],
        wall_s=time.monotonic()-started,performance_validated=False)
    (CASE/'result.json').write_text(json.dumps(result,indent=2));print('PASS',json.dumps(result),flush=True)
if __name__=='__main__':main()
