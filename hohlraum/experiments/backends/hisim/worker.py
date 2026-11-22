"""Paper trace client; no simulator timing/model replacements."""
import os,json,asyncio,time
from pathlib import Path
from hohlraum.experiments.common.diagnostics import hisim_request_metrics
from hohlraum.experiments.common.workload import reference_model_fold, validate_measurement_duration
CASE=Path(os.environ['HOHLRAUM_EXPERIMENT_CASE'])
def validate_empty_ssd_index(case):
    """Require a case-local empty SSD index before warmup."""
    raw = os.environ.get('SGLANG_SIMULATOR_HICACHE_STORAGE_KEYS_PATH')
    if not raw:
        raise RuntimeError('Cold-start experiment requires an explicit SSD index path')
    path = Path(raw).resolve()
    if path.parent != case.resolve():
        raise RuntimeError(f'SSD index must be case-local, not shared with another run: {path}')
    if path.exists() and path.stat().st_size:
        raise RuntimeError(f'Cold-start refused: nonempty SSD index {path}; use a fresh case. Never reuse completed-run SSD keys.')
    (case/'initial-cache-check.json').write_text(json.dumps(dict(
        ssd_index=str(path), initial_ssd_bytes=0, cold_start=True)))

validate_empty_ssd_index(CASE)
manifest=json.loads((CASE/'manifest.json').read_text())
validate_measurement_duration(manifest['formal_seconds'])
assert manifest['horizon_s']==manifest['warmup_seconds']+manifest['formal_seconds']
from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
from sglang.srt.server_args import ServerArgs
from sglang_simulator.dataset import GenericRequest,SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from hohlraum.experiments.backends.hisim.dispatch import install, install_horizon

# Spawned scheduler processes import this module before creating the engine.
rows=json.loads(Path(manifest['rows_file']).read_text())
contract=json.loads((CASE/'input_contract.json').read_text())
install(manifest,contract,rows)
install_horizon(manifest,rows,CASE)

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
    vocab=json.loads((Path(args['model_path'])/'config.json').read_text())['vocab_size']
    training=None
    if manifest['workload']=='playing' and json.loads((CASE/'formal_policy.json').read_text()).get('enabled'):
        training_path=Path(os.environ['HOHLRAUM_JOINT_MODELS'])/'training.json'
        training=json.loads(training_path.read_text())
    reqs=[]
    for serial,row in enumerate(rows):
        meta={k:row[k] for k in ['root','chain','index','owner','root_offer','first_delay','think_time']}
        meta["reference_fold"]=reference_model_fold(row,manifest["workload"],training)
        meta.update(trace_serial=serial,vocab_size=vocab)
        reqs.append(GenericRequest(token_ids=[7],input_length=row['raw_input_tokens'],output_length=1,custom_params={'agentx':meta}))
    started=time.monotonic();runner=Runner(ServerArgs(**args))
    try:runner.engine.loop.run_until_complete(run(runner,SimpleDataset(reqs=reqs)))
    finally:runner.shutdown()
    snap=json.loads((CASE/'horizon.json').read_text())
    end=manifest['horizon_s']
    start=end-manifest['formal_seconds']
    metrics=hisim_request_metrics(snap['requests'],start,end)
    exhausted=snap['progress']['budget_exhausted_at_s']
    physical=snap['progress']['physical_budget_exhausted_at_s']
    result=dict(manifest=manifest,**metrics,write_bytes=snap['progress']['write_bytes'],
        regular_write_bytes=snap['progress']['regular_write_bytes'],
        transition_write_bytes=snap['progress']['transition_write_bytes'],
        budget_exhausted_at_s=exhausted-start if exhausted is not None else None,
        physical_budget_exhausted_at_s=physical-start if physical is not None else None,
        wall_s=time.monotonic()-started,performance_validated=False)
    (CASE/'result.json').write_text(json.dumps(result,indent=2));print('PASS',json.dumps(result),flush=True)
if __name__=='__main__':main()
