"""Small public installation/replay check; synthetic inputs, no paper results."""
"""Dependency location check without importing CUDA/PyTorch packages."""
from pathlib import Path
from importlib.machinery import PathFinder
import sys
root = Path(__file__).resolve().parents[1]
vendor = root / 'third_party/sglang'
for name, paths in [('hohlraum', [root]), ('sglang', [vendor/'python']),
                    ('sglang_simulator', [vendor/'tools/sglang-simulator/src']),
                    ('benchmark', [vendor])]:
    spec = PathFinder.find_spec(name, [str(p) for p in paths])
    assert spec is not None, name
    locations = list(spec.submodule_search_locations or [])
    assert locations and all(Path(p).is_relative_to(root) for p in locations), (name, locations)
    print(name + ': ' + ', '.join(str(Path(p).relative_to(root)) for p in locations))
sys.path.insert(0, str(root))
from hohlraum.experiments.run import ROOT, VENDOR
assert ROOT == root and VENDOR == vendor
print('PASS: bundled dependency paths; no patch application required')


"""Synthetic checks only; fixtures are not paper data or trained models."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'hohlraum/experiments/backends/policy_replay'
subprocess.run(['make','-C',str(ROOT),'all','paper'],check=True)
with tempfile.TemporaryDirectory() as temp:
    d=Path(temp); models=d/'models'; models.mkdir()
    legacy="""HOHLRAUM_MODEL 2
window_seconds 600
growth_tokens 0
qg_remaining_seconds 60 60
length_table 1
0 0.5
session_coefficients 0 0 0 0
survival_single_qg 0 0 0
survival_multiple_qg 0 0 0
branch_coefficients 0 0 0 0
"""
    for i in range(6):
        (models/f'{i}.txt').write_text(('3 0 0.9 600 0.5 3600 0.01\n'*4)+'65 '+' '.join(['0.1']*65)+'\n')
        (models/f'legacy-interface-{i}.model').write_text(legacy)
    (models/'branch-prior.txt').write_text('3 0 30 3600\n'+('0.1 0.2\n'*6))
    (models/'length_curves.txt').write_text(('0.9 '+' '.join(['0.8']*8)+' 3 30 60 120\n')*24)
    env={k:v for k,v in os.environ.items() if not k.startswith('HOHLRAUM_')}
    env.update(HOHLRAUM_MODEL_COUNT='6',HOHLRAUM_JOINT_MODELS=str(models),HOHLRAUM_ONLINE_PRIOR='20',HOHLRAUM_BRANCH_PRIOR=str(models/'branch-prior.txt'),HOHLRAUM_LENGTH_CURVES=str(models/'length_curves.txt'))
    events=d/'events.tsv'; events.write_text('0\t0\t0\t1024\t0\n1\t1\t0\t1024\t1\n2\t2\t0\t1024\t2\n3\t0\t0\t1024\t0\n4\t0\t0\t0\t\n')
    for mode in ('full_nocopy','lru_nocopy','full_copy','lru_copy'):
        out=d/mode
        args=[sys.executable,str(ROOT/'run.py'),'--events',str(events),'--models',str(models),'--legacy-models',str(models),'--model-count','6','--blocks','32','--sessions','16','--read-blocks','1','--staging-blocks','1','--end-seconds','100','--threshold','0','--mode',mode,'--output',str(out)]
        subprocess.run(args,check=True)
        r=json.loads((out/'result.json').read_text())
        assert r['requests']==5 and r['refs']==4 and r['global_hits']==1 and r['ssd_reads']==1 and r['written_blocks']>0, r
        assert (out/'manifest.json').is_file()
        assert subprocess.run(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
print('PASS: four replay modes, output provenance and overwrite protection')


"""Tiny external-client example; no simulator, model, private inputs or output files."""
import json
import os
import tempfile
from types import SimpleNamespace as NS
from hohlraum.experiments.common.workload import TraceSource
from hohlraum.experiments.common.backpressure import RequestAdmission


def check_client_dependencies():
    def request(chain, index):
        meta = dict(root=0, chain=chain, index=index, root_offer=0,
                    first_delay=0, think_time=0)
        return NS(rid=f'{chain}:{index}', sampling_params=NS(
            custom_params={'simulation': {'agentx': meta}}))
    parent, child, continuation = request(0, 0), request(1, 0), request(0, 1)
    plan = {'0': {'deps': {'0:0': [], '1:0': [[0, 0, 3]],
                           '0:1': [[1, 0, 2]]},
                  'first_delays': {'0': 0, '1': 0}}}
    previous = os.environ.get('HISIM_AGENTX_DEPENDENCIES')
    try:
        with tempfile.NamedTemporaryFile(mode='w') as f:
            json.dump(plan, f); f.flush()
            os.environ['HISIM_AGENTX_DEPENDENCIES'] = f.name
            source = TraceSource([(0, i, r) for i, r in enumerate(
                [parent, child, continuation])], 1)
            assert source.pop_due(0) == [parent]
            source.finish(parent, 1)
            assert source.pop_due(3) == []
            assert source.pop_due(4) == [child]
            source.finish(child, 5)
            assert source.pop_due(7) == [continuation]
            source.finish(continuation, 8)
            assert len(source.completed) == 3 and not source.live
    finally:
        if previous is None:
            os.environ.pop('HISIM_AGENTX_DEPENDENCIES', None)
        else:
            os.environ['HISIM_AGENTX_DEPENDENCIES'] = previous
    gate = RequestAdmission(1)
    gate.offer(parent, 0); assert gate.release(0) == [parent]
    gate.offer(child, 1); assert gate.release(1) == []
    gate.finish(parent.rid); assert gate.release(4) == [child]
    stats = gate.export_stats([dict(rid=child.rid, created_time=4,
        gen_token_latencies=[2], queue_start=4, queue_end=4)])[0]
    assert gate.records[child.rid]['external_wait_s'] == 3
    assert stats['gen_token_latencies'][0] == 5
    print('Dependencies: parent -> child -> parent continuation')
    print('External wait: 3.0 s; engine TTFT: 2.0 s; end-to-end TTFT: 5.0 s')
    print('PASS')


check_client_dependencies()
