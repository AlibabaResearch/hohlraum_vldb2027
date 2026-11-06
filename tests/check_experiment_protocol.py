"""CPU protocol checks; no assertion of GPU performance equivalence."""
import sys
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hohlraum.experiments.common.workload import TraceSource,dump,restore
from hohlraum.experiments.common.backpressure import QueueDelayAdmission
from hohlraum.experiments.backends.sglang.feedback import NativeFeedback,atomic,read_events
from hohlraum.experiments.common.diagnostics import report


def source():
    q=[]
    for i in range(3):
        m=dict(root=1,chain=0,index=i,root_offer=0,first_delay=0,think_time=20,owner=0,reference_fold=0)
        q.append((0,i,NS(rid=str(i),sampling_params=NS(custom_params={'simulation':{'agentx':m},'hohlraum':m}))))
    return TraceSource(q,1)

s=source();req=s.pop_due(0)[0];s.finish(req,2)
ck=dump(s,10)
assert [1,0,0] not in ck['remaining_keys']
r=source();restore(r,json.loads(json.dumps(ck)))
assert r.pop_due(0)==[] and r.pop_due(11)==[]
next_req=r.pop_due(12)[0];assert next_req.rid=='1'
# An outstanding request restarts once at t=0 with its dependent still blocked.
ck2=dump(r,13);r2=source();restore(r2,json.loads(json.dumps(ck2)))
assert [x.rid for x in r2.pop_due(0)]==['1'] and r2.pop_due(1)==[]
r2.finish(r2.requests[1,0,1],2)
assert [x.rid for x in r2.pop_due(22)]==['2']
print('PASS: checkpoint excludes completed work, preserves remaining delay and outstanding dependencies')
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);transport=NativeFeedback(root);now=0.
    feedback=lambda:transport.refresh(now)
    gate=QueueDelayAdmission(feedback,10,8);transport.gate=gate
    atomic(root/'queue.json',dict(monotonic=time.monotonic(),waiting_rids=[]))
    for i in range(20):gate.offer(NS(rid=str(i)),0)
    assert len(gate.release(0))==8
    assert not gate.release(0)  # all slots reserved before receipt
    # Two received/running requests do not consume waiting quota.
    atomic(root/'queue.json',dict(monotonic=time.monotonic(),waiting_rids=[str(i) for i in range(2,8)]))
    with (root/'received.jsonl').open('a') as f:
        for i in range(8):f.write(json.dumps({'rid':str(i)})+'\n')
    assert len(gate.release(0))==2
    now=11
    assert not gate.release(now)  # oldest wait exceeds threshold
    atomic(root/'queue.json',dict(monotonic=time.monotonic(),waiting_rids=[]))
    now=12
    released=gate.release(now)
    assert len(released)==6 and all(gate.records[x.rid]['external_wait_s']==12 for x in released)
    # Partial messages must not be consumed before their terminating newline.
    path=root/'partial';path.write_text('{"rid":')
    with path.open() as f:
        assert read_events(f)==[]
        with path.open('a') as out:out.write('"x"}\n')
        assert read_events(f)==[{'rid':'x'}]
    transport.close()
    (root/'result.json').write_text(json.dumps({'throughput':12,'goodput':10,'ttft_p90_s':3}))
    diag=report(root)
    assert diag['metrics']['ssd_hit_fraction'] is None
    assert not diag['errors']
print('PASS: queue reservations, running-vs-waiting distinction, 10s gate, external wait and missing-counter diagnostics')

# Validate/reuse/rebuild is shared by native and HiSim entries.
from hohlraum.experiments.preprocessing.run import ensure_inputs, sha
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);source=root/'source.json';target=root/'cache/rows.json'
    row=dict(root=1,chain=0,index=0,owner=1,reference_fold=0,admission_fold=0,
             root_offer=0,first_delay=0,think_time=0,raw_input_tokens=1024,blocks=[17])
    source.write_text(json.dumps([row]))
    selection=dict(rows=str(target),seed=7,rebuild=dict(kind='prepared-source',rows=str(source),rows_sha256=sha(source)))
    _,result=ensure_inputs('playing',selection,7)
    assert result['action']=='rebuilt' and target.read_bytes()==source.read_bytes()
    _,result=ensure_inputs('playing',selection,7)
    assert result['action']=='reused' and result['validation']['expected_checksum_checked']
    target.write_text('corrupted')
    _,result=ensure_inputs('playing',selection,7)
    assert result['action']=='rebuilt' and target.read_bytes()==source.read_bytes()
    target.write_text('keep-on-failed-repair');source.write_text('corrupt-source')
    try:ensure_inputs('playing',selection,7)
    except ValueError:pass
    else:raise AssertionError('corrupt reconstruction source accepted')
    assert target.read_text()=='keep-on-failed-repair'
print('PASS: automatic data reuse, missing/corrupt-cache repair, trusted source checks and failed-repair preservation')


# All providers must reach the correct model and online-statistics group.
from hohlraum.experiments.common.workload import reference_model_fold
for provider, index, group in [('claude',0,'claude_13agents'),('gpt',1,'gpt_5agents'),('kimi',2,'kimi_5agents')]:
    for fold in [0,1]:
        trajectory=group+'/agent01'
        row=dict(source_trajectory=trajectory,reference_fold=fold)
        training=[dict(provider=provider,fold=fold,train=[group+'/agent02'],test=[trajectory])]
        model=reference_model_fold(row,'playing',training)
        assert model==index*2+fold and model//2==index
        assert row['reference_fold']==fold
        training[0]['train'].append(trajectory)
        try:reference_model_fold(row,'playing',training)
        except ValueError:pass
        else:raise AssertionError('training/test overlap accepted')
for row in [dict(reference_fold=0),dict(source_trajectory='unknown/a',reference_fold=0),dict(source_trajectory='gpt_5agents/a',reference_fold=2)]:
    try:reference_model_fold(row,'playing')
    except ValueError:pass
    else:raise AssertionError('ambiguous model metadata accepted')
for fold in range(4):assert reference_model_fold(dict(reference_fold=fold),'agentx')==fold
print('PASS: provider/local-fold mapping, held-out membership, no in-place mutation, AgentX unchanged')
