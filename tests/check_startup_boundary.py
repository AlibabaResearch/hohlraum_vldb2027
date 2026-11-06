import sys,json
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hohlraum.experiments.common.workload import TraceSource,dump,restore,bootstrap_at_first_ready

def source(offer=0,delay=0):
 q=[]
 for i in range(3):
  m=dict(root=1,chain=0,index=i,root_offer=offer,first_delay=delay,think_time=20,owner=0,reference_fold=0)
  q.append((0,i,NS(rid=str(i),sampling_params=NS(custom_params={'simulation':{'agentx':m},'hohlraum':m}))))
 plans={'1':{'deps':{'0:0':[],'0:1':[[0,0,20]],'0:2':[[0,1,20]]},'first_delays':{'0':delay}}}
 return TraceSource(q,1,plans)
# Cold start: original arrival 43.32 with an additional first-request delay.
s=source(43.32,5);ready,meta=bootstrap_at_first_ready(s)
assert abs(meta['origin_shift_s']-48.32)<1e-8 and len(ready)==1
assert s.arrivals['0']==0 and s.pop_due(0)==[]
s.finish(ready[0],2);assert s.pop_due(21)==[] and s.pop_due(22)[0].rid=='1'
# Restored idle checkpoint retains all dependency lags after a single origin shift.
s=source();req=s.pop_due(0)[0];s.finish(req,2);ck=dump(s,10)
r=source();restore(r,json.loads(json.dumps(ck)));ready,meta=bootstrap_at_first_ready(r)
assert meta['origin_shift_s']==12 and ready[0].rid=='1' and r.arrivals['1']==0
assert r.pop_due(0)==[];r.finish(ready[0],3);assert r.pop_due(22)==[] and r.pop_due(23)[0].rid=='2'
# Outstanding request: no origin shift and no double release.
r=source();restore(r,ck);req=r.pop_due(12)[0];ck2=dump(r,13)
t=source();restore(t,ck2);ready,meta=bootstrap_at_first_ready(t)
assert meta['origin_shift_s']==0 and [x.rid for x in ready]==['1'] and t.pop_due(0)==[]
# Empty client events before the first request do not initialize the model clock.
s=source(1500,0);ready,meta=bootstrap_at_first_ready(s);assert meta['origin_shift_s']==1500
print('PASS cold nonzero start, idle checkpoint, outstanding checkpoint, dependency delays, single delivery')
