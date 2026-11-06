from pathlib import Path
from types import SimpleNamespace as NS
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hohlraum.core.capacity_controller import CapacityController
cfg=json.loads((Path(__file__).resolve().parents[1]/'hohlraum/experiments/protocol.json').read_text())['capacity_control']
def make():
 h=NS(cfg={'capacity_control':cfg},cache=NS(page_size=1,tp_world_size=1),total_tokens=300,staging_limit=100,read_limit=200,stage={},kv_bytes_per_token=10**9,stats_path=None,t=0)
 h.now=lambda:h.t;h._budget_exhausted=lambda:False;h.read_allocated=lambda:0;h.waiting_staging_tokens=lambda:0
 h.resize_staging=lambda target:setattr(h,'staging_limit',target)
 return h,CapacityController(h)
def tick(h,c,t,fail=False):
 h.t=t
 if fail:c.failed(str(t),1,'host_capacity')
 c.tick(t)
for ratio,step in [(0,0),(.000001,1),(.009,1),(.01,10),(.019,10),(.02,10),(.049,10),(.05,15),(.099,15),(.1,30),(1.,30)]:
 h,c=make();c.accepted.extend([(0,100,ratio*100),(0,float('inf'),(1-ratio)*100)])
 for t in (300,600,900):tick(h,c,t);assert h.staging_limit==100
 tick(h,c,1200);assert h.staging_limit==100+step,(ratio,h.staging_limit)
 h,c=make();c.slack.extend([(0,1000,ratio*100),(0,0,(1-ratio)*100)])
 tick(h,c,300,True);assert h.staging_limit==100
 tick(h,c,600,True);assert h.staging_limit==100-step,(ratio,h.staging_limit)
print('PASS: both directions, four tiers, all boundaries, four-window expansion/two-window pressure')
h,c=make()
for t in (300,600):tick(h,c,t,True)
assert h.staging_limit==100 and len(c.history)==2
for t in (900,1200):tick(h,c,t,True)
assert h.staging_limit==99
for t in (1500,1800):tick(h,c,t,True)
assert h.staging_limit==89
for t in (2100,2400):tick(h,c,t,True)
assert h.staging_limit==74
for t in (2700,3000):tick(h,c,t,True)
assert h.staging_limit==44
print('PASS: sustained unimproved pressure escalates zero ->1 ->10 ->15 ->30 GB')
