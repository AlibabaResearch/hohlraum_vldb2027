"""CPU-only observer checks: no changes to scheduling or capacity decisions."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys, json, ast, threading
from functools import wraps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hohlraum.core.capacity_controller import CapacityController
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/'hohlraum/experiments/protocol.json').read_text())['capacity_control']
h=NS(cfg={'capacity_control':cfg},cache=NS(page_size=1,tp_world_size=1),total_tokens=300,
 staging_limit=30,read_limit=270,stage={},admission_threshold=.7,stats_path=None,
 kv_bytes_per_token=10**9,t=0.)
h.now=lambda:h.t;h._budget_exhausted=lambda:False
h.read_allocated=lambda:0;h.waiting_staging_tokens=lambda:0
h.sli=NS(profiles=lambda keys,t:{k:(.8,20) for k in keys},pending={})
c=CapacityController(h)
c.scheduler_begin(0);c.read_progress('a',True,0,True);c.read_progress('b',True,0,True)
c.scheduler_end(False,0);c.scheduler_begin(3)
assert c._idle_total==3 # Two readers, not six seconds.
c.read_progress('a',True,3,True);c.scheduler_end(True,3)
c.scheduler_begin(10);assert c._idle_total==3
# A compute result still in flight suppresses idle, despite no next batch.
r=object();c.batch_submitted(r);c.read_progress('a',True,10,True);c.scheduler_end(False,10)
c.scheduler_begin(20);assert c._idle_total==3
c.batch_completed(r);c.read_progress('a',True,20,True);c.scheduler_end(False,20)
c.end_read('a',24);assert c._idle_total==7
c.scheduler_begin(30);c.scheduler_end(False,30)
c.scheduler_begin(40);assert c._idle_total==7 # No checked SSD waiter; no demand idle.
# New epoch clears window accounting, preserves live in-flight bookkeeping.
c.read_progress('b',True,40,True);c.scheduler_end(False,40)
h.t=42;c._new_epoch(42,'expand');assert c._idle_total==9 and c._idle_seconds==0
c.end_read('b',45);assert c._idle_seconds==3 and c._idle_total==12
h.t=342;c.tick(342)
assert c.last_decision['reason'] in ('observation','collecting_history')
assert h.staging_limit==30 # Observer never drives control.
print('PASS: union time, compute overlap, completion/cancellation, no-demand idle, resize reset')

# Exercise actual adapter wrapper with fake native methods and result identity.
source=root/'hohlraum/core/staging_storage.py'
tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_install_scheduler_hooks')
import types
module=types.ModuleType('sglang.srt.managers.scheduler')
class Scheduler:
 def process_input_requests(self,requests):return requests
 def get_next_batch_to_run(self):
  c.read_progress('r',True,h.now(),True)
  return NS(batch_to_run=self.batch)
 def process_batch_result(self,batch,result):return result
 def run_batch(self,batch):return object()
 def _release_aborted_request(self,req):return req
module.Scheduler=Scheduler
saved=sys.modules.get(module.__name__);sys.modules[module.__name__]=module
try:
 env={'wraps':wraps};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
 env['_install_scheduler_hooks']()
 c.reset();h.capacity_controller=c;h.lock=threading.RLock()
 h.maintain=lambda:None;h.on_batch_complete=lambda reqs,now:None
 s=Scheduler();s.tree_cache=NS(hohlraum_host=h);s.batch=NS(reqs=[])
 assert s.get_next_batch_to_run().batch_to_run is s.batch
 result=s.run_batch(s.batch);assert id(result) in c._inflight_results
 s.batch=None;h.t+=5;s.get_next_batch_to_run();assert not c._idle_active
 # Keyword invocation must preserve native result signature; batch may be copied.
 assert s.process_batch_result(NS(reqs=[]),result=result) is result
 assert not c._inflight_results
 s.get_next_batch_to_run();assert c._idle_active
 h.t+=4;s.get_next_batch_to_run();assert c._idle_total==4
finally:
 if saved is None:sys.modules.pop(module.__name__,None)
 else:sys.modules[module.__name__]=saved
print('PASS: actual adapter, keyword signature, copied batches, native return values, result tracking')
