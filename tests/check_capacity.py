"""Tiered controller behavior and native adapter invariants; no GPU required."""
from pathlib import Path
import sys, ast, json
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hohlraum.core.capacity_controller import CapacityController

CONFIG=dict(period_s=300,history_windows=8,observation_windows=2,pressure_windows=2,
 additional_wait_windows=2,failure_cooldown_windows=4,min_read_fraction=.25,
 tiers_gb=[5,15,30],min_expansion_windows=4,
 expansion_thresholds=[.01,.05,.2],contraction_thresholds=[.2,.5])

def host():
    # One fake token is one GB: tier sizes remain physical byte quantities.
    h=NS(cfg={'capacity_control':dict(CONFIG)},cache=NS(page_size=1,tp_world_size=1),total_tokens=300,
         staging_limit=100,read_limit=200,stage={},admission_threshold=.7,
         stats_path=None,kv_bytes_per_token=10**9,exhausted=False,t=0.)
    h.read_allocated=lambda:0
    h.waiting_staging_tokens=lambda:0
    h.now=lambda:h.t
    h._budget_exhausted=lambda:h.exhausted
    def resize(target):h.staging_limit=target;h.read_limit=h.total_tokens-target
    h.resize_staging=resize
    h.sli=NS(profiles=lambda keys,t:{k:(.8,20,0,0,0) for k in keys},pending={})
    h.staging_policy=NS(requires_prefix_write=lambda e:False)
    return h

def tick(c,t,wait=None,fail=False):
    c.host.t=t
    if wait is not None:c.interval_waits.append(wait)
    if fail:c.failed(str(t),1,'host_capacity')
    c.tick(t)

# Confirmed exact tier boundaries and zero tier without clearing samples.
for fraction,expected in [(0,100),(.009,100),(.01,105),(.049,105),(.05,115),(.199,115),(.2,130),(1.,130)]:
    h=host();c=CapacityController(h)
    c.accepted.extend([(0,500,100*fraction),(0,float('inf'),100*(1-fraction))])
    for t in (300,600,900):
        tick(c,t);assert h.staging_limit==100
    tick(c,1200)
    assert h.staging_limit==expected,(fraction,h.staging_limit)
    assert bool(c.history)==(expected==100)
    assert c.last_decision['opportunity_bytes']==100*fraction
print('PASS: byte-weighted discrete tiers, exact boundaries, zero tier retains history')

# Reset all distributions/episodes; old asynchronous ACKs cannot contaminate epoch.
h=host();c=CapacityController(h);e=NS(id=1,hash_value=['x'],operation=None);h.stage[1]=e
sample=c.write_sample(e);c.accepted.append((0,10,1));c.slack.append((0,10,1))
for t in (300,600,900):
    tick(c,t);assert h.staging_limit==100
tick(c,1200);assert h.staging_limit==130
assert not c.accepted and not c.slack and not c.history
c.wrote(sample);assert not c.accepted
c.wrote(c.write_sample(e));assert len(c.accepted)==1
assert c.observe_until==2400
# New TTD opportunities do not cause repeated expansion during observation.
for t in (1500,1800,2100):
    tick(c,t);assert h.staging_limit==130
tick(c,2400);assert h.staging_limit==160
# Short rejection episode is restarted on capacity changes.
c.observe(e,(.1,),2450);h.t=2500;h.sli.profiles=lambda keys,t:{k:(.1,) for k in keys}
c.discarded(e);assert c.slack[-1][1]==50
print('PASS: reset on resize, delayed ACK isolation, observation, new rejection episodes')

# Pending read time does not trigger shrink in this version.
h=host();c=CapacityController(h);c.slack.append((0,1000,10))
tick(c,300,wait=100);tick(c,600,wait=100)
assert h.staging_limit==100
# One capacity spike doesn't shrink; a clean window breaks persistence.
tick(c,900,fail=True);assert h.staging_limit==100
tick(c,1200);assert c.capacity_failure_streak==0
tick(c,1500,fail=True);assert h.staging_limit==100
tick(c,1800,fail=True);assert h.staging_limit==70
assert not c.history and c.expand_after==3000
print('PASS: read blocking disabled, sustained capacity pressure, slack selects shrink tier')

# Continued pressure after full observation escalates; shrink restarts expansion clock.
h=host();h.staging_limit=200;c=CapacityController(h)
tick(c,300,fail=True);tick(c,600,fail=True);assert h.staging_limit==195
assert c.expand_after==1800
tick(c,900,fail=True);assert h.staging_limit==195
tick(c,1200,fail=True);assert h.staging_limit==180
assert c.last_decision['tier_gb']==15 and c.expand_after==2400
tick(c,1500,fail=True);tick(c,1800,fail=True);assert h.staging_limit==150
assert c.last_decision['tier_gb']==30
# Fewer failed requests -> improved pressure; do not escalate solely on persistence.
h=host();h.staging_limit=200;c=CapacityController(h)
for t in (300,600):
    c.failed('extra',1,'host_capacity');tick(c,t,fail=True)
for t in (900,1200):tick(c,t,fail=True)
assert c.last_decision['tier_gb']==5
print('PASS: observation, tier escalation on sustained unimproved pressure, improvement avoids escalation')

# Prefetch pressure is primary and requires two windows. Generic storage errors do not qualify.
h=host();c=CapacityController(h)
c.failed('unknown',5,'storage_transfer');tick(c,300);assert h.staging_limit==100
tick(c,600,fail=True);assert h.staging_limit==100
tick(c,900,fail=True);assert h.staging_limit==95 and c.expand_after==2100
# Pressure-free windows stop escalation, but cannot bypass contraction cooldown.
c.accepted.append((901,10,10));tick(c,1200);tick(c,1500);tick(c,1800)
assert h.staging_limit==95 and c.last_shrink is None
tick(c,2100);assert h.staging_limit==125
# A skipped window cannot be evidence of continuous pressure.
h=host();c=CapacityController(h);tick(c,300,fail=True);tick(c,900,fail=True)
assert h.staging_limit==100 and c.capacity_failure_streak==0
print('PASS: capacity-only failures, sustained failure, four-window expansion cooldown, skipped windows')

# TP uses aggregate GB, matching the existing staging_GB configuration.
h=host();h.cache.tp_world_size=2;h.kv_bytes_per_token=5*10**8
c=CapacityController(h);assert c.tiers==(5,15,30)
print('PASS: TP aggregate GB units')

# 8 same-capacity windows; bounds; failed resize must not clear history.
h=host();c=CapacityController(h)
for t in range(300,3001,300):tick(c,t)
assert len(c.history)==8
h.staging_limit=220;c=CapacityController(h);c.accepted.append((h.t,0,1))
for _ in range(4):tick(c,h.t+300)
assert h.staging_limit==225
h=host();c=CapacityController(h);h.resize_staging=lambda target:False
c.accepted.append((0,0,1))
for t in (300,600,900,1200):tick(c,t)
assert c.accepted and len(c.history)==4 and not c.last_decision['history_reset']
h.exhausted=True;h.resize_staging=lambda target:setattr(h,'staging_limit',target)
tick(c,1500);assert h.staging_limit==0 and not c.history
h=host();h.cfg['capacity_control']['window_s']=600
try:CapacityController(h)
except ValueError:pass
else:raise AssertionError('legacy config accepted')
print('PASS: bounded history and capacity, unsuccessful resize, exhaustion, obsolete config rejection')

# Four actual complete windows, not elapsed time alone, are required.
h=host();c=CapacityController(h);c.accepted.append((0,10,1))
for t in (300,900,1200,1500):tick(c,t)
assert h.staging_limit==100 and c.last_decision['complete_window_count']==3
tick(c,1800);assert h.staging_limit==130
print('PASS: skipped windows do not satisfy four-window expansion minimum')

integration_host=host
def host():
    h=integration_host();h.total_tokens=1000;h.staging_limit=300;h.read_limit=700
    return h

# Execute the actual storage resize method, recording quotas at eviction.
p=Path(__file__).resolve().parents[1]/'hohlraum/core/staging_storage.py'
tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeHost')
resize=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='resize_staging')
env={'return_waiting_to_read':lambda h:setattr(h,'returned',True)}
exec(compile(ast.Module(body=[resize],type_ignores=[]),str(p),'exec'),env)
h=host();seen=[];h.cache.evict_host=lambda n:seen.append((h.read_limit,h.staging_limit,n))
assert env['resize_staging'](h,400) is True
assert seen[-1]==(600,400,0)
assert env['resize_staging'](h,100) is True
assert seen[-1]==(900,100,0)
# Ordinary contraction to zero evicts; only exhaustion keeps old return behavior.
env['resize_staging'](h,0);assert seen[-1]==(1000,0,0) and not hasattr(h,'returned')
h.staging_limit=100;h.read_limit=900;h.exhausted=True;h.stage={1:NS(read_pins=1)}
assert env['resize_staging'](h,0) is False and h.staging_limit==100
h.stage.clear();env['resize_staging'](h,0);assert h.returned
print('PASS: both quotas visible before eviction; pin-safe exhaustion and normal zero contraction')

feedback=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_attach_capacity_feedback')
exec(compile(ast.Module(body=[feedback],type_ignores=[]),str(p),'exec'),env)
h=host();h.capacity_controller=CapacityController(h);calls=[]
for name in ['_record_storage_prefetch_hit','_resolve_storage_prefetch_tokens','_finish_storage_prefetch','discard_storage_prefetch_accounting']:
    setattr(h.cache,name,lambda *a,_name=name,**k:calls.append((_name,a,k)))
r=NS(rid='r');op=object();h.cache.ongoing_prefetch={r.rid:None}
# Handles are hashable in the native runtime.
class Handle:
    rid='r'
r=Handle();h.cache.ongoing_prefetch={r:NS(operation=op)}
h.storage=NS(leases={id(op):NS(entries={0:object()},keys=['staged','ssd'])})
h.cache.check_prefetch_progress=lambda req:False
env['_attach_capacity_feedback'](h)
h.cache._record_storage_prefetch_hit(r,num_tokens=100)
h.cache._resolve_storage_prefetch_tokens(r,num_tokens=30,reason='host_capacity')
h.cache._finish_storage_prefetch(r,fulfilled_tokens=50,reason='cache_admission_shortfall')
assert h.capacity_controller.events['host_capacity']==1
assert h.capacity_controller.events['cache_admission_shortfall']==1
assert len(calls)==3
assert h.cache.check_prefetch_progress(r) is False
assert 'r' in h.capacity_controller.waiting
h.t=3;h.cache.discard_storage_prefetch_accounting(r)
assert h.capacity_controller.interval_waits==[3] and not h.capacity_controller.waiting
print('PASS: native callbacks preserved; keyword arguments, mixed-read waiting and cancellation accounting')


# Read-to-staging changes usable quota, not physical freed-page accounting.
for method in ['_attach_read_reclaim','_attach_writeback_feedback']:
 fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),env)
env['ComponentType']=NS(FULL='full')
h=host();h.available=0;h.read_available=lambda:h.available
freed=[0]
def reclaim(n,ct):h.available+=1024;return freed[0]
h.cache.evict_host=reclaim
env['_attach_read_reclaim'](h)
assert h.cache.evict_host(1024)==1024 and freed[0]==0
h.available=0
# A logical quota change with no actual allocatable space must not claim progress.
h.cache.evict_host=lambda n,ct:0
env['_attach_read_reclaim'](h);assert h.cache.evict_host(1024)==0
# Only failed writes with an unresolved capacity shortfall emit feedback.
h.capacity_controller=CapacityController(h);h.cache.cache_controller=NS(write_policy='write_back')
h.cache._execute_kv_backup=lambda *a,**kw:None
env['_attach_writeback_feedback'](h)
assert h.cache._execute_kv_backup(9,[0]*20) is None
assert h.capacity_controller.writeback_failures=={9}
h.available=100;h.cache._execute_kv_backup(10,[0]*20)
assert h.capacity_controller.writeback_failures=={9}
print('PASS: effective read reclaim, no physical-free inflation, unresolved backup-only feedback')
