import ast,heapq
from pathlib import Path
from types import SimpleNamespace as NS
from collections import defaultdict,Counter
root=Path(__file__).resolve().parents[1]
p=root/'hohlraum/core/staging_storage.py';tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeHost')
env=dict(heapq=heapq,ComponentType=NS(FULL=0),defaultdict=defaultdict)
for name in ['_hbm_read_leaf','_drive_read_leaves','_take_read_leaf']:
 f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[f],type_ignores=[]),str(p),'exec'),env)
class N:
 def __init__(self,i,hbm=True):
  self.id=i;self.children={};self.evicted=not hbm;self.last_access_time=i;self.parent=None;self.hash_value=[str(i)];self.key='k';self.write_through_pending_id=None;self.load_back_pending_id=None
  self.component_data=[NS(value=object() if hbm else None,host_value=NS(numel=lambda:1),lock_ref=0,host_lock_ref=0)]
 def get_prefix_hash_values(self,p):return []
 def __lt__(self,o):return self.id<o.id
core=NS(is_write_back=False,evictable_host_leaves=set(),full_host_duplicates={})
core._can_reclaim_full_host_duplicate=lambda n:not n.evicted and n.component_data[0].host_value is not None and n.write_through_pending_id is None and n.load_back_pending_id is None and n.component_data[0].host_lock_ref==0
h=NS(cfg={'stage_hbm_leaves':True},cache=NS(tree_core=core,page_size=1),_read_ownership=Counter(),stage={},_next_stage_id=-1)
h._budget_exhausted=lambda:False
h._hbm_read_leaf=lambda n:env['_hbm_read_leaf'](h,n)
a=N(1);b=N(2);a.children['b']=b;b.parent=a
assert not h._hbm_read_leaf(a) and h._hbm_read_leaf(b)
for attr in ['lock_ref','host_lock_ref']:
 setattr(b.component_data[0],attr,1);assert not h._hbm_read_leaf(b);setattr(b.component_data[0],attr,0)
b.write_through_pending_id=1;assert not h._hbm_read_leaf(b);b.write_through_pending_id=None
b.load_back_pending_id=1;assert not h._hbm_read_leaf(b);b.load_back_pending_id=None
h.cfg['stage_hbm_leaves']=False;assert not h._hbm_read_leaf(b);h.cfg['stage_hbm_leaves']=True
# Transfer same payload, preserve device data and tree edge; never free device.
payload=b.component_data[0].host_value;device=b.component_data[0].value
core.node_by_id=lambda i:b
core._update_evictable_leaf_sets=lambda n:None
core.full_host_duplicates[b.id]=b
core._release_full_host_duplicate=lambda n,t,d,f:(f[0].append(n.component_data[0].host_value),t.__setitem__(0,1),setattr(n.component_data[0],'host_value',None))
env['StagedPage']=lambda *args,**kw:NS(payload=args[3])
h._tokens=lambda n:1
entry=env['_take_read_leaf'](h,b)
assert entry.payload is payload and b.component_data[0].value is device and a.children['b'] is b
assert b.component_data[0].host_value is None and b.id not in core.full_host_duplicates
assert not h._hbm_read_leaf(a)
# Mixed candidates use original ordering, not HBM-first priority.
b=N(2);c=N(3,False);core.full_host_duplicates={b.id:b};core.evictable_host_leaves={c}
h.component=NS(_ensure_eviction_strategy=lambda:None,session_ref_eviction_strategy=lambda n:n.last_access_time)
seen=[]
env['_drive_read_leaves'](h,2,defaultdict(int),{}, {},leaf_handler=lambda n:(seen.append(n.id) or True),should_stop=lambda:len(seen)==2)
assert seen==[2,3]
print('PASS leaf/lock/DMA constraints, ownership transfer, HBM and parent preservation, shared LRU order')
# Optional DRAM leaves: retain the HBM subtree and expose parents bottom-up.
h.cfg['stage_hbm_dram_leaves']=True
a=N(1);b=N(2);c=N(3);a.children={'b':b,'c':c};b.parent=a;c.parent=a
assert not h._hbm_read_leaf(a)
b.component_data[0].host_value=None
assert not h._hbm_read_leaf(a)  # The other child still owns DRAM.
c.component_data[0].host_value=None
assert h._hbm_read_leaf(a)
c.component_data[0].host_value=NS(numel=lambda:1)
assert not h._hbm_read_leaf(a)  # A new host child removes parent eligibility.
# Same eviction pass: after both child payloads move, parent becomes eligible.
b.component_data[0].host_value=NS(numel=lambda:1)
core.full_host_duplicates={n.id:n for n in (a,b,c)};core.evictable_host_leaves=set();seen=[]
def transfer(n):
 seen.append(n.id);n.component_data[0].host_value=None
 return True
env['_drive_read_leaves'](h,3,defaultdict(int),{}, {},leaf_handler=transfer,should_stop=lambda:len(seen)==3)
assert seen==[2,3,1] and a.children=={'b':b,'c':c}
assert all(n.component_data[0].value is not None for n in (a,b,c))
h.cfg['stage_hbm_dram_leaves']=False
assert not h._hbm_read_leaf(a)
print('PASS DRAM-leaf sibling exclusion, live recheck, bottom-up parent discovery, HBM tree retained')

# Budget exhaustion disables the expanded candidates, independent of staging quota.
n=N(9);h.cfg['stage_hbm_leaves']=True
assert h._hbm_read_leaf(n)
h._budget_exhausted=lambda:True
assert not h._hbm_read_leaf(n)
fallback=[]
h._original_drive_host_eviction=lambda *a,**k:fallback.append(True)
env['_drive_read_leaves'](h,1,{}, {}, {},leaf_handler=lambda n:True,should_stop=lambda:False)
assert fallback==[True]
print('PASS budget exhaustion restores native candidate path')
