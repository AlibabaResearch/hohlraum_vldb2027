"""Exercise the adapter against extracted native tree eviction methods."""
import ast, sys, types
from enum import Flag, auto
from collections import defaultdict, Counter
from pathlib import Path
from types import SimpleNamespace as NS

root=Path(__file__).resolve().parents[1]
class Layer(Flag):
    DEVICE=auto(); HOST=auto(); ALL=DEVICE|HOST
class Result:
    def __init__(self):
        self.tracker=defaultdict(int);self.device_frees=defaultdict(list);self.host_frees=defaultdict(list)
        self.backup_kv=None;self.unbacked_tokens=0
env=dict(EvictLayer=Layer,BASE_COMPONENT_TYPE=0,ComponentType=NS(FULL=0),StorageMedium=NS(GPU=0,CPU=1),EvictDeviceLeafResult=Result,defaultdict=defaultdict)
def extract(path,cls,names):
    c=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls)
    out={}
    for n in c.body:
        if isinstance(n,ast.FunctionDef) and n.name in names:
            static=any(isinstance(d,ast.Name) and d.id=='staticmethod' for d in n.decorator_list)
            n.decorator_list=[];n.returns=None
            for a in n.args.args+n.args.kwonlyargs:a.annotation=None
            exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),'exec'),env)
            out[n.name]=staticmethod(env[n.name]) if static else env[n.name]
    assert len(out)==len(names),(names,set(out))
    return out
native=root/'third_party/sglang/python/sglang/srt/mem_cache/unified_cache'
names={'evict_device_leaf','_delete_unbacked_device_leaf','_release_all_component_layers','_remove_leaf_from_parent','_unregister_node','_update_evictable_leaf_sets','_is_device_leaf','_is_host_leaf','_evict_component_and_detach_lru','_cascade_evict','_is_cascade_evict_leaf','_should_cascade_evict_component','_iteratively_delete_tombstone_leaf','_iteratively_delete_tombstone_ancestors','_evict_host_leaf','_begin_tracking_unbacked_tokens','_finish_tracking_unbacked_tokens'}
Core=type('Core',(),extract(native/'unified_tree_core.py','UnifiedTreeCore',names))
Full=type('Full',(),extract(native/'components/full.py','FullComponent',{'evict_component'}))
adapter=extract(root/'hohlraum/core/staging_storage.py','NativeHost',{'_attach_hbm_eviction'})['_attach_hbm_eviction']
for name,attrs in [('sglang.srt.disaggregation.kv_events',dict(StorageMedium=env['StorageMedium'])),('sglang.srt.mem_cache.unified_cache.components',dict(EvictLayer=Layer)),('sglang.srt.mem_cache.unified_cache.unified_tree_core_interface',dict(EvictDeviceLeafResult=Result))]:
    m=types.ModuleType(name);m.__dict__.update(attrs);sys.modules[name]=m
class Node:
    def __init__(self,i,d,h,parent=None):
        self.id=i;self.parent=parent;self.children={};self.key=NS(child_key=lambda _:i);self.hash_value=[str(i)]
        self.component_data=[NS(value=d,host_value=h,lock_ref=0,host_lock_ref=0,session_ref=0)]
    @property
    def evicted(self):return self.parent is not None and self.component_data[0].value is None
    @property
    def backuped(self):return self.component_data[0].host_value is not None
def setup(enabled):
    c=Core();r=Node(0,None,None);a=Node(1,[1],None,r);b=Node(2,None,[2],a);r.children[1]=a;a.children[2]=b
    c.root_node=r;c._node_arena={0:r,1:a,2:b};c.node_by_id=lambda i:c._node_arena[i];c.is_write_back=False
    full=Full();full.component_type=0;full.tree_core=c;full.eviction_priority=lambda *a,**k:0
    full.discard_deleted_session_leaf=lambda n:None
    full.node_has_component_data=lambda n,target=Layer.DEVICE:n.component_data[0].value is not None if target==Layer.DEVICE else n.backuped
    c.components=[full];c.components_by_type={0:full};c.page_size=1;c.full_host_duplicates={};c.evictable_device_leaves={a};c.evictable_host_leaves={b}
    c._is_tracking_unbacked_tokens=False;c._tracked_unbacked_tokens=0;c.component_evictable_size_=defaultdict(int,{0:1})
    c.kv_events=NS(record_remove=lambda *a,**k:None);c.lru_lists={0:NS(in_list=lambda n:False)};c.host_lru_lists=c.lru_lists
    h=NS(cfg={'stage_hbm_leaves':enabled},cache=NS(tree_core=c,page_size=1),stage={'1':object()},_read_ownership=Counter())
    adapter(h)
    return c,h,r,a,b
# Native assumption fails for a WT unbacked parent with host-only children.
c,h,r,a,b=setup(False);c.evict_device_leaf(a.id,False);assert a.id not in c._node_arena and b.parent is a
c._update_evictable_leaf_sets(b.parent);assert a in c.evictable_device_leaves
print('REPRODUCED native deletion + child parent refresh resurrects a deleted candidate')
# Adapter retains the structural parent and frees HBM exactly once.
c,h,r,a,b=setup(True);result=c.evict_device_leaf(a.id,False)
assert result.tracker[0]==1 and result.device_frees[0]==[[1]] and not result.host_frees
assert result.unbacked_tokens==1 and a.evicted and not a.backuped
assert c._node_arena[a.id] is a and r.children[1] is a and a.children[2] is b
assert h.stage['1'] is not None and h._read_ownership['hbm_evicted_still_staged_tokens']==1
c._update_evictable_leaf_sets(b.parent);assert a not in c.evictable_device_leaves
# Once its last host child leaves, native tombstone cleanup deletes both safely.
c._evict_host_leaf(b,defaultdict(int),defaultdict(list),defaultdict(list))
assert set(c._node_arena)=={0} and not c.evictable_device_leaves and not c.evictable_host_leaves
print('PASS native HBM free, staging retained, parent/child links, later native cleanup')
# Ordinary unbacked whole-tree leaves still use native deletion.
c,h,r,a,b=setup(True);a.children={};c._node_arena.pop(b.id);c.evictable_host_leaves.clear()
result=c.evict_device_leaf(a.id,False);assert a.id not in c._node_arena and result.tracker[0]==1
print('PASS ordinary leaf path unchanged; feature off preserves native behavior')
