"""Staging + SSD storage: page ownership, data movement and runtime adapters.

StagingPolicy owns decisions; native runtimes retain scheduling and I/O timing.
"""
from __future__ import annotations
from contextlib import nullcontext, contextmanager
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
import json
import heapq
from pathlib import Path
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from hohlraum.core.sli import SLI
from hohlraum.core.staging_policy import StagingPolicy
from collections import Counter
from dataclasses import dataclass
import os
import threading
import time
from functools import wraps, partial


class HostPayloadIndex:
    def __init__(self, core, component_type):
        self.core, self.ct = core, component_type
        self.nodes = {}
        self.tokens = {}
        self.host_keys = Counter()
        self.key_nodes = {}
        self._keys = {}
        self.total_tokens = 0
        self.on_payload_change = None
        self.on_key_membership_change = None
        for node in tuple(core._node_arena.values()):
            self.sync(node)

    def forget(self, node):
        nid = node.id
        old = self.tokens.pop(nid, 0)
        self.total_tokens -= old
        self.nodes.pop(nid, None)
        keys = self._keys.pop(nid, ())
        for key in keys:
            self.key_nodes[key].discard(nid)
            if not self.key_nodes[key]:del self.key_nodes[key]
            self.host_keys[key] -= 1
            if not self.host_keys[key]:
                del self.host_keys[key]
                if self.on_key_membership_change is not None:
                    self.on_key_membership_change(key)
        if self.on_payload_change is not None:
            self.on_payload_change(nid)

    def sync(self, node):
        if node is None:
            return
        nid = node.id
        cd = node.component_data[self.ct]
        value = cd.host_value
        tokens = 0 if value is None else value.numel()
        keys = tuple(node.hash_value or ()) if tokens else ()
        old_tokens = self.tokens.get(nid, 0)
        old_keys = self._keys.get(nid, ())
        if tokens:
            self.nodes[nid] = node
            self.tokens[nid] = tokens
        else:
            self.nodes.pop(nid, None)
            self.tokens.pop(nid, None)
        self.total_tokens += tokens - old_tokens
        if old_keys != keys:
            previous, current = Counter(old_keys), Counter(keys)
            for key in previous.keys() - current.keys():
                self.key_nodes[key].discard(nid)
                if not self.key_nodes[key]:
                    del self.key_nodes[key]
            for key in current.keys() - previous.keys():
                self.key_nodes.setdefault(key, set()).add(nid)
            changes = current.copy()
            changes.subtract(previous)
            for key, delta in changes.items():
                if not delta:
                    continue
                before = self.host_keys[key]
                after = before + delta
                assert after >= 0
                if after:
                    self.host_keys[key] = after
                else:
                    del self.host_keys[key]
                if bool(before) != bool(after) and self.on_key_membership_change is not None:
                    self.on_key_membership_change(key)
            if keys:
                self._keys[nid] = keys
            else:
                self._keys.pop(nid, None)

        if self.on_payload_change is not None:
            self.on_payload_change(nid)

    def validate(self):
        nodes = list(self.core._node_arena.values())
        tokens = {n.id: n.component_data[self.ct].host_value.numel() for n in nodes
                  if n.component_data[self.ct].host_value is not None
                  and n.component_data[self.ct].host_value.numel()}
        keys = Counter(k for n in nodes if n.id in tokens for k in n.hash_value or ())
        assert self.tokens == tokens, ('host tokens',self.tokens,tokens)
        assert self.total_tokens == sum(tokens.values())
        assert self.host_keys == keys

    def bind(self):
        core = self.core
        for name in ('_update_evictable_leaf_sets', '_update_duplicate_tracking',
                     '_evict_component_and_detach_lru'):
            original = getattr(core, name)
            def hook(node, *args, _original=original, **kwargs):
                result = _original(node, *args, **kwargs)
                self.sync(node)
                return result
            setattr(core, name, hook)
        original = core._unregister_node
        def unregister(node, _original=original):
            result = _original(node)
            self.forget(node)
            return result
        core._unregister_node = unregister
        # BACKUP_HOST installs payloads before the write-through ACK. Observe
        # the native commit, not just its later duplicate-tracking notification.
        component = core.components_by_type[self.ct]
        original = component.commit_hicache_transfer
        def commit(node, phase, transfers=(), _original=original, **kwargs):
            result = _original(node, phase, transfers, **kwargs)
            self.sync(node)
            for transfer in transfers:
                for nid in getattr(transfer, "nodes_to_load", None) or ():
                    other = core._node_arena.get(nid)
                    if other is not None:
                        self.sync(other)
            return result
        component.commit_hicache_transfer = commit


def return_waiting_to_read(h):
    core=h.cache.tree_core;page=h.cache.page_size
    entries=list(h.stage.values())
    if not entries:return
    before=dict(waiting_pages=len(entries),read_tokens=h.read_allocated(),read_limit=h.read_limit,
                physical_free=h._physical_available(),budget=h.budget_charged_bytes,time=h.now())
    recency={k:n.last_access_time for n in core._node_arena.values() for k in n.hash_value or ()}
    recency.update({e.hash_value[0]:e.last_access_time for e in entries})
    ordered=sorted(entries,key=lambda e:len(e.prefix_hashes))
    # Returning a role is not an access and must not trigger another eviction
    # while this ownership transaction is in progress.
    was_driving=h._in_drive;h._in_drive=True
    restored=[];blocked=[]
    try:
        for e in ordered:
            if e.owner!='staging':continue
            parent=e.prefix_hashes[-1] if e.prefix_hashes else None
            anchor=core.root_node if parent is None else next(
                (n for n in core._node_arena.values() if parent in (n.hash_value or ())),None)
            if anchor is not None and parent is not None:
                length=(anchor.hash_value.index(parent)+1)*page
                if length<len(anchor.key):
                    anchor,action=core._split_node(anchor.key,anchor,length)
                    if action is not None:h.cache._apply_cache_action(action)
            if anchor is None or not h._return_read(e,anchor):
                blocked.append(dict(key=e.hash_value[0],parent=parent,parent_staged=parent in h.stage,
                                    parent_buffer=False))
            else:restored.append(e)
        for n in core._node_arena.values():
            times=[recency[k] for k in n.hash_value or () if k in recency]
            if times:n.last_access_time=max(times)
        # Explicit user-requested transition exception: these payloads are
        # already allocated. Keep them until real native SSD completion, retaining
        # their existing staging allocation until ACK; ordinary future evictions are not exempt.
        h._staging_return_flush.update(x['key'] for x in blocked)
        for e in ordered:
            if e.hash_value[0] in h._staging_return_flush and e.operation is None and not h._has_ssd_copy(e):
                h._submit_stage_write(e,budget_exempt=True)
        after=dict(restored_pages=len(restored),blocked=blocked,read_tokens=h.read_allocated(),
                   physical_free=h._physical_available(),waiting_pages=h.waiting_staging_tokens()//page)
        record=dict(before=before,after=after)
        h.staging_return_records=getattr(h,'staging_return_records',[])+[record]
        if h.stats_path:
            with Path(h.stats_path).with_suffix('.staging-return.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        assert after['physical_free']==before['physical_free'], ('ownership_return_allocated_or_freed',record)
        assert all(e.hash_value[0] in h._staging_return_flush for e in h.stage.values()),record
        assert h.read_allocated()<=h.read_limit,record
    finally:
        h._in_drive=was_driving


@dataclass
class StagedPage:
    id: int
    key: object
    hash_value: list
    payload: object
    last_access_time: float
    prefix_hashes: tuple = ()
    operation: object = None
    owner: str = 'staging'
    native_lock: object = None
    read_pins: int = 0
    read_pin_lock: object = None
    @property
    def evicted(self): return True
    @property
    def component_data(self):
        return [SimpleNamespace(host_value=self.payload, value=None, host_lock_ref=self.read_pins, lock_ref=0)]


class NativeHost:
    def on_requests(self, requests, now):
        if self.lifecycle is not None or self.sli is not None:
            for req in requests:
                if req.__class__.__name__ in ("TokenizedGenerateReqInput", "BatchTokenizedGenerateReqInput"):
                    self.observe_request(req, now)

    def before_schedule(self, now):
        if self.sli is not None:
            self.staging_policy.maybe_control(now)

    def next_policy_deadline(self):
        deadlines=[]
        ref=self.staging_policy
        if ref.dynamic_admission and ref.next_tick<=ref.cfg['duration_s']:
            deadlines.append(ref.next_tick)
        if self.capacity_controller:
            deadlines.append(self.capacity_controller.next_tick)
        return min(deadlines) if deadlines else None

    def on_batch_complete(self, requests, now):
        if self.sli is not None:
            for req in requests:
                if len(req.output_ids) != 0:
                    self.sli.release_read(req.rid)
                if req.finished():
                    self.sli.complete(req, now)

    def __init__(self, cache, cfg, total_tokens, staging_tokens, *, clock=None, storage_factory=None):
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        storage_factory = storage_factory or StagingStorage.create
        read_limit, staging_limit = total_tokens-staging_tokens, staging_tokens
        stats_path = cfg.get('stats_path')
        clock = clock or (lambda: time.monotonic()-self.started_at)
        self._clock = clock
        if read_limit <= 0 or staging_limit < 0: raise ValueError('invalid residency capacity')
        self.cache=cache;self.cfg=cfg;self.read_limit=read_limit;self.staging_limit=staging_limit
        self.total_tokens=read_limit+staging_limit
        fraction=float(cfg.get('write_buffer_fraction',0.))
        if not 0<=fraction<1:raise ValueError('invalid write buffer fraction')
        self.configured_staging_limit=staging_limit
        if fraction:raise ValueError('fixed write buffer removed; use native async writes')
        self._staging_return_flush=set()
        self.staging_return_exempt_bytes=0
        self.admission_policy=cfg.get('admission_policy','write_all')
        self.materialization_timing=cfg.get('materialization_timing','on_evict')
        if self.materialization_timing!='on_evict':raise ValueError('Hohlraum supports staging-exit admission only')
        self.residency_policy=cfg.get('residency_policy','lru')
        if self.admission_policy not in ('agent','write_all','write_none'):raise ValueError('invalid admission')
        if self.residency_policy not in ('lru','joint_hts'):raise ValueError('overlay supports native LRU or paper HTS')
        if not cfg.get('reference_history'):raise ValueError('overlay requires the canonical reference history')
        self.admission_threshold=float(cfg.get('threshold',.7))
        self.kv_bytes_per_token=int(cfg['kv_bytes_per_token'])
        self.stats_path=Path(stats_path) if stats_path else None
        self.component=cache.components[ComponentType.FULL]
        self.stage={};self.absorbed=set();self._next_stage_id=-1;self.inflight={}
        self.admission_observed_rids=set();self.request_hashes={}
        self.budget_exhausted_at_s=None
        self._next_budget_snapshot_s=0.
        self._next_stats_snapshot_s=0.
        self.staging_pressure=dict(pre_retire=0,wait_existing_write=0,no_read_candidate=0,no_stage_candidate=0,no_retire_progress=0,transfer_attempts=0,transfer_succeeded=0,transfer_failed=0)
        self.last_transfer_pressure=None
        self.pressure_waiting=False;self._in_match=False;self._in_drive=False
        self.sli=SLI(self,cfg)
        self.lifecycle=self.sli if self.admission_policy=='agent' else None
        self.residency=self.sli if self.admission_policy=='agent' else None
        self.staging_policy=StagingPolicy(self)
        self.capacity_controller=None
        self._host_index=HostPayloadIndex(cache.tree_core,ComponentType.FULL);self._host_index.bind()
        self._bind_resident_index()
        fields=('ssd_written_tokens','ssd_write_failures','ssd_prefetched_tokens','staging_exits',
                'read_releases','duplicate_reclaimed_tokens','demotions','promotions','peak_read_tokens',
                'peak_staging_tokens','over_limit_events','admission_write_decisions','admission_drop_decisions',
                'discarded_tokens','staging_reclassified_tokens','persisted_reclaimed_tokens',
                'budget_rejected_tokens','storage_unconfirmed_tokens',
                'host_pressure_retries','host_pressure_reclaimed_tokens','host_pressure_write_wait_s',
                'overlay_transfers_in','overlay_transfers_read','overlay_write_returns_read',
                'overlay_write_releases','overlay_backup_reuses')
        for name in fields:setattr(self,name,0)
        self._original_drive_host_eviction=self.component.drive_host_eviction
        self.component.drive_host_eviction=self._drive_host_eviction
        self.allocation_progress = {}
        self._read_ownership = Counter()
        self.ssd_backend = cache.cache_controller.storage_backend
        self.storage = storage_factory(self, self.ssd_backend, cache.page_size,
                                      self.kv_bytes_per_token)
        cache.cache_controller.storage_backend = self.storage
        install_storage_adapter(self, self.storage)
        self._attach_write_ack()
        self._attach_partition_allocator()
        self._attach_read_reclaim()
        self._attach_hbm_eviction()
        if cfg.get('dynamic_capacity'):
            from hohlraum.core.capacity_controller import CapacityController
            self.capacity_controller=CapacityController(self)
            if not os.environ.get('SGLANG_SIMULATOR_CONFIG_PATH') and cache.tp_world_size>1:
                from hohlraum.core.capacity_controller import native_capacity_consensus
                self.capacity_consensus=native_capacity_consensus(self.capacity_controller)
            self._attach_capacity_feedback()
            self._attach_writeback_feedback()
        query = cache.cache_controller._storage_hit_query
        @wraps(query)
        def pinned_query(*args, **kwargs):
            with self.lock:
                return query(*args, **kwargs)
        cache.cache_controller._storage_hit_query = pinned_query

    def contains(self, key):
        return key in self.stage

    def accept_evicted(self, entry):
        key = entry.hash_value[0]
        assert key not in self.stage and entry.owner == 'staging'
        self.stage[key] = entry
        self.absorbed.add(entry.id)
        self.demotions += 1
        self.overlay_transfers_in += 1
        self._resident_key_changed(key)
        if self.capacity_controller:self.capacity_controller.entered(entry)

    def acquire_source(self, key):
        entry = self.stage.get(key)
        if entry is not None:
            entry.read_pins += 1
            self._read_ownership['source_pins'] += 1
        return entry

    def release_source(self, entry):
        entry.read_pins -= 1
        assert entry.read_pins >= 0
        self._read_ownership['source_unpins'] += 1
        self._release_transferred_source(entry)

    def confirm_read(self, keys):
        for key in keys:
            entry = self.stage.get(key)
            if entry is None or getattr(entry, 'copy_committed', False):
                continue
            nodes = self._host_index.key_nodes.get(key)
            if not nodes:
                continue
            node_id = next(iter(nodes))
            entry.copy_lock = (node_id, self.cache.inc_host_lock_ref(node_id).to_dec_params())
            entry.copy_committed = True
            self._read_ownership['committed_pages'] += 1
            self._release_transferred_source(entry)

    def _release_transferred_source(self, entry):
        if not getattr(entry, 'copy_committed', False) or entry.read_pins or entry.operation is not None:
            return
        if self.stage.get(entry.hash_value[0]) is not entry:
            return
        self._remove_stage(entry, free=True)
        self.cache.dec_host_lock_ref(*entry.copy_lock)
        entry.copy_lock = None
        self.promotions += 1
        self.overlay_transfers_read += 1
        self._read_ownership['released_sources'] += 1

    def read_ownership_stats(self):
        return dict(self._read_ownership)

    def _attach_partition_allocator(self):
        group=self.cache.cache_controller.mem_pool_host
        pool=group.anchor_entry.host_pool if hasattr(group,'anchor_entry') else group
        self._physical_available=pool.available_size
        native_alloc=pool.alloc
        def alloc(need_size,*args,**kwargs):
            if need_size>self.read_available():return None
            return native_alloc(need_size,*args,**kwargs)
        pool.available_size=self.read_available
        pool.alloc=alloc

    def _attach_read_reclaim(self):
        # The native caller asks for allocatable host tokens. Moving a page to
        # staging frees read quota without freeing its physical allocation.
        # Keep native physical-free trackers untouched; report effective space
        # only at the partition-facing allocation boundary.
        original = self.cache.evict_host
        def evict(num_tokens, component_type=ComponentType.FULL):
            before = self.read_available()
            freed = original(num_tokens, component_type)
            if component_type != ComponentType.FULL:
                return freed
            return max(freed, max(0, self.read_available() - before))
        self.cache.evict_host = evict

    def _attach_hbm_eviction(self):
        # Reclaiming a WT host duplicate allows an unbacked device parent
        # to retain host-only children. Preserve that structural parent when
        # evicting HBM, instead of using native WT's whole-node deletion.
        # This must remain installed after budget exhaustion: old mixed-tier
        # branches can outlive the switch back to native host candidates.
        if not self.cfg.get('stage_hbm_leaves', False):
            return
        core = self.cache.tree_core
        if core.is_write_back:
            return
        original = core.evict_device_leaf
        from sglang.srt.disaggregation.kv_events import StorageMedium
        from sglang.srt.mem_cache.unified_cache.components import EvictLayer
        from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import EvictDeviceLeafResult

        def evict(node_id, is_write_back):
            node = core.node_by_id(node_id)
            unbacked = not node.backuped and not is_write_back
            if unbacked:
                tokens = len(node.component_data[ComponentType.FULL].value)
                self._read_ownership['hbm_evicted_without_read_tokens'] += tokens
                keys = node.hash_value or ()
                staged = sum(k in self.stage for k in keys)
                self._read_ownership['hbm_evicted_still_staged_tokens'] += staged * self.cache.page_size
                if self.cfg.get('trace_hbm_backing', False):
                    persisted = self._copy_bits(keys)
                    missing = sum(k not in self.stage and not stored for k, stored in zip(keys, persisted))
                    self._read_ownership['hbm_evicted_no_backing_tokens'] += missing * self.cache.page_size
                    self._read_ownership['hbm_evicted_ssd_backed_tokens'] += sum(persisted) * self.cache.page_size
            if not unbacked or not node.children:
                return original(node_id, is_write_back)
            assert core._is_device_leaf(node)
            result = EvictDeviceLeafResult()
            core._begin_tracking_unbacked_tokens()
            try:
                trigger = core.components_by_type[ComponentType.FULL]
                core._evict_component_and_detach_lru(
                    node, trigger, target=EvictLayer.DEVICE,
                    tracker=result.tracker, device_frees=result.device_frees,
                    host_frees=result.host_frees)
                core._cascade_evict(node, trigger, result.tracker,
                    device_frees=result.device_frees, host_frees=result.host_frees)
                core.kv_events.record_remove(node, medium=StorageMedium.GPU)
                core._update_evictable_leaf_sets(node.parent)
                self._read_ownership['hbm_structural_parents_preserved'] += 1
                assert core.node_by_id(node.id) is node and node.evicted
                assert node not in core.evictable_device_leaves
                return result
            finally:
                result.unbacked_tokens = core._finish_tracking_unbacked_tokens()
        core.evict_device_leaf = evict

    def _attach_writeback_feedback(self):
        if self.cache.cache_controller.write_policy != 'write_back':
            return
        original = self.cache._execute_kv_backup
        def backup(node_id, device_value, *args, **kwargs):
            result = original(node_id, device_value, *args, **kwargs)
            # Record only remaining FULL host capacity failures after native
            # reclamation. Successful reclamation is not controller pressure.
            needed = len(device_value)
            if result is None and needed > self.read_available():
                self.capacity_controller.writeback_failed(node_id, needed)
            return result
        self.cache._execute_kv_backup = backup

    def read_allocated(self,deferred_free=0):
        # Everything outside staging belongs to read, including native IO buffers.
        return self.total_tokens-self._physical_available()-deferred_free-len(self.stage)*self.cache.page_size

    def read_available(self,deferred_free=0):
        physical=self._physical_available()+deferred_free
        return max(0,min(physical,self.read_limit-self.read_allocated(deferred_free)))


    def _reserve_read_transfer(self,entry,anchor,credit=0):
        deficit=self.read_allocated()+self.cache.page_size-credit-self.read_limit
        if deficit<=0:return True
        # The caller's anchor and source must survive reclamation.
        entry.read_pins+=1
        lock=self.cache.inc_host_lock_ref(anchor.id).to_dec_params()
        try:
            self.staging_pressure['transfer_attempts']+=1
            self.cache.evict_host(deficit)
            success=self.read_allocated()+self.cache.page_size-credit<=self.read_limit
            self.staging_pressure['transfer_succeeded' if success else 'transfer_failed']+=1
            if not success:
                self.last_transfer_pressure=dict(read_allocated=self.read_allocated(),read_limit=self.read_limit,
                    physical_free=self._physical_available(),stage_pages=len(self.stage),stage_limit=self.staging_limit,
                    evictable_read_leaves=len(self.cache.tree_core.evictable_host_leaves),inflight_writes=len(self.inflight),
                    inside_eviction=self._in_drive,source_read_pins=entry.read_pins)
            return success
        finally:
            self.cache.dec_host_lock_ref(anchor.id,lock)
            entry.read_pins-=1

    def now(self):
        return self._clock()

    @staticmethod
    def _tokens(node):
        v=node.component_data[ComponentType.FULL].host_value
        return 0 if v is None else v.numel()

    @staticmethod
    def _host_only(node): return node.evicted and NativeHost._tokens(node)>0

    def _nodes(self):
        yield from (n for n in self.cache.tree_core._node_arena.values() if n is not self.cache.tree_core.root_node)
        yield from self.stage.values()

    def _stage_nodes(self):
        return tuple(self.stage.values())

    def waiting_staging_tokens(self):
        return sum(e.operation is None for e in self.stage.values())*self.cache.page_size

    def _bind_resident_index(self):
        self._host_index.on_key_membership_change = self._resident_key_changed
        for key in self.resident_keys():
            self.staging_policy.index.set_resident(key, True)

    def _resident_key_changed(self, key):
        self.staging_policy.index.set_resident(key, key in self.stage or key in self._host_index.host_keys)

    def has_resident_descendant(self, key):
        return self.staging_policy.index.has_resident_descendant(key)

    def resident_keys(self): return set(self._host_index.host_keys)|set(self.stage)
    def _pending(self,node):
        if isinstance(node,StagedPage):return node.operation is not None
        return any(nid==node.id for nid,_ in self.cache.ongoing_backup.values())
    def _copy_bits(self,keys):
        return [self.ssd_backend.batch_exists([key], None) == 1 for key in keys]
    def _has_ssd_copy(self,node):
        keys=node.hash_value or []
        return bool(keys) and all(self._copy_bits(keys))
    def storage_active(self):return self.cache.enable_storage and self.cache.cache_controller.storage_backend is not None
    def _counts(self):return self._host_index.total_tokens,len(self.stage)*self.cache.page_size
    def _record_peaks(self):
        read,stage=self._counts();self.peak_read_tokens=max(self.peak_read_tokens,read);self.peak_staging_tokens=max(self.peak_staging_tokens,stage)
        self.over_limit_events+=int(read>self.read_limit or self.waiting_staging_tokens()>self.staging_limit)
        if self.cfg.get('validate_residency_index'):
            self._validation_checks=getattr(self,'_validation_checks',0)+1
            self._host_index.validate()
            assert self.absorbed=={e.id for e in self.stage.values()}
            resident = self.resident_keys()
            assert self.staging_policy.index._resident == resident
            expected = set()
            for key in resident:
                parent = self.staging_policy.index.parents.get(key)
                while parent is not None and parent not in expected:
                    expected.add(parent); parent = self.staging_policy.index.parents.get(parent)
            assert set(self.staging_policy.index._live_children) == expected
            assert all(getattr(self.stage[k], 'copy_committed', False) for k in set(self.stage).intersection(self._host_index.host_keys)), 'untracked read/staging copy overlap'
            slots=set()
            for node in self._host_index.nodes.values():
                values=node.component_data[ComponentType.FULL].host_value.tolist()
                assert not slots.intersection(values), 'duplicate native read allocation'
                slots.update(values)
            for entry in self.stage.values():
                values=entry.payload.tolist()
                assert not slots.intersection(values), 'read/staging allocation overlap'
                slots.update(values)
            native=set(self.cache.tree_core._node_arena)
            assert not native.intersection(self.absorbed)
            assert all(e.owner=='staging' and e.payload.numel()==self.cache.page_size for e in self.stage.values())

    def _hbm_read_leaf(self, node):
        """WT host candidate: tree leaf, or optional DRAM leaf; both tiers unlocked."""
        core = self.cache.tree_core
        return (self.cfg.get('stage_hbm_leaves', False)
                and not self._budget_exhausted()
                and not core.is_write_back
                and (not node.children or (
                    self.cfg.get('stage_hbm_dram_leaves', False)
                    and all(child.component_data[ComponentType.FULL].host_value is None
                            for child in node.children.values())))
                and core._can_reclaim_full_host_duplicate(node)
                and all(cd.lock_ref == 0 and cd.host_lock_ref == 0
                        for cd in node.component_data)
                and all(cd.host_value is None for ct, cd in enumerate(node.component_data)
                        if ct != int(ComponentType.FULL)))

    def _drive_read_leaves(self, num_tokens, tracker, device_frees, host_frees,
                           *, leaf_handler, should_stop):
        if (not self.cfg.get('stage_hbm_leaves', False)
                or self._budget_exhausted() or self.cache.tree_core.is_write_back):
            return self._original_drive_host_eviction(num_tokens, tracker, device_frees,
                host_frees, leaf_handler=leaf_handler, should_stop=should_stop)
        core = self.cache.tree_core
        self.component._ensure_eviction_strategy()
        candidates = set(core.evictable_host_leaves)
        candidates.update(n for n in core.full_host_duplicates.values() if self._hbm_read_leaf(n))
        heap = [(self.component.session_ref_eviction_strategy(n), n) for n in candidates]
        heapq.heapify(heap)
        while heap and not should_stop():
            _, node = heapq.heappop(heap)
            duplicate = self._hbm_read_leaf(node)
            if node not in core.evictable_host_leaves and not duplicate:
                continue
            handled = leaf_handler(node)
            if handled is False:
                break
            if handled is None:
                if duplicate:
                    core._release_full_host_duplicate(node, tracker, device_frees, host_frees)
                    core.full_host_duplicates.pop(node.id, None)
                    core._update_evictable_leaf_sets(node)
                else:
                    core._evict_host_leaf(node, tracker, device_frees, host_frees)
            # Host removal can expose a DRAM leaf while the HBM subtree remains.
            # Recheck eligibility on pop too: a new host child must block it.
            if node.parent is not None and (node.parent in core.evictable_host_leaves
                                             or self._hbm_read_leaf(node.parent)):
                heapq.heappush(heap, (self.component.session_ref_eviction_strategy(node.parent), node.parent))

    def _take_read_leaf(self,node):
        """Native leaf deletion, transferring its host allocation instead of freeing it."""
        core=self.cache.tree_core;page=self.cache.page_size
        assert node in core.evictable_host_leaves or self._hbm_read_leaf(node)
        if self._tokens(node)>page:
            # Ownership transfer is not a read access. Native splitting assigns
            # fresh timestamps; preserve the pre-eviction recency on both pieces.
            prior_access=node.last_access_time
            prefix,action=core._split_node(node.key,node,self._tokens(node)-page)
            prefix.last_access_time=node.last_access_time=prior_access
            if action is not None:self.cache._apply_cache_action(action)
        assert self._tokens(node)==page
        key=node.hash_value[0];payload=node.component_data[ComponentType.FULL].host_value
        assert key not in self.stage, 'read leaf is already staging-owned'
        entry=StagedPage(self._next_stage_id,node.key,list(node.hash_value),payload,node.last_access_time,
                         prefix_hashes=tuple(node.get_prefix_hash_values(node.parent)))
        self._next_stage_id-=1
        tracker=defaultdict(int);device=defaultdict(list);host=defaultdict(list)
        if self._hbm_read_leaf(node):
            core._release_full_host_duplicate(node,tracker,device,host)
            core.full_host_duplicates.pop(node.id,None)
            core._update_evictable_leaf_sets(node)
            self._read_ownership['hbm_leaf_transfers'] += 1
            if node.children:self._read_ownership['hbm_internal_transfers'] += 1
            assert core.node_by_id(node.id) is node and not node.evicted
        else:
            core._evict_host_leaf(node,tracker,device,host)
        # The native deletion owns all tree/index changes. Its one host free
        # is handed to the overlay, never passed to the allocator's free().
        assert tracker[ComponentType.FULL]==page and not any(device.values())
        assert sum(x.numel() for x in host[ComponentType.FULL])==page
        return entry

    def _remove_stage(self,entry,free=False):
        key=entry.hash_value[0]
        assert self.stage.get(key) is entry
        if self.capacity_controller:self.capacity_controller.removed(entry)
        self.stage.pop(key);self.absorbed.discard(entry.id)
        self._staging_return_flush.discard(key)
        self._resident_key_changed(key)
        if free:
            assert entry.operation is None and not entry.read_pins
            self.cache.cache_controller.mem_pool_host.free(entry.payload);entry.owner='released'

    def _finish_read_transfer(self,entry,node_id):
        entry.owner='read'
        if entry.operation is not None:
            entry.native_lock=(node_id,self.cache.inc_host_lock_ref(node_id).to_dec_params())
        self.promotions+=1;self.overlay_transfers_read+=1

    def _return_read(self,entry,anchor,read_credit=0):
        """Transfer the existing allocation; never create a second resident copy."""
        core=self.cache.tree_core;page=self.cache.page_size
        if not self._reserve_read_transfer(entry,anchor,read_credit):return False
        assert anchor.get_last_hash_value()==(entry.prefix_hashes[-1] if entry.prefix_hashes else None), 'staging parent mismatch'
        assert entry.hash_value[0] not in self._host_index.host_keys
        child=anchor.children.get(entry.key.child_key(page))
        if child is not None and child.key.match(entry.key,page_size=page)==page:
            # An HBM insertion may have recreated the structural path. Native
            # insert_host intentionally skips existing paths: use its backup
            # commit boundary to attach the already available host allocation.
            if len(child.key)>page:
                child,action=core._split_node(child.key,child,page)
                if action is not None:self.cache._apply_cache_action(action)
            assert child.component_data[ComponentType.FULL].host_value is None
            assert child.component_data[ComponentType.FULL].value is not None
            self._remove_stage(entry,free=False)
            core.commit_backup(child.id,entry.payload,{})
            core._update_evictable_leaf_sets(child)
            core._update_duplicate_tracking(child)
            self._finish_read_transfer(entry,child.id)
            self.overlay_backup_reuses+=1
            return True
        if anchor is not core.root_node and not anchor.backuped and not core.is_write_back:return False
        self._remove_stage(entry,free=False)
        result=self._native_insert_host(anchor.id,entry.key,entry.payload,entry.hash_value)
        self.cache._apply_cache_actions(result.cache_actions)
        assert not getattr(result,'host_insert_dropped',False) and result.prefix_len==0
        self._finish_read_transfer(entry,result.inserted_host_node)
        return True

    def _choose_stage(self):
        return self.staging_policy.choose()

    @property
    def write_budget_bytes(self):
        total = self.ssd_backend.validation_cap
        return None if total is None else total - self.cfg.get('transition_reserve_bytes', 0)

    @property
    def budget_charged_bytes(self):
        # Reporting view of the single SSD counter, not a second budget ledger.
        return self.ssd_backend.validation_write_bytes - self.staging_return_exempt_bytes

    def _budget_exhausted(self):
        exhausted = getattr(self, '_synced_budget_exhausted', None)
        if exhausted is None:
            exhausted = self.ssd_backend.budget_exhausted(self.cfg.get('transition_reserve_bytes', 0))
        if exhausted and self.budget_exhausted_at_s is None:
            self.budget_exhausted_at_s = self.now()
        return exhausted


    def _retire_staging(self,entry):
        if entry.operation is not None or entry.read_pins:return 0
        copied=self._has_ssd_copy(entry)
        if entry.hash_value[0] in self._staging_return_flush:
            if copied:
                self._remove_stage(entry,free=True);self.staging_exits+=1
                return self.cache.page_size
            return self._submit_stage_write(entry,budget_exempt=True)
        decision=self.staging_policy.retire(entry,copied)
        if not decision.write:
            if not copied:
                self.admission_drop_decisions+=1;self.discarded_tokens+=self.cache.page_size
                if decision.budget_exhausted:self.budget_rejected_tokens+=self.cache.page_size
            if self.capacity_controller and not copied:
                self.capacity_controller.discarded(entry,decision.budget_exhausted)
            self._remove_stage(entry,free=True);self.staging_exits+=1
            return self.cache.page_size
        return self._submit_stage_write(entry)

    def _submit_stage_write(self,entry,budget_exempt=False):
        if budget_exempt:
            self.ssd_backend.allow_transition(entry.hash_value[0])
        assert entry.operation is None and not self._has_ssd_copy(entry)
        if budget_exempt:
            assert entry.hash_value[0] in self._staging_return_flush, 'exemption only for blocked staging return'
        entry.capacity_sample = self.capacity_controller.write_sample(entry) if self.capacity_controller and not budget_exempt else None
        entry.transition_write = budget_exempt
        op=self.cache.cache_controller.write_storage(entry.payload,entry.key.raw_token_ids(),entry.hash_value,list(entry.prefix_hashes))
        entry.operation=op;self.inflight[op]=entry
        # Native queues perform the write and consume its ACK. The negative
        # owner ID dispatches only the staging receipt at the unlock boundary.
        self.cache.ongoing_backup[op]=(entry.id,None)
        self.admission_write_decisions += 1
        return 0

    def _record_budget_progress(self,force=False,event='periodic'):
        filename=self.cfg.get('budget_progress_log')
        if not filename or not self.sli.observed_requests:return
        now=self.now()
        if not force and now<self._next_budget_snapshot_s:return
        interval=float(self.cfg.get('budget_progress_interval_s',60.))
        if interval<=0:raise ValueError('budget observation interval must be positive')
        self._next_budget_snapshot_s=(int(now//interval)+1)*interval
        budget=self.write_budget_bytes;used=self.budget_charged_bytes
        timeline=getattr(self.cache.cache_controller,'sim_write_timeline',None)
        row=dict(unified_storage=self.storage.stats(),event=event,logical_time_s=now,
                 observed_requests=self.sli.observed_requests,completed_requests=len(self.sli.finished),
                 budget_bytes=budget,used_bytes=used,remaining_bytes=None if budget is None else max(0,budget-used),
                 budget_GB=None if budget is None else budget/1e9,used_GB=used/1e9,
                 remaining_GB=None if budget is None else max(0,budget-used)/1e9,
                 used_pct=None if budget is None else (100*used/budget if budget else 100.),
                 exhausted=self._budget_exhausted(),first_exhausted_at_s=self.budget_exhausted_at_s,
                 ssd_completed_GB=None if timeline is None else timeline.completed_bytes/1e9,
                 read_target_GB=self.read_limit*self.kv_bytes_per_token/1e9,
                 staging_target_GB=self.staging_limit*self.kv_bytes_per_token/1e9,
                  write_buffer_target_GB=0,
                  write_buffer_used_GB=0,
                  waiting_staging_used_GB=self.waiting_staging_tokens()*self.kv_bytes_per_token/1e9,
                 meaning='Used budget reads the shared SSD successful-write counter; HiSim writes complete immediately.')
        path=Path(filename)
        temporary=path.with_suffix(path.suffix+'.pending')
        temporary.write_text(json.dumps(row,indent=2));temporary.replace(path)
        with path.with_suffix('.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')

    def _attach_write_ack(self):
        original=self.cache.dec_host_lock_ref
        def finish(node_id,params):
            if node_id>=0:return original(node_id,params)
            entry=next((e for e in self.inflight.values() if e.id==node_id),None)
            if entry is None:raise RuntimeError('unknown staging write receipt')
            op=entry.operation;self.inflight.pop(op);entry.operation=None
            if self.capacity_controller and self._has_ssd_copy(entry):
                self.capacity_controller.wrote(getattr(entry,"capacity_sample",None))
            entry.capacity_sample=None
            if getattr(entry, 'transition_write', False) and self._has_ssd_copy(entry):
                self.staging_return_exempt_bytes += self.cache.page_size*self.kv_bytes_per_token
            entry.transition_write = False
            if getattr(entry, 'copy_committed', False):
                if self._has_ssd_copy(entry):self.ssd_written_tokens+=self.cache.page_size
                else:self.ssd_write_failures+=1
                self._release_transferred_source(entry)
                return
            if not self._has_ssd_copy(entry):
                self.ssd_write_failures+=1
                if entry.owner=='read' and entry.native_lock:original(*entry.native_lock);entry.native_lock=None
                return
            if getattr(self,'staging_policy',None) is not None:self.staging_policy.complete_prefix_write(entry)
            self.ssd_written_tokens+=self.cache.page_size
            if entry.owner=='read':
                if entry.native_lock:original(*entry.native_lock);entry.native_lock=None
                return
            if entry.read_pins:return
            # An eviction write is complete. Native storage already owns its
            # copy; release the source unless an actual read still holds it.
            # Completion itself is not a read or a reason to reinsert into LRU.
            self._remove_stage(entry,free=True);self.overlay_write_releases+=1;self.staging_exits+=1
        self.cache.dec_host_lock_ref=finish

    def _drive_host_eviction(self,required_free,tracker,device_frees,host_frees):
        if not self.storage_active():
            return self._original_drive_host_eviction(required_free,tracker,device_frees,host_frees)
        if self._in_drive:return
        self._in_drive=True
        target=self.read_available()+required_free
        # In-flight writes retain their physical source but no longer consume
        # the waiting-residency quota. Count their future frees only for limiting
        # submissions, never as allocator capacity or as returned freed tokens.
        pending_pages=sum(self.stage.get(e.hash_value[0]) is e
                          for e in self.inflight.values())
        pending_free=sum(self.cache.page_size for e in self.inflight.values()
                         if self.stage.get(e.hash_value[0]) is e and not e.read_pins)
        def retire(entry):
            nonlocal pending_pages,pending_free
            was_pending=entry.operation is not None
            freed=self._retire_staging(entry)
            if not was_pending and entry.operation is not None:
                pending_pages+=1
                if not entry.read_pins:pending_free+=self.cache.page_size
            tracker[ComponentType.FULL]+=freed
        def waiting_tokens():
            return (len(self.stage)-pending_pages)*self.cache.page_size
        def enough():
            deferred=sum(v.numel() for v in host_frees.get(ComponentType.FULL,()))
            eventual=min(self._physical_available()+deferred+pending_free,
                         self.read_limit-self.read_allocated(deferred))
            return eventual>=target and self.read_allocated(deferred)<=self.read_limit
        def stage_leaf(node):
            if not self.staging_policy.accepts_staging(node):
                self.read_releases+=1
                return None
            while waiting_tokens()>=max(self.cache.page_size,self.staging_limit):
                entry=self._choose_stage()
                if entry is None:
                    self.staging_pressure['no_stage_candidate']+=1
                    return False
                before=(len(self.stage),len(self.inflight))
                retire(entry)
                if before==(len(self.stage),len(self.inflight)):
                    self.staging_pressure['no_retire_progress']+=1
                    return False
                self.staging_pressure['pre_retire']+=1
                if enough():return False
            entry=self._take_read_leaf(node)
            self.storage.put(entry)
            if self.staging_limit==0:
                retire(entry)
            return True
        try:
            while waiting_tokens()>self.staging_limit:
                entry=self._choose_stage()
                if entry is None:break
                before=(len(self.stage),len(self.inflight))
                retire(entry)
                if before==(len(self.stage),len(self.inflight)):break
            self._drive_read_leaves(required_free,tracker,device_frees,host_frees,
                                               leaf_handler=stage_leaf,should_stop=enough)
            self._record_peaks()
        finally:
            self._in_drive=False

    def observe_request(self, req, now):
        if req.rid in self.admission_observed_rids:
            return
        params = req.sampling_params.custom_params
        if params is None:
            params = req.sampling_params.custom_params = {}
        if 'hohlraum' not in params:
            simulation_meta = (params.get('simulation') or {}).get('agentx')
            if simulation_meta is not None:
                # Preserve the trace's session identity.
                params['hohlraum'] = dict(simulation_meta)
        if 'hohlraum' not in params:
            # Requests without agent metadata are independent sessions; never
            # infer a shared agent lifecycle from unrelated HTTP requests.
            import hashlib
            params['hohlraum'] = dict(root=int.from_bytes(hashlib.sha256(req.rid.encode()).digest()[:8],'big'),
                                     chain=0,owner=0,reference_fold=0)
        meta = params['hohlraum']
        for key in ('root','chain','owner','reference_fold'):
            if key not in meta:
                raise ValueError(f'Hohlraum request missing lifecycle field {key}')
        hashes = self.sli.hashes_for_request(req, self.cache.page_size)
        self.staging_policy.observe((meta['root'],meta['owner']),hashes)
        self.request_hashes[req.rid]=hashes
        self.sli.arrive(req,now,hashes)
        self.admission_observed_rids.add(req.rid)

    def finish_request(self,req):
        if self.capacity_controller:self.capacity_controller.end_read(req.rid,self.now())
        self.storage.release_request(req.rid)
        self.request_hashes.pop(req.rid,None)
        self.sli.ready_hashes.pop(req.rid, None)

    def _advance_admission(self):
        if not self.sli.observed_requests:return True
        now=self.now()
        if now<self.sli.now-1e-8:
            raise RuntimeError('reference clock reversed')
        self.sli.advance(now)
        return True

    def maintain(self,required_free=0):
        if not self._advance_admission():return 0
        if self.sli.observed_requests:self.staging_policy.maybe_control(self.now())
        consensus=getattr(self,'capacity_consensus',None)
        if consensus is not None:
            consensus.advance()
        else:
            if self._budget_exhausted() and self.staging_limit:
                if self.resize_staging(0) is not False:
                    self._record_budget_progress(force=True,event="staging_disabled_after_exhaustion")
            if self.capacity_controller:self.capacity_controller.tick(self.now())
        self._record_budget_progress()
        freed=self.cache.evict_host(required_free)
        # Live progress snapshots are observational, not a scheduler decision.
        # The result exporter still reads fresh stats at experiment completion.
        now = self.now()
        if self.stats_path and now >= self._next_stats_snapshot_s:
            self.stats_path.write_text(json.dumps(self.stats()))
            self._next_stats_snapshot_s = now + 60.
        return freed

    def resize_staging(self,target):
        page=self.cache.page_size
        target=int(target)//page*page
        if not 0<=target<self.total_tokens:raise ValueError('capacity must preserve read residency')
        # Update both quotas before any eviction; existing SSD copies still use
        # native release, and unmaterialized read victims enter the expanded tier.
        previous=self.staging_limit
        if previous==target:return True
        exhausted=self._budget_exhausted()
        if target==0 and exhausted and (getattr(self,'_capacity_zero_blocked',False) or any(e.read_pins or getattr(e,'copy_committed',False) for e in self.stage.values())):
            return False
        self.staging_limit=target;self.read_limit=self.total_tokens-target
        if previous and target==0 and exhausted:
            return_waiting_to_read(self)
        else:
            # The existing eviction driver first retires staging overflow, then
            # shrinks read using its native victim order. Locked/in-flight pages
            # defer reclamation; they are never counted as free prematurely.
            self.cache.evict_host(0)
        return True

    def _attach_capacity_feedback(self):
        cache=self.cache;control=self.capacity_controller
        unresolved=control.unresolved
        original_hit=cache._record_storage_prefetch_hit
        def hit(request,num_tokens):
            if num_tokens>0:unresolved[request.rid]=num_tokens
            return original_hit(request,num_tokens)
        cache._record_storage_prefetch_hit=hit
        original_resolve=cache._resolve_storage_prefetch_tokens
        def resolve(request,num_tokens,reason=None):
            remaining=unresolved.get(request.rid,0)
            lost=min(max(0,num_tokens),remaining)
            control.failed(request.rid,lost,reason)
            if remaining:unresolved[request.rid]=remaining-lost
            return original_resolve(request,num_tokens,reason)
        cache._resolve_storage_prefetch_tokens=resolve
        original_finish=cache._finish_storage_prefetch
        def finish(request,fulfilled_tokens,reason):
            remaining=unresolved.pop(request.rid,0)
            control.failed(request.rid,max(0,remaining-fulfilled_tokens),reason)
            return original_finish(request,fulfilled_tokens,reason)
        cache._finish_storage_prefetch=finish
        original_discard=cache.discard_storage_prefetch_accounting
        def discard(request):
            unresolved.pop(request.rid,None)
            control.end_read(request.rid,self.now())
            return original_discard(request)
        cache.discard_storage_prefetch_accounting=discard
        original_check=cache.check_prefetch_progress
        def check(request):
            info=cache.ongoing_prefetch.get(request)
            lease=self.storage.leases.get(id(info.operation)) if info is not None else None
            has_ssd=lease is not None and len(lease.entries)<len(lease.keys)
            ready=original_check(request)
            control.read_progress(request.rid,not ready,self.now(),has_ssd)
            return ready
        cache.check_prefetch_progress=check

    def record_prefetch(self,tokens):self.ssd_prefetched_tokens+=tokens
    def reset(self):
        # Called AFTER native reset, which has cleared the shared host allocator.
        if self.capacity_controller:self.capacity_controller.reset()
        self.storage.reset()
        self.staging_policy.reset_prefix_obligations()
        self._staging_return_flush.clear()
        self.sli.ready_hashes.clear()
        self.stage.clear();self.absorbed.clear();self.inflight.clear();self.request_hashes.clear()
        self._host_index.__init__(self.cache.tree_core,ComponentType.FULL)
        self._next_stats_snapshot_s=0.
        self.staging_policy.index.clear_residents()
        self._bind_resident_index()

    def stats(self):
        read,stage=self._counts()
        fields=('ssd_written_tokens','ssd_write_failures','ssd_prefetched_tokens','staging_exits','read_releases',
                'duplicate_reclaimed_tokens','demotions','promotions','peak_read_tokens','peak_staging_tokens',
                'over_limit_events','admission_write_decisions','admission_drop_decisions','discarded_tokens',
                'staging_reclassified_tokens','persisted_reclaimed_tokens','budget_charged_bytes','budget_rejected_tokens',
                'storage_unconfirmed_tokens','host_pressure_retries','host_pressure_reclaimed_tokens','host_pressure_write_wait_s',
                'overlay_transfers_in','overlay_transfers_read','overlay_write_returns_read','overlay_write_releases','overlay_backup_reuses')
        result = dict(staging_pressure=dict(self.staging_pressure),last_transfer_pressure=self.last_transfer_pressure,allocation_progress=dict(self.allocation_progress),staging_return_exempt_bytes=self.staging_return_exempt_bytes,staging_return_pending_pages=len(self._staging_return_flush),total_storage_submitted_bytes=self.ssd_backend.validation_write_bytes,prefix_materialization=self.staging_policy.prefix_write_stats(),unified_storage=self.storage.stats(),budget_exhausted_at_s=self.budget_exhausted_at_s,**{k:getattr(self,k) for k in fields},admission_policy=self.admission_policy,
                    materialization_timing=self.materialization_timing,write_budget_bytes=self.write_budget_bytes,
                    kv_bytes_per_token=self.kv_bytes_per_token,residency_policy=self.residency_policy,
                    candidate_scope='native_read_leaves_external_staging_head_tail',
                    read_tokens=read,staging_tokens=stage,read_limit_tokens=self.read_limit,
                    read_allocated_tokens=self.read_allocated(),read_available_tokens=self.read_available(),
                    physical_free_tokens=self._physical_available(),
                    staging_limit_tokens=self.staging_limit,pending_write_tokens=len(self.inflight)*self.cache.page_size,
                    configured_staging_limit_tokens=self.configured_staging_limit,
                    waiting_staging_tokens=self.waiting_staging_tokens(),
                    dynamic_capacity=self.capacity_controller is not None,
                    capacity_control=self.capacity_controller.stats() if self.capacity_controller else None,
                    residency_selections=self.staging_policy.selections,indexed_host_nodes=len(self._host_index.nodes),
                    residency_index_validations=getattr(self,'_validation_checks',0),
                    ssd_backend_tokens=len(getattr(self.cache.cache_controller.storage_backend,'storage',{}))*self.cache.page_size,
                    storage_active=self.storage_active(),lifecycle_admission=dict(**self.sli.stats(),**self.staging_policy.stats(),threshold=self.admission_threshold),
                    storage_execution='hisim_offline',performance_validated=False)
        result['storage_execution'] = 'native_sglang'
        result['regular_write_bytes'] = self.ssd_backend.regular_write_bytes
        result['transition_write_bytes'] = self.ssd_backend.transition_write_bytes
        return result


def install_cache_hooks(target):
    # Only hand off unpersisted victims; native I/O and event draining stay intact.
    original_backup = target.write_backup_storage
    original_reset = target.reset

    def wrapped_backup(self, node_id):
        policy = getattr(self, 'hohlraum_host', None)
        if policy is None or not policy.storage_active():
            return original_backup(self, node_id)
        # The host-victim callback transfers ownership to staging. Staging
        # admission later submits a native write_storage operation.
        return None

    def wrapped_reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        policy = getattr(self, 'hohlraum_host', None)
        if policy is not None:
            policy.reset()
        return result

    target.write_backup_storage = wrapped_backup
    target.reset = wrapped_reset


def install_storage_adapter(h, backend):
    """Native notifications only; ownership decisions belong to residency."""
    cache, controller, core = h.cache, h.cache.cache_controller, h.cache.tree_core
    pending_commits = None
    query = controller._storage_hit_query
    def lookup(operation, *args, **kwargs):
        keys, tokens = query(operation, *args, **kwargs)
        backend.begin_read(operation, keys[:tokens // cache.page_size])
        return keys, tokens
    controller._storage_hit_query = lookup
    if backend._threaded:
        page_get = controller.page_get_func
        def read_batch(operation, *args, **kwargs):
            # Source accounting only; native reads and TP ACK reduction are unchanged.
            with backend.read_context(operation):
                return page_get(operation, *args, **kwargs)
        controller.page_get_func = read_batch

    # Preserve original native insertion. No splitting, grafting or child requests.
    insert = core.insert_host
    h._native_insert_host = insert
    def inserted(node_id, key, payload, hashes):
        result = insert(node_id, key, payload, hashes)
        if not result.host_insert_dropped:
            if pending_commits is None:
                backend.confirm_read(hashes)
            else:
                pending_commits.extend(hashes)
        return result
    core.insert_host = inserted
    backup = core.commit_backup
    def committed(node_id, *args, **kwargs):
        result = backup(node_id, *args, **kwargs)
        backend.confirm_read(core.node_by_id(node_id).hash_value or [])
        return result
    core.commit_backup = committed

    handle = cache._handle_prefetch_result
    def result(operation):
        nonlocal pending_commits
        assert pending_commits is None
        pending_commits = []
        try:
            value = handle(operation)
            backend.confirm_read(pending_commits)
            h.ssd_prefetched_tokens += backend.record_read(operation)
            return value
        finally:
            pending_commits = None
            backend.finish_read(operation)
    cache._handle_prefetch_result = result

    for name in ['release_aborted_request', 'revoke_pending_prefetch']:
        original = getattr(cache, name)
        def released(handle, *args, _original=original, **kwargs):
            try:
                return _original(handle, *args, **kwargs)
            finally:
                backend.cancel_request(handle.rid)
        setattr(cache, name, released)


# SGLang: actual page copies and native lifecycle hooks.


@dataclass
class ReadLease:
    operation: object
    keys: list
    entries: dict


class StagingStorage:
    """Public staging + SSD abstraction shared by both runtimes."""

    @classmethod
    def create(cls, residency, backing, page_size, token_bytes, *, simulated=False):
        """Select execution once at attachment; consumers use the same interface."""
        implementation = _HiSimStorage if simulated else _SGLangStorage
        return implementation(residency, backing, page_size, token_bytes)

    def __getattr__(self, name):
        return getattr(self.backing, name)

    def ssd_exists(self, keys):
        return [self.backing.batch_exists([key], None) == 1 for key in keys]

    def batch_exists(self, keys, extra_info=None):
        for i, key in enumerate(keys):
            if not self.exists(key):
                return i
        return len(keys)

    def __init__(self, residency, backing, page_size, token_bytes):
        self.residency, self.backing = residency, backing
        self.page_size, self.token_bytes = page_size, token_bytes
        self.lock = residency.lock if self._threaded else nullcontext()
        self.leases = {}
        self.events = Counter()

    _threaded = False

    def put(self, entry):
        with self.lock:
            self.residency.accept_evicted(entry)

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        from sglang.srt.mem_cache.hicache_storage import PoolTransferResult
        # FULL-only geometry is checked at native attachment.
        return PoolTransferResult(kv_hit_pages=self.batch_exists(keys,extra_info), extra_pool_hit_pages={})

    def begin_read(self, operation, keys):
        with self.lock:
            if id(operation) in self.leases:
                return
            entries = {}
            for i, key in enumerate(keys):
                entry = self.residency.acquire_source(key)
                if entry is not None:
                    entries[i] = entry
            self.leases[id(operation)] = ReadLease(operation, list(keys), entries)
            if not self._threaded:
                self.events['requests'] += 1

    def confirm_read(self, keys):
        with self.lock:
            self.residency.confirm_read(keys)

    def finish_read(self, operation):
        with self.lock:
            lease = self.leases.pop(id(operation), None)
            if lease is not None:
                for entry in lease.entries.values():
                    self.residency.release_source(entry)

    def cancel_request(self, rid):
        with self.lock:
            for lease in list(self.leases.values()):
                handle = getattr(lease.operation, 'handle', None)
                if getattr(handle, 'rid', None) == rid or getattr(lease.operation, 'request_id', None) == rid:
                    self.finish_read(lease.operation)

    def release_request(self, rid):
        self.cancel_request(rid)

    def stats(self):
        with self.lock:
            return dict(self.events, live_leases=len(self.leases), **self.residency.read_ownership_stats())

    def reset(self):
        with self.lock:
            self.leases.clear()


class _SGLangStorage(StagingStorage):
    _threaded = True

    def __init__(self, residency, backing, page_size, token_bytes):
        super().__init__(residency, backing, page_size, token_bytes)
        self.pool = residency.cache.cache_controller.storage_host_pool
        self._read_local = threading.local()

    def begin_read(self, operation, keys):
        # Queries are hints, not ownership changes. In particular, background
        # query timing must not change the scheduler's eviction candidates.
        with self.lock:
            self.leases.setdefault(id(operation), ReadLease(operation, list(keys), {}))

    def finish_read(self, operation):
        with self.lock:
            self.leases.pop(id(operation), None)

    @contextmanager
    def read_context(self, operation):
        with self.lock:
            lease = self.leases.get(id(operation))
        previous = getattr(self._read_local, 'lease', None)
        self._read_local.lease = lease
        try:
            yield
        finally:
            self._read_local.lease = previous


    def exists(self, key):
        with self.lock:
            if self.residency.contains(key):
                return True
        return self.backing.batch_exists([key], None) == 1


    def _staged_page(self, key):
        with self.lock:
            entry = self.residency.stage.get(key)
            if entry is None:
                return None
            # Clone while protected; the returned CPU copy owns its memory.
            # No source address or eviction pin survives this critical section.
            data = self.pool.get_data_page(int(entry.payload[0])).clone()
            self.events['staging_read_bytes'] += data.numel() * data.element_size()
            return data

    def _read_plan(self, keys):
        # A page evicted since lookup falls back to SSD (or a normal miss).
        pages = [self._staged_page(key) for key in keys]
        lease = getattr(self._read_local, 'lease', None)
        if lease is not None:
            sources = {key: page is not None for key, page in zip(keys, pages)}
            with self.lock:
                for i, key in enumerate(lease.keys):
                    if key in sources:
                        if sources[key]:
                            lease.entries[i] = True
                        else:
                            lease.entries.pop(i, None)
        return pages

    def batch_get(self, keys, destinations=None, extra_info=None):
        result = self._read_plan(keys)
        i = 0
        while i < len(keys):
            if result[i] is not None:
                i += 1
                continue
            end = i + 1
            while end < len(keys) and result[end] is None:
                end += 1
            dst = None if destinations is None else destinations[i:end]
            pages = self.backing.batch_get(keys[i:end], dst)
            if pages is None:
                pages = [None] * (end-i)
            if len(pages) != end-i:
                raise RuntimeError('SSD backend returned invalid read result length')
            result[i:end] = pages
            self.events['ssd_read_bytes'] += sum(x is not None for x in pages) * self.page_size * self.token_bytes
            i = end
        return result

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        if len(host_indices) != len(keys) * self.page_size:
            raise ValueError('read destination must contain one full page per key')
        pages = self._read_plan(keys)
        success = [False] * len(keys)
        i = 0
        while i < len(keys):
            if pages[i] is not None:
                self.pool.set_from_flat_data_page(host_indices[i*self.page_size], pages[i])
                success[i] = True
                i += 1
                continue
            end = i + 1
            while end < len(keys) and pages[end] is None:
                end += 1
            # Zero staging is exactly one native SSD batch, with original
            # destinations, metadata, completion results and native timing.
            result = self.backing.batch_get_v1(
                keys[i:end], host_indices[i*self.page_size:end*self.page_size], extra_info)
            if len(result) != end-i:
                raise RuntimeError('SSD backend returned invalid read result length')
            success[i:end] = result
            self.events['ssd_read_bytes'] += sum(bool(x) for x in result) * self.page_size * self.token_bytes
            i = end
        return success


    def record_read(self, operation):
        with self.lock:
            lease = self.leases.get(id(operation))
            if lease is None:
                return 0
            n = int(operation.completed_tokens) // self.page_size
            return sum(self.page_size for i in range(n) if i not in lease.entries)


class BudgetStorage:
    """One per-rank logical write budget, shared by Eager and Hohlraum."""
    def __init__(self, backing, cap, page_bytes):
        self.backing = backing
        self.validation_cap = cap
        self.validation_write_bytes = 0
        self.regular_write_bytes = 0
        self.transition_write_bytes = 0
        self.transition_keys = set()
        self.page_bytes = page_bytes
        self.lock = threading.RLock()

    def __getattr__(self, name):
        return getattr(self.backing, name)

    def budget_exhausted(self, reserve=0):
        with self.lock:
            return self.validation_cap is not None and self.regular_write_bytes + self.page_bytes > self.validation_cap - reserve

    def allow_transition(self, key):
        with self.lock:
            self.transition_keys.add(key)

    def _charge(self, key):
        self.validation_write_bytes += self.page_bytes
        if key in self.transition_keys:
            self.transition_write_bytes += self.page_bytes
            self.transition_keys.discard(key)
        else:
            self.regular_write_bytes += self.page_bytes

    def batch_set(self, keys, data, extra_info=None):
        with self.lock:
            for key, page in zip(keys, data):
                if self.backing.batch_exists([key],None) == 1:
                    continue
                if key not in self.transition_keys and self.budget_exhausted():
                    return False
                if not self.backing.batch_set([key], [page], extra_info):
                    return False
                self._charge(key)
            return len(keys) == len(data)

    def batch_set_v1(self, keys, indices, extra_info=None):
        with self.lock:
            if not keys:
                return []
            page = len(indices)//len(keys)
            result = []
            for i,key in enumerate(keys):
                if self.backing.batch_exists([key],None) == 1:
                    result.append(True)
                    continue
                if key not in self.transition_keys and self.budget_exhausted():
                    result.append(False)
                    continue
                ok = bool(self.backing.batch_set_v1([key],indices[i*page:(i+1)*page],extra_info)[0])
                if ok:
                    self._charge(key)
                result.append(ok)
            return result


def _install_scheduler_hooks():
    from sglang.srt.managers.scheduler import Scheduler
    if getattr(Scheduler,'_hohlraum_installed',False):
        return
    original_input=Scheduler.process_input_requests
    original_next=Scheduler.get_next_batch_to_run
    original_result=Scheduler.process_batch_result
    original_run=Scheduler.run_batch
    original_abort=Scheduler._release_aborted_request
    def policy(s):
        return getattr(getattr(s,'tree_cache',None),'hohlraum_host',None)
    @wraps(original_input)
    def inputs(s,requests,*args,**kwargs):
        h=policy(s)
        if h is None:return original_input(s,requests,*args,**kwargs)
        with h.lock:
            tokenized = [item for request in requests for item in s._tokenized_requests(request)]
            h.on_requests(tokenized,h.now())
            return original_input(s,requests,*args,**kwargs)
    @wraps(original_next)
    def next_batch(s,*args,**kwargs):
        h=policy(s)
        if h is None:return original_next(s,*args,**kwargs)
        with h.lock:
            control=h.capacity_controller
            if control:control.scheduler_begin(h.now())
            h.maintain()
            plan=original_next(s,*args,**kwargs)
            if control:control.scheduler_end(plan.batch_to_run is not None,h.now())
            return plan
    @wraps(original_result)
    def result(s,batch,result,*args,**kwargs):
        h=policy(s)
        if h is None:return original_result(s,batch,result,*args,**kwargs)
        with h.lock:
            value=original_result(s,batch,result,*args,**kwargs)
            h.on_batch_complete(batch.reqs,h.now())
            if h.capacity_controller:h.capacity_controller.batch_completed(result)
            return value
    @wraps(original_run)
    def run(s,batch,*args,**kwargs):
        h=policy(s)
        if h is None or h.capacity_controller is None:return original_run(s,batch,*args,**kwargs)
        with h.lock:
            h.capacity_controller.batch_launch(h.now())
            value=original_run(s,batch,*args,**kwargs)
            h.capacity_controller.batch_submitted(value)
            return value
    @wraps(original_abort)
    def aborted(s,req,*args,**kwargs):
        h=policy(s)
        if h is None:return original_abort(s,req,*args,**kwargs)
        with h.lock:
            value=original_abort(s,req,*args,**kwargs)
            # Queued cancellation does not produce a batch-result callback.
            # Close the same lifecycle record only after native cleanup succeeds.
            h.sli.complete(req,h.now())
            return value
    Scheduler.process_input_requests=inputs
    Scheduler.get_next_batch_to_run=next_batch
    Scheduler.process_batch_result=result
    Scheduler.run_batch=run
    Scheduler._release_aborted_request=aborted
    Scheduler._hohlraum_installed=True


def attach(cache, filename):
    simulated = bool(os.environ.get('SGLANG_SIMULATOR_CONFIG_PATH'))
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    cfg=json.loads(Path(filename).read_text())
    if not cache.enable_storage or cache.host_memory_mode!='cache':
        raise ValueError('Hohlraum requires native storage and host cache mode')
    if set(cache.components)!={ComponentType.FULL} or not isinstance(cache.tree_core,UnifiedTreeCore):
        raise ValueError('Initial native implementation requires FULL component and Python tree')
    controller=cache.cache_controller
    pool=controller.mem_pool_host.anchor_entry.host_pool
    if simulated:
        from sglang_simulator.simulation.sglang.mem_pool_host import get_refined_cache_size_per_token
        token_bytes=int(get_refined_cache_size_per_token(pool))
    else:
        page_data=controller.storage_host_pool.get_data_page(0)
        token_bytes=page_data.numel()*page_data.element_size()//cache.page_size
    total=int(pool.size)
    world=cache.tp_world_size
    cap=cfg.get('write_budget_bytes')
    cap=None if cap is None else int(cap)//world
    controller.storage_backend=BudgetStorage(controller.storage_backend,cap,cache.page_size*token_bytes)
    cache.hohlraum_host=None
    if not cfg.get('enabled',True):return
    if cache.cache_controller.write_policy not in ('write_back', 'write_through'):
        raise ValueError('Native Hohlraum requires write_back or write_through')
    for key in ('HOHLRAUM_JOINT_MODELS','HOHLRAUM_MODEL_COUNT','HOHLRAUM_ONLINE_PRIOR',
                'HOHLRAUM_BRANCH_PRIOR','HOHLRAUM_LENGTH_CURVES'):
        if not os.environ.get(key):
            raise ValueError(f'Native lifecycle policy requires {key}')
    if not Path(cfg['history_library']).is_file():
        raise ValueError('history_library does not exist')
    cfg['kv_bytes_per_token']=token_bytes
    cfg.setdefault('reference_history',True)
    cfg.setdefault('admission_policy','agent')
    cfg.setdefault('residency_policy','joint_hts')
    cfg.setdefault('duration_s',14400)
    cfg.setdefault('threshold',0.7)
    stage=int(float(cfg.get('staging_GB',30))*1e9/world/token_bytes)//cache.page_size*cache.page_size
    if cfg.get('stats_path'):
        cfg['stats_path']=cfg['stats_path']+f'.rank{controller.storage_config.tp_rank}.json'
    cls=type(cache)
    if not getattr(cls,'_hohlraum_cache_hooks',False):
        install_cache_hooks(cls)
        cls._hohlraum_cache_hooks=True
    if simulated:
        host_class = HiSimHost
    else:
        host_class = NativeHost
    cache.hohlraum_host=host_class(cache,cfg,total,stage)
    _install_scheduler_hooks()


# HiSim: source metadata and binding to native simulation time.


class _HiSimStorage(StagingStorage):
    def __init__(self, residency, backing, page_size, token_bytes):
        super().__init__(residency, backing, page_size, token_bytes)
        self._ssd_spans = {}


    def exists(self, key):
        return self.residency.contains(key) or self.backing.exists(key)


    def batch_get_v2(self, transfers, extra_info=None):
        # HiSim carries metadata rather than KV payloads. Transfer timing and
        # source ownership are handled by the operation's lease, not this query.
        return {transfer.name: [self.exists(self.backing._pool_storage_key(key, transfer.name))
                                for key in transfer.keys or []] for transfer in transfers}


    def read_source_ranges(self, operation):
        # Half-open token ranges; the lookup lease pins the source identity.
        lease = self.leases[id(operation)]
        ranges = []
        for i in range(len(lease.keys)):
            source = 'host_copy' if i in lease.entries else 'storage'
            start, end = i*self.page_size, (i+1)*self.page_size
            if ranges and ranges[-1][2] == source:
                ranges[-1] = (ranges[-1][0], end, source)
            else:
                ranges.append((start, end, source))
        return tuple(ranges)

    def ssd_spans(self, rid):
        """Stable list shared with the request's statistics adapter."""
        return self._ssd_spans.setdefault(rid, [])

    def record_read(self, operation):
        lease = self.leases.get(id(operation))
        if lease is None:
            return 0
        spans = self.ssd_spans(operation.handle.rid)
        tokens = 0
        for i in range(int(operation.completed_tokens) // self.page_size):
            if i not in lease.entries:
                start = operation.storage_start + i * self.page_size
                spans.append((start, start + self.page_size))
                tokens += self.page_size
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        spans[:] = merged
        return tokens

    def release_request(self, rid):
        super().release_request(rid)
        self._ssd_spans.pop(rid, None)


    def reset(self):
        super().reset()
        self._ssd_spans.clear()


# Lifecycle/ownership binding only. HiSim owns all execution and timing.


class HiSimHost(NativeHost):
    def __init__(self, cache, cfg, total_tokens, staging_tokens):
        from sglang_simulator.simulation.manager import StateManager
        super().__init__(cache, cfg, total_tokens, staging_tokens,
                         clock=StateManager.get_global_clock,
                         storage_factory=partial(StagingStorage.create, simulated=True))

    def _drive_host_eviction(self, required_free, tracker, device_frees, host_frees):
        if self._in_drive:
            return
        super()._drive_host_eviction(required_free, tracker, device_frees, host_frees)
        if not self.inflight:
            return
        # HiSim's existing storage worker completes writes without advancing
        # virtual time. Service newly queued writes before a synchronous host
        # allocation mistakes their retained source pages for a hard shortfall.
        controller = self.cache.cache_controller
        before = self._physical_available()
        controller.handle_backup_operation()
        # Do not consume ACKs here: an enclosing native drain may already have
        # snapshotted their count. Leave each ACK for native bookkeeping, but
        # settle our own completed receipt once so its source is allocatable.
        with controller.ack_backup_queue.mutex:
            completed = [op.id for op in controller.ack_backup_queue.queue
                         if op.id in self.inflight]
        for operation_id in completed:
            receipt = self.cache.ongoing_backup.pop(operation_id, None)
            if receipt is not None:
                self.cache.dec_host_lock_ref(*receipt)
        freed = self._physical_available() - before
        if freed:
            from sglang.srt.mem_cache.unified_cache.components import ComponentType
            tracker[ComponentType.FULL] += freed

    def _advance_admission(self):
        from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook
        # Native OFFLINE profiling resets its clock on stop. Idle callbacks
        # after that boundary must not advance the policy's completed epoch.
        # A reversal during active replay remains an error in the shared policy.
        if not C_SchedulerHook.REQ_DISPATCHER.profile_active:
            return False
        from sglang_simulator.simulation.manager import StateManager
        # Native benchmark uses START_PROFILE to export final metrics too.
        # It resets clock/iteration to zero while profile_active stays true.
        if (self.sli.observed_requests and self.sli.now > 0
                and self.now() == 0 and StateManager.get_iteration() == 0):
            return False
        return super()._advance_admission()

    def stats(self):
        result = super().stats()
        result['storage_execution'] = 'hisim_native_source_routes'
        result['performance_validated'] = False
        return result
