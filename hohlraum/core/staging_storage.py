"""Staging + SSD storage: page ownership, data movement and runtime adapters.

StagingPolicy owns decisions; native runtimes retain scheduling and I/O timing.
"""
from __future__ import annotations
from contextlib import nullcontext, contextmanager
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import heapq
from pathlib import Path
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from hohlraum.core.sli import SLI
from hohlraum.core.read_pressure import install_prefill_observer, observe_physical_wait
from hohlraum.core.staging_policy import StagingPolicy
import os
import threading
import time
from functools import wraps, partial


def native_completion_timestamp(req, fallback):
    """Read the scheduler's completion timestamp, captured before output."""
    completed = getattr(getattr(req, 'time_stats', None), 'completion_time', 0.)
    return completed if completed > 0 else fallback


class HostPayloadIndex:
    def __init__(self, core, component_type):
        self.core, self.ct = core, component_type
        self.nodes = {}
        self.tokens = {}
        self.host_keys = Counter()
        self.native_live_keys = Counter()
        self._native_live_keys = {}
        self.key_nodes = {}
        self._keys = {}
        self.total_tokens = 0
        self.on_payload_change = None
        self.on_key_membership_change = None
        self._batch_depth = 0
        self._batch_nodes = {}
        for node in tuple(core._node_arena.values()):
            self.sync(node)

    def forget(self, node):
        nid = node.id
        self._batch_nodes.pop(nid, None)
        self._sync_native_live(node, removed=True)
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

    def _sync_native_live(self, node, removed=False):
        # Mirror the old arrival predicate, including HBM-only native nodes.
        keys = (() if removed or node is self.core.root_node or getattr(node, 'evicted', True)
                else tuple(node.hash_value or ()))
        previous = self._native_live_keys.get(node.id, ())
        if previous == keys:
            return
        changes = Counter(keys)
        changes.subtract(previous)
        for key, delta in changes.items():
            value = self.native_live_keys.get(key, 0) + delta
            if value:
                self.native_live_keys[key] = value
            else:
                self.native_live_keys.pop(key, None)
        if keys:
            self._native_live_keys[node.id] = keys
        else:
            self._native_live_keys.pop(node.id, None)

    def sync(self, node):
        if node is None:
            return
        if self._batch_depth:
            self._batch_nodes[node.id] = node
            return
        self._sync_native_live(node)
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

    def begin_batch(self):
        self._batch_depth += 1

    def end_batch(self):
        assert self._batch_depth > 0
        self._batch_depth -= 1
        if not self._batch_depth:
            pending, self._batch_nodes = self._batch_nodes, {}
            self._sync_batch(pending)

    def _sync_batch(self, pending):
        # A split usually only changes which node owns the same prefix pages.
        # Coalesce aggregate membership before emitting any adapter callbacks.
        # This avoids removing/re-adding every prefix in the global counters.
        prepared = []
        old_host, new_host, old_live, new_live = [], [], [], []
        for nid, node in pending.items():
            value = node.component_data[self.ct].host_value
            tokens = 0 if value is None else value.numel()
            keys = tuple(node.hash_value or ()) if tokens else ()
            live = (() if node is self.core.root_node or getattr(node, 'evicted', True)
                    else tuple(node.hash_value or ()))
            previous = self._keys.get(nid, ())
            old_host.extend(previous);new_host.extend(keys)
            old_live.extend(self._native_live_keys.get(nid, ()));new_live.extend(live)
            prepared.append((nid,node,tokens,previous,keys,live))
        for nid, node, tokens, previous, keys, live in prepared:
            self.total_tokens += tokens-self.tokens.get(nid,0)
            if tokens:self.nodes[nid]=node;self.tokens[nid]=tokens
            else:self.nodes.pop(nid,None);self.tokens.pop(nid,None)
            if previous != keys:
                before, after = set(previous), set(keys)
                for key in before-after:
                    self.key_nodes[key].discard(nid)
                    if not self.key_nodes[key]:del self.key_nodes[key]
                for key in after-before:self.key_nodes.setdefault(key,set()).add(nid)
            if keys:self._keys[nid]=keys
            else:self._keys.pop(nid,None)
            if live:self._native_live_keys[nid]=live
            else:self._native_live_keys.pop(nid,None)
        changed=[]
        for counter, before, after, notify in [
                (self.host_keys,old_host,new_host,True),
                (self.native_live_keys,old_live,new_live,False)]:
            if before==after:continue
            delta=Counter(after);delta.subtract(before)
            for key, difference in delta.items():
                if not difference:continue
                previous=counter.get(key,0);current=previous+difference
                assert current>=0
                if current:counter[key]=current
                else:counter.pop(key,None)
                if notify and bool(previous)!=bool(current):changed.append(key)
        if self.on_key_membership_change is not None:
            for key in changed:self.on_key_membership_change(key)
        if self.on_payload_change is not None:
            for nid, *_ in prepared:self.on_payload_change(nid)

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
        # Storage may be enabled after an initial HBM-only phase. Backfill is
        # the native bulk hash mutation that does not emit leaf-set updates.
        backfill = getattr(core, 'backfill_missing_hash_values', None)
        if backfill is not None:
            def backfill_and_sync(*args, **kwargs):
                result = backfill(*args, **kwargs)
                for node in tuple(core._node_arena.values()):
                    self.sync(node)
                return result
            core.backfill_missing_hash_values = backfill_and_sync


def return_waiting_to_read(h):
    core=h.cache.tree_core;page=h.cache.page_size
    entries=list(h.stage.values())
    if not entries:
        h._budget_return_pending=False
        h._budget_return_started=True
        return
    before=dict(waiting_pages=len(entries),read_tokens=h.read_allocated(),read_limit=h.read_limit,
                physical_free=h._physical_available(),budget=h.budget_charged_bytes,time=h.now())
    recency={k:n.last_access_time for n in core._node_arena.values() for k in n.hash_value or ()}
    # A staged page can still have an HBM node with a more recent access.
    # Returning its host allocation must preserve that native LRU priority.
    for entry in entries:
        key=entry.hash_value[0]
        staged_access=entry.last_access_time
        recency[key]=max(staged_access,recency.get(key,staged_access))
    ordered=sorted(entries,key=lambda e:len(e.prefix_hashes))
    # Returning a role is not an access and must not trigger another eviction
    # while this ownership transaction is in progress.
    was_driving=h._in_drive;h._in_drive=True
    restored=[];blocked=[];deferred=set()
    try:
        for e in ordered:
            if e.owner!='staging':continue
            parent=e.prefix_hashes[-1] if e.prefix_hashes else None
            # Restore the quota immediately, but never transfer/free a source
            # still owned by a read or write. Retry these pages after its ACK.
            if (e.read_pins or e.operation is not None
                    or getattr(e,'copy_committed',False) or parent in deferred):
                deferred.add(e.hash_value[0])
                continue
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
        # Account for blocked ownership returns as cutover writes. Keep each
        # existing allocation until the native write acknowledgement.
        h._staging_return_flush.update(x['key'] for x in blocked)
        for e in ordered:
            if e.hash_value[0] in h._staging_return_flush and e.operation is None and not h._has_ssd_copy(e):
                h._submit_stage_write(e,transition=True)
        after=dict(restored_pages=len(restored),blocked=blocked,read_tokens=h.read_allocated(),
                   physical_free=h._physical_available(),waiting_pages=h.waiting_staging_tokens()//page,
                   deferred_pages=len(deferred))
        h._budget_return_pending=bool(deferred)
        record=dict(before=before,after=after)
        h.staging_return_records=getattr(h,'staging_return_records',[])+[record]
        if h.stats_path:
            with Path(h.stats_path).with_suffix('.staging-return.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        assert after['physical_free']==before['physical_free'], ('ownership_return_allocated_or_freed',record)
        assert all(e.hash_value[0] in h._staging_return_flush or e.hash_value[0] in deferred
                   for e in h.stage.values()),record
        assert h.read_allocated()<=h.read_limit,record
        h._budget_return_started=True
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


class NativeHost:
    def on_requests(self, requests, now):
        if self.lifecycle is not None or self.sli is not None:
            for req in requests:
                if req.__class__.__name__ in ("TokenizedGenerateReqInput", "BatchTokenizedGenerateReqInput"):
                    self.observe_request(req, now)

    def on_batch_complete(self, requests, now, *, defer_completion=False):
        if self.sli is not None:
            for req in requests:
                if len(req.output_ids) != 0:
                    self.sli.release_read(req.rid)
                if req.finished() and not defer_completion:
                    completed = now
                    if getattr(self, '_native_wall_clock', False):
                        completed = native_completion_timestamp(req, self.started_at + now) - self.started_at
                    self.sli.complete(req, completed)

    def __init__(self, cache, cfg, total_tokens, staging_tokens, *, clock=None, storage_factory=None):
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        self._native_wall_clock = clock is None
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
        self._budget_return_pending=False
        self._budget_return_started=False
        self._write_cutover_pages=0
        self.admission_policy=cfg.get('admission_policy','write_all')
        if cfg.get('materialization_timing','on_evict')!='on_evict':raise ValueError('Hohlraum supports staging-exit admission only')
        self.residency_policy=cfg.get('residency_policy','lru')
        if self.admission_policy not in ('agent','write_all','write_none'):raise ValueError('invalid admission')
        if self.residency_policy not in ('lru','joint_hts'):raise ValueError('overlay supports native LRU or paper HTS')
        if not cfg.get('reference_history', True):raise ValueError('overlay requires the canonical reference history')
        self.admission_threshold=float(cfg.get('threshold',.7))
        self.kv_bytes_per_token=int(cfg['kv_bytes_per_token'])
        self.stats_path=Path(stats_path) if stats_path else None
        self.component=cache.components[ComponentType.FULL]
        self.stage={};self._next_stage_id=-1;self.inflight={}
        self._waiting_stage_keys = set()
        self._prefix_read_release = {}
        self.admission_observed_rids=set();self.request_hashes={}
        self.budget_exhausted_at_s=None
        self._next_budget_snapshot_s=0.
        self._next_stats_snapshot_s=0.
        self.staging_pressure=dict(discard_on_entry=0,pre_retire=0,wait_existing_write=0,no_read_candidate=0,no_stage_candidate=0,no_retire_progress=0,transfer_attempts=0,transfer_succeeded=0,transfer_failed=0)
        self.last_transfer_pressure=None
        self._in_drive=False
        self.sli=SLI(self,cfg)
        self.lifecycle=self.sli if self.admission_policy=='agent' else None
        self.staging_policy=StagingPolicy(self)
        self.capacity_controller=None
        self.budget_consensus=None
        self._host_index=HostPayloadIndex(cache.tree_core,ComponentType.FULL);self._host_index.bind()
        self._bind_resident_index()
        fields=('ssd_written_tokens','ssd_write_failures','ssd_prefetched_tokens','staging_exits',
                'read_releases','duplicate_reclaimed_tokens','demotions','promotions','peak_read_tokens',
                'peak_staging_tokens','over_limit_events','admission_write_decisions','admission_drop_decisions',
                'discarded_tokens',
                'overlay_transfers_in','overlay_transfers_read','overlay_write_returns_read',
                'overlay_write_releases','overlay_backup_reuses')
        for name in fields:setattr(self,name,0)
        self._original_drive_host_eviction=self.component.drive_host_eviction
        import inspect
        self._native_cache_action_api = 'cache_actions' in inspect.signature(cache.tree_core._evict_host_leaf).parameters
        self._native_leaf_callback_api = 'leaf_handler' in inspect.signature(self._original_drive_host_eviction).parameters
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
        elif (not os.environ.get('SGLANG_SIMULATOR_CONFIG_PATH')
              and cache.tp_world_size>1 and cfg.get('write_budget_bytes') is not None):
            from hohlraum.core.capacity_controller import native_budget_consensus
            self.budget_consensus=native_budget_consensus(self)
        query = cache.cache_controller._storage_hit_query
        @wraps(query)
        def pinned_query(*args, **kwargs):
            with self.lock:
                return query(*args, **kwargs)
        cache.cache_controller._storage_hit_query = pinned_query

    def contains(self, key):
        return key in self.stage

    def accept_evicted(self, entry):
        assert len(entry.hash_value) == 1 and entry.payload.numel() == self.cache.page_size
        key = entry.hash_value[0]
        assert key not in self.stage and entry.owner == 'staging'
        self.stage[key] = entry
        self._staging_changed(key)
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
        if not entry.read_pins and entry.read_pin_lock is not None:
            self.cache.dec_host_lock_ref(*entry.read_pin_lock)
            entry.read_pin_lock = None
        self._release_transferred_source(entry)
        if (not entry.read_pins and entry.operation is None
                and getattr(entry, 'fallback_write_committed', False)
                and self.stage.get(entry.hash_value[0]) is entry):
            self._remove_stage(entry, free=True)
            self.overlay_write_releases += 1
            self.staging_exits += 1

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
        # Write-through backing belongs to the logical (staging + SSD) tier,
        # rather than requiring its copy to remain in read DRAM. Releasing a
        # read copy can therefore leave an HBM parent with host-only children.
        # Preserve that structural parent when evicting HBM, instead of native
        # WT's whole-node deletion. This adapts backing ownership to the tier
        # abstraction; it is not a workaround for staging prefetch failures.
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
                self._native_mutation('_evict_component_and_detach_lru',
                    node, trigger, target=EvictLayer.DEVICE,
                    tracker=result.tracker, device_frees=result.device_frees,
                    host_frees=result.host_frees, cache_actions=getattr(result, 'cache_actions', None))
                self._native_mutation('_cascade_evict', node, trigger, result.tracker,
                    device_frees=result.device_frees, host_frees=result.host_frees,
                    cache_actions=getattr(result, 'cache_actions', None))
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

    def _nodes(self):
        yield from (n for n in self.cache.tree_core._node_arena.values() if n is not self.cache.tree_core.root_node)
        yield from self.stage.values()

    def waiting_staging_tokens(self):
        return len(self._waiting_stage_keys)*self.cache.page_size

    def _staging_changed(self, key):
        entry = self.stage.get(key)
        if entry is not None and entry.operation is None:
            self._waiting_stage_keys.add(key)
        else:
            self._waiting_stage_keys.discard(key)
        self.staging_policy.waiting_changed(key)

    def _bind_resident_index(self):
        self._host_index.on_key_membership_change = self._resident_key_changed
        for key in self.resident_keys():
            self.staging_policy.index.set_resident(key, True)

    def _resident_key_changed(self, key):
        self.staging_policy.index.set_resident(key, key in self.stage or key in self._host_index.host_keys)

    def has_resident_descendant(self, key):
        return self.staging_policy.index.has_resident_descendant(key)

    def resident_keys(self): return set(self._host_index.host_keys)|set(self.stage)
    def _copy_bits(self,keys):
        # Independent per-key bits use the native single-key predicate.
        exists=self.ssd_backend.exists
        return [exists(key) for key in keys]
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
            assert self._waiting_stage_keys == {k for k, e in self.stage.items() if e.operation is None}
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
            assert not native.intersection(e.id for e in self.stage.values())
            assert all(e.owner=='staging' and e.payload.numel()==self.cache.page_size for e in self.stage.values())

    def _hbm_read_leaf(self, node):
        """WT read-copy candidate; the native Full host/DMA guards apply."""
        # The (staging + SSD) abstraction assumes the write-through backing
        # role: a read copy may move to staging, or be released if SSD-backed,
        # even while HBM retains its copy. Staging policy then controls retention
        # and SSD admission. This is an ownership rule, not a prefetch-failure
        # optimization; native lock and child-structure constraints still apply.
        core = self.cache.tree_core
        return (self.cfg.get('stage_hbm_leaves', False)
                and not self._budget_exhausted()
                and not core.is_write_back
                and (not node.children or (
                    self.cfg.get('stage_hbm_dram_leaves', False)
                    and all(child.component_data[ComponentType.FULL].host_value is None
                            for child in node.children.values())))
                and core._can_reclaim_full_host_duplicate(node)
                # Native duplicate reclamation may release Full host payloads
                # while HBM is in use. Keep actual host/DMA locks and every aux
                # pool's lock; the HBM data, locks and tree node remain intact.
                and all(cd.host_lock_ref == 0 and
                        (ct == int(ComponentType.FULL) or cd.lock_ref == 0)
                        for ct, cd in enumerate(node.component_data))
                and all(cd.host_value is None for ct, cd in enumerate(node.component_data)
                        if ct != int(ComponentType.FULL)))

    def _native_mutation(self, name, *args, cache_actions=None, **kwargs):
        """Preserve native eviction actions across the old/new tree APIs."""
        method = getattr(self.cache.tree_core, name)
        if not getattr(self, '_native_cache_action_api', False):
            return method(*args, **kwargs)
        actions = cache_actions
        if actions is None:
            actions = getattr(self, '_eviction_cache_actions', None)
        own_actions = actions is None
        if own_actions:
            actions = []
        result = method(*args, cache_actions=actions, **kwargs)
        if own_actions and actions:
            self.cache._apply_cache_actions(actions)
        return result

    def _drive_read_leaves(self, num_tokens, tracker, device_frees, host_frees,
                           *, leaf_handler, should_stop):
        extended = (self.cfg.get('stage_hbm_leaves', False)
                    and not self._budget_exhausted() and not self.cache.tree_core.is_write_back)
        if not extended and getattr(self, '_native_leaf_callback_api', True):
            return self._original_drive_host_eviction(num_tokens, tracker, device_frees,
                host_frees, leaf_handler=leaf_handler, should_stop=should_stop)
        core = self.cache.tree_core
        self.component._ensure_eviction_strategy()
        candidates = set(core.evictable_host_leaves)
        if extended:
            candidates.update(n for n in core.full_host_duplicates.values() if self._hbm_read_leaf(n))
        heap = [(self.component.session_ref_eviction_strategy(n), n) for n in candidates]
        heapq.heapify(heap)
        while heap and not should_stop():
            _, node = heapq.heappop(heap)
            duplicate = extended and self._hbm_read_leaf(node)
            if node not in core.evictable_host_leaves and not duplicate:
                continue
            handled = leaf_handler(node)
            if handled == 'skip_candidate':
                # Retirement may pin/split this victim. Other heap candidates
                # remain eligible; this is not a global capacity stop.
                continue
            if handled is False:
                break
            if handled is None:
                if duplicate:
                    if node.component_data[ComponentType.FULL].lock_ref:
                        self._read_ownership['locked_hbm_host_releases'] += 1
                        self._read_ownership['locked_hbm_host_released_tokens'] += int(node.component_data[ComponentType.FULL].host_value.numel())
                    self._native_mutation('_release_full_host_duplicate', node, tracker, device_frees, host_frees)
                    core.full_host_duplicates.pop(node.id, None)
                    core._update_evictable_leaf_sets(node)
                else:
                    self._native_mutation('_evict_host_leaf', node, tracker, device_frees, host_frees)
            # Host removal can expose a DRAM leaf while the HBM subtree remains.
            # Recheck eligibility on pop too: a new host child must block it.
            if node.parent is not None and (node.parent in core.evictable_host_leaves
                                             or (extended and self._hbm_read_leaf(node.parent))):
                heapq.heappush(heap, (self.component.session_ref_eviction_strategy(node.parent), node.parent))

    def _take_read_leaf(self,node):
        """Native leaf deletion, transferring its host allocation instead of freeing it."""
        core=self.cache.tree_core;page=self.cache.page_size
        assert node in core.evictable_host_leaves or self._hbm_read_leaf(node)
        if self._tokens(node)>page:
            # Ownership transfer is not a read access. Native splitting assigns
            # fresh timestamps; preserve the pre-eviction recency on both pieces.
            prior_access=node.last_access_time
            # Native splitting notifies several leaf/duplicate hooks for the
            # same nodes. Publish our final payload index once, before any
            # adapter consumer runs; native tree/LRU updates remain immediate.
            index = self._host_index
            index.begin_batch()
            try:
                prefix,action=core._split_node(node.key,node,self._tokens(node)-page)
            finally:
                index.end_batch()
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
            if node.component_data[ComponentType.FULL].lock_ref:
                self._read_ownership['locked_hbm_staging_transfers'] += 1
                self._read_ownership['locked_hbm_staging_transferred_tokens'] += page
            self._native_mutation('_release_full_host_duplicate',node,tracker,device,host)
            core.full_host_duplicates.pop(node.id,None)
            core._update_evictable_leaf_sets(node)
            self._read_ownership['hbm_leaf_transfers'] += 1
            if node.children:self._read_ownership['hbm_internal_transfers'] += 1
            assert core.node_by_id(node.id) is node and not node.evicted
        else:
            self._native_mutation('_evict_host_leaf',node,tracker,device,host)
        # The native deletion owns all tree/index changes. Its one host free
        # is handed to the overlay, never passed to the allocator's free().
        assert tracker[ComponentType.FULL]==page and not any(device.values())
        assert sum(x.numel() for x in host[ComponentType.FULL])==page
        return entry

    def _remove_stage(self,entry,free=False):
        key=entry.hash_value[0]
        assert self.stage.get(key) is entry
        if self.capacity_controller:self.capacity_controller.removed(entry)
        self.stage.pop(key)
        if entry.operation is None:
            self.ssd_backend.release_transition(key)
        self._staging_changed(key)
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

    @property
    def write_budget_bytes(self):
        return self.ssd_backend.validation_cap

    @property
    def budget_charged_bytes(self):
        return self.ssd_backend.regular_write_bytes

    def _reserve_write_pages(self, keys):
        if self.ssd_backend.reserve_pages(keys):
            self._write_cutover_pages=0
            return True
        self._write_cutover_pages=max(self._write_cutover_pages,len(set(keys)))
        return False

    def _unreserved_budget_short(self):
        needed=getattr(self,'_write_cutover_pages',0)
        available=self.ssd_backend.remaining_unreserved_pages() if needed else None
        return available is not None and needed > available

    def _budget_exhausted(self):
        exhausted = self.budget_exhausted_at_s is not None or getattr(self, '_synced_budget_exhausted', False)
        if not hasattr(self, '_synced_budget_exhausted'):
            exhausted = (exhausted or self._unreserved_budget_short()
                         or self.ssd_backend.admission_budget_exhausted())
        if exhausted and self.budget_exhausted_at_s is None:
            self.budget_exhausted_at_s = self.now()
            # Admission-only can still own a page while its prefix write is
            # waiting for credits. Its zero staging quota does not exempt it
            # from the scheduler's ownership handoff.
            self._budget_return_pending=bool(self.staging_limit or self.stage)
        if exhausted and self.staging_limit:
            # The same verdict disables extended eviction and restores read
            # quota. Pinned physical pages remain allocated until safe to move.
            self.staging_limit=0
            self.read_limit=self.total_tokens
        return exhausted


    def _retire_staging(self,entry,*,rejected_on_entry=False):
        if entry.operation is not None or entry.read_pins:return 0
        copied=self._has_ssd_copy(entry)
        if entry.hash_value[0] in self._staging_return_flush:
            if copied:
                self._remove_stage(entry,free=True);self.staging_exits+=1
                return self.cache.page_size
            return self._submit_stage_write(entry,transition=True)
        decision=self.staging_policy.retire(entry,copied,rejected_on_entry=rejected_on_entry)
        if decision.retry or decision.budget_exhausted:
            # A cutoff never retires untouched staging ownership. The next
            # scheduler maintenance pass returns these pages or queues their
            # separately accounted fallback writes.
            return 0
        if not decision.write:
            if not copied:
                self.admission_drop_decisions+=1;self.discarded_tokens+=self.cache.page_size
            if self.capacity_controller and not copied:
                self.capacity_controller.discarded(entry,decision.budget_exhausted)
            self._remove_stage(entry,free=True);self.staging_exits+=1
            return self.cache.page_size
        return self._submit_stage_write(entry)

    def _pending_prefix_read_releases(self):
        """Predict frees for submission control only; never allocator capacity.

        Reuse native host-leaf and WT duplicate ownership rules. Other locks,
        DMA, aux state and children outside this plan stay resident.
        Only this adapter's in-flight write pins may be discounted in a forecast.
        """
        plan = getattr(self, '_prefix_read_release', {})
        if not plan or self._budget_exhausted():return {}
        core = self.cache.tree_core
        if core.is_write_back or not self.cfg.get('stage_hbm_leaves', False):return {}
        pins = Counter(e.native_lock[0] for e in self.inflight.values()
                       if e.owner == 'read' and e.native_lock)
        pending = self.staging_policy._pending_prefix_keys()
        nodes = {}
        for nid, (payload, age, hashes) in tuple(plan.items()):
            node = core._node_arena.get(nid)
            if (node is None or node.component_data[ComponentType.FULL].host_value is not payload
                    or node.last_access_time != age or tuple(node.hash_value or ()) != hashes):
                plan.pop(nid, None)
                continue
            nodes[nid] = node
        result = {}; visited = set()
        def eligible(node):
            if node.id in result:return True
            if node.id in visited:return False
            visited.add(node.id)
            cd = node.component_data[ComponentType.FULL]
            if (node is core.root_node or cd.host_value is None
                    or node.write_through_pending_id is not None
                    or node.load_back_pending_id is not None
                    or cd.host_lock_ref != pins[node.id]
                    or any(x.host_lock_ref or x.lock_ref or x.host_value is not None
                           for ct, x in enumerate(node.component_data) if ct != int(ComponentType.FULL))):
                return False
            if cd.value is None and (cd.lock_ref or any(x.value is not None for x in node.component_data)):
                return False
            if any(not stored and key not in pending
                   for key, stored in zip(node.hash_value, self._copy_bits(node.hash_value))):return False
            if node.children:
                if cd.value is None:
                    # Native host-only eviction deletes leaves. A parent can
                    # follow only if each structural child will also be deleted.
                    if any(child.id not in nodes or not eligible(child)
                           or child.component_data[ComponentType.FULL].value is not None
                           for child in node.children.values()):return False
                else:
                    if not self.cfg.get('stage_hbm_dram_leaves', False):return False
                    for child in node.children.values():
                        if child.component_data[ComponentType.FULL].host_value is not None:
                            if child.id not in nodes or not eligible(child):return False
            result[node.id] = node
            return True
        for node in nodes.values():eligible(node)
        return result

    def _release_written_read_prefixes(self):
        """On ACK, recheck native guards; retain HBM trees, clean host-only leaves natively."""
        plan = getattr(self, '_prefix_read_release', {})
        if not plan:return
        while plan:
            progress = False
            for nid, node in tuple(self._pending_prefix_read_releases().items()):
                core = self.cache.tree_core
                duplicate = self._hbm_read_leaf(node)
                if not duplicate and not core._is_host_leaf(node):continue
                if not all(self._copy_bits(node.hash_value)):continue
                tracker = defaultdict(int); device = defaultdict(list); host = defaultdict(list)
                if duplicate:
                    self._native_mutation('_release_full_host_duplicate', node, tracker, device, host)
                    core.full_host_duplicates.pop(nid, None)
                    core._update_evictable_leaf_sets(node)
                else:
                    self._native_mutation('_evict_host_leaf', node, tracker, device, host)
                    self._read_ownership['prefix_read_host_only_released_nodes'] += 1
                assert not any(device.values()), 'prefix release must never free HBM'
                self.cache._drain_host_frees(host)
                self._read_ownership['prefix_read_released_nodes'] += 1
                self._read_ownership['prefix_read_released_tokens'] += tracker[ComponentType.FULL]
                plan.pop(nid, None); progress = True
            if not progress:break
        # Remaining locks/children or failed copies mean copy-only, not a
        # deferred right to evict this payload after an unrelated future access.
        if not any(e.owner == 'read' and e.native_lock for e in self.inflight.values()):
            self._read_ownership['prefix_read_copy_only_nodes'] += len(plan)
            plan.clear()

    def persist_prefixes(self, prefix_keys, *, head_key):
        """Persist ancestors in order; read payloads stay native and LRU is unchanged."""
        events = self.staging_policy._prefix_events
        missing = prefix_keys
        for retry in (False, True):
            pending = self.staging_policy._pending_prefix_keys()
            missing = [key for key, stored in zip(missing, self._copy_bits(missing))
                       if not stored and key not in pending]
            if any(key in self.stage for key in missing):
                return 'retry'
            if not retry:
                events['admitted_heads'] += 1
            sources = {}
            absent = []
            for key in missing:
                nodes = self._host_index.key_nodes.get(key)
                if nodes:
                    sources[key] = self._host_index.nodes[next(iter(nodes))]
                else:
                    absent.append(key)
            if not absent:
                break
            # A successful native backup is not proof that every requested hash
            # is now host-visible: native eviction may have reclaimed a payload.
            # Re-resolve once, then yield without discarding the staging tail.
            if retry:
                events['native_backup_not_visible_retries'] += 1
                return 'retry'
            # Only absent host payloads need the native device->host backup path.
            wanted=set(absent)
            device_nodes={k:n for n in self.cache.tree_core._node_arena.values()
                          for k in n.hash_value or () if k in wanted
                          and n.component_data[ComponentType.FULL].value is not None}
            unavailable=[k for k in absent if k not in device_nodes]
            if unavailable:
                events['unavailable_prefix_chains'] += 1
                raise RuntimeError(
                    "staging prefix invariant violated: required prefix is absent "
                    "from SSD, in-flight writes, staging, read and HBM; "
                    f"missing={unavailable}")
            seen=set()
            for key in absent:
                node=device_nodes[key]
                if node.id not in seen:
                    seen.add(node.id)
                    if not self.cache.backup_node_for_write_back(node.id):
                        events['native_backup_retries'] += 1
                        return 'retry'
            # Native backup can split nodes or change host ownership; resolve
            # sources afresh instead of keeping stale payload addresses.
        if not self._reserve_write_pages((*missing,head_key)):
            return 'retry'
        # Register whole native source nodes, once, before queuing page writes.
        # All of a node's pages must be SSD-backed/in-flight before it receives
        # any forecast credit; partial prefix copies cannot release a whole node.
        if self.cfg.get('stage_hbm_leaves', False) and self.staging_limit:
            plan = self._prefix_read_release
            for node in sources.values():
                cd = node.component_data[ComponentType.FULL]
                plan.setdefault(node.id, (cd.host_value, node.last_access_time, tuple(node.hash_value)))
        positions={key:i for i,key in enumerate(prefix_keys)}
        page = self.cache.page_size
        for key in missing:
            source = sources[key]
            # The write receipt retains the native I/O pin until storage ACK.
            offset = source.hash_value.index(key)
            page_slice = slice(offset * page, (offset + 1) * page)
            lock = (source.id, self.cache.inc_host_lock_ref(source.id).to_dec_params())
            entry = StagedPage(self._next_stage_id, source.key[page_slice], [key],
                               source.component_data[ComponentType.FULL].host_value[page_slice],
                               source.last_access_time, tuple(prefix_keys[:positions[key]]),
                               owner='read', native_lock=lock)
            self._next_stage_id -= 1
            try:
                self._submit_stage_write(entry)
            except BaseException:
                self.cache.dec_host_lock_ref(*lock)
                raise
            events['immediate_read_prefix_pages'] += 1
        return 'ready'

    def _submit_stage_write(self,entry,transition=False):
        assert entry.operation is None and not self._has_ssd_copy(entry)
        if transition:
            assert entry.hash_value[0] in self._staging_return_flush, 'fallback only for blocked staging return'
            if not self.ssd_backend.allow_transition(entry.hash_value[0]):
                raise RuntimeError('failed to classify a blocked staging return')
        entry.capacity_sample = self.capacity_controller.write_sample(entry) if self.capacity_controller and entry.owner=='staging' and not transition else None
        op=self.cache.cache_controller.write_storage(entry.payload,entry.key.raw_token_ids(),entry.hash_value,list(entry.prefix_hashes))
        entry.operation=op;self.inflight[op]=entry
        self._staging_changed(entry.hash_value[0])
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
                  waiting_staging_used_GB=self.waiting_staging_tokens()*self.kv_bytes_per_token/1e9,
                 meaning='Used budget reads the shared SSD successful-write counter; HiSim writes complete immediately.')
        path=Path(filename)
        temporary=path.with_suffix(path.suffix+'.pending')
        temporary.write_text(json.dumps(row,indent=2));temporary.replace(path)
        with path.with_suffix('.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')

    def _attach_write_ack(self):
        original=self.cache.dec_host_lock_ref
        def finish(node_id,params):
            self.staging_policy.invalidate_round()
            if node_id>=0:return original(node_id,params)
            entry=next((e for e in self.inflight.values() if e.id==node_id),None)
            if entry is None:raise RuntimeError('unknown staging write receipt')
            op=entry.operation;self.inflight.pop(op);entry.operation=None
            self._staging_changed(entry.hash_value[0])
            if self.capacity_controller and self._has_ssd_copy(entry):
                self.capacity_controller.wrote(getattr(entry,"capacity_sample",None))
            entry.capacity_sample=None
            if getattr(entry, 'copy_committed', False):
                if self._has_ssd_copy(entry):self.ssd_written_tokens+=self.cache.page_size
                else:self.ssd_write_failures+=1
                self._release_transferred_source(entry)
                return
            if not self._has_ssd_copy(entry):
                self.ssd_write_failures+=1
                if entry.owner == 'staging' and entry.hash_value[0] in self._staging_return_flush:
                    # Use the scheduler's failure path. A rank-local retry can
                    # reorder the native write receipts across the TP group.
                    raise RuntimeError(f"SSD write failed during staging budget fallback for {entry.hash_value[0]}")
                if entry.owner=='read':
                    self.ssd_backend.release_transition(entry.hash_value[0])
                    if entry.native_lock:original(*entry.native_lock);entry.native_lock=None
                self._release_written_read_prefixes()
                return
            self.ssd_written_tokens+=self.cache.page_size
            if entry.owner=='read':
                if entry.native_lock:original(*entry.native_lock);entry.native_lock=None
                self._release_written_read_prefixes()
                return
            if entry.read_pins:
                if entry.hash_value[0] in self._staging_return_flush:
                    entry.fallback_write_committed = True
                return
            # An eviction write is complete. Native storage already owns its
            # copy; release the source unless an actual read still holds it.
            # Completion itself is not a read or a reason to reinsert into LRU.
            self._remove_stage(entry,free=True);self.overlay_write_releases+=1;self.staging_exits+=1
        self.cache.dec_host_lock_ref=finish

    def _drive_host_eviction(self,required_free,tracker,device_frees,host_frees,cache_actions=None):
        storage_active=self.storage_active()
        if storage_active and self._in_drive:return
        exhausted=storage_active and self._budget_exhausted()
        if exhausted and self._budget_return_pending and not self._budget_return_started:
            # Preserve read anchors until the scheduler has classified the
            # staged pages. Subsequent read reclaims are safe with deferred
            # staging pages retained by the scheduler's handoff.
            return
        if not storage_active or exhausted:
            kwargs = {} if cache_actions is None else dict(cache_actions=cache_actions)
            return self._original_drive_host_eviction(required_free,tracker,device_frees,host_frees,**kwargs)
        self._in_drive=True
        previous_actions = getattr(self, '_eviction_cache_actions', None)
        self._eviction_cache_actions = cache_actions
        target=self.read_available()+required_free
        # In-flight writes retain their physical source but no longer consume
        # the waiting-residency quota. Count their future frees only for limiting
        # submissions, never as allocator capacity or as returned freed tokens.
        pending_pages=sum(self.stage.get(e.hash_value[0]) is e
                          for e in self.inflight.values())
        pending_free=sum(self.cache.page_size for e in self.inflight.values()
                         if self.stage.get(e.hash_value[0]) is e and not e.read_pins)
        def retire(entry, **kwargs):
            nonlocal pending_pages,pending_free
            was_pending=entry.operation is not None
            freed=self._retire_staging(entry,**kwargs)
            if not was_pending and entry.operation is not None:
                pending_pages+=1
                if not entry.read_pins:pending_free+=self.cache.page_size
            tracker[ComponentType.FULL]+=freed
            return freed
        def waiting_tokens():
            return (len(self.stage)-pending_pages)*self.cache.page_size
        def enough():
            if self._budget_exhausted():return True
            deferred=sum(v.numel() for v in host_frees.get(ComponentType.FULL,()))
            prefix_free = sum(self._tokens(node) for node in self._pending_prefix_read_releases().values())
            self._read_ownership['prefix_read_forecast_peak_tokens'] = max(
                self._read_ownership['prefix_read_forecast_peak_tokens'], prefix_free)
            eventual=min(self._physical_available()+deferred+pending_free+prefix_free,
                         self.read_limit-self.read_allocated(deferred)+prefix_free)
            enough_space = eventual>=target and self.read_allocated(deferred)-prefix_free<=self.read_limit
            if enough_space and prefix_free:self._read_ownership['prefix_read_forecast_stops'] += 1
            return enough_space
        def stage_leaf(node):
            if self._budget_exhausted():return False
            if not self.staging_policy.accepts_staging(node):
                self.read_releases+=1
                return None
            # A rejectable incoming tail needs no waiting-residency slot.
            # Resolve it before displacing a page that still benefits from waiting.
            if self.staging_policy.discard_on_entry(node):
                entry=self._take_read_leaf(node)
                self.storage.put(entry)
                freed=retire(entry,rejected_on_entry=True)
                assert freed==self.cache.page_size, 'rejected read tail did not release its page'
                self.staging_pressure["discard_on_entry"]+=1
                return True
            while waiting_tokens()>=max(self.cache.page_size,self.staging_limit):
                entry=self.staging_policy.choose()
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
            # Retiring staging may persist this read prefix and acquire a native
            # I/O pin, split its path, or exhaust the budget. The heap-pop check
            # preceded those operations; validate again at ownership transfer.
            core=self.cache.tree_core
            if node not in core.evictable_host_leaves and not self._hbm_read_leaf(node):
                self._read_ownership['read_candidate_changed_during_retire'] += 1
                if node.component_data[ComponentType.FULL].host_lock_ref:
                    self._read_ownership['read_candidate_pinned_during_retire'] += 1
                return 'skip_candidate'
            # Prefix completion may also have supplied the SSD copy after the
            # initial admission check. Native eviction now owns this candidate.
            if not self.staging_policy.accepts_staging(node):
                self._read_ownership['read_candidate_persisted_during_retire'] += 1
                self.read_releases+=1
                return None
            entry=self._take_read_leaf(node)
            self.storage.put(entry)
            if self.staging_limit==0:
                retire(entry)
            return True
        self.staging_policy.begin_round()
        try:
            while not self._budget_exhausted() and waiting_tokens()>self.staging_limit:
                entry=self.staging_policy.choose()
                if entry is None:break
                before=(len(self.stage),len(self.inflight))
                retire(entry)
                if before==(len(self.stage),len(self.inflight)):break
            if not self._budget_exhausted():
                self._drive_read_leaves(required_free,tracker,device_frees,host_frees,
                                                   leaf_handler=stage_leaf,should_stop=enough)
            self._record_peaks()
        finally:
            self.staging_policy.end_round()
            self._eviction_cache_actions = previous_actions
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
        # Resolve async write completion at a common native TP scheduler round.
        if getattr(self,'budget_consensus',None) is not None:
            self.budget_consensus.advance()
        if not self._advance_admission():return 0
        if self.sli.observed_requests:self.staging_policy.maybe_control(self.now())
        consensus=getattr(self,'capacity_consensus',None)
        if consensus is not None:
            consensus.advance()
        else:
            if self.capacity_controller:self.capacity_controller.tick(self.now())
        if self._budget_exhausted() and self._budget_return_pending:
            return_waiting_to_read(self)
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
        exhausted=self._budget_exhausted()
        if exhausted:target=0
        if previous==target and not self._budget_return_pending:return True
        self.staging_limit=target;self.read_limit=self.total_tokens-target
        if target==0 and exhausted and (previous or self._budget_return_pending):
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
            info = cache.ongoing_prefetch.get(request)
            lease = self.storage.leases.get(id(info.operation)) if info is not None else None
            has_ssd = lease is not None and len(lease.entries) < len(lease.keys)
            ready = original_check(request)
            now = self.now()
            control.read_progress(request.rid, not ready, now, has_ssd)
            current = cache.ongoing_prefetch.get(request) if not ready else None
            if current is not None and self.storage.physical_ssd_pending(current.operation):
                observe_physical_wait(control, request, current.operation, now)
            return ready
        cache.check_prefetch_progress=check

    def reset(self):
        # Called AFTER native reset, which has cleared the shared host allocator.
        if self.capacity_controller:self.capacity_controller.reset()
        self.storage.reset()
        self.staging_policy.reset_prefix_stats()
        self._staging_return_flush.clear()
        self._budget_return_pending=False
        self._budget_return_started=False
        self.sli.ready_hashes.clear()
        self.stage.clear();self._waiting_stage_keys.clear();self.inflight.clear();self.request_hashes.clear()
        self._host_index.__init__(self.cache.tree_core,ComponentType.FULL)
        self._next_stats_snapshot_s=0.
        self.staging_policy.index.clear_residents()
        self._bind_resident_index()

    def stats(self):
        read,stage=self._counts()
        fields=('ssd_written_tokens','ssd_write_failures','ssd_prefetched_tokens','staging_exits','read_releases',
                'duplicate_reclaimed_tokens','demotions','promotions','peak_read_tokens','peak_staging_tokens',
                'over_limit_events','admission_write_decisions','admission_drop_decisions','discarded_tokens',
                'budget_charged_bytes',
                'overlay_transfers_in','overlay_transfers_read','overlay_write_returns_read','overlay_write_releases','overlay_backup_reuses')
        result = dict(staging_pressure=dict(self.staging_pressure),last_transfer_pressure=self.last_transfer_pressure,allocation_progress=dict(self.allocation_progress),staging_return_pending_pages=len(self._staging_return_flush),total_storage_submitted_bytes=self.ssd_backend.validation_write_bytes,prefix_materialization=self.staging_policy.prefix_write_stats(),unified_storage=self.storage.stats(),budget_exhausted_at_s=self.budget_exhausted_at_s,**{k:getattr(self,k) for k in fields},admission_policy=self.admission_policy,
                    materialization_timing='on_evict',write_budget_bytes=self.write_budget_bytes,
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
                    storage_execution='native_sglang',performance_validated=False)
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
    active_read = None
    transfer = None
    if ((h.staging_limit or h.cfg.get('dynamic_capacity')) and cache.pp_size == 1
            and not core.is_eagle and getattr(controller, 'prefetch_buffer', None) is not None):
        from hohlraum.core.staging_read import StagingReadTransfers
        transfer = StagingReadTransfers(h, backend)
        transfer.install()
    if (cache.pp_size == 1 and not core.is_eagle and cache.host_memory_mode != 'buffer_only'
            and getattr(controller, 'prefetch_hit_queue', None) is not None
            and cache.tree_components == (ComponentType.FULL,)):
        from hohlraum.core.prefetch_reclaim import install_prefetch_reclaim
        install_prefetch_reclaim(cache, lambda: h.storage_active() and not h._budget_exhausted(),
                                 h._read_ownership)
    query = controller._storage_hit_query
    if backend._threaded:
        def lookup(operation, *args, **kwargs):
            with backend.observe_query_sources() as resident:
                keys, tokens = query(operation, *args, **kwargs)
            backend.begin_read(operation, keys[:tokens // cache.page_size], resident)
            return keys, tokens
    else:
        def lookup(operation, *args, **kwargs):
            keys, tokens = query(operation, *args, **kwargs)
            backend.begin_read(operation, keys[:tokens // cache.page_size])
            return keys, tokens
    controller._storage_hit_query = lookup
    if backend._threaded:
        page_get = controller.page_get_func
        def read_batch(operation, *args, **kwargs):
            with backend.read_context(operation):
                if transfer is not None:
                    return transfer.read_batch(page_get, operation, *args, **kwargs)
                return page_get(operation, *args, **kwargs)
        controller.page_get_func = read_batch

    # Preserve original native insertion. No splitting, grafting or child requests.
    insert = core.insert_host
    h._native_insert_host = insert
    def inserted(node_id, key, payload, hashes):
        if transfer is None or active_read is None:
            result = insert(node_id, key, payload, hashes)
        else:
            indices, replacements = transfer.substitute(active_read, payload, hashes)
            result = insert(node_id, key, indices, hashes)
            transfer.commit(result, payload, replacements, hashes)
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
        nonlocal pending_commits, active_read
        assert pending_commits is None and active_read is None
        pending_commits = []
        active_read = operation
        try:
            value = handle(operation)
            backend.confirm_read(pending_commits)
            h.ssd_prefetched_tokens += backend.record_read(operation)
            return value
        finally:
            pending_commits = None
            active_read = None
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
    ssd_read_active: bool = False
    claim: object = None
    staged_at_query: frozenset = frozenset()


def migrate_prefetch_leases(host):
    """Carry in-flight warmup SSD operations into the staging adapter."""
    migrated = 0
    for info in host.cache.ongoing_prefetch.values():
        operation = info.operation
        keys = list(operation.hash_value or ())
        if keys:
            host.storage.leases[id(operation)] = ReadLease(operation, keys, {})
            migrated += 1
    return migrated


class StagingStorage:
    """Public staging + SSD abstraction shared by both runtimes."""

    @classmethod
    def create(cls, residency, backing, page_size, token_bytes, *, simulated=False):
        """Select execution once at attachment; consumers use the same interface."""
        implementation = _HiSimStorage if simulated else _SGLangStorage
        return implementation(residency, backing, page_size, token_bytes)

    def __getattr__(self, name):
        return getattr(self.backing, name)

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

    def release_read_claim(self, lease):
        if lease.claim is not None and self._threaded:
            for entry in lease.claim.sources.values():
                self.residency.release_source(entry)
        lease.claim = None

    def finish_read(self, operation):
        with self.lock:
            lease = self.leases.pop(id(operation), None)
            if lease is not None:
                for entry in lease.entries.values():
                    self.residency.release_source(entry)
                self.release_read_claim(lease)

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

    @contextmanager
    def observe_query_sources(self):
        previous = getattr(self._read_local, 'query_sources', None)
        resident = set()
        self._read_local.query_sources = resident
        try:
            yield resident
        finally:
            self._read_local.query_sources = previous

    def begin_read(self, operation, keys, staged_at_query=()):
        # Record source observations without pinning from the query thread.
        with self.lock:
            lease = ReadLease(operation, list(keys), {},
                              staged_at_query=frozenset(staged_at_query).intersection(keys))
            self.leases.setdefault(id(operation), lease)

    def finish_read(self, operation):
        with self.lock:
            lease = self.leases.pop(id(operation), None)
            if lease is not None:
                self.release_read_claim(lease)

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


    def physical_ssd_pending(self, operation):
        with self.lock:
            lease = self.leases.get(id(operation))
            return lease is not None and lease.ssd_read_active

    @contextmanager
    def _physical_read(self):
        lease = getattr(self._read_local, 'lease', None)
        if lease is not None:
            with self.lock:
                lease.ssd_read_active = True
        try:
            yield
        finally:
            if lease is not None:
                with self.lock:
                    lease.ssd_read_active = False

    def exists(self, key):
        with self.lock:
            if self.residency.contains(key):
                sources = getattr(self._read_local, 'query_sources', None)
                if sources is not None:
                    sources.add(key)
                return True
        return self.backing.batch_exists([key], None) == 1

    def _staged_page(self, key, seen_in_staging=False):
        with self.lock:
            entry = self.residency.stage.get(key)
            if entry is not None:
                slot = entry.payload[0]
                counter = 'staging_read_bytes'
            elif seen_in_staging:
                # A different request may have moved this source after the
                # storage query. Read its current host payload by hash.
                index = self.residency._host_index
                owners = index.key_nodes.get(key)
                if not owners:
                    return None
                node = index.nodes[next(iter(owners))]
                offset = node.hash_value.index(key) * self.page_size
                slot = node.component_data[ComponentType.FULL].host_value[offset]
                counter = 'promoted_source_read_bytes'
            else:
                return None
            # The returned CPU copy owns its memory after releasing the lock.
            data = self.pool.get_data_page(int(slot)).clone()
            self.events[counter] += data.numel() * data.element_size()
            return data

    def _read_plan(self, keys):
        lease = getattr(self._read_local, 'lease', None)
        staged = lease.staged_at_query if lease is not None else ()
        pages = [self._staged_page(key, key in staged) for key in keys]
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
            with self._physical_read():
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
            with self._physical_read():
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
    """A per-rank admission budget and separate cutover writes."""
    def __init__(self, backing, cap, page_bytes):
        self.backing = backing
        self.validation_cap = cap
        self.validation_write_bytes = 0
        self.regular_write_bytes = 0
        self.transition_write_bytes = 0
        self.regular_reservations = set()
        # A native scheduler round protects the same free credits on all ranks.
        self.round_credits = None
        self.transition_keys = set()
        self.page_bytes = page_bytes
        self.lock = threading.RLock()

    def __getattr__(self, name):
        return getattr(self.backing, name)

    def physical_budget_exhausted(self):
        with self.lock:
            return (self.validation_cap is not None
                    and self.validation_write_bytes + self.page_bytes > self.validation_cap)

    def admission_budget_exhausted(self):
        with self.lock:
            return (self.validation_cap is not None
                    and self.regular_write_bytes + self.page_bytes > self.validation_cap)

    def budget_exhausted(self, reserve=0):
        with self.lock:
            reserved = max(reserve, self.regular_reserved_bytes)
            return (self.validation_cap is not None
                    and self.regular_write_bytes + reserved + self.page_bytes > self.validation_cap)

    @property
    def regular_reserved_bytes(self):
        with self.lock:
            return len(self.regular_reservations) * self.page_bytes

    def remaining_unreserved_pages(self):
        """Return whole pages still available, or None without a cap."""
        with self.lock:
            if self.validation_cap is None:
                return None
            remaining = self.validation_cap - self.regular_write_bytes - self.regular_reserved_bytes
            return max(0, remaining // self.page_bytes)

    def protect_round_credits(self):
        with self.lock:
            self.round_credits = self.remaining_unreserved_pages()
            return self.round_credits

    def limit_round_credits(self, pages):
        with self.lock:
            self.round_credits = pages

    def reserve_pages(self, keys):
        # Reserve the entire prefix chain before queuing any of its pages.
        keys = set(keys)
        with self.lock:
            if self.validation_cap is None:
                return True
            new = keys - self.regular_reservations
            if self.round_credits is not None:
                # Debit the same proposed keys on each scheduler rank even when
                # a local SSD completion already consumed a matching claim.
                if len(keys) > self.round_credits:
                    return False
                self.round_credits -= len(keys)
            elif len(new) > self.remaining_unreserved_pages():
                return False
            self.regular_reservations.update(new)
            return True

    def reset_budget(self, cap):
        with self.lock:
            self.validation_cap = cap
            self.validation_write_bytes = self.regular_write_bytes = self.transition_write_bytes = 0
            self.transition_keys.clear()
            self.regular_reservations.clear()
            self.round_credits = None
            self.__dict__.pop('budget_exhausted_at_s', None)

    def release_transition(self, key):
        """Release any write accounting when a page exits or completes."""
        with self.lock:
            self.transition_keys.discard(key)
            self.regular_reservations.discard(key)

    def allow_transition(self, key):
        """Classify a blocked cutover page before submitting its native write."""
        with self.lock:
            self.regular_reservations.discard(key)
            if self.backing.batch_exists([key], None) == 0:
                self.transition_keys.add(key)
            return True

    def _charge(self, key):
        self.validation_write_bytes += self.page_bytes
        if key in self.transition_keys:
            self.transition_write_bytes += self.page_bytes
        else:
            self.regular_write_bytes += self.page_bytes
        self.release_transition(key)

    def _write_page(self, key, write):
        if self.backing.batch_exists([key], None) == 1:
            self.release_transition(key)
            return True
        additional = 0 if key in self.regular_reservations else self.page_bytes
        if (key not in self.transition_keys and self.validation_cap is not None
                and self.regular_write_bytes + self.regular_reserved_bytes +
                    (self.round_credits or 0) * self.page_bytes + additional > self.validation_cap):
            return False
        if not write():
            return False
        self._charge(key)
        return True

    def batch_set(self, keys, data, extra_info=None):
        with self.lock:
            for key, page in zip(keys, data):
                if not self._write_page(key, lambda: self.backing.batch_set([key], [page], extra_info)):
                    return False
            return len(keys) == len(data)

    def batch_set_v1(self, keys, indices, extra_info=None):
        with self.lock:
            if not keys:
                return []
            page = len(indices)//len(keys)
            result = []
            for i,key in enumerate(keys):
                page_indices = indices[i*page:(i+1)*page]
                result.append(self._write_page(
                    key, lambda: self.backing.batch_set_v1([key], page_indices, extra_info)[0]))
            return result


def _install_scheduler_hooks():
    from sglang.srt.managers import scheduler
    Scheduler = scheduler.Scheduler
    if getattr(Scheduler,'_hohlraum_installed',False):
        return
    install_prefill_observer(scheduler)
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
            h.maintain()
            try:
                return original_next(s,*args,**kwargs)
            finally:
                if control is not None:
                    control.read_pressure.finish_round()
    @wraps(original_result)
    def result(s,batch,result,*args,**kwargs):
        h=policy(s)
        if h is None:return original_result(s,batch,result,*args,**kwargs)
        with h.lock:
            # Native postprocessing may filter or replace batch.reqs. Keep the
            # submitted request identities until completion references close.
            requests=tuple(batch.reqs)
            defer_completion = getattr(s, '_experiment_feedback', False)
            # Fallback for runtimes or paths without a scheduler completion
            # timestamp. Sample before native output can reach the client.
            before_output = h.now() if not defer_completion and getattr(h, '_native_wall_clock', False) else None
            value=original_result(s,batch,result,*args,**kwargs)
            if defer_completion:
                h.on_batch_complete(requests, h.now(), defer_completion=True)
            else:
                h.on_batch_complete(requests,h.now() if before_output is None else before_output)
            return value
    @wraps(original_run)
    def run(s,batch,*args,**kwargs):
        h=policy(s)
        if h is None or h.capacity_controller is None:return original_run(s,batch,*args,**kwargs)
        with h.lock:
            return original_run(s,batch,*args,**kwargs)
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


def create_host(cache, cfg, host_class):
    """Create the runtime adapter for the rank's existing budget and host pool."""
    controller = cache.cache_controller
    pool = controller.mem_pool_host.anchor_entry.host_pool
    page_size = cache.page_size
    token_bytes = controller.storage_backend.page_bytes // page_size
    cfg['kv_bytes_per_token'] = token_bytes
    stage_tokens = int(float(cfg.get('staging_GB', 0)) * 1e9 / cache.tp_world_size / token_bytes)
    stage_tokens = stage_tokens // page_size * page_size
    if cfg.get('stats_path'):
        cfg['stats_path'] += f'.rank{controller.storage_config.tp_rank}.json'
    cls = type(cache)
    if not getattr(cls, '_hohlraum_cache_hooks', False):
        install_cache_hooks(cls)
        cls._hohlraum_cache_hooks = True
    return host_class(cache, cfg, int(pool.size), stage_tokens)


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
    world=cache.tp_world_size
    cap=cfg.get('write_budget_bytes')
    cap=None if cap is None else int(cap)//world
    controller.storage_backend=BudgetStorage(controller.storage_backend,cap,cache.page_size*token_bytes)
    cache.hohlraum_host=None
    if not cfg.get('enabled',True):return
    if cache.cache_controller.write_policy not in ('write_back', 'write_through'):
        raise ValueError('Native Hohlraum requires write_back or write_through')
    for key in ('HOHLRAUM_JOINT_MODELS','HOHLRAUM_MODEL_COUNT','HOHLRAUM_ONLINE_UPDATES',
                'HOHLRAUM_BRANCH_PRIOR','HOHLRAUM_LENGTH_CURVES'):
        if not os.environ.get(key):
            raise ValueError(f'Native lifecycle policy requires {key}')
    if not Path(cfg['history_library']).is_file():
        raise ValueError('history_library does not exist')
    cfg.setdefault('admission_policy','agent')
    cfg.setdefault('residency_policy','joint_hts')
    cfg.setdefault('duration_s',14400)
    cfg.setdefault('threshold',0.7)
    cfg.setdefault('staging_GB',30)
    host_class = HiSimHost if simulated else NativeHost
    cache.hohlraum_host = create_host(cache, cfg, host_class)
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

    def physical_ssd_pending(self, operation):
        """Check the current incomplete transfer, including a pure SSD route."""
        lease = self.leases.get(id(operation))
        if lease is None or operation.host_indices is None:
            return False
        completed = operation.completed_tokens
        if completed >= len(operation.host_indices):
            return False
        route = getattr(operation, '_sim_read_route', None)
        if route is None:
            page_index = completed // self.page_size
            return page_index < len(lease.keys) and page_index not in lease.entries
        if route.index >= len(route.steps):
            return False
        start, end, resource, _, _ = route.steps[route.index]
        return resource == 'storage_read' and route.progress < end - start

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

    def _drive_host_eviction(self, required_free, tracker, device_frees, host_frees, cache_actions=None):
        if self._in_drive:
            return
        super()._drive_host_eviction(required_free, tracker, device_frees, host_frees, cache_actions)
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
