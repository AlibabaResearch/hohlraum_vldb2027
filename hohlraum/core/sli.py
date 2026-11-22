"""Session-Level Index (SLI): history and compressed staging branch ranges.

No CLOCK reclamation or history persistence; the experiment session set is bounded.
"""
from __future__ import annotations
import ctypes as C
from collections import Counter
from dataclasses import dataclass


class RelativePositionRisk:
    """Observed next-request loss by distance from the current request's tail.

    Statistics use completed context transitions only, never future trace labels.
    Positions refer to request pages, not to the staging queue. This signal is
    consumed only by residency ranking; it cannot accept or discard a page.
    """
    def __init__(self):
        self.current = {}
        self.locations = {}
        self.counts = [[0, 0] for _ in range(4)]

    @staticmethod
    def bucket(distance):
        return 0 if distance == 0 else 1 if distance < 4 else 2 if distance < 16 else 3

    def observe(self, session, handles):
        positions = {key: i for i, key in enumerate(handles)}
        previous = self.current.get(session, {})
        n = len(previous)
        for key, i in previous.items():
            count = self.counts[self.bucket(n - 1 - i)]
            count[0] += 1
            count[1] += key not in positions
            owners = self.locations[key]
            owners.pop(session)
            if not owners:
                del self.locations[key]
        n = len(positions)
        for key, i in positions.items():
            self.locations.setdefault(key, {})[session] = self.bucket(n - 1 - i)
        self.current[session] = positions

    def risk(self, handle):
        # Multiple current owners: retain the existing max-risk convention.
        return max((self.counts[b][1] / self.counts[b][0]
                    if self.counts[b][0] else 0.
                    for b in self.locations.get(handle, {}).values()), default=0.)


class SLI:
    def __init__(self, host, cfg):
        self.host,self.cfg=host,cfg
        self.lib=C.CDLL(cfg['history_library'])
        l=self.lib
        if not hasattr(l,'hh_sli_version') or l.hh_sli_version()!=1:
            raise RuntimeError('Rebuild history_library for the session-range SLI ABI')
        for name in ('hh_sli_ranges','hh_sli_updates'):
            fn=getattr(l,name);fn.argtypes=[C.c_void_p];fn.restype=C.c_ulonglong
        if hasattr(l,'hh_statistics_updates'):
            l.hh_statistics_updates.argtypes=[C.c_void_p];l.hh_statistics_updates.restype=C.c_ulonglong
            for name in ('hh_statistics_last_update','hh_statistics_period'):
                fn=getattr(l,name);fn.argtypes=[C.c_void_p];fn.restype=C.c_double
        l.hh_error.restype=C.c_char_p
        l.hh_create_joint.argtypes=[C.c_double];l.hh_create_joint.restype=C.c_void_p
        l.hh_clock.argtypes=[C.c_void_p,C.c_uint,C.c_double,C.c_int]
        l.hh_observe.argtypes=[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,C.c_double,C.POINTER(C.c_uint),C.c_uint]
        l.hh_advance.argtypes=[C.c_void_p,C.c_double]
        l.hh_threshold.argtypes=[C.c_void_p,C.c_double]
        l.hh_query.argtypes=[C.c_void_p,C.c_double,C.POINTER(C.c_uint),C.c_uint,C.POINTER(C.c_double)]
        self.ptr=l.hh_create_joint(host.admission_threshold)
        if not self.ptr:raise RuntimeError(l.hh_error().decode())
        self.now=0.;self.observed_requests=0
        self.prefix_ids={};self.native_prefixes={};self.ids={};self.sessions={};self.observed=set();self.started=set();self.finished=set();self.client_ready_ids=set()
        self.position_risk=RelativePositionRisk()
        self.pending=Counter();self.request_pending={};self.cache={};self.ready_hashes={}
    def check(self,rc):
        if rc:raise RuntimeError(self.lib.hh_error().decode())
    @staticmethod
    def metadata(req):
        params = req.sampling_params.custom_params or {}
        if 'hohlraum' in params:
            return params['hohlraum']
        return params['simulation']['agentx']

    def sid(self,meta):
        key=(int(meta['root']),int(meta['chain']))
        return self.sessions.setdefault(key,len(self.sessions))
    @staticmethod
    def request_hashes(req, page_size):
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.utils import get_storage_hash_str
        key=RadixKey(req.input_ids,getattr(req,'extra_key',None),cache_salt=getattr(req,'cache_salt',None)).page_aligned(page_size)
        return get_storage_hash_str(key,None,page_size=page_size)

    @staticmethod
    def hash_input(req, page_size):
        # Exact snapshot catches in-place token edits as well as namespace changes.
        return (page_size, getattr(req, 'extra_key', None),
                getattr(req, 'cache_salt', None), tuple(req.input_ids))

    def remember_request_hashes(self, req, page_size, hashes):
        self.ready_hashes[req.rid] = (self.hash_input(req, page_size), list(hashes))

    def hashes_for_request(self, req, page_size):
        prefixes = self.input_prefixes(req)
        if prefixes is not None:
            from sglang.srt.mem_cache.radix_cache import RadixKey
            from sglang.srt.mem_cache.utils import get_storage_hash_str
            result = []; parent = None; prior = None
            for i, identity in enumerate(prefixes):
                cached = self.native_prefixes.get(identity)
                if cached is not None:
                    if cached[0] != parent:
                        raise ValueError('input prefix identity reused under another parent')
                    native = cached[1]
                else:
                    key = RadixKey(req.input_ids[i*page_size:(i+1)*page_size],
                                   getattr(req, 'extra_key', None),
                                   cache_salt=getattr(req, 'cache_salt', None))
                    native = get_storage_hash_str(key, prior, page_size=page_size)[0]
                    self.native_prefixes[identity] = (parent, native)
                result.append(native); parent = identity; prior = native
            return result
        cached = self.ready_hashes.pop(req.rid, None)
        if cached is not None and cached[0] == self.hash_input(req, page_size):
            return cached[1]
        return self.request_hashes(req, page_size)

    def input_prefixes(self, req):
        meta = self.metadata(req)
        if 'prefix_blocks' not in meta:
            return None
        page = int(meta['prefix_page_size'])
        if page != self.host.cache.page_size:
            raise ValueError('input prefix page size differs from cache')
        length = req.input_length if hasattr(req, 'input_length') else len(req.input_ids)
        if len(meta['prefix_blocks']) != length // page:
            raise ValueError('input prefix count differs from request length')
        namespace = (int(meta['root']), getattr(req, 'extra_key', None),
                     getattr(req, 'cache_salt', None), page)
        return [namespace + (int(block),) for block in meta['prefix_blocks']]

    def bind_input_prefixes(self, req, hashes):
        prefixes = self.input_prefixes(req)
        if prefixes is None:
            return
        if len(prefixes) != len(hashes):
            raise ValueError('native/input prefix count mismatch')
        for identity, native in zip(prefixes, hashes):
            handle = self.prefix_ids[identity]
            previous = self.ids.get(native)
            if previous is not None and previous != handle:
                raise ValueError('native hash aliases different input prefix identities')
            self.ids[native] = handle

    def invalidate_profiles(self):
        self.revision = getattr(self, "revision", 0) + 1
        self.cache.clear()

    def client_ready(self, req, now, hashes=None):
        # Freeze idle time and update the actual context together when available.
        # Metadata-only callers retain the engine-arrival fallback.
        if req.rid not in self.client_ready_ids:
            self.check(self.lib.hh_clock(self.ptr,self.sid(self.metadata(req)),now,0))
            self.client_ready_ids.add(req.rid)
            self.invalidate_profiles()
        prefixes = self.input_prefixes(req)
        if prefixes is not None:
            self._observe_context(req, now, prefixes, input_identity=True)
        elif hashes is not None:
            self._observe_context(req,now,hashes)

    def _observe_context(self,req,now,hashes,input_identity=False):
        if req.rid in self.observed:return
        block_ids=[]
        mapping = self.prefix_ids if input_identity else self.ids
        for key in hashes:
            if key not in mapping:
                if len(self.ids)+len(getattr(self,"prefix_ids",{}))>=2**32:
                    raise OverflowError('SLI block handle space exhausted')
                mapping[key]=len(self.ids)+len(getattr(self,"prefix_ids",{}))
            block_ids.append(mapping[key])
        meta=self.metadata(req)
        self.check(self.lib.hh_observe(self.ptr,self.sid(meta),int(meta['reference_fold']),(req.input_length if hasattr(req,'input_length') else len(req.input_ids)),now,(C.c_uint*len(block_ids))(*block_ids),len(block_ids)))
        self.position_risk.observe(self.sid(meta), block_ids)
        self.observed.add(req.rid)
        self.observed_requests+=1;self.now=max(self.now,now);self.invalidate_profiles()

    def arrive(self,req,now,hashes):
        if req.rid in self.started:return
        self.client_ready(req,now,hashes)
        self.bind_input_prefixes(req,hashes)
        self.started.add(req.rid)
        if self.cfg.get('pending_read_admission',True):
            # Pending reuse is per page, including resident pages after a gap.
            # A missing earlier page does not erase known demand for later KV.
            page=self.host.cache.page_size
            keys=hashes[:max(0,(len(req.input_ids)-1)//page)]
            index=self.host._host_index
            resident=lambda key: (key in self.host.stage or key in index.host_keys
                                  or key in index.native_live_keys)
            resident_bits=[resident(key) for key in keys]
            missing=[key for key,live in zip(keys,resident_bits) if not live]
            stored=dict(zip(missing,self.host._copy_bits(missing)))
            pending=[key for key,live in zip(keys,resident_bits) if live or stored.get(key,False)]
            self.pending.update(pending);self.request_pending[req.rid]=pending
            self.invalidate_profiles()
    def complete(self,req,now):
        if req.rid not in self.started or req.rid in self.finished:return
        self.finished.add(req.rid)
        meta=self.metadata(req)
        self.check(self.lib.hh_clock(self.ptr,self.sid(meta),now,1))
        self.release_read(req.rid);self.invalidate_profiles()
        if hasattr(self.host,'finish_request'):self.host.finish_request(req)
    def release_read(self,rid):
        self.revision = getattr(self, "revision", 0) + 1
        self.ready_hashes.pop(rid, None)
        for h in self.request_pending.pop(rid,[]):
            self.pending[h]-=1
            if self.pending[h]==0:del self.pending[h]
    def advance(self,now):
        if now<self.now-1e-8:raise RuntimeError('reference history time reversal')
        self.check(self.lib.hh_advance(self.ptr,now))
        if now!=self.now:self.cache.clear()
        self.now=now
    def profiles(self,keys,now):
        keys=list(keys);missing=[h for h in keys if (h,now) not in self.cache]
        if missing:
            ids=[self.ids[h] for h in missing]
            out=(C.c_double*(5*len(ids)))()
            self.check(self.lib.hh_query(self.ptr,now,(C.c_uint*len(ids))(*ids),len(ids),out))
            for i,h in enumerate(missing):self.cache[h,now]=tuple(out[5*i+j] for j in range(5))
        return {h:self.cache[h,now] for h in keys}
    def position_risks(self, keys):
        return {key: self.position_risk.risk(self.ids[key]) for key in keys}

    def context_risks(self, keys):
        keys=list(keys)
        if not keys:return {}
        if not hasattr(self.lib, 'hh_context_risks'):
            raise RuntimeError('Rebuild history_library for context-risk staging ordering')
        fn=self.lib.hh_context_risks
        fn.argtypes=[C.c_void_p,C.POINTER(C.c_uint),C.c_uint,C.POINTER(C.c_double)]
        out=(C.c_double*len(keys))()
        self.check(fn(self.ptr,(C.c_uint*len(keys))(*(self.ids[k] for k in keys)),len(keys),out))
        return dict(zip(keys,out))

    def score(self,key,now):return self.profiles([key],now)[key][0]
    def set_threshold(self, threshold):
        self.check(self.lib.hh_threshold(self.ptr, threshold))
        self.invalidate_profiles()
    def stats(self):
        updates={}
        if hasattr(self.lib,'hh_statistics_updates'):
            updates=dict(statistics_updates=self.lib.hh_statistics_updates(self.ptr),statistics_last_update_s=self.lib.hh_statistics_last_update(self.ptr),statistics_period_s=self.lib.hh_statistics_period(self.ptr))
        return dict(**updates,implementation='session_range_sli',history_ranges=self.lib.hh_sli_ranges(self.ptr),history_range_updates=self.lib.hh_sli_updates(self.ptr),session_clock=True,observed_requests=self.observed_requests,
                    sessions=len(self.sessions),pending_read_references=sum(self.pending.values()))


@dataclass(eq=False)
class BranchRange:
    keys: list
    children: list


class BranchTable:
    def __init__(self):
        self.root = BranchRange([], [])
        self.locations = {}
        self._topology = None

    def endpoint_topology(self):
        """Cache topology only; active membership, locks and scores stay live."""
        if self._topology is None:
            parents, order = {}, []
            work = [(self.root, None)]
            while work:
                branch, parent = work.pop()
                parents[branch] = parent
                order.append(branch)
                work.extend((child, branch) for child in branch.children)
            leaves = {}
            for branch in reversed(order):
                leaves[branch] = sum(leaves[c] for c in branch.children) if branch.children else 1
            self._topology = parents, order, leaves
        return self._topology

    def append_suffix(self, parent, keys):
        if not keys:
            return
        if parent is None:
            anchor = self.root
        else:
            anchor, offset = self.locations[parent]
            if offset + 1 < len(anchor.keys):
                self._topology = None
                suffix = BranchRange(anchor.keys[offset + 1:], anchor.children)
                anchor.keys = anchor.keys[:offset + 1]
                anchor.children = [suffix]
                for i, key in enumerate(suffix.keys):
                    self.locations[key] = (suffix, i)
        if anchor is not self.root and not anchor.children:
            offset = len(anchor.keys)
            anchor.keys.extend(keys)
            for i, key in enumerate(keys, offset):
                self.locations[key] = (anchor, i)
        else:
            self._topology = None
            branch = BranchRange(list(keys), [])
            anchor.children.append(branch)
            for i, key in enumerate(keys):
                self.locations[key] = (branch, i)


class StagingRangeIndex:
    """Compressed session branches; payloads stay in the native KV/staging index.

    Block handles locate branch offsets. Only topology changes split branches;
    ordinary requests do not refresh per-block history or native LRU state.
    """
    def __init__(self):
        self.parents = {}
        self.tables = {}
        self.key_sessions = {}
        self._resident = set()
        self._live_children = {}
        self._endpoint_cache = {}
        self._active_endpoints = None
        self.waiting = HTSEndpoints(self)

    def observe(self, session, keys):
        self._active_endpoints = None
        table = self.tables.setdefault(session, BranchTable())
        parent = None
        first_new = None
        for i, key in enumerate(keys):
            if key in self.parents and self.parents[key] != parent:
                raise RuntimeError("conflicting observed KV prefix")
            if key not in self.parents:
                self.parents[key] = parent
                if key in self._resident or self._live_children.get(key, 0):
                    self._propagate_live(parent, 1)
            if key not in table.locations:
                if first_new is None:
                    first_new = i
                self.key_sessions.setdefault(key, []).append(session)
                # A newly observed owner may share a page already waiting in
                # another session. Keep this lookup current before rebuilding.
                if key in self.waiting.active:
                    self.waiting.session_active.setdefault(session, set()).add(key)
            elif first_new is not None:
                raise RuntimeError("prefix hash rejoined after an unknown prefix")
            parent = key
        if first_new is not None:
            self._endpoint_cache.pop(session, None)
            table.append_suffix(keys[first_new-1] if first_new else None, keys[first_new:])
            if self.waiting.active:
                self.waiting.rebuild_session(session)

    def _propagate_live(self, key, delta):
        # Count immediate children whose subtree contains a resident page.
        # Stop when this subtree's boolean liveness remains unchanged.
        while key is not None:
            count = self._live_children.get(key, 0)
            before = key in self._resident or count > 0
            count += delta
            assert count >= 0, "negative live-child count"
            if count:
                self._live_children[key] = count
            else:
                self._live_children.pop(key, None)
            after = key in self._resident or count > 0
            if before == after:
                break
            key = self.parents.get(key)

    def set_resident(self, key, resident):
        before = key in self._resident
        if before == resident:
            return
        if resident:
            self._resident.add(key)
        else:
            self._resident.remove(key)
        if not self._live_children.get(key, 0):
            self._propagate_live(self.parents.get(key), 1 if resident else -1)

    def has_resident_descendant(self, key):
        return self._live_children.get(key, 0) > 0

    def clear_residents(self):
        self._resident.clear()
        self._live_children.clear()
        self._endpoint_cache.clear()
        self._active_endpoints = None

    def endpoints(self, active, *, with_roles=False):
        # Candidate, legality and retirement checks may repeat one active set.
        # Locks are reflected in that set; history observations invalidate it.
        active = frozenset(active)
        cached = self._active_endpoints
        if cached is not None and cached[0] == active:
            heads, tails = set(cached[1]), set(cached[2])
            return (heads, tails) if with_roles else heads | tails
        selected = {}
        for key in active:
            for session in self.key_sessions.get(key, ()):
                branch, offset = self.tables[session].locations[key]
                bounds = selected.setdefault(session, {})
                if branch in bounds:
                    low, high = bounds[branch]
                    bounds[branch] = (min(low, offset), max(high, offset))
                else:
                    bounds[branch] = (offset, offset)
        heads, tails = set(), set()
        for session, bounds in selected.items():
            # Only topology and active range bounds affect HTS endpoints.
            # Never cache profiles, time, locks or admission decisions here.
            signature = frozenset((b, low, high) for b, (low, high) in bounds.items())
            cached = self._endpoint_cache.get(session)
            if cached is not None and cached[0] == signature:
                heads.update(cached[1]); tails.update(cached[2])
                continue
            local_heads, local_tails = set(), set()
            root = self.tables[session].root
            work = [(root, False, False)]
            summaries = {}
            while work:
                branch, ancestor_selected, leaving = work.pop()
                own = branch in bounds
                if not leaving:
                    if own and not ancestor_selected:
                        local_heads.add(branch.keys[bounds[branch][0]])
                    work.append((branch, ancestor_selected, True))
                    work.extend((child, ancestor_selected or own, False) for child in branch.children)
                else:
                    leaves = sum(summaries[c][0] for c in branch.children) if branch.children else 1
                    covered = sum(summaries[c][1] for c in branch.children)
                    if own and covered < leaves:
                        local_tails.add(branch.keys[bounds[branch][1]])
                    summaries[branch] = (leaves, leaves if own else covered)
            self._endpoint_cache[session] = (signature, local_heads, local_tails)
            heads.update(local_heads); tails.update(local_tails)
        self._active_endpoints = (active, frozenset(heads), frozenset(tails))
        return (heads, tails) if with_roles else heads | tails

class HTSEndpoints:
    """Incremental HTS boundaries for one synchronous reclamation round.

    Each compressed branch owns a sorted list of active offsets. Moving a
    boundary updates that branch only; becoming empty/nonempty updates coverage
    along its ancestors. Topology changes rebuild only the affected session.
    Locks and payloads are deliberately not cached here.
    """
    def __init__(self, index):
        self.index = index
        self.clear()

    def clear(self):
        self.active = set()
        self.session_active = {}
        self.sessions = {}
        self.heads = Counter()
        self.tails = Counter()

    def reset(self, active):
        self.clear()
        self.active = set(active)
        grouped = {}
        for key in self.active:
            for sid in self.index.key_sessions.get(key, ()):
                grouped.setdefault(sid, []).append(key)
                self.session_active.setdefault(sid, set()).add(key)
        for sid, keys in grouped.items():
            self.rebuild_session(sid, keys)

    @staticmethod
    def _replace(counter, old, new):
        for key in old - new:
            counter[key] -= 1
            if not counter[key]:
                del counter[key]
        for key in new - old:
            counter[key] += 1

    def _publish(self, state):
        bounds = state['bounds']
        heads = {b.keys[bounds[b][0]] for b in state['head_branches']}
        tails = {b.keys[bounds[b][-1]] for b in state['tail_branches']}
        self._replace(self.heads, state['heads'], heads)
        self._replace(self.tails, state['tails'], tails)
        state['heads'], state['tails'] = heads, tails

    def _refresh_roles(self, state):
        bounds, parents = state['bounds'], state['parents']
        heads, tails = set(), set()
        for branch in bounds:
            parent = parents[branch]
            while parent is not None and parent not in bounds:
                parent = parents[parent]
            if parent is None:
                heads.add(branch)
            if sum(state['covered'][c] for c in branch.children) < state['leaves'][branch]:
                tails.add(branch)
        state['head_branches'], state['tail_branches'] = heads, tails
        self._publish(state)

    def rebuild_session(self, sid, keys=None):
        old = self.sessions.pop(sid, None)
        if old:
            self._replace(self.heads, old['heads'], set())
            self._replace(self.tails, old['tails'], set())
        if keys is None:
            keys = self.session_active.get(sid, ())
        if not keys:
            return
        table = self.index.tables[sid]
        parents, order, leaves = table.endpoint_topology()
        bounds = {}
        for key in keys:
            branch, offset = table.locations[key]
            bounds.setdefault(branch, []).append(offset)
        for offsets in bounds.values():
            offsets.sort()
        covered = {}
        for branch in reversed(order):
            covered[branch] = (leaves[branch] if branch in bounds
                               else sum(covered[c] for c in branch.children))
        state = dict(parents=parents, bounds=bounds, leaves=leaves, covered=covered,
                     heads=set(), tails=set(), head_branches=set(), tail_branches=set())
        self.sessions[sid] = state
        self._refresh_roles(state)

    def set_active(self, key, active):
        from bisect import bisect_left, insort
        if (key in self.active) == active:
            return
        if active:
            self.active.add(key)
        else:
            self.active.remove(key)
        for sid in self.index.key_sessions.get(key, ()):
            if active:
                self.session_active.setdefault(sid, set()).add(key)
            else:
                members = self.session_active[sid]
                members.remove(key)
                if not members:
                    del self.session_active[sid]
            state = self.sessions.get(sid)
            if state is None:
                self.rebuild_session(sid)
                continue
            branch, offset = self.index.tables[sid].locations[key]
            bounds = state['bounds']; was_active = branch in bounds
            if active:
                insort(bounds.setdefault(branch, []), offset)
            else:
                offsets = bounds[branch]; at = bisect_left(offsets, offset)
                assert at < len(offsets) and offsets[at] == offset
                offsets.pop(at)
                if not offsets:
                    del bounds[branch]
            if was_active == (branch in bounds):
                self._publish(state)
                continue
            current = branch
            while current is not None:
                value = (state['leaves'][current] if current in bounds
                         else sum(state['covered'][c] for c in current.children))
                if value == state['covered'][current]:
                    break
                state['covered'][current] = value
                current = state['parents'][current]
            self._refresh_roles(state)

    def roles(self):
        return set(self.heads), set(self.tails)
