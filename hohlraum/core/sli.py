"""Session-Level Index (SLI): history and compressed staging branch ranges.

No CLOCK reclamation or history persistence; the experiment session set is bounded.
"""
from __future__ import annotations
import ctypes as C
from collections import Counter
from dataclasses import dataclass

class SLI:
    def __init__(self, host, cfg):
        self.host,self.cfg=host,cfg
        self.lib=C.CDLL(cfg['history_library'])
        l=self.lib
        if not hasattr(l,'hh_sli_version') or l.hh_sli_version()!=1:
            raise RuntimeError('Rebuild history_library for the session-range SLI ABI')
        for name in ('hh_sli_ranges','hh_sli_updates'):
            fn=getattr(l,name);fn.argtypes=[C.c_void_p];fn.restype=C.c_ulonglong
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

    def client_ready(self, req, now, hashes=None):
        # Freeze idle time and update the actual context together when available.
        # Metadata-only callers retain the engine-arrival fallback.
        if req.rid not in self.client_ready_ids:
            self.check(self.lib.hh_clock(self.ptr,self.sid(self.metadata(req)),now,0))
            self.client_ready_ids.add(req.rid)
            self.cache.clear()
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
        self.observed.add(req.rid)
        self.observed_requests+=1;self.now=max(self.now,now);self.cache.clear()

    def arrive(self,req,now,hashes):
        if req.rid in self.started:return
        self.client_ready(req,now,hashes)
        self.bind_input_prefixes(req,hashes)
        self.started.add(req.rid)
        if self.cfg.get('pending_read_admission',True):
            # Pending reuse is per page, including resident pages after a gap.
            # A missing earlier page does not erase known demand for later KV.
            resident=self.host.resident_keys()
            # Preserve pending reuse protection for device-only cache pages too.
            resident.update(h for n in self.host._nodes() if not n.evicted
                            for h in n.hash_value or ())
            page=self.host.cache.page_size
            pending=[key for key in hashes[:max(0,(len(req.input_ids)-1)//page)]
                     if key in resident or self.host._copy_bits([key])[0]]
            self.pending.update(pending);self.request_pending[req.rid]=pending
    def complete(self,req,now):
        if req.rid not in self.started or req.rid in self.finished:return
        self.finished.add(req.rid)
        meta=self.metadata(req)
        self.check(self.lib.hh_clock(self.ptr,self.sid(meta),now,1))
        self.release_read(req.rid);self.cache.clear()
        if hasattr(self.host,'finish_request'):self.host.finish_request(req)
    def release_read(self,rid):
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
    def score(self,key,now):return self.profiles([key],now)[key][0]
    def time_to_discard(self,key,now,threshold):return self.profiles([key],now)[key][1]
    def set_threshold(self, threshold):
        self.check(self.lib.hh_threshold(self.ptr, threshold))
        self.cache.clear()
    def stats(self):
        return dict(implementation='session_range_sli',history_ranges=self.lib.hh_sli_ranges(self.ptr),history_range_updates=self.lib.hh_sli_updates(self.ptr),session_clock=True,observed_requests=self.observed_requests,
                    sessions=len(self.sessions),pending_read_references=sum(self.pending.values()))


@dataclass(eq=False)
class BranchRange:
    keys: list
    children: list


class BranchTable:
    def __init__(self):
        self.root = BranchRange([], [])
        self.locations = {}

    def append_suffix(self, parent, keys):
        if not keys:
            return
        if parent is None:
            anchor = self.root
        else:
            anchor, offset = self.locations[parent]
            if offset + 1 < len(anchor.keys):
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

    def observe(self, session, keys):
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
            elif first_new is not None:
                raise RuntimeError("prefix hash rejoined after an unknown prefix")
            parent = key
        if first_new is not None:
            table.append_suffix(keys[first_new-1] if first_new else None, keys[first_new:])

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

    def endpoints(self, active):
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
        result = set()
        for session, bounds in selected.items():
            root = self.tables[session].root
            work = [(root, False, False)]
            summaries = {}
            while work:
                branch, ancestor_selected, leaving = work.pop()
                own = branch in bounds
                if not leaving:
                    if own and not ancestor_selected:
                        result.add(branch.keys[bounds[branch][0]])
                    work.append((branch, ancestor_selected, True))
                    work.extend((child, ancestor_selected or own, False) for child in branch.children)
                else:
                    leaves = sum(summaries[c][0] for c in branch.children) if branch.children else 1
                    covered = sum(summaries[c][1] for c in branch.children)
                    if own and covered < leaves:
                        result.add(branch.keys[bounds[branch][1]])
                    summaries[branch] = (leaves, leaves if own else covered)
        return result
