"""Local experiment transport; reads queue state, never changes scheduling."""
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from functools import wraps
from hohlraum.experiments.common.backpressure import waiting_queue_feedback


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


def read_events(stream):
    result = []
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line or not line.endswith('\n'):
            stream.seek(position)
            return result
        result.append(json.loads(line))


class NativeFeedback:
    """Client and server run on the same machine, sharing monotonic time."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ready = (self.directory/'ready.jsonl').open('a', buffering=1)
        (self.directory/'received.jsonl').touch(exist_ok=True)
        self.received = (self.directory/'received.jsonl').open()
        self.gate = None
        self.stale = False

    def notify_restore(self, checkpoint, rows, dataset):
        token = str(time.monotonic_ns())
        self.ready.write(json.dumps(dict(kind='restore_history', token=token, checkpoint=str(Path(checkpoint).resolve()), rows=str(Path(rows).resolve()), dataset=dataset, monotonic=time.monotonic()))+'\n')
        return token

    async def wait_restored(self, token, timeout=120.):
        return await self.wait_reports(token, 'history-restored', timeout)

    async def wait_reports(self, token, prefix, timeout=120.):
        # No HTTP requests may race the per-rank lifecycle restoration.
        deadline = time.monotonic() + timeout
        while True:
            reports = []
            for path in self.directory.glob(prefix+'.rank*.json'):
                row = json.loads(path.read_text())
                if row.get('token') == token:
                    reports.append(row)
            if reports:
                sizes = {row['tp_size'] for row in reports}
                if len(sizes) != 1:
                    raise RuntimeError('History restore rank-count mismatch')
                size = sizes.pop()
                if size > 0 and {row['rank'] for row in reports} == set(range(size)):
                    return reports
            if time.monotonic() >= deadline:
                raise RuntimeError('Timed out waiting for all ranks: '+prefix)
            await asyncio.sleep(.05)

    def notify_ready(self, req):
        meta = req.sampling_params.custom_params['hohlraum']
        length = req.input_length if hasattr(req, 'input_length') else len(req.input_ids)
        self.ready.write(json.dumps(dict(rid=req.rid, meta=meta, monotonic=time.monotonic(),
                                        input_length=length,page_size=1024))+'\n')

    async def activate(self, config, origin, end_seconds):
        token=str(time.monotonic_ns())
        self.ready.write(json.dumps(dict(kind='activate',token=token,config=str(Path(config).resolve()),
                                        origin=origin,end_seconds=end_seconds))+'\n')
        return await self.wait_reports(token,'activated')

    async def snapshot(self):
        token=str(time.monotonic_ns())
        self.ready.write(json.dumps(dict(kind='snapshot',token=token))+'\n')
        return await self.wait_reports(token,'snapshot')

    def refresh(self, now):
        for event in read_events(self.received):
            self.gate.received(event['rid'])
        path = self.directory/'queue.json'
        if not path.exists():
            raise RuntimeError('Native queue feedback missing; launch with the experiment server entry')
        row = json.loads(path.read_text())
        age = time.monotonic()-row['monotonic']
        stale = age > 30
        if stale != self.stale:
            with (self.directory/'feedback-health.jsonl').open('a') as out:
                out.write(json.dumps(dict(monotonic=time.monotonic(), stale=stale,
                                          snapshot_age_s=age))+'\n')
            self.stale = stale
        if stale:
            # Fail closed for new admissions, not for already executing requests.
            # Never present an old queue snapshot as a fresh/empty queue.
            return float('inf'), len(row['waiting_rids'])
        return waiting_queue_feedback(row['waiting_rids'], self.gate.records, now)

    def close(self):
        self.ready.close(); self.received.close()


def install(directory):
    from sglang.srt.managers.scheduler import Scheduler
    if getattr(Scheduler, '_experiment_feedback', False):
        return
    # This transport serves the trace experiment, not arbitrary HTTP traffic.
    # Native startup/health requests must execute normally without becoming
    # lifecycle samples or preventing the cold-start history restoration.
    from hohlraum.core.staging_storage import NativeHost, _install_scheduler_hooks, native_completion_timestamp
    if not getattr(NativeHost, '_experiment_trace_requests', False):
        original_on_requests = NativeHost.on_requests
        def trace_requests(host, requests, now):
            traced = []
            for req in requests:
                params = getattr(req.sampling_params, 'custom_params', None) or {}
                if 'hohlraum' in params or (params.get('simulation') or {}).get('agentx') is not None:
                    traced.append(req)
            return original_on_requests(host, traced, now)
        NativeHost.on_requests = trace_requests
        NativeHost._experiment_trace_requests = True

    # Install the policy hooks first, so feedback remains the outer wrapper:
    # restore history / deliver readiness before observing formal engine input.
    # The later cache setup reuses these hooks via their existing install guard.
    _install_scheduler_hooks()
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    old_input = Scheduler.process_input_requests
    old_next = Scheduler.get_next_batch_to_run
    old_result = Scheduler.process_batch_result
    completion_audit = os.environ.get('HOHLRAUM_NATIVE_COMPLETION_AUDIT')

    def debug(s, event, requests=()):
        path=os.environ.get('HOHLRAUM_NATIVE_DEBUG_LOG')
        if not path:return
        from sglang.srt.runtime_context import get_parallel
        host=getattr(s.tree_cache,'hohlraum_host',None)
        row=dict(event=event,monotonic=time.monotonic(),requests=[dict(
            rid=r.rid,prefix_tokens=len(getattr(r,'prefix_indices',())),
            extend_tokens=getattr(r,'extend_input_len',None)) for r in requests])
        if host is not None:
            row.update(raw_budget_bytes=host.ssd_backend.validation_write_bytes,
                budget_agreed=getattr(host,'_synced_budget_exhausted',None),
                read_limit=host.read_limit,read_tokens=host._host_index.total_tokens,
                stage_keys=list(host.stage),inflight=len(host.inflight))
        with open(path+f'.rank{get_parallel().tp_rank}.jsonl','a') as out:
            out.write(json.dumps(row)+'\n')

    def rank_report(s, prefix, token, value):
        from sglang.srt.runtime_context import get_parallel
        rank=get_parallel().tp_rank
        atomic(root/f'{prefix}.rank{rank}.json',dict(token=token,rank=rank,
            tp_size=s.server_args.tp_size,monotonic=time.monotonic(),**value))

    def activate(s, event):
        from hohlraum.core.staging_storage import (
            BudgetStorage, NativeHost, create_host, migrate_prefetch_leases,
        )
        from hohlraum.experiments.common.history_restore import restore_resident_history
        tree=s.tree_cache;backend=tree.cache_controller.storage_backend
        if s._experiment_formal or not isinstance(backend,BudgetStorage) or getattr(tree,'hohlraum_host',None) is not None:
            raise RuntimeError('Formal activation requires native full-admission warmup')
        if backend.validation_cap is not None:
            raise RuntimeError('Warmup must have an unlimited write budget')
        cfg=json.loads(Path(event['config']).read_text())
        pool=tree.cache_controller.mem_pool_host.anchor_entry.host_pool
        free_before=pool.available_size();history=None
        if cfg.get('enabled',True):
            cfg['duration_s']=event['end_seconds']
            host=create_host(tree,cfg,NativeHost)
            host.started_at=event['origin']
            if host.capacity_controller is not None:
                # Start the first read-pressure window at formal activation.
                host.capacity_controller.reset()
            tree.hohlraum_host=host
            migrate_prefetch_leases(host)
            now=time.monotonic()-event['origin']
            history=restore_resident_history(host,s._experiment_warm_history,event['origin'],now,
                                             s._experiment_engine_requests)
            assert host._physical_available()==free_before and not host.stage
        with backend.lock:
            warm_bytes=backend.regular_write_bytes
            cap=cfg.get('write_budget_bytes')
            backend.reset_budget(None if cap is None else int(cap)//tree.tp_world_size)
        s._experiment_formal=True
        s._experiment_warm_history.clear();s._experiment_engine_requests.clear()
        rank_report(s,'activated',event['token'],dict(warmup_write_bytes=warm_bytes,
            formal_budget_bytes_per_rank=backend.validation_cap,history=history,
            kv_preserved=True,physical_free_before=free_before,physical_free_after=free_before))

    def primary():
        from sglang.srt.runtime_context import get_parallel
        return get_parallel().tp_rank == 0

    def update(s, force=False, events=()):
        host = getattr(s.tree_cache, 'hohlraum_host', None)
        if not hasattr(s, '_experiment_ready'):
            (root/'ready.jsonl').touch(exist_ok=True)
            s._experiment_ready = (root/'ready.jsonl').open()
            s._experiment_last_publish = 0.
            s._experiment_warm_history=[]
            s._experiment_warm_rids=set()
            s._experiment_warm_completed=set()
            s._experiment_engine_requests={}
            s._experiment_completion_events=[]
            s._experiment_formal=os.environ.get('HOHLRAUM_RESIDENT_WARMUP')!='1'
        for event in events:
            if event.get('kind')=='complete':
                rid=event['rid']
                if not s._experiment_formal and rid in s._experiment_warm_rids:
                    if rid not in s._experiment_warm_completed:
                        s._experiment_warm_history.append(event)
                        s._experiment_warm_completed.add(rid)
                        s._experiment_engine_requests.pop(rid,None)
                if host is not None:
                    req=SimpleNamespace(rid=rid,sampling_params=SimpleNamespace(custom_params={'hohlraum':event['meta']}))
                    with host.lock:
                        host.sli.complete(req,event['monotonic']-host.started_at)
                continue
            if event.get('kind')=='activate':
                activate(s,event)
                host=getattr(s.tree_cache,'hohlraum_host',None)
                continue
            if event.get('kind')=='snapshot':
                backend=s.tree_cache.cache_controller.storage_backend
                budget=host.ssd_backend if host is not None else backend
                pool=s.tree_cache.cache_controller.mem_pool_host.anchor_entry.host_pool
                rank_report(s,'snapshot',event['token'],dict(
                    policy=host.stats() if host is not None else None,
                    host_capacity_tokens=int(pool.size),
                    kv_bytes_per_token=budget.page_bytes//s.tree_cache.page_size,
                    write_bytes=budget.validation_write_bytes,
                    regular_write_bytes=budget.regular_write_bytes,
                    transition_write_bytes=budget.transition_write_bytes,
                    budget_bytes=budget.validation_cap))
                continue
            if event.get('kind') == 'restore_history':
                report = None
                if host is not None:
                    from hohlraum.experiments.common.history_restore import load_checkpoint, restore_sli
                    with host.lock:
                        report = restore_sli(host.sli,load_checkpoint(event['checkpoint']),json.loads(Path(event['rows']).read_text()),event['dataset'],formal_start=event['monotonic']-host.started_at)
                from sglang.srt.runtime_context import get_parallel
                rank = get_parallel().tp_rank
                atomic(root/f'history-restored.rank{rank}.json', dict(
                    token=event['token'], rank=rank, tp_size=s.server_args.tp_size,
                    monotonic=time.monotonic(), history=report))
                continue
            if not s._experiment_formal:
                s._experiment_warm_history.append(dict(event,kind='ready'))
                s._experiment_warm_rids.add(event['rid'])
            if host is not None:
                req = SimpleNamespace(rid=event['rid'], input_length=event.get('input_length',0), sampling_params=SimpleNamespace(custom_params={'hohlraum':event['meta']}))
                with host.lock:
                    if event.get('hashes') is not None and event['page_size']!=host.cache.page_size:
                        raise ValueError('client-ready hash page size differs from engine')
                    host.sli.client_ready(req, event['monotonic']-host.started_at,event.get('hashes'))
        now = time.monotonic()
        if primary() and (force or now-s._experiment_last_publish >= .05):
            waiting_rids = [r.rid for r in s.waiting_queue
                            if 'hohlraum' in (r.sampling_params.custom_params or {}) and not r.finished()]
            atomic(root/'queue.json', dict(monotonic=now, waiting_rids=waiting_rids))
            s._experiment_last_publish = now

    @wraps(old_input)
    def inputs(s, requests, *args, **kwargs):
        update(s)
        # Every TP rank calls process_input_requests after the native receive
        # broadcast, including empty polls. Deliver our control/readiness at
        # this common boundary; independent file polling can activate a new
        # collective/partition on only one rank in a scheduler round.
        events=read_events(s._experiment_ready) if primary() else []
        if s._experiment_completion_events:
            # Merge engine completions and client events into the existing TP
            # control exchange. Use event time, with completions first on ties.
            events=s._experiment_completion_events+events
            s._experiment_completion_events=[]
            events.sort(key=lambda event: (event['monotonic'] if 'monotonic' in event
                                           else int(event['token'])/1e9))
        if s.server_args.tp_size>1:
            import torch.distributed as dist
            group=s.tree_cache.tp_group
            leader=dist.get_process_group_ranks(group)[0]
            payload=[events]
            dist.broadcast_object_list(payload,src=leader,group=group)
            events=payload[0]
        update(s,events=events)
        traced=[]
        for request in requests:
            for item in s._tokenized_requests(request):
                params=getattr(item.sampling_params,'custom_params',None) or {}
                if 'hohlraum' in params:
                    traced.append(item)
                    if not s._experiment_formal:s._experiment_engine_requests[item.rid]=item
        rids=[item.rid for item in traced]
        debug(s,'input',traced)
        value = old_input(s, requests, *args, **kwargs)
        update(s, True)
        if primary() and rids:
            with (root/'received.jsonl').open('a') as out:
                for rid in rids: out.write(json.dumps(dict(rid=rid))+'\n')
        return value

    @wraps(old_next)
    def next_batch(s, *args, **kwargs):
        update(s)
        value = old_next(s, *args, **kwargs)
        batch=getattr(value,'batch_to_run',None)
        if batch is not None:debug(s,'batch',batch.reqs)
        update(s)
        return value

    @wraps(old_result)
    def result(s, batch, *args, **kwargs):
        requests=tuple(batch.reqs)
        leader=primary()
        before_output=time.monotonic() if leader or completion_audit else None
        value=old_result(s,batch,*args,**kwargs)
        debug(s,'complete',requests)
        if leader or completion_audit:
            for req in requests:
                params=getattr(req.sampling_params,'custom_params',None) or {}
                meta=params.get('hohlraum')
                if meta is None or not req.finished():
                    continue
                completed=native_completion_timestamp(req,before_output)
                if leader:
                    s._experiment_completion_events.append(dict(
                        kind='complete',rid=req.rid,monotonic=completed,
                        meta={key:meta[key] for key in ('root','chain')}))
                if completion_audit:
                    from sglang.srt.runtime_context import get_parallel
                    with open(f'{completion_audit}.rank{get_parallel().tp_rank}.jsonl','a') as out:
                        out.write(json.dumps(dict(rid=req.rid,monotonic=completed,authoritative=leader))+'\n')
        return value
    Scheduler.process_input_requests = inputs
    Scheduler.get_next_batch_to_run = next_batch
    Scheduler.process_batch_result = result
    Scheduler._experiment_feedback = True
