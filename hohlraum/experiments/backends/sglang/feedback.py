"""Local experiment transport; reads queue state, never changes scheduling."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from functools import wraps


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

    def notify_ready(self, req):
        meta = req.sampling_params.custom_params['hohlraum']
        self.ready.write(json.dumps(dict(rid=req.rid, meta=meta, monotonic=time.monotonic(),
                                        input_length=len(req.input_ids),page_size=1024))+'\n')

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
        waiting = row['waiting_rids']
        ages = []
        for rid in waiting:
            rec = self.gate.records.get(rid)
            if rec is None or 'engine_admitted_at' not in rec:
                raise RuntimeError('Unexpected request in dedicated experiment engine: '+rid)
            ages.append(max(0., now-rec['engine_admitted_at']))
        return max(ages, default=0.), len(waiting)

    def close(self):
        self.ready.close(); self.received.close()


def install(directory):
    from sglang.srt.managers.scheduler import Scheduler
    if getattr(Scheduler, '_experiment_feedback', False):
        return
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    old_input = Scheduler.process_input_requests
    old_next = Scheduler.get_next_batch_to_run

    def primary():
        from sglang.srt.runtime_context import get_parallel
        return get_parallel().tp_rank == 0

    def update(s, force=False):
        host = getattr(s.tree_cache, 'hohlraum_host', None)
        if not hasattr(s, '_experiment_ready'):
            (root/'ready.jsonl').touch(exist_ok=True)
            s._experiment_ready = (root/'ready.jsonl').open()
            s._experiment_last_publish = 0.
        for event in read_events(s._experiment_ready):
            if host is not None:
                req = SimpleNamespace(rid=event['rid'], input_length=event.get('input_length',0), sampling_params=SimpleNamespace(custom_params={'hohlraum':event['meta']}))
                with host.lock:
                    if event.get('hashes') is not None and event['page_size']!=host.cache.page_size:
                        raise ValueError('client-ready hash page size differs from engine')
                    host.sli.client_ready(req, event['monotonic']-host.started_at,event.get('hashes'))
        now = time.monotonic()
        if primary() and (force or now-s._experiment_last_publish >= .05):
            atomic(root/'queue.json', dict(monotonic=now, waiting_rids=[r.rid for r in s.waiting_queue if not r.finished()]))
            s._experiment_last_publish = now

    @wraps(old_input)
    def inputs(s, requests, *args, **kwargs):
        update(s)
        rids = [item.rid for request in requests for item in s._tokenized_requests(request)]
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
        update(s)
        return value
    Scheduler.process_input_requests = inputs
    Scheduler.get_next_batch_to_run = next_batch
    Scheduler._experiment_feedback = True
