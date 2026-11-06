"""Unified dependency-driven external request source; no simulator clock changes."""
import heapq, os, json
from pathlib import Path
from collections import defaultdict, deque


"""Normalize prepared request rows into one dependency graph for both backends."""
import copy

def normalize(rows, dependencies=None):
    plans = copy.deepcopy(dependencies) if dependencies is not None else {}
    keys = {(int(r["root"]), int(r["chain"]), int(r["index"])) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate request identity")
    if dependencies is None:
        for row in rows:
            root, chain, index = (int(row[k]) for k in ("root", "chain", "index"))
            plan = plans.setdefault(str(root), {"deps": {}, "first_delays": {}})
            if index:
                if (root, chain, index-1) not in keys:
                    raise ValueError("missing preceding request")
                edges = [[chain, index-1, max(0., float(row["think_time"]))]]
            else:
                edges = []
                plan["first_delays"][str(chain)] = float(row["first_delay"])
            plan["deps"][f"{chain}:{index}"] = edges
    degree = {}; children = defaultdict(list)
    for root, chain, index in keys:
        plan = plans[str(root)]
        edges = plan["deps"][f"{chain}:{index}"]
        degree[root,chain,index] = len(edges)
        seen = set()
        for pc, pi, lag in edges:
            parent = (root, int(pc), int(pi))
            if parent not in keys or float(lag) < 0 or parent in seen:
                raise ValueError("invalid dependency")
            seen.add(parent); children[parent].append((root,chain,index))
        if not edges and (index != 0 or float(plan["first_delays"][str(chain)]) < 0):
            raise ValueError("invalid initial request")
    todo = deque(k for k,v in degree.items() if v == 0); visited = 0
    while todo:
        key = todo.popleft(); visited += 1
        for child in children[key]:
            degree[child] -= 1
            if degree[child] == 0: todo.append(child)
    if visited != len(keys): raise ValueError("cyclic request dependencies")
    return plans

def input_contract(dataset):
    # Retain historical token-tail encoding and expansion stage during migration.
    if dataset == "agentx":
        return dict(schema_version=1, token_tail="serial_token", encode_on_ready=False,
                    detailed_audit=True)
    if dataset == "playing":
        return dict(schema_version=1, token_tail="serial_bytes", encode_on_ready=True,
                    detailed_audit=False)
    raise ValueError("unsupported source")


class TraceSource:

    def __init__(self, queued_requests, concurrency, plans=None):
        self.concurrency = concurrency
        self.idle_timeout = float(os.environ.get('HISIM_NEW_ROOT_IDLE_TIMEOUT_S', '1200'))
        self.active_generation = {}
        self.admission_audit = []
        self.requests = {}
        self.remaining = defaultdict(int)
        self.first = defaultdict(list)
        self.offers = {}
        self.events = []
        self.pending = deque()
        self.live = set()
        self.completed = set()
        self.serial = 0
        self.arrivals = {}
        self.audit = []
        self.offered_count = 0
        for (_, _, req) in queued_requests:
            m = req.sampling_params.custom_params['simulation']['agentx']
            (root, chain, index) = (int(m[k]) for k in ('root', 'chain', 'index'))
            self.requests[root, chain, index] = req
            self.remaining[root] += 1
            self.offers[root] = float(m['root_offer'])
            if index == 0:
                self.first[root].append((chain, float(m['first_delay'])))
        for (root, when) in self.offers.items():
            self._event(when, 'offer', root)
        if plans is None:
            path = os.environ.get('HOHLRAUM_TRACE_DEPENDENCIES') or os.environ.get('HISIM_AGENTX_DEPENDENCIES')
            if path:
                plans = json.loads(Path(path).read_text())
            else:
                rows = [req.sampling_params.custom_params['simulation']['agentx'] for req in self.requests.values()]
                plans = normalize(rows)
        self.plans = plans
        self.successors = defaultdict(list)
        self.pending_deps = {}
        self.ready_at = {}
        self.key_by_rid = {r.rid: k for (k, r) in self.requests.items()}
        self.initial = defaultdict(list)
        for (key, req) in self.requests.items():
            (root, chain, index) = key
            edges = plans[str(root)]['deps'][f'{chain}:{index}']
            self.pending_deps[key] = len(edges)
            self.ready_at[key] = 0.0
            for (pc, pi, lag) in edges:
                parent = (root, pc, pi)
                assert parent in self.requests and lag >= 0
                self.successors[parent].append((key, lag))
            if not edges:
                assert index == 0
                self.initial[root].append((key, float(plans[str(root)]['first_delays'][str(chain)])))
        assert set(self.initial) == set(self.offers)

    def _event(self, when, kind, payload):
        self.serial += 1
        heapq.heappush(self.events, (when, self.serial, kind, payload))

    def _start_pending(self, now):
        while self.pending and len(self.live) < self.concurrency:
            root = self.pending.popleft()
            self.live.add(root)
            self.admission_audit.append(dict(t=now, event='start', root=root, active=len(self.live), external_arrival=self.offers[root], admission_delay_s=now - self.offers[root]))
            self.active_generation[root] = 0
            self._event(now + self.idle_timeout, 'idle_release', (root, 0))
            for (key, lag) in self.initial[root]:
                self._event(now + lag, 'request', key)

    def pop_due(self, now):
        due = []
        while self.events and self.events[0][0] <= now:
            (when, _, kind, payload) = heapq.heappop(self.events)
            if kind == 'offer':
                self.pending.append(payload)
                self.offered_count += 1
                self.admission_audit.append(dict(t=when, event='offer', root=payload, active=len(self.live)))
                self._start_pending(when)
            elif kind == 'idle_release':
                (root, generation) = payload
                if self.active_generation.get(root) == generation and root in self.live:
                    self.live.remove(root)
                    self.admission_audit.append(dict(t=when, event='idle_release', root=root, active=len(self.live)))
                    self._start_pending(when)
            else:
                req = self.requests[payload]
                root = payload[0]
                returning = root not in self.live
                self.live.add(root)
                self.admission_audit.append(dict(t=when, event='request', root=root, active=len(self.live), returning_bypass=returning))
                generation = self.active_generation[root] + 1
                self.active_generation[root] = generation
                self._event(when + self.idle_timeout, 'idle_release', (root, generation))
                req.sampling_params.custom_params['simulation']['created_time'] = when
                self.arrivals[req.rid] = when
                due.append(req)
        return due

    def next_time(self):
        return self.events[0][0] if self.events else None

    def finish(self, req, now):
        if req.rid in self.completed:
            return
        key = self.key_by_rid[req.rid]
        (root, chain, index) = key
        self.completed.add(req.rid)
        assert self.pending_deps[key] == 0
        self.audit.append(dict(root=root, chain=chain, index=index, arrival=self.arrivals[req.rid], completion=now, dependency_ready_at_s=self.ready_at[key]))
        for (child, lag) in self.successors.get(key, []):
            self.pending_deps[child] -= 1
            assert self.pending_deps[child] >= 0
            self.ready_at[child] = max(self.ready_at[child], now + lag)
            if self.pending_deps[child] == 0:
                self._event(self.ready_at[child], 'request', child)
        self.remaining[root] -= 1
        if self.remaining[root] == 0:
            occupied = root in self.live
            self.live.discard(root)
            self.active_generation[root] += 1
            self.admission_audit.append(dict(t=now, event='trace_end_release', root=root, active=len(self.live), occupied=occupied))
            self._start_pending(now)

def dump(source, now):

    def key(req):
        m = req.sampling_params.custom_params['simulation']['agentx']
        return [m['root'], m['chain'], m['index']]
    return dict(clock=now, events=source.events, pending=list(source.pending), live=list(source.live), active_generation=source.active_generation, offers=source.offers, offered_count=source.offered_count, serial=source.serial, remaining_keys=[key(r) for r in source.requests.values() if r.rid not in source.completed], outstanding=[dict(key=key(r), ready=source.arrivals[r.rid]) for r in source.requests.values() if r.rid in source.arrivals and r.rid not in source.completed], pending_deps=[[list(k), v] for (k, v) in source.pending_deps.items()], ready_at=[[list(k), v] for (k, v) in source.ready_at.items()])

def restore(source, ck):
    ck = dict(ck)
    if 'pending_deps' not in ck:
        keep = {tuple(k) for k in ck['remaining_keys']}
        ready = {tuple(e[3]) for e in ck['events'] if e[2] == 'request'}
        ready.update(tuple(x['key']) for x in ck['outstanding'])
        pending = []
        for key in source.requests:
            root, chain, index = key
            edges = source.plans[str(root)]['deps'][f'{chain}:{index}']
            count = 0 if key in ready else sum((root, pc, pi) in keep for pc, pi, lag in edges)
            pending.append([list(key), count])
        ck['pending_deps'] = pending
        ck['ready_at'] = [[list(k), 0.] for k in source.requests]
    cut = ck['clock']
    keep = {tuple(k) for k in ck['remaining_keys']}
    source.requests = {k: r for (k, r) in source.requests.items() if k in keep}
    assert set(source.requests) == keep
    source.remaining = defaultdict(int)
    for k in keep:
        source.remaining[k[0]] += 1
    source.pending = deque(ck['pending'])
    source.live = set(ck['live'])
    source.active_generation = {int(k): v for (k, v) in ck['active_generation'].items()}
    source.offers = {int(k): v - cut for (k, v) in ck['offers'].items()}
    source.offered_count = ck['offered_count']
    source.serial = ck['serial']
    source.pending_deps = {tuple(k): v for (k, v) in ck['pending_deps']}
    source.ready_at = {tuple(k): max(0, v - cut) for (k, v) in ck['ready_at']}
    source.events = []
    for (when, serial, kind, payload) in ck['events']:
        if kind in ('request', 'idle_release'):
            payload = tuple(payload)
        source.events.append((max(0, when - cut), serial, kind, payload))
    heapq.heapify(source.events)
    outstanding = sorted(ck['outstanding'], key=lambda x: x['ready'])
    for item in outstanding:
        k = tuple(item['key'])
        assert source.pending_deps[k] == 0
        source._event(0, 'request', k)
    scheduled = [tuple(e[3]) for e in source.events if e[2] == 'request']
    assert len(scheduled) == len(set(scheduled))
    assert all((k in keep and source.pending_deps[k] == 0 for k in scheduled))
    assert all((v >= 0 for v in source.pending_deps.values()))
    source.admission_audit.append(dict(t=0, event='restore_cold', active=len(source.live), root=None))
    return dict(remaining=len(keep), outstanding=len(outstanding), scheduled=len(scheduled), live=len(source.live), dependency_check='passed')

PAGE = 1024
def encode_serial_token(root, blocks, raw_tokens, serial, vocab_size=4000):
    """每个完整块可逆编码；跨抽样任务隔离，不伪造尾块复用。"""
    if not 0 <= serial < vocab_size - 3000 or raw_tokens // PAGE != len(blocks):
        raise ValueError('请求长度或序号越界')
    ids = []
    for block in blocks:
        marker = int(root).to_bytes(4, 'big') + int(block).to_bytes(8, 'big')
        ids.extend([16 + b for b in marker] + [7] * (PAGE - len(marker)))
    ids.extend([3000 + serial] * (raw_tokens % PAGE))
    return ids

def encode_serial_bytes(root, blocks, raw_tokens, serial, vocab_size=4000):
    if not 0 <= root < 2 ** 32 or serial < 0 or raw_tokens < 0:
        raise ValueError('Invalid request identity or length')
    if raw_tokens // PAGE != len(blocks) or vocab_size < 272:
        raise ValueError('Invalid page count or vocabulary')
    ids = []
    for block in blocks:
        marker = root.to_bytes(4, 'big') + int(block).to_bytes(8, 'big')
        ids.extend([16 + b for b in marker] + [7] * (PAGE - len(marker)))
    tail = raw_tokens % PAGE
    marker = [16 + b for b in serial.to_bytes(8, 'big')]
    ids.extend((marker + [7] * tail)[:tail])
    return ids

def encode(root, blocks, raw_tokens, serial, vocab_size=4000, mode="serial_bytes"):
    if mode == "serial_token":
        return encode_serial_token(root, blocks, raw_tokens, serial, vocab_size)
    if mode == "serial_bytes":
        return encode_serial_bytes(root, blocks, raw_tokens, serial, vocab_size)
    raise ValueError("unknown token-tail encoding")


def reference_model_fold(row, dataset, training=None):
    """Convert dataset-local folds to the scoring model's provider/fold index.

    Playing rows retain their original local 0/1 fold. Never rewrite those rows
    in place: both serving backends must perform exactly this conversion once.
    """
    fold=row['reference_fold']
    if type(fold) is not int:
        raise ValueError('reference_fold must be an integer')
    if dataset=='agentx':
        if not 0<=fold<4:
            raise ValueError('AgentX expects a global model index in [0,4)')
        return fold
    if dataset!='playing':
        raise ValueError('Unknown dataset for model selection')
    trajectory=row.get('source_trajectory','')
    source=trajectory.split('/',1)[0]
    provider_ids={'claude_13agents':0,'gpt_5agents':1,'kimi_5agents':2}
    if source not in provider_ids or '/' not in trajectory:
        raise ValueError('Playing requires a recognized source_trajectory')
    if fold not in (0,1):
        raise ValueError('Playing reference_fold must be local 0/1, not a global model index')
    provider=source.split('_',1)[0]
    if training is not None:
        matches=[m for m in training if m['provider']==provider and m['fold']==fold]
        if len(matches)!=1 or trajectory not in matches[0]['test'] or trajectory in matches[0]['train']:
            raise ValueError('Playing trajectory does not belong to the selected held-out fold')
    return 2*provider_ids[source]+fold


def bootstrap_at_first_ready(source):
    """Start OFFLINE measurement at its first ready request, without clock writes.

    Consume only pre-request client events. Preserve the order and relative
    times of all later events and dependency lags. Works for cold starts and
    restored checkpoints with or without outstanding requests.
    """
    ready = []
    origin = 0.0
    while source.next_time() is not None:
        origin = source.next_time()
        ready = source.pop_due(origin)
        if ready:
            break
    if not ready:
        return [], dict(origin_shift_s=origin, initial_ready=0)
    source.events = [(t-origin, serial, kind, payload)
                     for t, serial, kind, payload in source.events]
    heapq.heapify(source.events)
    source.offers = {k: t-origin for k,t in source.offers.items()}
    source.ready_at = {k: max(0.,t-origin) for k,t in source.ready_at.items()}
    source.arrivals = {k: t-origin for k,t in source.arrivals.items()}
    for item in source.admission_audit:
        for key in ('t', 'external_arrival'):
            if key in item:
                item[key] -= origin
    for req in ready:
        req.sampling_params.custom_params['simulation']['created_time'] = 0.
        assert source.arrivals[req.rid] == 0.
    source.admission_audit.append(dict(t=0., event='measurement_origin',
                                       origin_shift_s=origin, initial_ready=len(ready)))
    return ready, dict(origin_shift_s=origin, initial_ready=len(ready))
