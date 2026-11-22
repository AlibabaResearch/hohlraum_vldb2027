"""Unified dependency-driven external request source; no simulator clock changes."""
import heapq, os, json, math
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
    # Prepared graphs may contain stale same-chain lags. Trace think_time
    # is authoritative; preserve all cross-chain dependency edges.
    for row in rows:
        root,chain,index=(int(row[k]) for k in ("root","chain","index"))
        if not index:continue
        lag=float(row["think_time"])
        if not math.isfinite(lag) or lag<0:
            raise ValueError("invalid think_time")
        edges=plans[str(root)]["deps"][f"{chain}:{index}"]
        for edge in edges:
            if int(edge[0])==chain and int(edge[1])==index-1:edge[2]=lag
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
    # Preserve each input format's incomplete-page identity.
    if dataset == "agentx":
        return dict(schema_version=1, token_tail="serial_token")
    if dataset == "playing":
        return dict(schema_version=1, token_tail="serial_bytes")
    raise ValueError("unsupported source")


class TraceSource:

    def __init__(self, queued_requests, concurrency, plans=None):
        self.concurrency = concurrency
        self.idle_timeout = float(os.environ.get('HISIM_NEW_ROOT_IDLE_TIMEOUT_S', '1200'))
        self.active_generation = {}
        self.outstanding_by_root = defaultdict(int)
        self.last_root_activity = {}
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
        self.lifecycle_history = []
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
            path = os.environ.get('HOHLRAUM_TRACE_DEPENDENCIES')
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
        # External hourly-session dependencies: successors are offered only
        # after the preceding hourly session finishes plus its recorded gap.
        self.hourly_successors = defaultdict(list)
        dependent = set()
        for child, plan in plans.items():
            if 'after_root' not in plan: continue
            parent, lag = plan['after_root']; child=int(child); parent=int(parent)
            if parent not in self.offers or child==parent or lag<0:
                raise ValueError('invalid hourly predecessor')
            self.hourly_successors[parent].append((child,float(lag)))
            dependent.add(child)
        self.events=[e for e in self.events if not (e[2]=='offer' and e[3] in dependent)]
        heapq.heapify(self.events)


    def _event(self, when, kind, payload):
        self.serial += 1
        heapq.heappush(self.events, (when, self.serial, kind, payload))

    def _start_pending(self, now):
        while self.pending and len(self.live) < self.concurrency:
            root = self.pending.popleft()
            self.live.add(root)
            self.admission_audit.append(dict(t=now, event='start', root=root, active=len(self.live), external_arrival=self.offers[root], admission_delay_s=now - self.offers[root]))
            self.active_generation[root] = 0
            self.last_root_activity[root] = now
            self._release_idle_root(root, now)
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
                if (self.active_generation.get(root) == generation and root in self.live
                        and not self.outstanding_by_root[root]):
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
                self.outstanding_by_root[root] += 1
                self.last_root_activity[root] = when
                req.sampling_params.custom_params['simulation']['created_time'] = when
                self.arrivals[req.rid] = when
                self.lifecycle_history.append(dict(key=list(payload),t=when,kind="ready"))
                due.append(req)
        # Returning roots already bypass admission above. The configured limit
        # bounds new-root admissions, not roots returning after idle-slot release.
        return due

    def next_time(self):
        return self.events[0][0] if self.events else None

    def finish(self, req, now):
        if req.rid in self.completed:
            return
        key = self.key_by_rid[req.rid]
        (root, chain, index) = key
        self.completed.add(req.rid)
        self.outstanding_by_root[root] -= 1
        assert self.outstanding_by_root[root] >= 0
        self.lifecycle_history.append(dict(key=list(key),t=now,kind="complete"))
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
            self.admission_audit.append(dict(t=now, event='trace_end',
                                            root=root, active=len(self.live)))
        # Only explicit hourly-session chains release at segment completion.
        # Ordinary root workloads retain their existing idle-slot semantics.
        if self.remaining[root] == 0 and self.hourly_successors:
            for child, lag in self.hourly_successors.get(root, []):
                self.offers[child] = now + lag
                self._event(now + lag, 'offer', child)
            if root in self.live:
                self.live.remove(root)
                self.active_generation[root] += 1
                self.admission_audit.append(dict(t=now,event='hourly_complete_release',root=root,active=len(self.live)))
                self._start_pending(now)
        self.last_root_activity[root] = now
        self._release_idle_root(root, now)

    def _release_idle_root(self, root, now):
        if root in self.live and not self.outstanding_by_root[root]:
            # Every session under this root must be idle. A completed trace
            # keeps its slot too; only 20 elapsed idle minutes release it.
            # Pending/executing requests suppress the timer until completion.
            self.active_generation[root] += 1
            generation = self.active_generation[root]
            deadline = max(now, self.last_root_activity.get(root, now)
                           + self.idle_timeout)
            self._event(deadline, 'idle_release', (root, generation))

def dump(source, now):

    def key(req):
        m = req.sampling_params.custom_params['simulation']['agentx']
        return [m['root'], m['chain'], m['index']]
    history = sorted(source.lifecycle_history, key=lambda event: event['t'])
    return dict(clock=now, last_root_activity=source.last_root_activity, lifecycle_history=history, events=source.events, pending=list(source.pending), live=list(source.live), active_generation=source.active_generation, offers=source.offers, offered_count=source.offered_count, serial=source.serial, remaining_keys=[key(r) for r in source.requests.values() if r.rid not in source.completed], outstanding=[dict(key=key(r), ready=source.arrivals[r.rid]) for r in source.requests.values() if r.rid in source.arrivals and r.rid not in source.completed], pending_deps=[[list(k), v] for (k, v) in source.pending_deps.items()], ready_at=[[list(k), v] for (k, v) in source.ready_at.items()])

def restore(source, ck):
    cut = ck['clock']
    source.lifecycle_history = [dict(e,t=e['t']-cut) for e in ck['lifecycle_history']]
    # Outstanding old requests restart at time zero, after cancellation.
    pending_history={tuple(e['key']) for e in ck['outstanding']}
    source.lifecycle_history=[e for e in source.lifecycle_history if tuple(e['key']) not in pending_history]
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
    source.outstanding_by_root = defaultdict(int)
    source.last_root_activity = {int(k): v-cut for k,v in
                                 ck['last_root_activity'].items()}
    for (when, serial, kind, payload) in ck['events']:
        # Rebuild idle timers from root activity on the resumed clock.
        if kind == 'idle_release':
            continue
        if kind == 'request':
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
    # Rebuild idle deadlines from last activity, including ended roots.
    for root in list(source.live):
        source._release_idle_root(root, 0.)
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


def encode_array(root, blocks, raw_tokens, serial, vocab_size=4000, mode='serial_bytes'):
    """Construct the same token buffer without an intermediate Python list."""
    from array import array
    if mode != 'serial_bytes':
        return array('q', encode(root, blocks, raw_tokens, serial, vocab_size, mode))
    if not 0 <= root < 2 ** 32 or serial < 0 or raw_tokens < 0:
        raise ValueError('Invalid request identity or length')
    if raw_tokens // 1024 != len(blocks) or vocab_size < 272:
        raise ValueError('Invalid page count or vocabulary')
    serial_marker = array('q', (16 + b for b in serial.to_bytes(8, 'big')))
    ids = array('q', [7]) * raw_tokens
    prefix = root.to_bytes(4, 'big')
    for (i, block) in enumerate(blocks):
        marker = prefix + int(block).to_bytes(8, 'big')
        ids[i * 1024:i * 1024 + 12] = array('q', (16 + b for b in marker))
    tail = raw_tokens % 1024
    if tail:
        marker = serial_marker
        n = min(tail, 8)
        ids[len(blocks) * 1024:len(blocks) * 1024 + n] = marker[:n]
    return ids


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
            raise ValueError('AgentX expects a local fold or legacy encoded fold in [0,4)')
        # Legacy rows encoded a main/child bit. It is not a scoring feature.
        return fold % 2
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
        if isinstance(training, dict):
            training = training['folds']
        matches=[m for m in training if m['provider']==provider and m['fold']==fold]
        if len(matches)!=1 or trajectory not in matches[0]['test'] or trajectory in matches[0]['train']:
            raise ValueError('Playing trajectory does not belong to the selected held-out fold')
    return 2*provider_ids[source]+fold


def bootstrap_at_first_ready(source):
    """Start OFFLINE measurement at its first ready request, without clock writes.

    Consume only pre-request client events. Preserve the order and relative
    times of all later events and dependency lags.
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
    source.lifecycle_history = [dict(e,t=e['t']-origin) for e in getattr(source,'lifecycle_history',[])]
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


def validate_measurement_duration(seconds, *, warmup=False):
    """Formal performance measurements use the fixed four-hour protocol."""
    if warmup and seconds == 3600:
        return
    if seconds != 14400:
        raise ValueError('Formal measurements require 14400 seconds (4 hours)')


def uniform_warmup_offers(rows, plans, concurrency, warmup_seconds=3600):
    """Spread initial roots over warmup; retain all within-root dependencies/delays."""
    if concurrency < 1 or warmup_seconds <= 0:
        raise ValueError('Positive concurrency and warmup duration required')
    roots = sorted({r['root'] for r in rows
                    if 'after_root' not in plans[str(r['root'])]})
    rank = {root: i for i, root in enumerate(roots)}
    return [dict(row, root_offer=(rank[row['root']] * warmup_seconds / concurrency
                                 if row['root'] in rank else 0.)) for row in rows]
