"""Staging policy: HTS candidates and constraints, ranking, and admission."""
import math
from bisect import bisect_left, insort
from collections import Counter, deque
from dataclasses import dataclass
from hohlraum.core.sli import StagingRangeIndex

@dataclass(frozen=True)
class RetirementDecision:
    write: bool
    budget_exhausted: bool
    retry: bool = False


class StagingPolicy:
    """Coordinate HTS candidates, ranking, admission and prefix obligations.

    SLI supplies lifecycle profiles. Residency executes the returned decisions.
    HTS and dynamic admission are methods of this policy; read LRU is unchanged.
    """
    def __init__(self, host):
        self.host = host
        self.cfg = cfg = host.cfg
        self.index = StagingRangeIndex()
        self._round_active = False
        self._round_snapshot = None
        self._waiting_ready = False
        self.reset_prefix_stats()
        self.selections = 0
        self.histogram = [0] * 100
        self.override_count = 0
        self.threshold_log = []
        self.next_tick = float(cfg.get('threshold_period_s', 1800.))
        self.dynamic_admission = bool(cfg.get("dynamic_admission_threshold", False))
        if self.dynamic_admission:
            self._init_admission_feedback(host.admission_threshold, cfg.get("duration_s",14400),
                                          cfg.get("threshold_period_s",1800), cfg.get("threshold_window_s",1800))

    def accepts_staging(self, node):
        """Reject existing SSD copies before transferring read ownership."""
        return not self.host._has_ssd_copy(node)

    def discard_on_entry(self, node):
        """An already-rejectable tail requires no waiting-residency slot."""
        h = self.host
        if h.lifecycle is None or h.admission_policy != "agent" or h.residency_policy != "joint_hts":
            return False
        key = node.hash_value[-1]
        if key not in self.index.key_sessions or h.has_resident_descendant(key):
            return False
        keys = [key]
        now, snapshot = self._ranking_snapshot(h.now())
        profiles = self._rank_profiles(keys, now, snapshot)
        if self.decisions_from_profiles(keys, profiles)[0]:
            return False
        self.decisions_from_profiles(keys, profiles, record=True)
        return True

    def choose(self):
        host = self.host
        if (host.residency_policy == 'joint_hts' and host.lifecycle is not None
                and not host._staging_return_flush):
            if not any(e.operation is None and not e.read_pins for e in host.stage.values()):
                return None
            if not host._budget_exhausted():
                return self.choose_ranked()
        candidates = [e for e in host.stage.values() if e.operation is None and not e.read_pins]
        if not candidates:
            return None
        returning = [e for e in candidates if e.hash_value[0] in host._staging_return_flush]
        if returning:
            return min(returning, key=lambda e:(len(e.prefix_hashes), e.last_access_time, e.id))
        if host.residency_policy == 'joint_hts' and host.lifecycle is not None and not host._budget_exhausted():
            return self.choose_ranked()
        return min(candidates, key=lambda e:(e.last_access_time, e.id))

    def begin_round(self):
        self._round_active = True
        self._round_snapshot = None
        self._waiting_ready = False

    def end_round(self):
        self._round_active = False
        self._round_snapshot = None
        self._waiting_ready = False
        self.index.waiting.clear()

    def invalidate_round(self):
        self._round_snapshot = None

    def _ranking_snapshot(self, now):
        if not self._round_active:
            return now, None
        sli = self.host.sli
        updates = (sli.lib.hh_statistics_updates(sli.ptr)
                   if hasattr(sli, 'lib') and hasattr(sli.lib, 'hh_statistics_updates') else 0)
        epoch = (getattr(sli, 'revision', 0), self.host.admission_threshold, updates)
        snapshot = self._round_snapshot
        # Strictly less than 0.1s; never let a sliding window extend a snapshot.
        if (snapshot is None or snapshot['epoch'] != epoch
                or now < snapshot['time'] or now >= snapshot['time'] + 0.1):
            snapshot = dict(time=now, epoch=epoch, profiles={}, context={}, position={},
                            order=[], sortkeys={})
            self._round_snapshot = snapshot
        return snapshot['time'], snapshot

    def _rank_profiles(self, keys, now, snapshot):
        if snapshot is None:
            return self.host.sli.profiles(keys, now)
        missing = [key for key in keys if key not in snapshot['profiles']]
        if missing:
            snapshot['profiles'].update(self.host.sli.profiles(missing, now))
        return {key:snapshot['profiles'][key] for key in keys}

    def _rank_risks(self, keys, snapshot):
        if snapshot is None:
            return self.host.sli.context_risks(keys), self.host.sli.position_risks(keys)
        missing = [key for key in keys if key not in snapshot['context']]
        if missing:
            snapshot['context'].update(self.host.sli.context_risks(missing))
            snapshot['position'].update(self.host.sli.position_risks(missing))
        return ({key:snapshot['context'][key] for key in keys},
                {key:snapshot['position'][key] for key in keys})

    def _ordered_waiting(self, waiting, weights, snapshot):
        sortkey = lambda key:(-weights[key],self.host.sli.ids[key])
        if snapshot is None:
            return sorted(waiting, key=sortkey)
        # Preserve stable-sort ties even if several native hashes share a handle.
        counts = Counter(self.host.sli.ids[key] for key in waiting)
        current = {key:sortkey(key)+(i if counts[self.host.sli.ids[key]]>1 else 0,key)
                   for i,key in enumerate(waiting)}
        previous = snapshot['sortkeys']; order = snapshot['order']
        for key, value in previous.items():
            if current.get(key) != value:
                at = bisect_left(order, value)
                assert at < len(order) and order[at] == value
                order.pop(at)
        for key, value in current.items():
            if previous.get(key) != value:
                insort(order, value)
        snapshot['sortkeys'] = current
        return [value[-1] for value in order]

    def choose_ranked(self):
        host = self.host
        now, snapshot = self._ranking_snapshot(host.now())
        roles = self._exit_roles()
        # A staging entry is always a single page. In-flight writes are
        # outside the waiting range and pinned pages cannot exit.
        candidates = [key for key in roles[0] | roles[1]
                      if (entry := host.stage.get(key)) is not None
                      and entry.operation is None and not entry.read_pins]
        if not candidates:
            return None
        threshold = host.admission_threshold
        profiles = self._rank_profiles(candidates, now, snapshot)
        def accepts(key):
            return profiles[key][0]>=threshold or host.sli.pending.get(key,0)>0
        candidates = [key for key in candidates
                      if not self.protect_head(key, profiles[key][0], threshold)
                      and self._endpoint_exit_allowed(key, accepts(key), roles)]
        if not candidates:return None

        keys = candidates
        context,position=self._rank_risks(keys,snapshot)
        # Session context risk and per-KV position risk add in residency
        # ordering. Admission probabilities and TTD remain unchanged.
        risks={key:context[key]+position[key] for key in keys}
        # A combined risk >= 1 has zero survival weight; retain the raw sum
        # for risk ordering. Context risk 1 already has zero base weight.
        weights={key:(0. if risks[key]>=1. else
                      profiles[key][2]*(1-risks[key])/(1-context[key])) for key in keys}
        # Legal tails already eligible for rejection always reach final HTS
        # ordering. Risk protects KV that still needs waiting; it must not hide
        # a discardable tail by reducing its prefilter weight to zero.
        # Pending reuse still overrides rejection, as in admission.
        discardable=[key for key in keys if not accepts(key)]
        waiting=[key for key in keys if accepts(key)]
        waiting=self._ordered_waiting(waiting,weights,snapshot)
        candidates=discardable+waiting[:(len(waiting)+1)//2]
        # Infinite TTD separates discards from writes in the final ordering.
        def exit_priority(key):
            score,wait,*_=profiles[key]
            weight=weights[key]
            if math.isinf(wait):
                if not accepts(key):return (wait,1,-score,weight,host.sli.ids[key])
                # Low risk exits first; probability roundoff must not erase
                # protection for KV close to a context change.
                return (wait,0,-risks[key],weight,host.sli.ids[key])
            return (wait,0,-score,0.,host.sli.ids[key])
        selected=max(candidates,key=exit_priority)
        self.selections+=1
        return host.stage[selected]


    def decisions(self,keys,now,record=False):
        now,snapshot=self._ranking_snapshot(now)
        return self.decisions_from_profiles(keys,self._rank_profiles(keys,now,snapshot),record)
    def decisions_from_profiles(self,keys,p,record=False):
        out=[]
        for h in keys:
            score=p[h][0]
            if record:self.histogram[min(99,int(score*100))]+=1
            if record and self.dynamic_admission:self._observe_admission(h,score,self.host.now())
            override=score<self.host.admission_threshold and self.host.sli.pending.get(h,0)>0
            if record:self.override_count+=int(override)
            out.append(score>=self.host.admission_threshold or override)
        return out
    def maybe_control(self,now):
        if not self.dynamic_admission or now < self.next_tick or self.next_tick > self.cfg['duration_s']:return
        scheduled=self.next_tick
        page_bytes=self.host.ssd_backend.page_bytes
        budget=self.host.write_budget_bytes
        if budget is None:raise ValueError('Dynamic admission requires a finite write budget')
        row=self._update_admission(now,self.host.admission_threshold,self.host.budget_charged_bytes//page_bytes,budget//page_bytes)
        row.update(scheduled_t=scheduled,page_bytes=page_bytes)
        self.host.admission_threshold=row['after']
        self.host.sli.set_threshold(row['after'])
        self.threshold_log.append(row)
        if self.cfg.get('stats_path'):
            import json
            with open(self.cfg['stats_path']+'.threshold.jsonl','a') as trace:trace.write(json.dumps(row)+'\n')
        period=self.cfg.get('threshold_period_s',1800.)
        while self.next_tick<=now:self.next_tick+=period

    def retire(self, entry, copied, *, rejected_on_entry=False):
        host = self.host
        over = host._budget_exhausted()
        # Existing copies and exhausted budgets use cleanup, never new writes.
        if copied or over or host.admission_policy == 'write_none':
            return RetirementDecision(False, over)
        # A rejected read tail enters and exits in the same eviction call.
        # Its admission observation has already been recorded.
        if rejected_on_entry:
            assert self._endpoint_exit_allowed(entry.hash_value[0], False)
            return RetirementDecision(False, over)
        decisions = self.decisions(entry.hash_value, host.now(), record=True) if host.lifecycle else [True]
        accepted = any(decisions)
        if not self._endpoint_exit_allowed(entry.hash_value[0], accepted):
            return RetirementDecision(False, over, retry=True)
        if accepted:
            status = self.prepare_head_write(entry)
            if status == 'retry':
                return RetirementDecision(False, over, retry=True)
        return RetirementDecision(accepted, over)

    def waiting_changed(self, key):
        if not self._round_active or not self._waiting_ready:
            return
        entry = self.host.stage.get(key)
        self.index.waiting.set_active(key, entry is not None and entry.operation is None)

    def _exit_roles(self):
        if self._round_active:
            if not self._waiting_ready:
                active = {key:entry for key,entry in self.host.stage.items() if entry.operation is None}
                self.index.waiting.reset(active)
                self._waiting_ready = True
            return self.index.waiting.roles()
        active={key:entry for key,entry in self.host.stage.items() if entry.operation is None}
        return self.index.endpoints(active, with_roles=True)

    def _endpoint_exit_allowed(self, key, accepted, roles=None):
        heads,tails=self._exit_roles() if roles is None else roles
        if not accepted:
            return key in tails and not self.host.has_resident_descendant(key)
        return key in heads

    def stats(self):
        return dict(pending_read_overrides=self.override_count,
                    histogram=self.histogram, threshold_log=self.threshold_log)

    # HTS: candidates, head protection and tail-prefix obligations.
    def observe(self, session, hashes):
        self.index.observe(session, hashes)

    def protect_head(self, key, score, threshold):
        """Do not reject a prefix still needed by resident descendants."""
        return score < threshold and self.host.has_resident_descendant(key)

    def reset_prefix_stats(self):
        self._prefix_events = Counter()

    def _pending_prefix_keys(self):
        return {k for e in self.host.inflight.values() for k in e.hash_value}

    def prepare_head_write(self, head):
        # Only a staging head may be admitted. Copy its missing read ancestors
        # without evicting read payloads; never force another waiting page out.
        return self.host.persist_prefixes(head.prefix_hashes, head_key=head.hash_value[0])

    def prefix_write_stats(self):
        return dict(self._prefix_events,
                    semantics='head_admission_copies_read_prefixes_with_native_async_io')


    # Dynamic admission: demand tracking and threshold feedback.
    def _init_admission_feedback(self, initial, duration, period=1800, window=1800):
        assert 0 <= initial <= 1 and duration > 0 and (period > 0) and (window > 0)
        self._admission_initial = initial
        self._admission_duration = duration
        self._admission_period = period
        self._admission_window = window
        self._admission_bins = {}
        self._admission_counts = [0] * 21
        self._admission_first = None
        self._admission_events = 0
        self._admission_history = deque([(0.0, 0)])
        self._admission_streak = 0
        self._admission_pending_target = None

    def _observe_admission(self, key, score, now):
        if self._admission_first is None:
            self._admission_first = now
        self._admission_events += 1
        j = max(0, min(20, math.floor(score * 20)))
        old = self._admission_bins.get(key)
        if old is not None:
            self._admission_counts[old] -= 1
        self._admission_bins[key] = j
        self._admission_counts[j] += 1

    def _update_admission(self, now, before, used, budget):
        remaining = max(0, budget - used)
        self._admission_history.append((now, used))
        while len(self._admission_history) > 1 and self._admission_history[1][0] <= now - self._admission_window:
            self._admission_history.popleft()
        span = now - max(self._admission_history[0][0], self._admission_first if self._admission_first is not None else now)
        recent_writes = used - self._admission_history[0][1]
        long_span = 0 if self._admission_first is None else now - self._admission_first
        target = before
        after = before
        ratio = None
        projected = None
        pressure = False
        reason = 'hold'
        if now >= self._admission_duration:
            reason = 'budget_period_end'
        elif not remaining:
            reason = 'budget_exhausted'
        elif self._admission_first is None or span < self._admission_period:
            reason = 'warmup'
        else:
            ratio = recent_writes / span * (self._admission_duration - now) / remaining
            rates = [0.0] * 21
            suffix = 0
            for j in range(20, -1, -1):
                suffix += self._admission_counts[j]
                rates[j] = suffix / max(1.0, long_span)
            (proposed, pressure, projected) = self._select_threshold(before, self._admission_initial, rates, remaining, self._admission_duration - now, ratio)
            if long_span < self._admission_window:
                reason = 'warmup_window'
            else:
                target = proposed
                reason = 'approach_normal_pending' if pressure else 'long_demand_surplus_pending'
            direction = -1 if target < before - 1e-12 else 1 if target > before + 1e-12 else 0
            if not direction:
                self._admission_streak = 0
                self._admission_pending_target = None
                reason = 'hold_target'
            else:
                self._admission_streak = self._admission_streak + 1 if self._admission_pending_target is not None and (self._admission_pending_target > before) == (target > before) else 1
                self._admission_pending_target = target
                # Apply established headroom at this tick; rising pressure still needs confirmation.
                if direction < 0 or self._admission_streak >= 2:
                    assert -1e-12 <= target <= self._admission_initial + 1e-12
                    after = max(0.0, min(self._admission_initial, target))
                    self._admission_streak = 0
                    self._admission_pending_target = None
                    reason = 'approach_normal' if pressure else 'long_demand_surplus'
        return dict(t=now, before=before, after=after, target=target, reason=reason, recent_ratio=ratio, long_normal_projected=projected, unique_cumulative_candidates=len(self._admission_bins), candidate_events=self._admission_events, window_writes=recent_writes, window_span=span, long_span=long_span, used_pages=used, remaining_pages=remaining, ceiling=self._admission_initial)

    @staticmethod
    def _select_threshold(before, normal, rates, remaining, seconds_left, recent_ratio):
        k = max(0, min(20, math.ceil(normal * 20 - 1e-09)))
        projected = rates[k] * seconds_left
        pressure = recent_ratio > 1.2 or projected > remaining
        target = before
        if pressure:
            if before < normal:
                target = min(normal, before + 0.05)
            elif before > normal:
                target = max(normal, before - 0.05)
        else:
            target = normal
            for j in range(21):
                if j * 0.05 > normal + 1e-12:
                    break
                if rates[j] * seconds_left <= remaining:
                    target = j * 0.05
                    break
        return (target, pressure, projected)
