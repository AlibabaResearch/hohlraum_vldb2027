"""Staging policy: HTS candidates and constraints, ranking, and admission."""
import math
from collections import Counter, deque
from dataclasses import dataclass
from hohlraum.core.sli import StagingRangeIndex

@dataclass(frozen=True)
class Endpoint:
    node: object
    offset: int
    key: str


@dataclass(frozen=True)
class RetirementDecision:
    write: bool
    budget_exhausted: bool


class StagingPolicy:
    """Coordinate HTS candidates, ranking, admission and prefix obligations.

    SLI supplies lifecycle profiles. Residency executes the returned decisions.
    HTS and dynamic admission are methods of this policy; read LRU is unchanged.
    """
    def __init__(self, host):
        self.host = host
        self.cfg = cfg = host.cfg
        self.index = StagingRangeIndex()
        self.nonleaf_writes = 0
        self.nonleaf_releases = 0
        self.cascade_tokens = 0
        self.prefix_completion_enabled = host.cfg.get('prefix_complete_admission', True)
        self.reset_prefix_obligations()
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

    def choose(self):
        host = self.host
        candidates = [e for e in host.stage.values() if e.operation is None and not e.read_pins]
        if not candidates:
            return None
        returning = [e for e in candidates if e.hash_value[0] in host._staging_return_flush]
        if returning:
            return min(returning, key=lambda e:(len(e.prefix_hashes), e.last_access_time, e.id))
        marked = [e for e in candidates if self.requires_prefix_write(e)]
        if marked and not host._budget_exhausted():
            return min(marked, key=lambda e:(len(e.prefix_hashes), e.last_access_time, e.id))
        if host.residency_policy == 'joint_hts' and host.lifecycle is not None and not host._budget_exhausted():
            return self.choose_ranked()
        return min(candidates, key=lambda e:(e.last_access_time, e.id))

    def choose_ranked(self, deferred=()):
        host = self.host
        now = host.now()
        candidates = self.candidates(deferred)
        if not candidates:
            return None
        threshold = host.admission_threshold
        profiles=host.sli.profiles([ep.key for ep in candidates],now)
        candidates = [ep for ep in candidates
                      if not self.protect_head(ep.key, profiles[ep.key][0], threshold)]
        if not candidates:return None
        candidates.sort(key=lambda ep:(-profiles[ep.key][2],host.sli.ids[ep.key]))
        candidates=candidates[:(len(candidates)+1)//2]
        selected=max(candidates,key=lambda ep:(profiles[ep.key][1],-profiles[ep.key][0],host.sli.ids[ep.key]))
        self.selections+=1
        return self.isolate(selected)


    def decisions(self,keys,now,record=False):
        return self.decisions_from_profiles(keys,self.host.sli.profiles(keys,now),record)
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

    def retire(self, entry, copied):
        host = self.host
        over = host._budget_exhausted()
        decisions = self.decisions(entry.hash_value, host.now(), record=True) if host.lifecycle else [True]
        marked = self.requires_prefix_write(entry)
        if marked and over:
            self.record_prefix_budget_block()
        write = not (copied or host.admission_policy == 'write_none' or over or (not marked and not any(decisions)))
        if write:
            self.prepare_tail_write(entry)
        return RetirementDecision(write, over)

    def stats(self):
        return dict(pending_read_overrides=self.override_count,
                    histogram=self.histogram, threshold_log=self.threshold_log)

    # HTS: candidates, head protection and tail-prefix obligations.
    def observe(self, session, hashes):
        self.index.observe(session, hashes)

    @staticmethod
    def unlocked(node):
        return not any(cd.host_lock_ref or cd.lock_ref for cd in node.component_data)

    def candidates(self, deferred=()):
        host = self.host
        pages = {}
        for node in host._stage_nodes():
            if node.id not in host.absorbed or not host._host_only(node):
                continue
            for offset,key in enumerate(node.hash_value or []):
                pages[key] = Endpoint(node,offset,key)
        keys = self.index.endpoints(pages)
        candidates = [pages[k] for k in keys
                      if pages[k].node.id not in deferred and self.unlocked(pages[k].node)
                      and not host._pending(pages[k].node)]
        return candidates

    def isolate(self, endpoint):
        host=self.host
        node=endpoint.node
        page=host.cache.page_size
        assert endpoint.offset==0 and node.id in host.absorbed, 'HTS only selects standalone staging pages'
        assert host._tokens(node)==page and node.hash_value==[endpoint.key]
        return node

    def protect_head(self, key, score, threshold):
        """Do not reject a prefix still needed by resident descendants."""
        return score < threshold and self.host.has_resident_descendant(key)

    def reset_prefix_obligations(self):
        self._prefix_events = Counter()
        self._required_prefixes = set()

    def record_prefix_budget_block(self):
        self._prefix_events['marked_pages_blocked_by_budget'] += 1

    def _pending_prefix_keys(self):
        h=self.host
        keys={k for e in h.inflight.values() for k in e.hash_value}
        return keys

    def requires_prefix_write(self,entry):
        if not self.prefix_completion_enabled:return False
        return any(k in self._required_prefixes for k in entry.hash_value)

    def prepare_tail_write(self,tail):
        if not self.prefix_completion_enabled:return None
        h=self.host
        keys=list(tail.prefix_hashes)
        pending=self._pending_prefix_keys()
        missing=[k for k,stored in zip(keys,h._copy_bits(keys)) if not stored and k not in pending]
        self._prefix_events['admitted_suffixes']+=1
        self._prefix_events['new_prefix_marks']+=len(set(missing)-self._required_prefixes)
        self._required_prefixes.update(missing)
        return 'ready'

    def complete_prefix_write(self,entry):
        if not self.prefix_completion_enabled:return
        resolved=self._required_prefixes.intersection(entry.hash_value)
        self._required_prefixes.difference_update(resolved)
        self._prefix_events['completed_marked_pages']+=len(resolved)

    def prefix_write_stats(self):
        if not self.prefix_completion_enabled:return None
        # Keep failed/pending writes marked. Clear only confirmed stored keys.
        keys=list(self._required_prefixes)
        self._required_prefixes.difference_update(k for k,stored in zip(keys,self.host._copy_bits(keys)) if stored)
        return dict(self._prefix_events,pending_native_tokens=0,outstanding_prefix_marks=len(self._required_prefixes),
                    semantics='deferred_prefix_marks_subject_to_endurance_budget')


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
                if self._admission_streak >= 2:
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
