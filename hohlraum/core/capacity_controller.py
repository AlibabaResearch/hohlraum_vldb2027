"""Windowed, tiered capacity control; native storage/timing remain unchanged.

Five-minute windows and eight-window history follow the paper. Tier amounts,
reset-on-resize and sustained-pressure escalation are explicitly agreed design
choices. Unspecified numerical thresholds must be supplied in configuration.
"""
from collections import Counter, deque
import json
import math
from pathlib import Path


class CapacityController:
    def __init__(self, host):
        self.host = host
        c = host.cfg.get('capacity_control', {})
        required = ('expansion_thresholds', 'contraction_thresholds',
                    'additional_wait_windows', 'failure_cooldown_windows',
                    'min_read_fraction')
        missing = [k for k in required if c.get(k) is None]
        if missing:
            raise ValueError('capacity_control needs explicitly confirmed parameters: ' + ', '.join(missing))
        legacy = {'window_s', 'additional_wait_s', 'read_cooldown_s', 'failure_cooldown_s',
                  'max_step_fraction', 'min_opportunity_fraction', 'preserve_savings_fraction',
                  'capacity_failure_intervals', 'max_staging_fraction', 'read_wait_floor_s',
                  'read_wait_growth_ratio', 'read_pressure_intervals'} & c.keys()
        if legacy:
            raise ValueError('obsolete capacity parameters: ' + ', '.join(sorted(legacy)))
        self.period = float(c.get('period_s', 300))
        self.history_windows = int(c.get('history_windows', 8))
        self.observation_windows = int(c.get('observation_windows', 2))
        self.min_expansion_windows = int(c.get('min_expansion_windows', 4))
        self.sustained_intervals = int(c.get('pressure_windows', 2))
        self.wait_windows = int(c['additional_wait_windows'])
        self.failure_cooldown_windows = int(c['failure_cooldown_windows'])
        self.wait = self.period * self.wait_windows
        self.min_read_fraction = float(c['min_read_fraction'])
        self.expansion_thresholds = tuple(map(float, c['expansion_thresholds']))
        self.contraction_thresholds = tuple(map(float, c['contraction_thresholds']))
        self.tiers_gb = tuple(map(float, c.get('tiers_gb', (1, 10, 15, 30))))
        if (not math.isfinite(self.period) or self.period <= 0 or
            min(self.history_windows, self.observation_windows, self.sustained_intervals, self.wait_windows, self.min_expansion_windows) < 1 or
            self.history_windows < max(self.observation_windows, self.sustained_intervals, self.min_expansion_windows) or
            self.failure_cooldown_windows < self.observation_windows or
            not 0 < self.min_read_fraction < 1):
            raise ValueError('invalid capacity window, bounds or pressure parameters')
        count=len(self.tiers_gb)
        if not count or len(self.contraction_thresholds) not in (count-1,count):
            raise ValueError('contraction thresholds must select all configured tiers')
        for values, n, upper in ((self.expansion_thresholds, count, 1),
                                  (self.contraction_thresholds, len(self.contraction_thresholds), 1), (self.tiers_gb, count, math.inf)):
            if len(values) != n or any(not math.isfinite(v) or not 0 <= v <= upper for v in values) or any(a >= b for a,b in zip(values,values[1:])):
                raise ValueError('capacity thresholds must be nonnegative and strictly increasing')
        if self.tiers_gb[0] <= 0:raise ValueError('capacity tier sizes must be positive')
        page=host.cache.page_size
        self.tiers = tuple(max(page, int(gb * 1e9 / getattr(host.cache, 'tp_world_size', 1) / host.kv_bytes_per_token) // page * page) for gb in self.tiers_gb)
        self.maximum = host.total_tokens - math.ceil(host.total_tokens * self.min_read_fraction / page) * page
        if self.maximum < page or host.staging_limit > self.maximum:
            raise ValueError('initial staging exceeds minimum read capacity')
        self.reset()

    def reset(self):
        self.accepted=deque(); self.slack=deque(); self.states={}
        self.history=deque(maxlen=self.history_windows)
        self.waiting={}; self.interval_waits=[]; self.failure_requests=set()
        self.capacity_failures=set(); self.other_failures=set(); self.writeback_failures=set()
        self.writeback_failed_tokens=0
        self.unresolved=getattr(self, 'unresolved', {}); self.unresolved.clear()
        self.events=Counter(); self.last_decision=None
        self.epoch=0; self.epoch_start=self.host.now(); self.window_start=self.epoch_start
        self.next_tick=self.window_start+self.period
        self.expand_after=self.epoch_start; self.observe_until=self.epoch_start
        self.capacity_failure_streak=0
        self.last_shrink=None; self.known_capacity=self.host.staging_limit
        self.last_direction=None
        self._idle_at=self.host.now(); self._idle_active=False
        self._idle_since=None; self._idle_seconds=0.; self._idle_longest=0.
        self._idle_intervals=0; self._idle_total=0.
        self._checked_ssd_waiters=set(); self._inflight_results=set()

    def _clear_interval(self):
        self._idle_seconds=0.; self._idle_longest=0.; self._idle_intervals=0
        if self._idle_active:self._idle_since=self._idle_at
        self.interval_waits.clear(); self.failure_requests.clear()
        self.capacity_failures.clear(); self.other_failures.clear(); self.writeback_failures.clear()
        self.writeback_failed_tokens=0

    def _new_epoch(self, now, direction, pressure_summary=None, tier=None, failure=False):
        self._advance_idle(now)
        # Keep only a comparison summary, never the old distributions/windows.
        if direction == 'shrink':
            self.last_shrink=dict(pressure_summary or {}, tier=tier)
        else:
            self.last_shrink=None
        self.last_direction=direction
        self.epoch += 1; self.epoch_start=now; self.window_start=now
        self.next_tick=now+self.period; self.known_capacity=self.host.staging_limit
        self.accepted.clear(); self.slack.clear(); self.states.clear(); self.history.clear()
        self.capacity_failure_streak=0
        self._clear_interval()
        # Retain live handles for correct completion accounting, but exclude
        # all waiting time accumulated under the previous partition.
        self.waiting={rid:now for rid in self.waiting}
        windows=max(self.observation_windows,self.wait_windows,self.min_expansion_windows) if direction == 'expand' else self.observation_windows
        self.observe_until=now+windows*self.period
        self.expand_after=max(self.expand_after,self.observe_until,
                              now+self.failure_cooldown_windows*self.period if direction == 'shrink' else now)
        self._observe_residents(now)

    def _observe_residents(self, now):
        entries=[e for e in self.host.stage.values() if e.operation is None]
        if entries:
            profiles=self.host.sli.profiles([e.hash_value[0] for e in entries],now)
            for e in entries:self.observe(e,profiles[e.hash_value[0]],now)

    def observe(self, entry, profile, now):
        # First observed rejection in the current rejection episode. A return
        # to acceptance invalidates prior slack; threshold changes are included.
        accepted=profile[0] >= self.host.admission_threshold
        state=self.states.setdefault(entry.id, [False,None])
        if accepted: state[:]=[True,None]
        elif state[0] and state[1] is None: state[1]=now

    def entered(self, entry):
        key=entry.hash_value[0]
        self.observe(entry,self.host.sli.profiles([key],self.host.now())[key],self.host.now())

    def write_sample(self, entry):
        key=entry.hash_value[0];now=self.host.now()
        profile=self.host.sli.profiles([key],now)[key]
        self.observe(entry,profile,now)
        # Forced prefix/pending-read writes still count in the byte denominator,
        # but are not counted as opportunities removable by idle waiting alone.
        eligible=(profile[0]>=self.host.admission_threshold
                  and not self.host.staging_policy.requires_prefix_write(entry)
                  and not self.host.sli.pending.get(key,0))
        wait=profile[1] if eligible and math.isfinite(profile[1]) else math.inf
        return (now,wait,self.host.cache.page_size*self.host.kv_bytes_per_token,self.epoch)

    def wrote(self, sample):
        if sample is not None and sample[3] == self.epoch:self.accepted.append(sample[:3])

    def discarded(self, entry, budget_exhausted=False):
        now=self.host.now();key=entry.hash_value[0]
        profile=self.host.sli.profiles([key],now)[key]
        self.observe(entry,profile,now)
        accepted,rejected_at=self.states[entry.id]
        if not budget_exhausted and accepted and rejected_at is not None:
            self.slack.append((now,max(0.,now-rejected_at),self.host.cache.page_size*self.host.kv_bytes_per_token))

    def removed(self, entry):
        self.states.pop(entry.id,None)

    def _advance_idle(self, now):
        elapsed=max(0.,now-self._idle_at)
        if self._idle_active:
            self._idle_seconds+=elapsed; self._idle_total+=elapsed
            self._idle_longest=max(self._idle_longest,max(0.,now-self._idle_since))
        self._idle_at=now

    def _set_idle(self, active, now):
        self._advance_idle(now)
        if active and not self._idle_active:
            self._idle_since=now; self._idle_intervals+=1
        elif not active:self._idle_since=None
        self._idle_active=active

    def scheduler_begin(self, now):
        self._advance_idle(now)
        self._checked_ssd_waiters.clear()

    def scheduler_end(self, batch_available, now):
        # Conservative: an unconsumed result is treated as in-flight even if
        # the GPU has finished. No CUDA synchronization or timing-model changes.
        self._set_idle(not batch_available and not self._inflight_results and
                       bool(self._checked_ssd_waiters & self.waiting.keys()),now)

    def batch_launch(self, now):
        self._set_idle(False,now)

    def batch_submitted(self, result):
        self._inflight_results.add(id(result))

    def batch_completed(self, result):
        self._inflight_results.discard(id(result))

    def read_progress(self, rid, waiting, now, has_ssd):
        if waiting and has_ssd:self._checked_ssd_waiters.add(rid)
        if waiting and has_ssd:self.waiting.setdefault(rid,now)
        elif not waiting:self.end_read(rid,now)

    def end_read(self, rid, now):
        self._advance_idle(now)
        self._checked_ssd_waiters.discard(rid)
        start=self.waiting.pop(rid,None)
        if start is not None:self.interval_waits.append(max(0.,now-start))
        if not self._checked_ssd_waiters & self.waiting.keys():self._set_idle(False,now)

    def failed(self, rid, tokens, reason):
        if tokens>0 and reason in ('host_capacity','storage_transfer','cache_admission_shortfall'):
            self.failure_requests.add(rid)
            (self.capacity_failures if reason == "host_capacity" else self.other_failures).add(rid)
            self.events[reason]+=1

    def writeback_failed(self, node_id, tokens):
        if tokens <= 0:return
        self.writeback_failures.add(node_id)
        self.writeback_failed_tokens += tokens
        self.events['writeback_host_failure'] += 1

    def _histogram(self, samples):
        bounds=sorted(set([0.,self.period,self.wait,4*self.period,8*self.period]))
        bins=[0]*(len(bounds)+2)
        for _,wait,size in samples:
            bucket=next((i for i,bound in enumerate(bounds) if wait<=bound),len(bounds)) if math.isfinite(wait) else len(bounds)+1
            bins[bucket]+=size
        return dict(upper_bounds_s=bounds,bytes_by_bin=bins)

    def tick(self, now, *, force=False, synchronize=None):
        # Budget exhaustion may resize storage before this callback.
        if self.host.staging_limit != self.known_capacity:
            self._new_epoch(now, 'external')
        self._advance_idle(now)
        if not force and now < self.next_tick:return
        complete = now < self.next_tick + self.period
        if not complete:
            self.capacity_failure_streak=0
        cutoff=now-self.history_windows*self.period
        self.accepted=deque(x for x in self.accepted if x[0]>=cutoff)
        self.slack=deque(x for x in self.slack if x[0]>=cutoff)
        self._observe_residents(now)
        # Pending-read duration is diagnostic only; it is NOT overlap-free wait.
        waits=self.interval_waits+[max(0.,now-t) for t in self.waiting.values()]
        mean_wait=sum(waits)/len(waits) if waits else None
        # Only confirmed capacity-related failures drive capacity control.
        # Other failures remain observable; they are not assumed to be capacity failures.
        failure=bool(self.capacity_failures or self.writeback_failures)
        self.capacity_failure_streak=self.capacity_failure_streak+1 if complete and failure else 0
        if complete and not failure:self.last_shrink=None
        cutoff=self.window_start
        current_accepted=[x for x in self.accepted if x[0]>=cutoff]
        current_slack=[x for x in self.slack if x[0]>=cutoff]
        summary=dict(mean_wait=mean_wait, capacity_failures=len(self.capacity_failures),
                     writeback_failed_tokens=self.writeback_failed_tokens)
        window=dict(start=self.window_start,end=now,complete=complete,epoch=self.epoch,
                    staging_tokens=self.host.staging_limit,accepted_distribution=self._histogram(current_accepted),
                    slack_distribution=self._histogram(current_slack),read_wait_sum_s=sum(waits),read_wait_count=len(waits),
                    capacity_pressure=failure,
                    ssd_wait_idle_s=self._idle_seconds,
                    ssd_wait_idle_fraction=self._idle_seconds/max(now-self.window_start,1e-9),
                    ssd_wait_idle_longest_s=self._idle_longest,
                    ssd_wait_idle_intervals=self._idle_intervals,**summary)
        self.history.append(window)
        accepted_bytes=sum(x[2] for x in self.accepted)
        opportunity_bytes=sum(x[2] for x in self.accepted if x[1]<=self.wait)
        slack_bytes=sum(x[2] for x in self.slack)
        safe_bytes=sum(x[2] for x in self.slack if x[1]>=self.wait)
        opportunity=opportunity_bytes/accepted_bytes if accepted_bytes else 0.
        retained=safe_bytes/slack_bytes if slack_bytes else 0.
        before=self.host.staging_limit;target=before;reason='hold';tier=None
        failure_pressure=self.capacity_failure_streak>=self.sustained_intervals
        sustained=failure_pressure
        # A resize must physically settle before another allocation change.
        pending=(self.host.read_allocated()>self.host.read_limit or self.host.waiting_staging_tokens()>before)
        if self.host._budget_exhausted():
            target=0;reason='budget_exhausted'
        elif not complete:
            reason='incomplete_window'
        elif sustained and now>=max(self.observe_until if self.last_shrink is not None else 0, self.epoch_start+self.observation_windows*self.period) and not pending:
            tier=sum(retained>=v for v in self.contraction_thresholds)
            if len(self.contraction_thresholds)==len(self.tiers):
                tier=tier-1 if retained>0 else -1
            previous=self.last_shrink
            if previous is not None and previous.get('tier') is not None:
                recent=list(self.history)[-self.sustained_intervals:]
                no_failure_improvement=(failure_pressure and
                    (sum(w['capacity_failures'] for w in recent)/len(recent)>=previous.get('capacity_failures',0)) and
                    (sum(w['writeback_failed_tokens'] for w in recent)/len(recent)>=previous.get('writeback_failed_tokens',0)))
                tier=max(tier,previous['tier'])
                if no_failure_improvement:tier=max(tier,min(len(self.tiers)-1,previous['tier']+1))
            target=max(0,before-self.tiers[tier]) if tier>=0 else before
            if tier<0:
                # A zero-slack tier must not permanently defeat persistent read
                # protection. Observe this zero step, then escalate if unimproved.
                self.last_shrink=dict(summary,tier=-1)
                self.observe_until=now+self.observation_windows*self.period
                self.expand_after=now+self.failure_cooldown_windows*self.period
            reason='writeback_pressure' if self.writeback_failures else 'prefetch_pressure'
        elif pending:
            reason='resize_pending'
        elif now<max(self.observe_until,self.expand_after):
            reason='observation'
        elif sustained:
            reason='pressure_at_lower_bound'
        elif sum(w['complete'] for w in self.history)<self.min_expansion_windows:
            reason='collecting_history'
        else:
            level=sum(opportunity>=v for v in self.expansion_thresholds) if accepted_bytes and opportunity>0 else 0
            if level:
                tier=level-1;target=min(self.maximum,before+self.tiers[tier]);reason='write_saving_opportunity'
        if synchronize is not None:
            # All ranks reach this callback in the same scheduler round. The
            # leader chooses the action; local wall-time/sample jitter must not
            # produce different quotas or different cooldowns.
            decision=synchronize(dict(target=target,reason=reason,tier=tier,
                failure_pressure=failure_pressure,summary=summary,
                last_shrink=self.last_shrink,observe_until=self.observe_until,
                expand_after=self.expand_after,streak=self.capacity_failure_streak))
            target=decision['target'];reason=decision['reason'];tier=decision['tier']
            failure_pressure=decision['failure_pressure'];summary=decision['summary']
            self.last_shrink=decision['last_shrink']
            self.observe_until=decision['observe_until'];self.expand_after=decision['expand_after']
            self.capacity_failure_streak=decision['streak']
        applied=True
        if target!=before:applied=self.host.resize_staging(target) is not False
        changed=self.host.staging_limit!=before
        self.last_decision=dict(t=now,epoch=self.epoch,before_tokens=before,target_tokens=target,
            actual_target_tokens=self.host.staging_limit,applied=applied,changed=changed,reason=reason,
            tier_gb=self.tiers_gb[tier] if tier is not None and tier>=0 else 0,
            opportunity_fraction=opportunity,retained_savings_fraction=retained,
            accepted_bytes=accepted_bytes,opportunity_bytes=opportunity_bytes,
            slack_bytes=slack_bytes,safe_slack_bytes=safe_bytes,mean_pending_ssd_wait_s=mean_wait,
            prefetch_failure_requests=len(self.failure_requests),capacity_failure_requests=len(self.capacity_failures),
            writeback_failure_nodes=len(self.writeback_failures),writeback_failed_tokens=self.writeback_failed_tokens,
            capacity_failure_streak=self.capacity_failure_streak,
            window_count=len(self.history),complete_window_count=sum(w['complete'] for w in self.history),
            min_expansion_windows=self.min_expansion_windows,complete_window=complete)
        self.events[reason]+=1
        if changed:
            direction='expand' if self.host.staging_limit>before else 'shrink'
            if reason=='budget_exhausted':direction='external'
            self._new_epoch(now,direction,summary,tier,failure_pressure)
        else:
            self.window_start=now;self.next_tick=now+self.period;self._clear_interval()
        self.last_decision.update(expand_after=self.expand_after,observe_until=self.observe_until,
                                  next_tick=self.next_tick,history_reset=changed)
        if self.host.stats_path:
            with Path(str(self.host.stats_path)+'.capacity.jsonl').open('a') as f:
                f.write(json.dumps(dict(self.last_decision,window=window),allow_nan=False)+'\n')

    def stats(self):
        self._advance_idle(self.host.now())
        return dict(ssd_wait_idle_total_s=self._idle_total,
                    ssd_wait_idle_window_s=self._idle_seconds,
                    ssd_wait_idle_active=self._idle_active,
                    ssd_wait_idle_observation_only=True,
                    observed_inflight_batches=len(self._inflight_results),next_tick=self.next_tick,expand_after=self.expand_after,observe_until=self.observe_until,
                    events=dict(self.events),last_decision=self.last_decision,epoch=self.epoch,
                    history_windows=len(self.history),accepted_samples=len(self.accepted),slack_samples=len(self.slack),
                    resize_mode='evict_from_shrinking_region',maximum_staging_tokens=self.maximum,
                    tiers_gb=self.tiers_gb)


class CapacityConsensus:
    """Agree on capacity actions at an existing scheduler entry, never a worker.

    Callbacks isolate native TP transport from policy and permit fast race tests.
    No engine clock is changed. Only controller window decisions use leader time.
    """
    def __init__(self, control, broadcast, reduce_flags):
        self.control=control;self.host=control.host
        self.broadcast=broadcast;self.reduce_flags=reduce_flags
        self.host._synced_budget_exhausted=False
        self.host._capacity_zero_blocked=False

    def advance(self):
        h=self.host;c=self.control
        exhausted,blocked=self.reduce_flags(
            h.ssd_backend.budget_exhausted(h.cfg.get('transition_reserve_bytes',0)),
            any(e.read_pins or getattr(e,'copy_committed',False) for e in h.stage.values()))
        h._synced_budget_exhausted=bool(exhausted)
        h._capacity_zero_blocked=bool(blocked)
        now,due=self.broadcast((h.now(),h.now()>=c.next_tick))
        if exhausted and h.staging_limit:
            if h.resize_staging(0) is not False:
                h._record_budget_progress(force=True,event='staging_disabled_after_exhaustion')
                c._new_epoch(now,'external')
                return
        if due:c.tick(now,force=True,synchronize=self.broadcast)


def native_capacity_consensus(control):
    """Bind only our controller to the native TP CPU process group."""
    import torch
    import torch.distributed as dist
    group=control.host.cache.tp_group
    ranks=dist.get_process_group_ranks(group)
    leader=ranks[0]
    def broadcast(value):
        values=[value if dist.get_rank()==leader else None]
        dist.broadcast_object_list(values,src=leader,group=group)
        return values[0]
    def reduce_flags(exhausted,blocked):
        flags=torch.tensor([int(exhausted),int(blocked)],dtype=torch.int64)
        dist.all_reduce(flags,op=dist.ReduceOp.MAX,group=group)
        return flags.tolist()
    return CapacityConsensus(control,broadcast,reduce_flags)
