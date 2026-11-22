"""Windowed, tiered capacity control; native storage/timing remain unchanged.

Five-minute windows and eight-window history follow the paper. Tier amounts,
reset-on-resize and sustained-pressure escalation are explicitly agreed design
choices. Unspecified numerical thresholds must be supplied in configuration.
"""
from collections import Counter, deque
import json
import math
from pathlib import Path

from .read_pressure import ReadPressure


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
        allowed = set(required) | {
            'period_s', 'history_windows', 'observation_windows', 'min_expansion_windows',
            'pressure_windows', 'exposed_ssd_wait_threshold_s', 'tiers_gb', 'max_expansion_gb',
        }
        unknown = c.keys() - allowed
        if unknown:
            raise ValueError('unknown capacity parameters: ' + ', '.join(sorted(unknown)))
        self.period = float(c.get('period_s', 300))
        self.history_windows = int(c.get('history_windows', 8))
        self.observation_windows = int(c.get('observation_windows', 2))
        self.min_expansion_windows = int(c.get('min_expansion_windows', 4))
        self.sustained_intervals = int(c.get('pressure_windows', 2))
        self.wait_windows = int(c['additional_wait_windows'])
        self.failure_cooldown_windows = int(c['failure_cooldown_windows'])
        self.wait = self.period * self.wait_windows
        self.min_read_fraction = float(c['min_read_fraction'])
        self.exposed_ssd_wait_threshold_s = float(c.get('exposed_ssd_wait_threshold_s', 2))
        if not math.isfinite(self.exposed_ssd_wait_threshold_s) or self.exposed_ssd_wait_threshold_s <= 0:
            raise ValueError('exposed SSD read feedback requires a positive wait threshold')
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
        expansion_limit = c.get('max_expansion_gb', self.tiers_gb[-1])
        if expansion_limit not in self.tiers_gb:
            raise ValueError('max_expansion_gb must match a configured capacity tier')
        self.max_expansion_tier = self.tiers_gb.index(expansion_limit)
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
        self._exposed_ssd_wait_pressure_streak = 0
        self._exposed_ssd_wait_reference_s = None
        self._last_exposed_ssd_shrink = None
        self.last_shrink=None; self.known_capacity=self.host.staging_limit
        self.last_direction=None
        self.read_pressure = ReadPressure()

    def _clear_interval(self):
        self.read_pressure.clear()
        self.interval_waits.clear(); self.failure_requests.clear()
        self.capacity_failures.clear(); self.other_failures.clear(); self.writeback_failures.clear()
        self.writeback_failed_tokens=0

    def _new_epoch(self, now, direction, pressure_summary=None, tier=None, failure=False):
        # Keep only a comparison summary, never the old distributions/windows.
        hard_shrink = direction == 'shrink' and failure
        if hard_shrink:
            self.last_shrink=dict(pressure_summary or {}, tier=tier)
        else:
            self.last_shrink=None
        self.last_direction=direction
        self.epoch += 1; self.epoch_start=now; self.window_start=now
        self.next_tick=now+self.period; self.known_capacity=self.host.staging_limit
        self.accepted.clear(); self.slack.clear(); self.states.clear(); self.history.clear()
        self.capacity_failure_streak=0
        self._exposed_ssd_wait_pressure_streak = 0
        soft_mean = (pressure_summary or {}).get('sustained_exposed_ssd_wait_mean_s')
        self._last_exposed_ssd_shrink = ((tier, soft_mean)
                                         if direction == 'shrink' and not failure and soft_mean is not None
                                         else None)
        if direction == 'expand':
            self._exposed_ssd_wait_reference_s = (pressure_summary or {}).get('pre_expand_exposed_ssd_wait_s')
        elif direction == 'external':
            self._exposed_ssd_wait_reference_s = None
        self._clear_interval()
        # Retain live handles for correct completion accounting, but exclude
        # all waiting time accumulated under the previous partition.
        self.waiting={rid:now for rid in self.waiting}
        windows=max(self.observation_windows,self.wait_windows,self.min_expansion_windows) if direction == 'expand' else self.observation_windows
        self.observe_until=now+windows*self.period
        self.expand_after=max(self.expand_after,self.observe_until,
                              now+self.failure_cooldown_windows*self.period if hard_shrink else now)
        self._observe_residents(now)

    def _observe_residents(self, now):
        entries=[e for e in self.host.stage.values() if e.operation is None]
        if entries:
            profiles=self.host.sli.profiles([e.hash_value[0] for e in entries],now)
            for e in entries:self.observe(e,profiles[e.hash_value[0]],now)

    def observe(self, entry, profile, now):
        # First observed rejection in the current rejection episode. A return
        # to acceptance invalidates prior slack; threshold changes are included.
        accepted=(profile[0] >= self.host.admission_threshold
                  or self.host.sli.pending.get(entry.hash_value[0],0)>0)
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
        # Pending-read writes still count in the byte denominator,
        # but are not counted as opportunities removable by idle waiting alone.
        eligible=(profile[0]>=self.host.admission_threshold
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

    def read_progress(self, rid, waiting, now, has_ssd):
        if waiting and has_ssd:
            self.waiting.setdefault(rid, now)
        elif not waiting:
            self.end_read(rid, now)

    def end_read(self, rid, now):
        start = self.waiting.pop(rid, None)
        if start is not None:
            self.interval_waits.append(max(0., now - start))

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

    def _contraction_tier(self, retained):
        tier = sum(retained >= threshold for threshold in self.contraction_thresholds)
        if len(self.contraction_thresholds) == len(self.tiers):
            return tier - 1 if retained > 0 else -1
        return tier

    def _ordinary_contraction_tier(self, accepted_bytes, opportunity, retained):
        # Low short-TTD demand alone is insufficient. Require observed rejection
        # slack indicating some writes would still be avoided with less staging.
        min_growth = next((value for value in self.expansion_thresholds if value > 0), None)
        min_retained = next((value for value in self.contraction_thresholds if value > 0), None)
        if (accepted_bytes and min_growth is not None and min_retained is not None
                and opportunity < min_growth and retained >= min_retained):
            return self._contraction_tier(retained)
        return None

    def _pre_expansion_wait_reference(self):
        windows = [window for window in self.history if window['exposed_ssd_wait_complete']][-2:]
        if len(windows) < 2:
            return None
        return self._pooled_exposed_ssd_wait(windows)

    @staticmethod
    def _pooled_exposed_ssd_wait(windows):
        count = sum(window['exposed_ssd_wait_request_count'] for window in windows)
        exposed = sum(window['exposed_ssd_wait_sum_s'] for window in windows)
        # Recorded windows without any physical SSD waiters give a zero baseline.
        return exposed/count if count else 0.

    def _observe_exposed_ssd_pressure(self, exposure, complete):
        threshold = self.exposed_ssd_wait_threshold_s
        reference = self._exposed_ssd_wait_reference_s
        mean = exposure['exposed_ssd_wait_mean_s']
        if complete and reference is None:
            # A fixed initial partition can encounter read pressure before expanding.
            reference = min(mean, threshold) if mean is not None else 0.
            self._exposed_ssd_wait_reference_s = reference
        if complete and mean is not None and mean >= threshold and (
                reference <= threshold or mean > reference):
            self._exposed_ssd_wait_pressure_streak += 1
        else:
            self._exposed_ssd_wait_pressure_streak = 0
            self._last_exposed_ssd_shrink = None
        exposure.update(exposed_ssd_wait_reference_s=reference,
                        exposed_ssd_wait_complete=complete,
                        exposed_ssd_wait_pressure_streak=self._exposed_ssd_wait_pressure_streak)
        return self._exposed_ssd_wait_pressure_streak >= self.sustained_intervals

    def tick(self, now, *, force=False, synchronize=None):
        # Budget exhaustion may resize storage before this callback.
        if self.host.staging_limit != self.known_capacity:
            self._new_epoch(now, 'external')
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
        # Distinguish confirmed capacity failures from general prefetch misses.
        failure=bool(self.capacity_failures or self.writeback_failures)
        self.capacity_failure_streak=self.capacity_failure_streak+1 if complete and failure else 0
        if complete and not failure:self.last_shrink=None
        cutoff=self.window_start
        current_accepted=[x for x in self.accepted if x[0]>=cutoff]
        current_slack=[x for x in self.slack if x[0]>=cutoff]
        summary=dict(mean_wait=mean_wait, capacity_failures=len(self.capacity_failures),
                     writeback_failed_tokens=self.writeback_failed_tokens)
        exposure = self.read_pressure.stats()
        exposed_read_pressure = self._observe_exposed_ssd_pressure(exposure, complete and now >= self.next_tick)
        window=dict(start=self.window_start,end=now,complete=complete,epoch=self.epoch,
                    staging_tokens=self.host.staging_limit,accepted_distribution=self._histogram(current_accepted),
                    slack_distribution=self._histogram(current_slack),read_wait_sum_s=sum(waits),read_wait_count=len(waits),
                    capacity_pressure=failure,**summary,**exposure)
        self.history.append(window)
        accepted_bytes=sum(x[2] for x in self.accepted)
        opportunity_bytes=sum(x[2] for x in self.accepted if x[1]<=self.wait)
        slack_bytes=sum(x[2] for x in self.slack)
        safe_bytes=sum(x[2] for x in self.slack if x[1]>=self.wait)
        opportunity=opportunity_bytes/accepted_bytes if accepted_bytes else 0.
        retained=safe_bytes/slack_bytes if slack_bytes else 0.
        before=self.host.staging_limit;target=before;reason='hold';tier=None
        failure_pressure=self.capacity_failure_streak>=self.sustained_intervals
        sustained=failure_pressure or exposed_read_pressure
        # A resize must physically settle before another allocation change.
        pending=(self.host.read_allocated()>self.host.read_limit or self.host.waiting_staging_tokens()>before)
        if self.host._budget_exhausted():
            target=0;reason='budget_exhausted'
        elif not complete:
            reason='incomplete_window'
        elif sustained and now>=max(self.observe_until if self.last_shrink is not None else 0, self.epoch_start+self.observation_windows*self.period) and not pending:
            # Confirmed read pressure always reclaims at least the smallest tier.
            tier = max(0,self._contraction_tier(retained))
            previous=self.last_shrink
            if failure_pressure and previous is not None and previous.get('tier') is not None:
                recent=list(self.history)[-self.sustained_intervals:]
                no_failure_improvement=(
                    (sum(w['capacity_failures'] for w in recent)/len(recent)>=previous.get('capacity_failures',0)) and
                    (sum(w['writeback_failed_tokens'] for w in recent)/len(recent)>=previous.get('writeback_failed_tokens',0)))
                tier=max(tier,previous['tier'])
                if no_failure_improvement:tier=max(tier,min(len(self.tiers)-1,previous['tier']+1))
            if not failure_pressure:
                recent_wait = self._pooled_exposed_ssd_wait(list(self.history)[-self.sustained_intervals:])
                summary['sustained_exposed_ssd_wait_mean_s'] = recent_wait
                previous_soft = self._last_exposed_ssd_shrink
                if previous_soft is not None:
                    prior_tier, prior_wait = previous_soft
                    if recent_wait >= prior_wait:
                        tier=max(tier,min(len(self.tiers)-1,prior_tier+1))
            target=max(0,before-self.tiers[tier])
            if failure_pressure:
                reason='writeback_pressure' if self.writeback_failures else 'prefetch_pressure'
            else:
                reason='exposed_ssd_read_pressure'
        elif pending:
            reason='resize_pending'
        elif now<max(self.observe_until,self.expand_after):
            reason='observation'
        elif sustained:
            reason='pressure_at_lower_bound'
        elif self._exposed_ssd_wait_pressure_streak:
            reason='observing_exposed_ssd_wait'
        elif sum(w['complete'] for w in self.history)<self.min_expansion_windows:
            reason='collecting_history'
        else:
            # A single capacity failure can mature into persistent pressure on
            # the next window. Do not reset that observation with an ordinary shrink.
            ordinary_tier = self._ordinary_contraction_tier(accepted_bytes, opportunity, retained) if before and not failure else None
            if ordinary_tier is not None:
                tier = ordinary_tier
                target = max(0, before - self.tiers[tier])
                reason = 'low_ttd_opportunity'
            else:
                level=sum(opportunity>=v for v in self.expansion_thresholds) if accepted_bytes and opportunity>0 else 0
                if level:
                    tier=min(level-1,self.max_expansion_tier)
                    target=min(self.maximum,before+self.tiers[tier])
                    reason='write_saving_opportunity'
        summary['exposed_ssd_wait_pressure_streak'] = self._exposed_ssd_wait_pressure_streak
        summary['exposed_ssd_wait_reference_s'] = self._exposed_ssd_wait_reference_s
        summary['last_exposed_ssd_shrink'] = self._last_exposed_ssd_shrink
        if target > before:
            summary['pre_expand_exposed_ssd_wait_s'] = self._pre_expansion_wait_reference()
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
            self._exposed_ssd_wait_pressure_streak = summary['exposed_ssd_wait_pressure_streak']
            self._exposed_ssd_wait_reference_s = summary['exposed_ssd_wait_reference_s']
            self._last_exposed_ssd_shrink = summary['last_exposed_ssd_shrink']
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
        self.last_decision.update(sustained_exposed_ssd_wait_mean_s=summary.get('sustained_exposed_ssd_wait_mean_s'),
                                  previous_exposed_ssd_shrink=summary['last_exposed_ssd_shrink'])
        self.events[reason]+=1
        if changed:
            direction='expand' if self.host.staging_limit>before else 'shrink'
            if reason=='budget_exhausted':direction='external'
            self._new_epoch(now,direction,summary if applied or direction != 'expand' else None,tier,failure_pressure)
        else:
            self.window_start=now;self.next_tick=now+self.period;self._clear_interval()
        self.last_decision.update(expand_after=self.expand_after,observe_until=self.observe_until,
                                  next_tick=self.next_tick,history_reset=changed)
        if self.host.stats_path:
            with Path(str(self.host.stats_path)+'.capacity.jsonl').open('a') as f:
                f.write(json.dumps(dict(self.last_decision,window=window),allow_nan=False)+'\n')

    def stats(self):
        return dict(next_tick=self.next_tick,expand_after=self.expand_after,observe_until=self.observe_until,
                    events=dict(self.events),last_decision=self.last_decision,epoch=self.epoch,
                    history_windows=len(self.history),accepted_samples=len(self.accepted),slack_samples=len(self.slack),
                    resize_mode='evict_from_shrinking_region',maximum_staging_tokens=self.maximum,
                    tiers_gb=self.tiers_gb,**self.read_pressure.stats())


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
            h._unreserved_budget_short() or h.ssd_backend.admission_budget_exhausted(),
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


class BudgetConsensus:
    """Publish one budget verdict per scheduler round for fixed partitions too."""
    def __init__(self, host, reduce_flags):
        self.host=host;self.reduce_flags=reduce_flags
        self.host._synced_budget_exhausted=False

    def advance(self):
        exhausted,_=self.reduce_flags(
            self.host._unreserved_budget_short()
            or self.host.ssd_backend.admission_budget_exhausted(),False)
        self.host._synced_budget_exhausted=bool(exhausted)


def _native_consensus_transport(host):
    """Bind our control messages to the existing native TP CPU process group."""
    import torch
    import torch.distributed as dist
    group=host.cache.tp_group
    ranks=dist.get_process_group_ranks(group)
    leader=ranks[0]
    def broadcast(value):
        values=[value if dist.get_rank()==leader else None]
        dist.broadcast_object_list(values,src=leader,group=group)
        return values[0]
    def reduce_flags(exhausted,blocked):
        credits=host.ssd_backend.protect_round_credits()
        unlimited=2**62
        flags=torch.tensor([int(exhausted),int(blocked),-(unlimited if credits is None else credits)],
                           dtype=torch.int64)
        dist.all_reduce(flags,op=dist.ReduceOp.MAX,group=group)
        exhausted,blocked,neg_credits=flags.tolist()
        host.ssd_backend.limit_round_credits(None if -neg_credits==unlimited else -neg_credits)
        return exhausted,blocked
    return broadcast,reduce_flags


def native_capacity_consensus(control):
    broadcast,reduce_flags=_native_consensus_transport(control.host)
    return CapacityConsensus(control,broadcast,reduce_flags)


def native_budget_consensus(host):
    _,reduce_flags=_native_consensus_transport(host)
    return BudgetConsensus(host,reduce_flags)
