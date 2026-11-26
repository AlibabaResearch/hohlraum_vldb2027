"""Unified marginal-throughput capacity controller for competing DRAM residencies.

Balances Read DRAM and Staging DRAM on one physical scale: marginal GPU prefill
recomputation saved per second (tokens / second).

- Read loss (L_read): tracks the most recent Delta-C keys evicted out of Host
  DRAM (_read_ghost). Hits on SSD-backed ghost keys cost alpha_ssd = w_ssd /
  t_prefill prefill-equivalent tokens; full misses and capacity failures cost
  1.0 token.
- Write gain (G_write): applies Little's Law (Delta-tau = Delta-C / lambda_stage)
  to the empirical TTD (accepted) and rejection-slack (slack) windows, weighted
  by budget utilization min(1, write_rate / sustainable_rate), mean admission
  reuse probability, and net compute saved per SSD hit max(0, 1 - alpha_ssd).
  Enforcing G_+ <= G_- gives the concave subdifferential [G_+, G_-] of write
  savings, so expand (L_read < G_+ - eps) and shrink (L_read > G_- + eps) are
  mutually exclusive around the 1-page-per-tick floor eps = page_size / period_s.
"""
from collections import Counter, OrderedDict, defaultdict, deque
import json
import math
from pathlib import Path

from .read_pressure import ReadPressure


class UnifiedCapacityController:
    def __init__(self, host):
        self.host = host
        c = host.cfg.get('capacity_control', {})
        allowed = {'mode', 'period_s', 'window_s', 'step_fraction', 'min_read_fraction'}
        unknown = c.keys() - allowed
        if unknown:
            raise ValueError('unknown unified capacity parameters: ' + ', '.join(sorted(unknown)))

        self.mode = c.get('mode', 'throughput')
        self.period = float(c.get('period_s', 60.0))
        self.window_s = float(c.get('window_s', 300.0))
        self.step_fraction = float(c.get('step_fraction', 0.025))
        self.min_read_fraction = float(c.get('min_read_fraction', 0.25))

        if (self.mode != 'throughput'
                or not math.isfinite(self.period) or self.period <= 0
                or not math.isfinite(self.window_s) or self.window_s < self.period
                or not 0 < self.step_fraction < 1
                or not 0 < self.min_read_fraction < 1):
            raise ValueError('invalid unified capacity controller parameters')

        page = host.cache.page_size
        self.page = page
        self.step_tokens = max(page, int(host.total_tokens * self.step_fraction) // page * page)
        self.ghost_max_pages = max(1, self.step_tokens // page)
        self.noise_floor_tok_s = page / self.period
        self.maximum = host.total_tokens - math.ceil(host.total_tokens * self.min_read_fraction / page) * page
        if self.maximum < page or host.staging_limit > self.maximum:
            raise ValueError('initial staging exceeds minimum read capacity')
        self.reset()

    def reset(self):
        now = self.host.now()
        self.accepted = deque()           # (t, remaining_ttd_s, tokens, reuse_prob)
        self.slack = deque()              # (t, rejection_slack_s, tokens, reuse_prob)
        self.states = {}                  # entry.id -> [ever_accepted, last_acc_at, first_rej_at]
        self._read_ghost = OrderedDict()  # last Delta-C keys evicted out of Host DRAM
        self.unresolved = getattr(self, 'unresolved', {})
        self.unresolved.clear()

        self.events = Counter()
        self.last_decision = None
        self.epoch = 0
        self._reset_at = now
        self.epoch_start = now
        self.window_start = now
        self.next_tick = now + self.period
        self.known_capacity = self.host.staging_limit
        self._in_resize = False
        self.read_pressure = ReadPressure()

        self._seen_budget = self.host.write_budget_bytes
        self._prev_charged_bytes = self.host.budget_charged_bytes
        self._prev_ssd_prefetched_tokens = getattr(self.host, 'ssd_prefetched_tokens', 0)
        self._raw_rates = defaultdict(float)
        self._decay = 1.0
        self._admit_prob = float(self.host.admission_threshold)
        self._clear_interval()

    def _clear_interval(self):
        self.read_pressure.clear()
        self.capacity_failed_tokens = 0
        self.writeback_failed_tokens = 0
        self._interval_ghost_ssd_tokens = 0
        self._interval_ghost_miss_tokens = 0
        self._interval_prefill_tokens = 0
        self._interval_prefill_time_s = 0.0
        self._interval_stage_exits_tokens = 0

    def _new_epoch(self, now, *_args, **_kwargs):
        self.epoch += 1
        self.epoch_start = now
        self.window_start = now
        self.next_tick = now + self.period
        self.known_capacity = self.host.staging_limit
        self._clear_interval()
        self._observe_residents(now)

    def _smooth(self, name, sample, alpha, norm):
        raw = alpha * sample + (1.0 - alpha) * self._raw_rates[name]
        self._raw_rates[name] = raw
        return raw * norm

    # --- Staging TTD & Slack Observation ---

    def _observe_residents(self, now):
        entries = [e for e in self.host.stage.values() if e.operation is None]
        if entries:
            profiles = self.host.sli.profiles([e.hash_value[0] for e in entries], now)
            for e in entries:
                self.observe(e, profiles[e.hash_value[0]], now)

    def observe(self, entry, profile, now):
        accepted = (profile[0] >= self.host.admission_threshold
                    or self.host.sli.pending.get(entry.hash_value[0], 0) > 0)
        state = self.states.setdefault(entry.id, [False, now, None])
        if accepted:
            state[:] = [True, now, None]
        elif state[0] and state[2] is None:
            state[2] = 0.5 * (state[1] + now)

    def entered(self, entry):
        key = entry.hash_value[0]
        self._read_ghost.pop(key, None)  # transferred into Staging, still in Host DRAM
        now = self.host.now()
        self.observe(entry, self.host.sli.profiles([key], now)[key], now)

    def write_sample(self, entry):
        if self._in_resize:
            return None
        key = entry.hash_value[0]
        now = self.host.now()
        profile = self.host.sli.profiles([key], now)[key]
        self.observe(entry, profile, now)
        eligible = (profile[0] >= self.host.admission_threshold
                    and not self.host.sli.pending.get(key, 0))
        wait = profile[1] if eligible and math.isfinite(profile[1]) else math.inf
        return (now, wait, self.page, float(profile[0]))

    def wrote(self, sample):
        if sample is not None:
            self.accepted.append(sample)
            self._interval_stage_exits_tokens += sample[2]

    def discarded(self, entry, budget_exhausted=False):
        now = self.host.now()
        key = entry.hash_value[0]
        self.observe(entry, self.host.sli.profiles([key], now)[key], now)
        state = self.states.get(entry.id)
        self._interval_stage_exits_tokens += self.page
        if (not self._in_resize and not budget_exhausted
                and state is not None and state[0] and state[2] is not None):
            self.slack.append((now, max(0.0, now - state[2]), self.page, self._admit_prob))

    def removed(self, entry):
        self.states.pop(entry.id, None)

    # --- Read Ghost & Compute/I/O Feedback ---

    def observe_read_membership(self, key, in_read):
        if in_read:
            self._read_ghost.pop(key, None)
        elif key in self._read_ghost:
            self._read_ghost.move_to_end(key)
        else:
            self._read_ghost[key] = None
            while len(self._read_ghost) > self.ghost_max_pages:
                self._read_ghost.popitem(last=False)

    def observe_request_hits(self, hashes, _now):
        host = self.host
        host_index = host._host_index
        live_hbm = host_index.native_live_keys
        host_keys = host_index.host_keys
        stage = host.stage
        ssd_backend = getattr(host, 'ssd_backend', None)
        page = self.page
        ghost = self._read_ghost

        for k in hashes:
            if k in live_hbm or k in host_keys or k in stage:
                continue
            if k in ghost:
                ghost.pop(k, None)
                if ssd_backend is not None and ssd_backend.exists(k):
                    self._interval_ghost_ssd_tokens += page
                else:
                    self._interval_ghost_miss_tokens += page

    def observe_batch_prefill(self, extend_tokens, duration_s):
        if extend_tokens > 0 and duration_s > 0.0:
            self._interval_prefill_tokens += extend_tokens
            self._interval_prefill_time_s += duration_s

    def read_progress(self, rid, waiting, now, has_ssd):
        pass

    def end_read(self, rid, now):
        pass

    def failed(self, _rid, tokens, reason):
        if tokens > 0 and reason in ('host_capacity', 'storage_transfer', 'cache_admission_shortfall'):
            if reason == 'host_capacity':
                self.capacity_failed_tokens += int(tokens)
            self.events[reason] += 1

    def writeback_failed(self, _node_id, tokens):
        if tokens > 0:
            self.writeback_failed_tokens += int(tokens)
            self.events['writeback_host_failure'] += 1

    # --- Marginal Balance Scale & Counterfactual Shift ---

    def _budget_ratio(self, now, write_rate_bps):
        budget = self.host.write_budget_bytes
        if budget is None:
            return 0.0
        remaining = budget - self.host.budget_charged_bytes
        if remaining <= 0:
            return 0.0
        end_s = float(self.host.cfg.get('duration_s', 14400.0))
        sustainable_bps = remaining / max(1.0, end_s - now)
        return min(1.0, max(0.0, write_rate_bps / sustainable_bps))

    def _shift_ttd_on_resize(self, delta_tokens, stage_rate):
        if delta_tokens == 0:
            return
        dtau = (max(1.0, min(3600.0, abs(delta_tokens) / stage_rate))
                if stage_rate > 1e-3 else self.window_s)
        if delta_tokens > 0:
            acc = [(t, w - dtau, tok, p) for t, w, tok, p in self.accepted if w > dtau]
            slk = ([(t, s + dtau, tok, p) for t, s, tok, p in self.slack]
                   + [(t, dtau - w, tok, p) for t, w, tok, p in self.accepted if w <= dtau])
        else:
            acc = ([(t, w + dtau, tok, p) for t, w, tok, p in self.accepted]
                   + [(t, dtau - s, tok, p) for t, s, tok, p in self.slack if s <= dtau])
            slk = [(t, s - dtau, tok, p) for t, s, tok, p in self.slack if s > dtau]
        self.accepted = deque(sorted(acc, key=lambda x: x[0]))
        self.slack = deque(sorted(slk, key=lambda x: x[0]))

    def tick(self, now, *, force=False, synchronize=None):
        if self.host.staging_limit != self.known_capacity:
            self._new_epoch(now, 'external')
        if not force and now < self.next_tick:
            return

        dt = max(1e-3, now - self.window_start)
        alpha = min(1.0, 2.0 * dt / (self.window_s + dt))
        self._decay *= (1.0 - alpha)
        norm = 1.0 / max(1e-9, 1.0 - self._decay)

        cutoff = now - self.window_s
        while self.accepted and self.accepted[0][0] < cutoff:
            self.accepted.popleft()
        while self.slack and self.slack[0][0] < cutoff:
            self.slack.popleft()
        self._observe_residents(now)

        # 1. Volume-weighted physical rates via bias-corrected EWMA.
        r_prefill_tok = self._smooth('prefill_tok', self._interval_prefill_tokens / dt, alpha, norm)
        r_prefill_time = self._smooth('prefill_time', self._interval_prefill_time_s / dt, alpha, norm)
        t_prefill = r_prefill_time / r_prefill_tok if r_prefill_tok > 0.0 else 2e-5

        exposure = self.read_pressure.stats()
        wall_wait_s = float(exposure.get('exposed_ssd_wall_wait_sum_s') or 0.0)
        ssd_now = getattr(self.host, 'ssd_prefetched_tokens', 0)
        if ssd_now < self._prev_ssd_prefetched_tokens:
            self._prev_ssd_prefetched_tokens = 0
        interval_ssd_tokens = ssd_now - self._prev_ssd_prefetched_tokens
        self._prev_ssd_prefetched_tokens = ssd_now

        r_ssd_tok = self._smooth('ssd_tok', interval_ssd_tokens / dt, alpha, norm)
        r_ssd_wait = self._smooth('ssd_wait', wall_wait_s / dt, alpha, norm)
        w_ssd = r_ssd_wait / r_ssd_tok if r_ssd_tok >= self.noise_floor_tok_s else 0.0

        budget_now = self.host.write_budget_bytes
        charged_now = self.host.budget_charged_bytes
        if budget_now != self._seen_budget or charged_now < self._prev_charged_bytes:
            self._seen_budget = budget_now
            self._prev_charged_bytes = 0
        interval_charged = charged_now - self._prev_charged_bytes
        self._prev_charged_bytes = charged_now

        obs_span = min(self.window_s, max(1e-3, now - self._reset_at))
        kv_bpt = float(self.host.kv_bytes_per_token)
        admitted_bps = (sum(x[2] for x in self.accepted) / obs_span) * kv_bpt if self.accepted else 0.0

        write_bps = self._smooth('write_bps', max(interval_charged / dt, admitted_bps), alpha, norm)
        stage_rate = self._smooth('stage_rate', self._interval_stage_exits_tokens / dt, alpha, norm)
        ghost_ssd_rate = self._smooth('ghost_ssd', self._interval_ghost_ssd_tokens / dt, alpha, norm)
        ghost_miss_rate = self._smooth('ghost_miss', self._interval_ghost_miss_tokens / dt, alpha, norm)

        if self.accepted:
            mean_p = sum(x[3] for x in self.accepted) / len(self.accepted)
            self._admit_prob = alpha * mean_p + (1.0 - alpha) * self._admit_prob
        admit_prob = max(0.0, min(1.0, self._admit_prob))

        # 2. Marginal throughput balance scale (recomputed prefill tokens / second).
        alpha_ssd = w_ssd / max(1e-9, t_prefill)
        net_ssd_saving = max(0.0, 1.0 - alpha_ssd)
        failed_tokens = self.capacity_failed_tokens + self.writeback_failed_tokens

        loss_read = ghost_ssd_rate * alpha_ssd + ghost_miss_rate + (failed_tokens / dt)

        delta_tau_s = (max(1.0, min(3600.0, self.step_tokens / stage_rate))
                       if stage_rate > 1e-3 else self.window_s)
        dw_expand = sum(x[2] for x in self.accepted if x[1] <= delta_tau_s) / obs_span
        dw_shrink = sum(x[2] for x in self.slack if x[1] <= delta_tau_s) / obs_span

        rho_expand = self._budget_ratio(now, write_bps)
        rho_shrink = self._budget_ratio(now, write_bps + dw_shrink * kv_bpt)

        gain_write_expand = dw_expand * rho_expand * admit_prob * net_ssd_saving
        gain_write_shrink = max(gain_write_expand, dw_shrink * rho_shrink * admit_prob * net_ssd_saving)

        net_expand = gain_write_expand - loss_read
        net_shrink = loss_read - gain_write_shrink

        # 3. Single symmetric step decision.
        before = self.host.staging_limit
        target = before
        reason = 'balanced'
        pending = (self.host.read_allocated() > self.host.read_limit
                   or self.host.waiting_staging_tokens() > before)

        if self.host._budget_exhausted():
            target = 0
            reason = 'budget_exhausted'
        elif pending:
            reason = 'resize_pending'
        elif before > 0 and net_shrink > self.noise_floor_tok_s:
            target = max(0, (before - self.step_tokens) // self.page * self.page)
            reason = 'throughput_shrink'
        elif before < self.maximum and net_expand > self.noise_floor_tok_s:
            target = min(self.maximum, before + self.step_tokens)
            reason = 'throughput_expand'

        if synchronize is not None:
            decision = synchronize(dict(target=target, reason=reason))
            target = decision['target']
            reason = decision['reason']

        applied = True
        if target != before:
            self._in_resize = True
            try:
                applied = self.host.resize_staging(target) is not False
            finally:
                self._in_resize = False
        actual_delta = self.host.staging_limit - before
        changed = actual_delta != 0

        self.last_decision = dict(
            t=now, mode=self.mode, epoch=self.epoch,
            before_tokens=before, target_tokens=target,
            actual_target_tokens=self.host.staging_limit,
            applied=applied, changed=changed, reason=reason,
            t_prefill_s_per_tok=t_prefill, w_ssd_s_per_tok=w_ssd, alpha_ssd=alpha_ssd,
            delta_tau_s=delta_tau_s, dw_expand_tok_s=dw_expand, dw_shrink_tok_s=dw_shrink,
            rho_expand=rho_expand, rho_shrink=rho_shrink, admit_prob=admit_prob,
            ghost_ssd_tok_s=ghost_ssd_rate, ghost_miss_tok_s=ghost_miss_rate,
            failed_tokens=failed_tokens,
            loss_read_tok_s=loss_read,
            gain_write_expand_tok_s=gain_write_expand,
            gain_write_shrink_tok_s=gain_write_shrink,
            net_expand_tok_s=net_expand, net_shrink_tok_s=net_shrink,
            noise_floor_tok_s=self.noise_floor_tok_s,
            exposed_ssd_wall_wait_sum_s=wall_wait_s, interval_ssd_tokens=interval_ssd_tokens,
        )
        if changed:
            self._shift_ttd_on_resize(actual_delta, stage_rate)
            self.epoch += 1
            self.epoch_start = now
            self.known_capacity = self.host.staging_limit
        self.events[reason] += 1
        self.window_start = now
        self.next_tick = now + self.period
        self._clear_interval()

        if self.host.stats_path:
            with Path(str(self.host.stats_path) + '.capacity.jsonl').open('a') as f:
                f.write(json.dumps(self.last_decision, allow_nan=False) + '\n')

    def stats(self):
        return dict(
            mode=self.mode, next_tick=self.next_tick,
            events=dict(self.events), last_decision=self.last_decision, epoch=self.epoch,
            accepted_samples=len(self.accepted), slack_samples=len(self.slack),
            read_ghost_pages=len(self._read_ghost),
            maximum_staging_tokens=self.maximum, step_tokens=self.step_tokens,
            noise_floor_tok_s=self.noise_floor_tok_s,
            **self.read_pressure.stats(),
        )
