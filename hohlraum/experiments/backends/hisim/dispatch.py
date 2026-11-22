"""Experiment-only closed-loop request source using native HiSim queue/time.

No clock writes, CPU subtraction, predictor replacement, or cache hooks.
Install after importing the simulator engine and before constructing it.
"""
import heapq
import json
import os
from pathlib import Path
from functools import wraps
from hohlraum.experiments.common.workload import TraceSource, encode_array, bootstrap_at_first_ready
from hohlraum.experiments.common.backpressure import QueueDelayAdmission, queue_feedback


def reset_formal_admission(host, start_s):
    """Reset experiment feedback at measurement start, honoring policy config."""
    from collections import deque
    cfg=host.cfg
    threshold=float(cfg.get('threshold',.7))
    dynamic=bool(cfg.get('dynamic_admission_threshold',False))
    policy=host.staging_policy
    policy.dynamic_admission=dynamic
    if dynamic:
        period=float(cfg.get('threshold_period_s',1800.))
        policy._init_admission_feedback(threshold,cfg.get('duration_s',14400),
                                        period,cfg.get('threshold_window_s',1800.))
        policy._admission_history=deque([(float(start_s),0)])
        policy.next_tick=float(start_s)+period
    host.admission_threshold=threshold
    host.sli.set_threshold(threshold)
    assert host.admission_threshold==threshold and policy.dynamic_admission==dynamic
    return dict(formal_admission_threshold=threshold,formal_dynamic_admission=dynamic)


def install(manifest, contract, rows):
    from sglang_simulator.simulation.sglang.scheduler import ReqDispatcher
    from sglang_simulator.simulation.manager import StateManager
    from sglang.srt.managers.scheduler import Scheduler
    if getattr(ReqDispatcher, '_experiment_closed_loop', False):
        raise RuntimeError('closed-loop source already installed')
    horizon = manifest['horizon_s']
    old_add, old_dispatch = ReqDispatcher.add, ReqDispatcher.dispatch
    old_has, old_next, old_reset = ReqDispatcher.has_next, ReqDispatcher.next_req_from_future_ts, ReqDispatcher.reset
    scheduler_ref = [None]
    old_init = Scheduler.__init__
    def capture_scheduler(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        scheduler_ref[0] = self
    Scheduler.__init__ = capture_scheduler
    old_enqueue = Scheduler._add_request_to_queue
    def received_request(self, req, *args, **kwargs):
        value = old_enqueue(self, req, *args, **kwargs)
        source = getattr(ReqDispatcher._instance, '_source', None)
        if source is not None:
            source.request_admission.received(req.rid)
        return value
    Scheduler._add_request_to_queue = received_request
    old_result = Scheduler.process_batch_result

    def add(self, reqs):
        old_add(self, reqs)
        if self.offline_recv_all_requests and getattr(self, '_source', None) is None:
            # Native add has received the entire declared input and built its heap.
            self._source = TraceSource(self.future_queue, manifest['concurrency'])
            self._source._initial_ready, startup = bootstrap_at_first_ready(self._source)
            Path(os.environ['HOHLRAUM_EXPERIMENT_CASE'], 'startup_boundary.json').write_text(json.dumps(startup))
            self.future_queue.clear()
            self._source.request_admission = QueueDelayAdmission(lambda: queue_feedback(scheduler_ref[0], self._source.request_admission, StateManager.get_global_clock()), float(os.environ["HISIM_CLIENT_QUEUE_DELAY_S"]), int(os.environ["HISIM_CLIENT_WAITING_LIMIT"]))

    def expand(req):
        meta = req.sampling_params.custom_params['simulation']['agentx']
        row = rows[meta['trace_serial']]
        meta['prefix_blocks'] = row['blocks']
        meta['prefix_page_size'] = 1024
        policy_meta = req.sampling_params.custom_params.get('hohlraum')
        if policy_meta is not None:
            policy_meta['prefix_blocks'] = row['blocks']
            policy_meta['prefix_page_size'] = 1024
        req.input_ids = encode_array(row['root'], row['blocks'], row['raw_input_tokens'],
                                     meta['trace_serial'], meta['vocab_size'], contract['token_tail'])

    def dispatch(self):
        source = getattr(self, '_source', None)
        if source is not None and StateManager.get_global_clock() < horizon:
            now = StateManager.get_global_clock()
            gate = source.request_admission
            initial = source._initial_ready
            source._initial_ready = []
            for req in initial + source.pop_due(now):
                expand(req)
                # Freeze policy idle time at client readiness, before backpressure.
                host = getattr(getattr(scheduler_ref[0], 'tree_cache', None), 'hohlraum_host', None)
                if host is not None:
                    host.sli.client_ready(req, source.arrivals[req.rid])
                gate.offer(req, source.arrivals[req.rid])
            for req in gate.release(now):
                req.sampling_params.custom_params["simulation"]["created_time"] = now
                when = req.sampling_params.custom_params['simulation']['created_time']
                source.serial += 1
                heapq.heappush(self.future_queue, (when, source.serial, req))
        return old_dispatch(self)

    def has_next(self):
        source = getattr(self, '_source', None)
        return old_has(self) or (source is not None and StateManager.get_global_clock() < horizon and (source.next_time() is not None or source.request_admission.can_release()))

    def next_time(self):
        values = [old_next(self)] if old_has(self) else []
        source = getattr(self, '_source', None)
        if source is not None and source.next_time() is not None:
            values.append(source.next_time())
        if source is not None and source.request_admission.can_release():
            values.append(StateManager.get_global_clock())
        return min(min(values), horizon)

    def reset(self):
        source = getattr(self, '_source', None)
        if source is not None:
            self._last_source = source
            output = Path(os.environ['SGLANG_SIMULATOR_OUTPUT_DIR'])
            output.mkdir(parents=True, exist_ok=True)
            (output/'closed_loop_audit.json').write_text(json.dumps(dict(
                requests=source.audit, admission=source.admission_audit,
                concurrency=source.concurrency, idle_timeout=source.idle_timeout)))
        self._source = None
        return old_reset(self)

    @wraps(old_result)
    def result(self, batch, *args, **kwargs):
        requests = list(batch.reqs)
        value = old_result(self, batch, *args, **kwargs)
        dispatcher = ReqDispatcher._instance
        source = getattr(dispatcher, '_source', None)
        if source is not None:
            now = StateManager.get_global_clock()
            for req in requests:
                if req.finished() and req.rid not in source.completed:
                    source.request_admission.finish(req.rid)
                    source.finish(req, now)
                    source.audit[-1].update(source.request_admission.records[req.rid])
                    # Native SGLang computes consumed-source breakdown on the
                    # first chunk; recording it avoids equating prefetch with use.
                    source.audit[-1].update(rid=req.rid,
                        cached_tokens=req.cached_tokens,
                        device_hit_tokens=req.cached_tokens_device,
                        host_hit_tokens=req.cached_tokens_host,
                        storage_hit_tokens=req.cached_tokens_storage)
                    if rows is not None:
                        meta = req.sampling_params.custom_params['simulation']['agentx']
                        source.requests.pop((meta['root'], meta['chain'], meta['index']), None)
        return value

    ReqDispatcher.add, ReqDispatcher.dispatch = add, dispatch
    ReqDispatcher.has_next, ReqDispatcher.next_req_from_future_ts = has_next, next_time
    ReqDispatcher.reset = reset
    Scheduler.process_batch_result = result
    ReqDispatcher._experiment_closed_loop = True


"""Read-only experiment snapshots at native scheduling boundaries."""
import json
import os
from dataclasses import asdict
from pathlib import Path
from functools import wraps


def install_horizon(manifest, rows, output):
    from sglang.srt.managers.scheduler import Scheduler
    from sglang_simulator.simulation.manager import StateManager
    from sglang_simulator.simulation.sglang.scheduler import ReqDispatcher, C_SchedulerHook
    from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
    output = Path(output)
    horizon = manifest['horizon_s']
    warmup_end = manifest['warmup_seconds']
    from hohlraum.core.staging_storage import BudgetStorage
    old_charge = BudgetStorage._charge
    def charged(self, key):
        old_charge(self, key)
        if self.physical_budget_exhausted() and not hasattr(self, 'budget_exhausted_at_s'):
            self.budget_exhausted_at_s = StateManager.get_global_clock()
            output.mkdir(parents=True, exist_ok=True)
            (output/'budget_exhausted.json').write_text(json.dumps(dict(
                clock_s=self.budget_exhausted_at_s, write_bytes=self.validation_write_bytes)))
    BudgetStorage._charge = charged
    old_next = Scheduler.get_next_batch_to_run
    old_result = Scheduler.process_batch_result
    written = False
    next_progress = 0.0
    measurement_started = False

    def observe(scheduler):
        nonlocal written, next_progress, measurement_started
        if written:
            return
        now = StateManager.get_global_clock()
        source = getattr(ReqDispatcher._instance, '_source', None)
        if source is None:
            return
        tree = scheduler.tree_cache
        host = getattr(tree, 'hohlraum_host', None)
        backend = host.ssd_backend if host is not None else tree.cache_controller.storage_backend
        assert isinstance(backend, BudgetStorage), type(backend)
        if host is not None:
            assert tree.cache_controller.storage_backend.backing is backend
        if not measurement_started and now >= warmup_end:
            from hohlraum.experiments.backends.hisim.warmup import activate
            activation=activate(tree,source,now,output,rows,manifest['workload'])
            host = getattr(tree, 'hohlraum_host', None)
            warm = dict(clock_s=now, warmup_write_bytes=backend.regular_write_bytes,
                        completed=len(source.completed), cache_preserved=True,
                        pending_client=len(source.request_admission.pending),
                        staged_pages=len(host.stage) if host is not None else 0,
                        host_stats=host.stats() if host is not None else None)
            formal_cap=manifest['budget_bytes']
            backend.reset_budget(None if formal_cap is None else formal_cap // tree.tp_world_size)
            assert not backend.budget_exhausted()
            if host is not None:
                assert host.budget_exhausted_at_s is None
                activation.update(reset_formal_admission(host,warmup_end))
            (output/'activation-checks.json').write_text(json.dumps(activation))
            assert tree.cache_controller.storage_backend.validation_cap == backend.validation_cap
            if host is not None:
                assert host.write_budget_bytes == backend.validation_cap
                assert host.budget_charged_bytes == 0
            warm['formal_budget_bytes_per_rank']=backend.validation_cap
            (output/'warm_boundary.json').write_text(json.dumps(warm))
            measurement_started=True
        storage_exhausted_at = getattr(backend, 'budget_exhausted_at_s', None)
        cutover_at = host.budget_exhausted_at_s if host is not None else storage_exhausted_at
        progress = dict(clock_s=now, completed=len(source.completed),
                        write_bytes=backend.validation_write_bytes,
                        regular_write_bytes=backend.regular_write_bytes,
                        transition_write_bytes=backend.transition_write_bytes,
                        budget_bytes=backend.validation_cap,
                        budget_exhausted_at_s=cutover_at,
                        physical_budget_exhausted_at_s=storage_exhausted_at)
        if now >= next_progress:
            if host is not None:
                progress['admission_threshold']=host.admission_threshold
                progress['dynamic_admission']=host.staging_policy.dynamic_admission
                if not host.cfg.get('dynamic_admission_threshold',False):
                    assert not host.staging_policy.dynamic_admission
                    assert abs(host.admission_threshold-float(host.cfg.get('threshold',.7)))<1e-12, 'Fixed threshold changed during experiment'
            output.mkdir(parents=True, exist_ok=True)
            with (output/'progress.jsonl').open('a') as f:
                f.write(json.dumps(progress)+'\n')
            next_progress = now + 60
        if now < horizon:
            return
        written = True
        stats = [asdict(x) for x in request_stats_manager.get_all_req_stats()
                 if x.rid in source.completed]
        gate = getattr(source, "request_admission", None)
        if gate is not None:
            stats = gate.export_stats(stats)
        snapshot = dict(horizon_s=horizon, observed_clock_s=now, requests=stats,
                        completed_audit=source.audit, admission_events=source.admission_audit,
                        progress=progress, iterations=C_SchedulerHook.ITERATION_STATS,
                        performance_validated=False, timing='unchanged native HiSim')
        if gate is not None:
            snapshot["client_admission"] = dict(waiting_limit=gate.waiting_limit, pending=len(gate.pending), inflight=len(gate.inflight), records=gate.records, ttft_includes_external_wait=True)
        host = getattr(tree, 'hohlraum_host', None)
        if host is not None:
            snapshot['hohlraum'] = host.stats()
        probe = getattr(tree, 'prefetch_outcome_stats_snapshot', None)
        if probe is not None:
            snapshot['prefetch_outcomes'] = probe()
        tmp = output/'horizon.pending.json'
        tmp.write_text(json.dumps(snapshot))
        tmp.replace(output/'horizon.json')

    @wraps(old_next)
    def next_batch(self, *args, **kwargs):
        observe(self)
        return old_next(self, *args, **kwargs)

    @wraps(old_result)
    def result(self, *args, **kwargs):
        value = old_result(self, *args, **kwargs)
        observe(self)
        return value

    Scheduler.get_next_batch_to_run = next_batch
    Scheduler.process_batch_result = result
