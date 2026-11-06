"""Experiment-only closed-loop request source using native HiSim queue/time.

No clock writes, CPU subtraction, predictor replacement, or cache hooks.
Install after importing the simulator engine and before constructing it.
"""
import heapq
import json
import os
from pathlib import Path
from functools import wraps
from hohlraum.experiments.common.workload import TraceSource, encode, dump, restore, bootstrap_at_first_ready
from hohlraum.experiments.common.backpressure import QueueDelayAdmission, queue_feedback


def install(concurrency, contract):
    from sglang_simulator.simulation.sglang.scheduler import ReqDispatcher
    from sglang_simulator.simulation.manager import StateManager
    from sglang.srt.managers.scheduler import Scheduler
    if getattr(ReqDispatcher, '_experiment_closed_loop', False):
        raise RuntimeError('closed-loop source already installed')
    horizon = float(os.environ.get('HISIM_AGENTX_HORIZON_S', 'inf'))
    rows_path = os.environ.get('HISIM_AGENTX_COMPACT_ROWS')
    rows = json.loads(Path(rows_path).read_text()) if rows_path else None
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
            self._source = TraceSource(self.future_queue, concurrency)
            if os.environ.get('HISIM_RESTORE_CLIENT_CHECKPOINT'):
                checks=restore(self._source,json.loads(Path(os.environ['HISIM_RESTORE_CLIENT_CHECKPOINT']).read_text()))
                Path(os.environ['HOHLRAUM_EXPERIMENT_CASE'],'resume_checks.json').write_text(json.dumps(checks))
            self._source._initial_ready, startup = bootstrap_at_first_ready(self._source)
            Path(os.environ['HOHLRAUM_EXPERIMENT_CASE'], 'startup_boundary.json').write_text(json.dumps(startup))
            self.future_queue.clear()
            self._source.request_admission = QueueDelayAdmission(lambda: queue_feedback(scheduler_ref[0], self._source.request_admission, StateManager.get_global_clock()), float(os.environ["HISIM_CLIENT_QUEUE_DELAY_S"]), int(os.environ["HISIM_CLIENT_WAITING_LIMIT"]))

    def expand(req):
        if rows is None:
            return
        from array import array
        meta = req.sampling_params.custom_params['simulation']['agentx']
        row = rows[meta['trace_serial']]
        meta['prefix_blocks'] = row['blocks']
        meta['prefix_page_size'] = 1024
        policy_meta = req.sampling_params.custom_params.get('hohlraum')
        if policy_meta is not None:
            policy_meta['prefix_blocks'] = row['blocks']
            policy_meta['prefix_page_size'] = 1024
        req.input_ids = array('q', encode(row['root'], row['blocks'], row['raw_input_tokens'],
                                        meta['trace_serial'], meta['vocab_size'], contract['token_tail']))

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
                    if contract['detailed_audit']:
                        meta = req.sampling_params.custom_params['simulation']['agentx']
                        source.audit[-1].update(think_time_s=float(meta['think_time']), input_tokens=int(rows[meta['trace_serial']]['raw_input_tokens']))
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


def install_horizon(horizon, output):
    from sglang.srt.managers.scheduler import Scheduler
    from sglang_simulator.simulation.manager import StateManager
    from sglang_simulator.simulation.sglang.scheduler import ReqDispatcher, C_SchedulerHook
    from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
    output = Path(output)
    from hohlraum.core.staging_storage import BudgetStorage
    old_charge = BudgetStorage._charge
    def charged(self, key):
        old_charge(self, key)
        if self.budget_exhausted() and not hasattr(self, 'budget_exhausted_at_s'):
            self.budget_exhausted_at_s = StateManager.get_global_clock()
            output.mkdir(parents=True, exist_ok=True)
            (output/'budget_exhausted.json').write_text(json.dumps(dict(
                clock_s=self.budget_exhausted_at_s, write_bytes=self.regular_write_bytes)))
    BudgetStorage._charge = charged
    old_next = Scheduler.get_next_batch_to_run
    old_result = Scheduler.process_batch_result
    written = False
    next_progress = 0.0

    def observe(scheduler):
        nonlocal written, next_progress
        if written:
            return
        now = StateManager.get_global_clock()
        source = getattr(ReqDispatcher._instance, '_source', None)
        if source is None:
            return
        tree = scheduler.tree_cache
        backend = tree.cache_controller.storage_backend
        progress = dict(clock_s=now, completed=len(source.completed),
                        write_bytes=getattr(backend, 'regular_write_bytes', 0),
                        budget_bytes=getattr(backend, 'validation_cap', None), budget_exhausted_at_s=getattr(backend, 'budget_exhausted_at_s', None))
        if now >= next_progress:
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
            snapshot["client_admission"] = dict(limit=gate.limit, pending=len(gate.pending), inflight=len(gate.inflight), records=gate.records, ttft_includes_external_wait=True)
        host = getattr(tree, 'hohlraum_host', None)
        if host is not None:
            snapshot['hohlraum'] = host.stats()
        probe = getattr(tree, 'prefetch_outcome_stats_snapshot', None)
        if probe is not None:
            snapshot['prefetch_outcomes'] = probe()
        if os.environ.get('HISIM_SAVE_CLIENT_CHECKPOINT') == '1':
            (output/'client_checkpoint.json').write_text(json.dumps(dump(source,now)))
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
