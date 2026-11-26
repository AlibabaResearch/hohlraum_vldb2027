"""Measure request-path waits for physical SSD reads during prefill scheduling."""

from contextvars import ContextVar
from functools import wraps
from math import lcm
from types import SimpleNamespace


_prefill = ContextVar('hohlraum_prefill', default=None)


def _fits_prefill(adder, scheduler, request):
    """Bound prefill demand using the full prompt and the live budget."""
    if adder.prefill_max_requests is not None and len(adder.can_run_list) >= adder.prefill_max_requests:
        return False
    if adder.rem_input_tokens <= 0 or adder.rem_total_tokens <= adder.per_req_token_overhead:
        return False
    if adder.rem_chunk_tokens is not None and adder.rem_chunk_tokens <= 0:
        return False
    if (adder.prefill_delayer_single_pass is not None or adder.dllm_config is not None
            or adder.is_hybrid_swa or adder.rem_mamba_slots is not None
            or adder.kv_shard_scratch_spec is not None or scheduler.enable_priority_preemption
            or (request.sampling_params.ignore_eos and getattr(scheduler.tree_cache, 'disable', True))
            or scheduler.tree_cache.buffer_pipeline is not None):
        return False
    length = len(request.full_untruncated_fill_ids)
    if not length:
        return False
    page = adder.page_size
    padded = adder.ceil_paged_tokens(length)
    chunk = adder.rem_chunk_tokens
    if chunk is None:
        if adder.can_run_list and padded >= adder.rem_input_tokens:
            return False
        tile_tokens = padded
    else:
        if padded > chunk:
            # Any active chunked request may preclude starting another chunk.
            if scheduler.chunked_req is not None:
                return False
            alignment = lcm(page, scheduler.truncation_align_size or page)
            if chunk < alignment:
                return False
        tile_tokens = min(padded, chunk)
    if adder._check_prefill_tile_budget(tile_tokens) is not None:
        return False

    decode = max(0, request.sampling_params.max_new_tokens - len(request.output_ids))
    full_cost = padded + decode + adder.per_req_token_overhead
    budget = adder.memory_budget
    if full_cost >= min(budget.remaining_total, budget.remaining_current):
        return False
    return True


class ReadPressure:
    def __init__(self):
        self.clear()

    def clear(self):
        self.previous = {}
        self.current = {}
        self.waiters = set()
        self.seconds = 0.
        self.wall_seconds = 0.
        self._round_wall = 0.

    def record(self, rid, route, fits, now):
        self.waiters.add(rid)
        if not fits or rid in self.current:
            return
        prior = self.previous.get(rid)
        if prior is not None and prior[0] == route:
            now = max(now, prior[1])
            dt = now - prior[1]
            self.seconds += dt
            if dt > self._round_wall:
                self._round_wall = dt
        self.current[rid] = route, now

    def finish_round(self):
        self.wall_seconds += self._round_wall
        self._round_wall = 0.
        self.previous, self.current = self.current, {}

    def stats(self):
        count = len(self.waiters)
        return dict(exposed_ssd_wait_request_count=count, exposed_ssd_wait_sum_s=self.seconds,
                    exposed_ssd_wall_wait_sum_s=self.wall_seconds,
                    exposed_ssd_wait_mean_s=self.seconds / count if count else None)


def install_prefill_observer(scheduler_module):
    """Expose the live prefill budget to the storage check within this scheduler round."""
    scheduler_cls = scheduler_module.Scheduler
    adder_cls = scheduler_module.PrefillAdder
    raw_prefill = scheduler_cls._get_new_batch_prefill_raw
    init_adder = adder_cls.__init__

    @wraps(raw_prefill)
    def prefill(scheduler, *args, **kwargs):
        host = getattr(scheduler.tree_cache, 'hohlraum_host', None)
        control = getattr(host, 'capacity_controller', None)
        if control is None:
            return raw_prefill(scheduler, *args, **kwargs)
        active = SimpleNamespace(scheduler=scheduler, control=control, adder=None, requests=None)
        token = _prefill.set(active)
        try:
            return raw_prefill(scheduler, *args, **kwargs)
        finally:
            _prefill.reset(token)

    @wraps(init_adder)
    def init(adder, *args, **kwargs):
        init_adder(adder, *args, **kwargs)
        active = _prefill.get()
        if active is not None:
            active.adder = adder

    scheduler_cls._get_new_batch_prefill_raw = prefill
    adder_cls.__init__ = init


def observe_physical_wait(control, handle, operation, now):
    active = _prefill.get()
    if active is None or active.control is not control or active.adder is None:
        return
    if active.requests is None:
        active.requests = {req.rid: req for req in active.scheduler.waiting_queue}
    request = active.requests.get(handle.rid)
    if request is None:
        return
    route = getattr(operation, '_sim_read_route', None)
    # Mixed transfers must stay in the same physical segment. Page progress
    # within a pure SSD transfer keeps the same operation identity.
    identity = handle.attempt_id, operation.id, route.index if route is not None else None
    control.read_pressure.record(handle.rid, identity,
                                 _fits_prefill(active.adder, active.scheduler, request), now)
