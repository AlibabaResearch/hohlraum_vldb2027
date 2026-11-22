"""Limit native FULL prefetch reclamation to its missing read capacity."""

from threading import local


def install_prefetch_reclaim(cache, is_partition_active, counters):
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    controller = cache.cache_controller
    pool = controller.mem_pool_host
    hits = controller.prefetch_hit_queue
    original_drain = cache._drain_storage_control_queues_impl
    original_get = hits.get
    original_alloc = pool.alloc
    original_evict = cache.evict_host
    scope = local()

    def drain(n_storage_hit, *args, **kwargs):
        if n_storage_hit == 0 or getattr(scope, 'state', None) is not None:
            return original_drain(n_storage_hit, *args, **kwargs)
        scope.state = [None, None]
        try:
            return original_drain(n_storage_hit, *args, **kwargs)
        finally:
            scope.state = None

    def take_hit(*args, **kwargs):
        operation = original_get(*args, **kwargs)
        state = getattr(scope, 'state', None)
        if state is not None:
            state[:] = [operation.storage_hit_count, None]
        return operation

    def allocate(tokens, *args, **kwargs):
        state = getattr(scope, 'state', None)
        if state is None or tokens != state[0]:
            return original_alloc(tokens, *args, **kwargs)
        state[0] = None
        indices = original_alloc(tokens, *args, **kwargs)
        if indices is None:
            state[1] = tokens
        return indices

    def evict(tokens, component_type=ComponentType.FULL):
        state = getattr(scope, 'state', None)
        if state is None or component_type != ComponentType.FULL or tokens != state[1]:
            return original_evict(tokens, component_type)
        state[1] = None
        if not is_partition_active():
            return original_evict(tokens, component_type)
        # Native passes the total receive size after its first alloc fails.
        # evict_host expects additional space. available_size is already
        # limited by both the read partition and the physical slots.
        shortage = max(0, tokens - pool.available_size())
        counters['prefetch_allocation_reclaim_attempts'] += 1
        if shortage < tokens:
            counters['prefetch_allocation_reclaim_reduced'] += 1
            counters['prefetch_allocation_reclaim_excess_tokens'] += tokens - shortage
        return original_evict(shortage, component_type) if shortage else 0

    cache._drain_storage_control_queues_impl = drain
    hits.get = take_hit
    pool.alloc = allocate
    cache.evict_host = evict
