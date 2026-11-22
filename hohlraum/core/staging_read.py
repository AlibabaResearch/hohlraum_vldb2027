"""Return staging slots to the native read cache at prefetch completion."""

from dataclasses import dataclass, replace
from functools import cache
from itertools import groupby, islice


@dataclass
class ReadClaim:
    sources: dict
    submitted: bool = False


@cache
def metadata_route_type():
    from sglang_simulator.simulation.read_routes import ReadRoute

    class MetadataReadRoute(ReadRoute):
        def consume_resident(self):
            while self.index < len(self.steps) and self.steps[self.index][2] == 'staging_resident':
                self.completed = self.steps[self.index][1]
                self.index += 1

        def advance(self, duration, token_bytes, estimate):
            def advance_step(tokens, width, available, bandwidth):
                return (tokens, 0.0) if bandwidth is None else estimate(tokens, width, available, bandwidth)

            self.consume_resident()
            _, spent = super().advance(duration, token_bytes, advance_step)
            # A resident segment after an SSD segment is already available
            # even when that physical read used the final transfer credit.
            self.consume_resident()
            self.bytes_by_resource.pop('staging_resident', None)
            return self.completed, spent

    return MetadataReadRoute


class StagingReadTransfers:
    def __init__(self, host, backend):
        self.host = host
        self.backend = backend
        self.cache = host.cache
        self.controller = self.cache.cache_controller
        self.page = self.cache.page_size
        self.threaded = backend._threaded

    def install(self):
        drain = self.cache._drain_storage_control_queues_impl
        put = self.controller.prefetch_buffer.put

        def drain_with_claims(n_storage_hit, *args, **kwargs):
            claims = self.prepare(n_storage_hit) if n_storage_hit else []
            try:
                return drain(n_storage_hit, *args, **kwargs)
            finally:
                with self.backend.lock:
                    for lease in claims:
                        if lease.claim is not None and not lease.claim.submitted:
                            self.backend.release_read_claim(lease)

        def put_with_claim(operation, *args, **kwargs):
            lease = self.backend.leases.get(id(operation))
            if lease is not None and lease.claim is not None:
                if not self.threaded:
                    self.install_simulated_route(operation, lease)
                lease.claim.submitted = True
            return put(operation, *args, **kwargs)

        self.cache._drain_storage_control_queues_impl = drain_with_claims
        self.controller.prefetch_buffer.put = put_with_claim

    def candidates(self, operation, hashes, lease):
        if (lease is None or lease.claim is not None or
                tuple(lease.keys[:len(hashes)]) != hashes or
                operation.is_terminated() or operation.storage_hit_count != len(hashes)*self.page or
                operation.storage_hit_count < self.cache.prefetch_threshold or
                operation.pool_transfers or self.host._budget_return_pending):
            return {}
        info = self.cache.ongoing_prefetch.get(operation.handle)
        if info is None or info.operation is not operation or info.host_indices is not None:
            return {}
        key = info.prefetch_key
        candidates = {}
        for index, name in enumerate(hashes):
            entry = self.host.stage.get(name)
            if entry is None or entry.operation is not None or getattr(entry, 'copy_committed', False):
                continue
            parent = hashes[index-1] if index else operation.last_hash
            if ((entry.prefix_hashes[-1] if entry.prefix_hashes else None) != parent or
                    entry.key.extra_key != key.extra_key or entry.key.cache_salt != key.cache_salt):
                continue
            if self.threaded:
                if entry.read_pins:
                    continue
            elif entry.read_pins != 1 or lease.entries.get(index) is not entry:
                continue
            candidates[index] = entry
        return candidates

    def common_positions(self, batch):
        groups = tuple(group for group in (getattr(self.cache, 'attn_cp_group', None),
                                          getattr(self.cache, 'attn_tp_group', None))
                       if group is not None)
        if self.cache.tp_world_size == 1 and not groups:
            return [set(candidates) for _, _, candidates in batch]
        import hashlib
        import torch
        if self.cache.tp_world_size == 1 and not any(
                torch.distributed.get_world_size(group=group)>1 for group in groups):
            return [set(candidates) for _, _, candidates in batch]

        # The drain count has already been reduced by the native scheduler. First
        # establish equal FIFO operation identities and mask sizes on every rank.
        headers = []
        for operation, hashes, candidates in batch:
            digest = hashlib.sha256()
            if operation is not None:
                digest.update(operation.handle.rid.encode())
            for name in hashes:
                digest.update(name.encode())
            signed = int.from_bytes(digest.digest()[:8], 'little') >> 1
            headers.extend((len(hashes), -len(hashes), signed, -signed, int(bool(candidates))))
        agreement = torch.tensor(headers, dtype=torch.int64, device='cpu')
        self.cache._all_reduce(agreement, torch.distributed.ReduceOp.MIN)
        verified = agreement.tolist()
        active = []
        for i in range(len(batch)):
            count, negative_count, digest, negative_digest, available = verified[5*i:5*i+5]
            active.append(available and count == -negative_count and digest == -negative_digest)
        if not any(active):
            return [set() for _ in batch]
        mask = [int(i in candidates) for (_, hashes, candidates), ok in zip(batch, active)
                if ok for i in range(len(hashes))]
        votes = torch.tensor(mask, dtype=torch.int32, device='cpu')
        self.cache._all_reduce(votes, torch.distributed.ReduceOp.MIN)
        accepted = iter(votes.tolist())
        return [{i for i in range(len(hashes)) if next(accepted)} if ok else set()
                for (_, hashes, _), ok in zip(batch, active)]

    def prepare(self, count):
        queue = self.controller.prefetch_hit_queue
        with queue.mutex:
            operations = list(islice(queue.queue, count))
        operations.extend([None] * (count-len(operations)))
        batch = []
        with self.backend.lock:
            for operation in operations:
                if operation is None:
                    batch.append((None, (), {}))
                    continue
                hashes = tuple(operation.hash_value or ())
                lease = self.backend.leases.get(id(operation))
                batch.append((operation, hashes, self.candidates(operation, hashes, lease)))
        shared = self.common_positions(batch)
        reserved = set()
        claimed = []
        with self.backend.lock:
            for (operation, hashes, candidates), positions in zip(batch, shared):
                sources = {hashes[i]: candidates[i] for i in sorted(positions) if hashes[i] not in reserved}
                if not sources:
                    continue
                reserved.update(sources)
                lease = self.backend.leases[id(operation)]
                if self.threaded:
                    for name, source in sources.items():
                        assert self.host.acquire_source(name) is source
                lease.claim = ReadClaim(sources)
                claimed.append(lease)
        return claimed

    def read_batch(self, native, operation, hashes, destinations, extra_info=None):
        with self.backend.lock:
            lease = self.backend.leases.get(id(operation))
            sources = lease.claim.sources if lease is not None and lease.claim is not None else {}
        if not sources:
            return native(operation, hashes, destinations, extra_info)

        completed = 0
        for staged, positions in groupby(range(len(hashes)), lambda i: hashes[i] in sources):
            start, *tail = positions
            end = tail[-1]+1 if tail else start+1
            if operation.is_terminated():
                break
            if staged:
                with self.backend.lock:
                    for i in range(start, end):
                        # Each native batch is a contiguous part of one hash chain.
                        lease.entries[lease.keys.index(hashes[i])] = True
                completed += end-start
                continue
            hints = extra_info
            if start and extra_info is not None and extra_info.prefix_keys is not None:
                hints = replace(extra_info, prefix_keys=extra_info.prefix_keys+list(hashes[:start]))
            received = native(operation, hashes[start:end], destinations[start*self.page:end*self.page], hints)
            completed += received
            if received != end-start:
                break
        return completed

    def substitute(self, operation, destination, hashes):
        lease = self.backend.leases.get(id(operation))
        if lease is None or lease.claim is None:
            return destination, ()
        sources = lease.claim.sources
        replacements = [(i*self.page, sources[name]) for i, name in enumerate(hashes) if name in sources]
        if not replacements:
            return destination, ()
        indices = destination.clone()
        for offset, source in replacements:
            indices[offset:offset+self.page] = source.payload
        return indices, replacements

    def commit(self, result, destination, replacements, hashes):
        if result.host_insert_dropped or not replacements:
            return
        installed = [(i, source) for i, source in replacements if i >= result.prefix_len]
        with self.backend.lock:
            for i, source in installed:
                name = hashes[i//self.page]
                assert result.inserted_host_node is not None
                assert self.host.stage[name] is source and source.read_pins >= 1
                assert not source.operation and not getattr(source, 'copy_committed', False)
                # Preserve the source for another in-flight simulated read until
                # that reader retires its legacy copy claim.
                shared = source.read_pins > 1
                self.host._remove_stage(source, free=False)
                self.host._finish_read_transfer(source, result.inserted_host_node)
                if shared:
                    pin = self.cache.inc_host_lock_ref(result.inserted_host_node)
                    source.read_pin_lock = (result.inserted_host_node, pin.to_dec_params())
                self.host._read_ownership['metadata_prefetch_pages'] += 1
            self.host._read_ownership['metadata_prefetch_operations'] += bool(installed)
        if installed:
            import torch
            retired = [destination[i:i+self.page] for i, _ in installed]
            self.controller.mem_pool_host.free(torch.cat(retired) if len(retired)>1 else retired[0])

    def install_simulated_route(self, operation, lease):
        from sglang_simulator.simulation.manager import ConfigManager

        platform = ConfigManager.get_platform_config()
        steps = []
        hashes = operation.hash_value[:len(operation.host_indices)//self.page]

        def resource(index):
            if hashes[index] in lease.claim.sources:
                return 'resident'
            return 'copy' if index in lease.entries else 'storage'

        for kind, positions in groupby(range(len(hashes)), resource):
            start, *tail = positions
            end = tail[-1]+1 if tail else start+1
            start, end = start*self.page, end*self.page
            if kind == 'storage':
                steps.append((start, end, 'storage_read', platform.disk_read_bandwidth, True))
            elif kind == 'copy':
                steps.append((start, end, 'host_read', platform.memory_read_bandwidth, False))
                steps.append((start, end, 'host_write', platform.memory_write_bandwidth, True))
            else:
                steps.append((start, end, 'staging_resident', None, True))

        route = metadata_route_type()(steps, self.page)
        # Make the first physical segment visible to read-pressure accounting
        # while it waits behind other operations, before it gets I/O credit.
        route.consume_resident()
        operation._sim_read_route = route
