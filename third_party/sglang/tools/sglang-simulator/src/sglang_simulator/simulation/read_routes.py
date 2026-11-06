# Source-routing adapter, not a transfer estimator or event scheduler.
# A backend returns (start_token, end_token, source) ranges. Host copies submit
# a read followed by a write to the existing byte/bandwidth estimator.
# No new overhead, efficiency, overlap, clock update or operation reordering.
from dataclasses import dataclass, field


@dataclass
class ReadRoute:
    steps: list
    page_size: int
    index: int = 0
    progress: float = 0.0
    completed: int = 0
    bytes_by_resource: dict = field(default_factory=dict)

    def advance(self, duration, token_bytes, estimate):
        left = max(0.0, duration)
        while self.index < len(self.steps) and left > 0:
            start, end, resource, bandwidth, publishes = self.steps[self.index]
            remaining = end-start-self.progress
            amount, used = estimate(remaining, token_bytes, left, bandwidth)
            amount = min(remaining, max(0.0, amount))
            used = min(left, max(0.0, used))
            self.progress += amount
            self.bytes_by_resource[resource] = self.bytes_by_resource.get(resource, 0.0) + amount*token_bytes
            left -= used
            if publishes:
                self.completed = start + int(self.progress)//self.page_size*self.page_size
            if self.progress >= end-start:
                if publishes:
                    self.completed = end
                self.index += 1
                self.progress = 0.0
            else:
                break
        return self.completed, duration-left


def make_route(ranges, allocated, page_size, disk_bandwidth, platform):
    steps = []
    previous = 0
    for start, end, source in ranges:
        if start != previous or end <= start or start % page_size or end % page_size:
            raise ValueError('read-source ranges must be contiguous complete pages')
        previous = end
        if source not in ('storage', 'host_copy'):
            raise ValueError('unsupported read source: '+str(source))
        if start >= allocated:
            continue
        end = min(end, allocated)
        if source == 'storage':
            steps.append((start, end, 'storage_read', disk_bandwidth, True))
        else:
            steps.append((start, end, 'host_read', platform.memory_read_bandwidth, False))
            steps.append((start, end, 'host_write', platform.memory_write_bandwidth, True))
    if previous < allocated or allocated % page_size:
        raise ValueError('source description does not cover the receive buffer')
    if any(step[3] <= 0 for step in steps):
        raise ValueError('transfer bandwidth must be positive')
    return ReadRoute(steps, page_size)


def advance_read(controller, operation, tokens, duration, token_bytes,
                 disk_bandwidth, estimate, platform):
    describe = getattr(controller.storage_backend, 'read_source_ranges', None)
    if describe is None:
        return estimate(tokens, token_bytes, duration, disk_bandwidth)
    state = getattr(operation, '_sim_read_route', None)
    if state is None:
        ranges = describe(operation)
        if all(source == 'storage' for _, _, source in ranges):
            return estimate(tokens, token_bytes, duration, disk_bandwidth)
        state = make_route(ranges, len(operation.host_indices), controller.page_size,
                           disk_bandwidth, platform)
        operation._sim_read_route = state
    completed, spent = state.advance(duration, token_bytes, estimate)
    return min(tokens, max(0, completed-operation.completed_tokens)), spent
