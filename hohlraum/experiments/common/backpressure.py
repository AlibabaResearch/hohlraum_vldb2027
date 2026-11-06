"""Experiment-owned FIFO admission; no simulator clocks or cache hooks."""
from collections import deque

class RequestAdmission:
    def __init__(self, limit=8):
        if limit < 0:
            raise ValueError('request admission limit must be >= 0 (0 disables it)')
        self.limit = limit
        self.pending = deque()
        self.inflight = set()
        self.records = {}

    def offer(self, req, ready_at):
        if req.rid in self.records:
            raise ValueError('duplicate request admission')
        self.records[req.rid] = dict(ready_at=ready_at)
        self.pending.append(req)

    def can_release(self):
        return bool(self.pending) and (not self.limit or len(self.inflight) < self.limit)

    def release(self, now):
        result = []
        while self.can_release():
            req = self.pending.popleft()
            rec = self.records[req.rid]
            assert now >= rec['ready_at']
            rec.update(engine_admitted_at=now, external_wait_s=now-rec['ready_at'])
            self.inflight.add(req.rid)
            result.append(req)
        return result

    def finish(self, rid):
        self.inflight.remove(rid)

    def export_stats(self, stats):
        # Export end-to-end TTFT without changing native timing/statistics.
        for row in stats:
            rec = self.records[row['rid']]
            row.update(rec)
            row['engine_created_time'] = row['created_time']
            row['engine_ttft_s'] = row['gen_token_latencies'][0]
            row['gen_token_latencies'] = list(row['gen_token_latencies'])
            row['gen_token_latencies'][0] += rec['external_wait_s']
            row['created_time'] = rec['ready_at']
        return stats


class QueueDelayAdmission(RequestAdmission):
    def __init__(self, feedback, threshold=10.0, waiting_limit=8):
        super().__init__(0)
        self.feedback=feedback
        self.threshold=threshold
        self.waiting_limit=waiting_limit
        self.submitted_not_received=set()
    def can_release(self):
        age, queued=self.feedback()
        return bool(self.pending) and age<=self.threshold and queued+len(self.submitted_not_received)<self.waiting_limit
    def release(self,now):
        result=[]
        while self.can_release():
            req=self.pending.popleft();rec=self.records[req.rid]
            assert now>=rec['ready_at']
            rec.update(engine_admitted_at=now,external_wait_s=now-rec['ready_at'])
            self.inflight.add(req.rid);self.submitted_not_received.add(req.rid)
            result.append(req)
        return result
    def received(self,rid):
        self.submitted_not_received.discard(rid)
    def finish(self,rid):
        super().finish(rid);self.submitted_not_received.discard(rid)

def queue_feedback(scheduler,gate,now):
    if scheduler is None:raise RuntimeError('Scheduler feedback unavailable')
    waiting=[req for req in scheduler.waiting_queue if not req.finished()]
    ages=[]
    for req in waiting:
        rec=gate.records.get(req.rid)
        if rec is None or 'engine_admitted_at' not in rec:raise RuntimeError('Missing waiting request arrival')
        ages.append(max(0.0,now-rec['engine_admitted_at']))
    return max(ages,default=0.0),len(waiting)
