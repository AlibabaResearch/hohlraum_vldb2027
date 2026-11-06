"""Fast staging races and native prefix/ACK contract checks; no model or GPU.

Uses the actual adapter classes and native controller methods via AST extraction.
Fake pages replace tensors/files only; no timing/performance claims are made.
"""
import ast
import logging
import queue
import random
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0,str(Path(__file__).parent))
from check_staging_native_reads import NativeReads, Page

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'third_party/sglang/python/sglang/srt/managers/cache_controller.py'

def load_native():
    tree=ast.parse(NATIVE.read_text())
    methods={}
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name in {'_page_get_zero_copy','_page_transfer'}:
            node.decorator_list=[]
            methods[node.name]=node
    ns=dict(PrefetchOperation=object,STORAGE_BATCH_SIZE=2,
            HiCacheStorageExtraInfo=NS,PrefetchAck=NS,
            logger=logging.getLogger('staging-test'))
    ns['logger'].addHandler(logging.NullHandler());ns['logger'].propagate=False
    exec(compile(ast.Module(body=list(methods.values()),type_ignores=[]),str(NATIVE),'exec'),ns)
    return ns

class Interleavings(NativeReads):
    def replica(self):
        r=NativeReads();r.setUp();return r
    def test_delayed_rank_query_does_not_change_eviction_eligibility(self):
        left,right=self.replica(),self.replica()
        left.storage.begin_read(left.op,['a'])
        self.assertEqual(left.entry.read_pins,right.entry.read_pins)
        self.assertEqual(left.entry.read_pins,0)
        right.storage.begin_read(right.op,['a'])
        left.storage.finish_read(left.op)
        self.assertEqual(left.entry.read_pins,right.entry.read_pins)
    def test_overlapping_reads_and_cancel_are_independent(self):
        first=NS(completed_tokens=1,request_id='r1');second=NS(completed_tokens=1,request_id='r2')
        self.storage.begin_read(first,['a']);self.storage.begin_read(second,['a'])
        with self.storage.read_context(first):self.storage.batch_get_v1(['a'],[10])
        self.storage.cancel_request('r1')
        self.host.stage.clear();self.disk['a']=7
        with self.storage.read_context(second):self.storage.batch_get_v1(['a'],[11])
        self.assertEqual(self.storage.record_read(second),1)
        self.storage.cancel_request('r2');self.assertFalse(self.storage.leases)
        self.assertEqual(self.entry.read_pins,0)
    def test_cancel_during_copy_does_not_release_source_early(self):
        self.storage.begin_read(self.op,['a'])
        with self.storage.read_context(self.op):
            self.storage.cancel_request('r')
            self.assertEqual(self.storage.batch_get_v1(['a'],[10]),[True])
        self.assertEqual(self.writes[10],7);self.assertFalse(self.storage.leases)
    def test_500_reordered_query_read_evict_cancel_schedules(self):
        for seed in range(500):
            rng=random.Random(seed);r=self.replica()
            ops=[NS(completed_tokens=0,request_id=str(i)) for i in range(4)]
            for op in ops:r.storage.begin_read(op,['a'])
            events=[('read',op) for op in ops]+[('cancel',op) for op in ops]+[('evict',None)]
            rng.shuffle(events)
            for event,op in events:
                if event=='evict':
                    with r.host.lock:
                        r.host.stage.clear()
                        if seed%2:r.disk['a']=7
                        r.pages[0].value=99
                elif event=='cancel':r.storage.cancel_request(op.request_id)
                else:
                    with r.storage.read_context(op):result=r.storage.batch_get_v1(['a'],[10])
                    if result[0]:self.assertEqual(r.writes[10],7,(seed,events))
                self.assertEqual(r.entry.read_pins,0)
            for op in ops:r.storage.finish_read(op)
            self.assertFalse(r.storage.leases)
    def test_native_partial_miss_preserves_ack_count_across_ranks(self):
        native=load_native();acks=[]
        for fail in [False,True]:
            r=self.replica();keys=['a','b','c','d','e','f','g','h']
            r.disk.update({k:7 for k in keys})
            if fail:del r.disk['b']
            op=NS(hash_value=keys,prefix_keys=[],host_indices=list(range(10,18)),
                  pool_transfers=None,request_id='r',completed_tokens=0,is_terminated=lambda:False)
            r.storage.begin_read(op,keys)
            controller=NS(page_size=1,storage_backend=r.storage,prefetch_sync_queue=queue.Queue())
            def page_batch(operation,hashes,indices,extra,sidecars):
                with r.storage.read_context(operation):
                    return native['_page_get_zero_copy'](controller,operation,hashes,indices,extra)
            controller._page_transfer_kv_batch=page_batch
            count=native['_page_transfer'](controller,op)
            progress=[a.completed_tokens for a in list(controller.prefetch_sync_queue.queue)]
            self.assertEqual(count,1 if fail else 8)
            self.assertEqual(progress,[1,1,1,1] if fail else [2,4,6,8])
            acks.append(progress)
        self.assertEqual(len(acks[0]),len(acks[1]))
        self.assertEqual([min(a,b) for a,b in zip(*acks)],[1,1,1,1])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Interleavings)
    result=unittest.TextTestRunner().run(suite)
    sys.exit(not result.wasSuccessful())
