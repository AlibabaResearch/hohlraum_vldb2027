"""Focused adapter checks without loading a model or modifying native SGLang."""
import ast
import threading
from collections import Counter
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

source = Path(__file__).resolve().parents[1]/'hohlraum/core/staging_storage.py'
if not source.exists():source=Path('/tmp/staging_storage_fixed.py')
names={'ReadLease','StagingStorage','_SGLangStorage'}
tree=ast.parse(source.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names],type_ignores=[]),str(source),'exec'))

class Page:
    def __init__(self,value):self.value=value
    def clone(self):return Page(self.value)
    def numel(self):return 1
    def element_size(self):return 1

class NativeReads(unittest.TestCase):
    def setUp(self):
        self.entry=NS(payload=[0],read_pins=0)
        self.host=NS(lock=threading.RLock(),stage={'a':self.entry})
        self.host.contains=lambda k:k in self.host.stage
        def acquire(key):
            entry=self.host.stage.get(key)
            if entry is not None:entry.read_pins+=1
            return entry
        def release(entry):entry.read_pins-=1
        self.host.acquire_source=acquire;self.host.release_source=release
        self.host.read_ownership_stats=lambda:{}
        self.writes={};self.pages={0:Page(7)};self.disk={}
        pool=NS(get_data_page=lambda i:self.pages[i],set_from_flat_data_page=lambda i,p:self.writes.update({i:p.value}))
        self.host.cache=NS(cache_controller=NS(storage_host_pool=pool))
        def disk_get(keys,indices,extra=None):
            result=[]
            for k,i in zip(keys,indices):
                result.append(k in self.disk)
                if k in self.disk:self.writes[i]=self.disk[k]
            return result
        backing=NS(batch_exists=lambda ks,extra:sum(k in self.disk for k in ks),batch_get_v1=disk_get)
        self.storage=_SGLangStorage(self.host,backing,1,1)
        self.op=NS(completed_tokens=0,request_id='r')
    def test_query_does_not_pin_or_change_residency(self):
        self.storage.begin_read(self.op,['a']);self.storage.begin_read(self.op,['a'])
        self.assertEqual(self.entry.read_pins,0)
        self.assertEqual(list(self.host.stage),['a'])
        self.storage.cancel_request('r');self.storage.finish_read(self.op)
        self.assertEqual(self.entry.read_pins,0);self.assertFalse(self.storage.leases)
    def test_evicted_query_hit_falls_back_to_ssd(self):
        self.storage.begin_read(self.op,['a']);self.host.stage.clear();self.disk['a']=9
        with self.storage.read_context(self.op):self.assertEqual(self.storage.batch_get_v1(['a'],[10]),[True])
        self.op.completed_tokens=1
        self.assertEqual(self.writes[10],9);self.assertEqual(self.storage.record_read(self.op),1)
    def test_missing_page_is_reported_not_fabricated(self):
        self.storage.begin_read(self.op,['a','b']);self.host.stage.clear();self.disk['b']=8
        with self.storage.read_context(self.op):result=self.storage.batch_get_v1(['a','b'],[10,11])
        self.assertEqual(result,[False,True]);self.assertNotIn(10,self.writes)
        # Native completion consumes only the continuous prefix before the miss.
        self.op.completed_tokens=0;self.assertEqual(self.storage.record_read(self.op),0)
    def test_clone_owns_memory_and_actual_source_is_counted(self):
        self.storage.begin_read(self.op,['a'])
        with self.storage.read_context(self.op):pages=self.storage._read_plan(['a'])
        with self.host.lock:self.host.stage.clear();self.pages[0].value=99
        self.assertEqual(pages[0].value,7)
        self.op.completed_tokens=1;self.assertEqual(self.storage.record_read(self.op),0)
    def test_reclamation_cannot_overlap_source_copy(self):
        entered=threading.Event();release=threading.Event();reclaimed=threading.Event()
        def clone():entered.set();self.assertTrue(release.wait(2));return Page(7)
        self.pages[0].clone=clone
        reader=threading.Thread(target=lambda:self.storage._staged_page('a'))
        def reclaim():
            with self.host.lock:self.host.stage.clear();reclaimed.set()
        reader.start();self.assertTrue(entered.wait(2));writer=threading.Thread(target=reclaim);writer.start()
        self.assertFalse(reclaimed.wait(.05));release.set();reader.join(2);writer.join(2)
        self.assertTrue(reclaimed.is_set())

if __name__=='__main__':unittest.main()
