"""SLI range and HTS correctness checks; synthetic inputs, no CUDA or private data."""
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hohlraum.core.sli import StagingRangeIndex


def check_endpoints():
    rng = random.Random(194)
    index = StagingRangeIndex()
    trees = {}
    paths = {}
    serial = 10
    known = set()
    resident = set()
    for step in range(1500):
        session = rng.randrange(6)
        path = paths.get(session, [session % 2])
        path = path[:rng.randrange(1, len(path) + 1)]
        added = rng.randrange(1, 7)
        path = path + list(range(serial, serial + added))
        serial += added
        paths[session] = path
        known.update(path)
        index.observe(session, path)
        tree = trees.setdefault(session, {None: set()})
        parent = None
        for key in path:
            tree.setdefault(parent, set()).add(key)
            tree.setdefault(key, set())
            parent = key
        for key in rng.sample(sorted(known), min(8, len(known))):
            live = rng.choice([True, False])
            index.set_resident(key, live)
            if live:
                resident.add(key)
            else:
                resident.discard(key)
        active = set(rng.sample(sorted(resident), min(60, len(resident))))
        expected = set()
        for tree in trees.values():
            todo = [(None, [])]
            while todo:
                node, candidates = todo.pop()
                if node in active:
                    candidates = candidates + [node]
                if tree[node]:
                    todo.extend((child, candidates) for child in tree[node])
                elif candidates:
                    expected.update([candidates[0], candidates[-1]])
        assert index.endpoints(active) == expected, step
    print('PASS: 1500 compressed-branch endpoint transitions against path oracle')


def check_history_ranges():
    source = r'''
#include "session_level_index.hpp"
#include <cassert>
#include <random>
using namespace hohlraum;
int main() {
    SessionLevelIndex index;
    std::map<uint32_t, SessionLevelIndex::State> expected;
    std::vector<uint32_t> previous;
    std::mt19937 rng(41);
    for(unsigned step=1; step<3000; ++step) {
        std::vector<uint32_t> keys;
        for(unsigned k=0;k<100;++k)if(rng()%3==0)keys.push_back(k);
        auto next=SessionLevelIndex::normalize(index.locate(0,keys));
        auto& table=index.tables[0];
        auto removed=SessionLevelIndex::subtract(table.current,next);
        for(auto span:removed)table.mark_old(span,step,step);
        for(auto key:previous)if(std::find(keys.begin(),keys.end(),key)==keys.end()) {
            expected[key].old_since=step;expected[key].old_gap=step;
        }
        SessionLevelIndex::State state{0,double(step),-1,step,-1};
        for(auto span:next)table.assign(span,state);
        for(auto key:keys)expected[key]=state;
        table.current=next;previous=keys;
        for(auto pair:expected)assert(index.owners(pair.first).at(0)==pair.second);
    }
    SessionLevelIndex large;
    std::vector<uint32_t> keys(10000);for(unsigned k=0;k<keys.size();++k)keys[k]=k;
    auto spans=SessionLevelIndex::normalize(large.locate(0,keys));
    for(auto span:spans)large.tables[0].assign(span,{0,1,-1,1,-1});
    assert(large.range_count()==1);
}
'''
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp)
        (p/'test.cpp').write_text(source)
        subprocess.run(['c++', '-O2', '-std=c++17',
                        '-I'+str(ROOT/'hohlraum/core/cpp'),
                        str(p/'test.cpp'), '-o', str(p/'test')], check=True)
        subprocess.run([str(p/'test')], check=True)
    print('PASS: 2999 history transitions against per-block oracle; 10000 blocks share one range')


def check_scoring_identity():
    """Exercise real Python ingestion/query mapping with a recording ABI."""
    from types import SimpleNamespace as NS
    from hohlraum.core.sli import SLI
    sli = SLI.__new__(SLI)
    sli.ids = {}; sli.sessions = {}; sli.started = set(); sli.observed = set()
    sli.client_ready_ids = set(); sli.observed_requests = 0
    sli.now = 0; sli.cache = {}; sli.ptr = None
    sli.cfg = {'pending_read_admission': False}
    sli.check = lambda rc: None
    current = {}; observed = []; queried = []
    def observe(ptr, session, fold, length, now, ids, count):
        blocks = [ids[i] for i in range(count)]
        current[session] = set(blocks)
        observed.append(blocks)
        return 0
    def query(ptr, now, ids, count, out):
        queried.extend(ids[i] for i in range(count))
        for i in range(count):
            # Synthetic active-owner score isolates identity from fitted models.
            out[5*i] = float(any(ids[i] in blocks for blocks in current.values()))
        return 0
    sli.lib = NS(hh_clock=lambda *args: 0, hh_observe=observe, hh_query=query)
    def request(rid, root):
        return NS(rid=rid, input_ids=[1]*1024,
                  sampling_params=NS(custom_params={'hohlraum':
                      {'root':root, 'chain':0, 'reference_fold':0, 'source_blocks':[17]}}))
    sli.arrive(request('a0', 1), 0, ['A-old'])
    sli.arrive(request('b0', 2), 1, ['B-old'])
    assert sli.ids['A-old'] != sli.ids['B-old']
    req=request('a1',1)
    sli.client_ready(req,2,['A-new'])  # Compression is known before admission.
    assert req.rid not in sli.started
    assert sli.score('A-old',2)==0 and sli.score('A-new',2)==1
    count=len(observed)
    sli.client_ready(req,3,['A-new'])
    sli.arrive(req,32,['A-new'])
    assert len(observed)==count  # No extra QG/history update at engine ingress.
    assert sli.score('A-old', 2) == 0
    assert sli.score('B-old', 2) == 1
    # Same physical hash still shares identity despite different source labels.
    sli.arrive(request('b1', 2), 3, ['A-new'])
    assert observed[-1] == observed[-2]
    sli.arrive(request('a2', 1), 4, ['A-new'])
    assert observed[-1] == observed[-2]
    assert all(key in sli.ids.values() for key in queried)
    print('PASS: client-ready compression updates only actual KV; engine ingress does not double-observe')
    print('PASS: isolated replicas cannot refresh each other; genuine sharing and query identity preserved')


def check_hash_reuse():
    from types import SimpleNamespace as NS
    from collections import Counter
    from hohlraum.core.sli import SLI
    sli=SLI.__new__(SLI); sli.ready_hashes={}; sli.request_pending={}; sli.pending=Counter()
    calls=[]
    def compute(req, page):
        calls.append(req.rid)
        return ['recomputed']
    sli.request_hashes=compute
    def request(): return NS(sampling_params=NS(custom_params={'hohlraum':{}}), rid='a', input_ids=[1,2,3], extra_key='branch', cache_salt='session-A')
    req=request();sli.remember_request_hashes(req,1024,['original'])
    # Engine may construct a new request object with equal input.
    assert sli.hashes_for_request(request(),1024)==['original'] and not calls
    assert not sli.ready_hashes
    for field,value in [('input_ids',[1,2,4]),('extra_key','other'),('cache_salt','session-B')]:
        req=request();sli.remember_request_hashes(req,1024,['original']);setattr(req,field,value)
        assert sli.hashes_for_request(req,1024)==['recomputed']
    req=request();sli.remember_request_hashes(req,1024,['original']);req.input_ids[0]=9
    assert sli.hashes_for_request(req,1024)==['recomputed']
    req=request();sli.remember_request_hashes(req,1024,['original'])
    assert sli.hashes_for_request(req,2048)==['recomputed']
    sli.remember_request_hashes(req,1024,['original']);sli.release_read(req.rid)
    assert not sli.ready_hashes
    assert sli.hashes_for_request(req,1024)==['recomputed']
    assert len(calls)==6
    print('PASS: exact-input hash reuse; token/namespace/page changes recompute; cancellation releases snapshot')


def check_input_prefix_identity():
    from types import SimpleNamespace as NS
    from unittest.mock import patch
    import hashlib
    from hohlraum.core.sli import SLI
    from hohlraum.experiments.common.workload import encode
    sli=SLI.__new__(SLI)
    sli.host=NS(cache=NS(page_size=1024));sli.prefix_ids={};sli.ids={}
    sli.native_prefixes={};sli.sessions={};sli.observed=set();sli.observed_requests=0
    sli.client_ready_ids=set();sli.now=0.;sli.cache={};sli.ptr=None
    observed=[]
    sli.lib=NS(hh_clock=lambda *a:0,hh_observe=lambda *a:observed.append(list(a[-2])) or 0)
    sli.check=lambda rc:None
    def req(root, blocks, rid):
        return NS(rid=rid,input_ids=encode(root,blocks,len(blocks)*1024,0),
                  sampling_params=NS(custom_params={'hohlraum':dict(root=root,chain=0,
                  reference_fold=0,prefix_blocks=blocks,prefix_page_size=1024)}))
    a=req(1,[10,20],'a');b=req(2,[10,20],'b');compressed=req(1,[10,30],'c')
    sli.client_ready(a,0);sli.client_ready(b,0);sli.client_ready(compressed,1)
    assert observed[0][0]==observed[2][0]
    assert not set(observed[0]) & set(observed[1])
    assert observed[0][1]!=observed[2][1]
    calls=[]
    class Key(list):
        def __init__(self,tokens,*args,**kw):super().__init__(tokens)
    def hashing(tokens,prior,page_size):
        calls.append(len(tokens));out=[]
        for i in range(0,len(tokens),page_size):
            prior=hashlib.sha256((prior or '').encode()+repr(list(tokens[i:i+page_size])).encode()).hexdigest()
            out.append(prior)
        return out
    modules={'sglang.srt.mem_cache.radix_cache':NS(RadixKey=Key),
             'sglang.srt.mem_cache.utils':NS(get_storage_hash_str=hashing)}
    with patch.dict(sys.modules,modules):
        native=sli.hashes_for_request(a,1024)
        assert len(calls)==2
        assert sli.hashes_for_request(a,1024)==native and len(calls)==2
        changed=sli.hashes_for_request(compressed,1024)
        assert changed[0]==native[0] and len(calls)==3
        assert native==hashing(a.input_ids,None,1024)
        other=sli.hashes_for_request(b,1024)
        sli.bind_input_prefixes(a,native);sli.bind_input_prefixes(b,other)
        assert sli.ids[native[0]]!=sli.ids[other[0]]
        try:sli.bind_input_prefixes(b,native)
        except ValueError:pass
        else:raise AssertionError('cross-session alias accepted')
        try:sli.hashes_for_request(req(1,[30],'invalid-parent'),1024)
        except ValueError:pass
        else:raise AssertionError('prefix reused under changed parent')
    assert not sli.native_prefixes.keys() & sli.ids.keys()
    print('PASS: input identity isolation, retained prefix after compaction, native mapping, incremental conversion and alias rejection')


if __name__ == '__main__':
    check_input_prefix_identity()
    check_hash_reuse()
    check_scoring_identity()
    check_history_ranges()
    check_endpoints()
