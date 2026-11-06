"""Deterministic two-rank capacity control races; no GPU/model required."""
import copy,json,sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hohlraum.core.capacity_controller import CapacityController,CapacityConsensus

class Transport:
    def __init__(self):self.barrier=threading.Barrier(2,timeout=3);self.slots=[None,None]
    def exchange(self,rank,value,reduce=False):
        self.slots[rank]=value;self.barrier.wait()
        result=[max(v[i] for v in self.slots) for i in range(2)] if reduce else copy.deepcopy(self.slots[0])
        self.barrier.wait();return result

class ConsensusTests(unittest.TestCase):
    def setUp(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'hohlraum/experiments/protocol.json').read_text())['capacity_control']
        self.transport=Transport();self.hosts=[];self.controls=[];self.syncs=[]
        for rank in range(2):
            h=NS(cfg={'capacity_control':cfg},cache=NS(page_size=1,tp_world_size=1),total_tokens=300,staging_limit=30,read_limit=270,stage={},kv_bytes_per_token=10**9,stats_path=None,t=rank*.1,exhausted=False)
            h.now=lambda h=h:h.t;h.read_allocated=lambda:0;h.waiting_staging_tokens=lambda:0
            h.ssd_backend=NS(budget_exhausted=lambda reserve,h=h:h.exhausted)
            h._budget_exhausted=lambda h=h:getattr(h,'_synced_budget_exhausted',h.exhausted)
            h._record_budget_progress=lambda **kw:None
            def resize(target,h=h):
                if target==0 and h._budget_exhausted() and h._capacity_zero_blocked:return False
                h.staging_limit=target;h.read_limit=300-target;return True
            h.resize_staging=resize
            c=CapacityController(h);c._observe_residents=lambda now:None
            sync=CapacityConsensus(c,lambda value,r=rank:self.transport.exchange(r,value),lambda a,b,r=rank:self.transport.exchange(r,[a,b],True))
            self.hosts.append(h);self.controls.append(c);self.syncs.append(sync)
    def advance(self,left,right):
        errors=[]
        for h,t in zip(self.hosts,[left,right]):h.t=t
        def run(s):
            try:s.advance()
            except BaseException as e:errors.append(e)
        threads=[threading.Thread(target=run,args=(s,),daemon=True) for s in self.syncs]
        for t in threads:t.start()
        for t in threads:t.join(4)
        self.assertFalse(any(t.is_alive() for t in threads),'collective order deadlock')
        if errors:raise errors[0]
        self.assertEqual(self.hosts[0].staging_limit,self.hosts[1].staging_limit)
    def test_follower_crossing_window_early_does_not_resize(self):
        self.advance(299.9,301)
        self.assertEqual([len(c.history) for c in self.controls],[0,0])
        self.advance(300.01,299.95)
        self.assertEqual([len(c.history) for c in self.controls],[1,1])
    def test_expansion_uses_same_target_despite_samples_and_clock_skew(self):
        for c,ratio in zip(self.controls,[.02,.5]):c.accepted.extend([(0,100,ratio*100),(0,float('inf'),(1-ratio)*100)])
        for i in range(1,5):self.advance(300*i+.01,300*i-.1)
        self.assertEqual(self.hosts[0].staging_limit,40)
        self.assertEqual(self.controls[0].next_tick,self.controls[1].next_tick)
        self.assertEqual(self.controls[0].expand_after,self.controls[1].expand_after)
        self.assertEqual([len(c.history) for c in self.controls],[0,0])
    def test_contraction_uses_leader_tier_for_both_ranks(self):
        for c,ratio in zip(self.controls,[.06,0]):c.slack.extend([(0,1000,ratio*100),(0,0,(1-ratio)*100)])
        for i in range(1,3):
            for c in self.controls:c.failed(str(i),1,'host_capacity')
            self.advance(300*i+.01,300*i-.1)
        self.assertEqual(self.hosts[0].staging_limit,15)
        self.assertEqual(self.controls[0].last_shrink,self.controls[1].last_shrink)
    def test_one_rank_budget_exhaustion_zeroes_both(self):
        self.hosts[1].exhausted=True;self.advance(10,9.9)
        self.assertEqual([h.staging_limit for h in self.hosts],[0,0])
        self.assertTrue(all(h._budget_exhausted() for h in self.hosts))
    def test_zero_waits_until_all_ranks_can_release(self):
        self.hosts[0].exhausted=True
        self.hosts[1].stage['p']=NS(read_pins=1,copy_committed=False)
        self.advance(300.01,299.9)
        self.assertEqual(self.hosts[0].staging_limit,30)
        self.hosts[1].stage.clear();self.advance(301,300.9)
        self.assertEqual(self.hosts[0].staging_limit,0)
    def test_200_rounds_of_clock_skew(self):
        for i in range(200):self.advance(i*10,i*10+(-.2 if i%2 else .2))

if __name__=='__main__':unittest.main()
