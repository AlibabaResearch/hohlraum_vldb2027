import sys,os,importlib.util
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hohlraum.core.capacity_controller import native_capacity_consensus
from check_capacity_consensus import ConsensusTests

def worker(rank,initfile):
    dist.init_process_group('gloo',init_method='file://'+initfile,rank=rank,world_size=2)
    t=ConsensusTests();t.setUp();h=t.hosts[rank];c=t.controls[rank]
    h.cache.tp_group=dist.group.WORLD
    s=native_capacity_consensus(c)
    ratio=.02 if rank==0 else .5
    c.accepted.extend([(0,100,ratio*100),(0,float('inf'),(1-ratio)*100)])
    for i in range(1,5):
        h.t=300*i+(.01 if rank==0 else -.1);s.advance()
    assert h.staging_limit==40,(rank,h.staging_limit)
    for i in range(5,7):
        c.failed(str(i),1,'host_capacity')
        c.slack.extend([(h.t,1000,6),(h.t,0,94)])
        h.t=300*i+(.01 if rank==0 else -.1);s.advance()
    assert h.staging_limit==25,(rank,h.staging_limit)
    h.exhausted=rank==1;h.t+=1;s.advance()
    assert h.staging_limit==0,(rank,h.staging_limit)
    for i in range(50):h.t+=1;s.advance()
    print('PASS rank',rank,'expansion 40, contraction 25, budget zero, 50 rounds',flush=True)
    dist.destroy_process_group()
if __name__=='__main__':
    import tempfile
    initfile=tempfile.mktemp(prefix='capacity-gloo-')
    mp.spawn(worker,args=(initfile,),nprocs=2,join=True)
