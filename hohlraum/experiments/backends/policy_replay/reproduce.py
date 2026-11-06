"""Recompute paper Section 6.4 from frozen inputs; never substitute reference results."""
import argparse, concurrent.futures, datetime, hashlib, json, math, os
from pathlib import Path
import subprocess, sys, time, csv
ROOT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(p.read_text())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/paper')
    p.add_argument('--output',type=Path)
    p.add_argument('--jobs',type=int,default=2,help='Concurrent replay processes (default 2; memory intensive)')
    p.add_argument('--resume',action='store_true',help='Resume only an identical source/input/config run')
    p.add_argument('--check-only',action='store_true',help='Verify inputs and run the smoke check; no paper sweep')
    p.add_argument('--reference',type=Path,help='Optional external historical counter JSON; not needed to run experiments')
    p.add_argument('--no-plots',action='store_true',help='Metrics only; default also produces figures')
    a=p.parse_args()
    if a.jobs<1:p.error('--jobs must be positive')
    a.data=a.data.resolve(); config=read(ROOT/'paper/config.json'); inputs=read(ROOT/'paper/input_manifest.json')
    for name,h in inputs['files'].items():
        f=a.data/name
        if not f.is_file():p.error(f'Missing paper input {f}. Supply the policy inputs through --data.')
        if sha(f)!=h:p.error(f'Paper input hash mismatch: {f}')
    print('PASS input hashes:',len(inputs['files']),flush=True)
    if not a.no_plots:
        try:import matplotlib,numpy
        except ImportError:p.error('Install plotting dependencies: python3 -m pip install -r '+str(ROOT/'requirements.txt'))
    subprocess.run([sys.executable,str(ROOT.parents[3]/'tests/smoke.py')],check=True)
    if a.check_only:return
    out=(a.output or ROOT/'runs'/('paper-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    source={str(f.relative_to(ROOT)):sha(f) for folder in ('src','include','paper') for f in (ROOT/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    for n in ('Makefile','reproduce.py','run.py','requirements.txt'):source[n]=sha(ROOT/n)
    source['../../../../tests/smoke.py']=sha(ROOT.parents[3]/'tests/smoke.py')
    identity=dict(config=config,input_hashes=inputs['files'],source_hashes=source,binaries={n:sha(ROOT/'build'/n) for n in ('policy_runner','baseline_runner','feedback_runner')})
    if out.exists():
        if not a.resume or not (out/'manifest.json').is_file():p.error('Output exists; choose a new output or use --resume')
        if read(out/'manifest.json')['identity']!=identity:p.error('Cannot resume: sources, binaries, inputs or configuration changed')
    else:
        out.mkdir(parents=True);dump(out/'manifest.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],data=str(a.data),timing='none'))
    print('OUTPUT',out,flush=True)
    dump(out/'status.json',dict(state='running',phase='calibration'))
    envbase={k:v for k,v in os.environ.items() if not k.startswith('HOHLRAUM_')}
    def env(c,signal='full'):
        m=a.data/'models'/c['workload'];e=envbase.copy()
        e.update(HOHLRAUM_MODEL_COUNT='4' if c['workload']=='agentx' else '6',HOHLRAUM_JOINT_MODELS=str(m),HOHLRAUM_LEGACY_MODELS=str(m),HOHLRAUM_BRANCH_PRIOR=str(m/'branch-prior.txt'),HOHLRAUM_LENGTH_CURVES=str(m/'length_curves.txt'),HOHLRAUM_ONLINE_PRIOR='20',HOHLRAUM_SIGNAL_ABLATION=signal)
        return e
    def execute(path,cmd,e,meta):
        if path.exists():
            r=read(path)
            if r.get('complete') and r.get('command')==cmd:return r
            raise RuntimeError(f'Incomplete or conflicting cached output: {path}')
        path.parent.mkdir(parents=True,exist_ok=True);t=time.monotonic()
        with path.with_suffix('.log').open('w') as log:r=subprocess.run(cmd,env=e,text=True,stdout=subprocess.PIPE,stderr=log)
        path.with_suffix('.stdout').write_text(r.stdout)
        r.check_returncode();row=json.loads(r.stdout)
        if not 0<=row['global_hits']<=row.get('theory_hits',row['refs'])<=row['refs']:raise RuntimeError('invalid hit counts')
        row.update(meta,command=cmd,complete=True,elapsed_s=time.monotonic()-t)
        temp=path.with_suffix('.tmp');dump(temp,row);temp.replace(path);return row
    def run(c,total,mode='lru_nocopy',threshold=.7,ratio=0,signal='full',family='policy',level=None):
        s,w=c['seed'],c['workload'];d=a.data/f'streams/seed{s}/{w}';m=a.data/'models'/w;stage=round(total*ratio)
        directory=out/f'policy/seed{s}/{w}'/(f'coverage{level}' if level else 'calibration')
        name=f'{family}-{mode.replace(":","-")}-t{threshold:.2f}-r{ratio:g}-{signal}-c{total}'
        if family=='baseline':cmd=[str(ROOT/'build/baseline_runner'),str(d/'baseline.tsv'),mode,str(total)]
        else:cmd=[str(ROOT/'build/policy_runner'),mode,str(threshold),str(d/'events.tsv'),str(c['blocks']),str(c['sessions']),str(total-stage),str(stage),str(m),str(m),str(c['end_seconds'])]
        row=execute(directory/(name+'.json'),cmd,env(c,signal),dict(seed=s,workload=w,total_blocks=total,staging_blocks=stage,staging_fraction=ratio,variant=signal,family=family))
        if family=='policy':
            assert row['requests']==c['requests'] and row['global_hits']==row['dram_hits']+row['ssd_reads']
            assert row['fallbacks']==row['ssd_evictions']==0
        return row
    def calibrate(c):
        levels={}
        for level,cap in c['capacities'].items():
            # Published C is a starting bracket, not a cached measured result.
            # Independently verify C and C-1 on the actual engine/input.
            low=run(c,cap-1); high=run(c,cap);target=int(level)/100
            if not low['dram_hits']/low['theory_hits']<target<=high['dram_hits']/high['theory_hits']:
                raise RuntimeError(f'Published capacity boundary no longer holds: {c["seed"]}/{c["workload"]}/c{level}; see calibration results. Do not silently change paper settings.')
            levels[level]=dict(capacity_blocks=cap,coverage=high['dram_hits']/high['theory_hits'],previous_capacity_coverage=low['dram_hits']/low['theory_hits'])
        dump(out/f'policy/seed{c["seed"]}/{c["workload"]}/capacities.json',levels)
        print('CALIBRATED',c['seed'],c['workload'],flush=True)
    def budget(c,t,mode,dwpd):
        cap=c['capacities']['80'];ssd=round(12.8*cap);period=c['end_seconds'];limit=math.floor(ssd*period/86400)*dwpd
        folder=out/f'budget/seed{c["seed"]}/dwpd{dwpd}-{mode}-{t:.1f}';folder.mkdir(parents=True,exist_ok=True)
        cmd=[str(ROOT/'build/feedback_runner'),str(folder),str(t),str(a.data/f'streams/seed{c["seed"]}/playing/events.tsv'),str(c['blocks']),str(c['sessions']),mode,'playing','c80',str(period),str(limit),str(cap),str(round(.5*cap)),str(ssd),str(period)]
        row=execute(folder/'result.json',cmd,env(c),dict(seed=c['seed'],initial_threshold=t,mode=mode,dwpd=dwpd,budget_blocks=limit,ssd_blocks=ssd,budget_period_seconds=period))
        assert row['requests']==c['requests'] and row['writes']<=limit and row['global_hits']==row['dram_hits']+row['ssd_reads']
        for line in (folder/'trajectory.jsonl').read_text().splitlines():
            x=json.loads(line);assert 0<=x['threshold']<=t+1e-12
            if mode=='fixed':assert x['threshold']==t
        return row
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:list(pool.map(calibrate,config['cases']))
        jobs=[]
        for c in config['cases']:
            for level,cap in c['capacities'].items():
                configurations=[('policy','full_nocopy',.7,r,'full') for r in config['staging_fractions'] if r]+[('policy','lru_nocopy',.7,0,'full'),('baseline','all',0,0,'full')]
                configurations += [('baseline',f'larc:{max(1,round(cap*r))}',0,0,'full') for r in config['larc_ratios']]
                if level=='80':
                    configurations += [('policy',m,t,r,'full') for t in config['thresholds'] for m,r in [('full_nocopy',.5),('lru_nocopy',0)]]
                    configurations += [('baseline',f'access:{n}',0,0,'full') for n in config['access_counts']]
                    configurations += [('policy','lru_nocopy',.7,.5,'full')]
                    if c['workload']=='playing':
                        configurations += [('policy',m,.7,.5,'full') for m in ('full_copy','lru_copy')]
                        configurations += [('policy','full_nocopy',.7,.5,v) for v in ('no_return','no_branch')]
                for family,m,t,r,v in dict.fromkeys(configurations):jobs.append((c,cap,m,t,r,v,family,level))
        dump(out/'status.json',dict(state='running',phase='policy',jobs=len(jobs)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            fs=[pool.submit(run,*x) for x in jobs]
            for i,f in enumerate(concurrent.futures.as_completed(fs),1):
                f.result();print('POLICY',i,'/',len(jobs),flush=True);dump(out/'status.json',dict(state='running',phase='policy',done=i,jobs=len(jobs)))
        dump(out/'policy/completed.json',dict(jobs=len(jobs)))
        bj=[(c,t,m,d) for c in config['cases'] if c['workload']=='playing' for t in config['budget_starts'] for m in ('fixed','dynamic') for d in config['dwpd']]
        rows=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for i,f in enumerate(concurrent.futures.as_completed([pool.submit(budget,*x) for x in bj]),1):
                rows.append(f.result());print('BUDGET',i,'/',len(bj),flush=True);dump(out/'status.json',dict(state='running',phase='budget',done=i,jobs=len(bj)))
        dump(out/'budget/results.json',rows)
        from paper.report import report
        comparison=report(out,ROOT,a.reference)
        if not a.no_plots:subprocess.run([sys.executable,str(ROOT/'paper/plot.py'),str(out)],check=True)
        dump(out/'status.json',dict(state='complete',paper_reference_match=comparison['exact_match'],differences=len(comparison['differences'])))
        print('COMPLETE',out,'reference match:',comparison['exact_match'],flush=True)
        if comparison['exact_match'] is False:sys.exit(3)
    except Exception as exc:
        dump(out/'status.json',dict(state='failed',error=str(exc)));raise

if __name__=='__main__':main()
