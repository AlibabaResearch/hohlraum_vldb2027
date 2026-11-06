"""Portable entry point for fixed-stream policy replay; no inference timing."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
MODES = ('full_nocopy', 'lru_nocopy', 'full_copy', 'lru_copy')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True, help='Joint curves, branch-prior.txt, length_curves.txt')
    p.add_argument('--legacy-models', type=Path, required=True, help='Six v2 legacy-interface-N.model compatibility files')
    p.add_argument('--model-count', type=int, choices=(4, 6), required=True)
    p.add_argument('--blocks', type=int, required=True, help='Block ID upper bound, exclusive')
    p.add_argument('--sessions', type=int, required=True, help='Session ID upper bound, exclusive')
    p.add_argument('--read-blocks', type=int, required=True)
    p.add_argument('--staging-blocks', type=int, required=True)
    p.add_argument('--end-seconds', type=float, required=True)
    p.add_argument('--threshold', type=float, default=.7)
    p.add_argument('--mode', choices=MODES, default='full_nocopy')
    p.add_argument('--signal', choices=('full', 'no_return', 'no_branch'), default='full')
    p.add_argument('--output', type=Path, required=True, help='New run directory; never overwrites')
    a = p.parse_args()
    if not (0 <= a.threshold <= 1 and math.isfinite(a.end_seconds) and a.end_seconds > 0):
        p.error('threshold must be in [0,1] and end-seconds must be positive and finite')
    if min(a.blocks, a.sessions) <= 0 or min(a.read_blocks, a.staging_blocks) < 0:
        p.error('invalid block/session capacity')
    a.events=a.events.resolve();a.models=a.models.resolve();a.legacy_models=a.legacy_models.resolve()
    models = [a.models/f'{i}.txt' for i in range(a.model_count)] + [a.models/'branch-prior.txt', a.models/'length_curves.txt']
    models += [a.legacy_models/f'legacy-interface-{i}.model' for i in range(6)]
    for f in [a.events]+models:
        if not f.is_file():p.error(f'missing input: {f}')
    # Validate IDs/timestamps before C++ indexed access. No future labels are used.
    last=-1.;count=0
    with a.events.open() as f:
        for line_number,line in enumerate(f,1):
            fields=line.rstrip('\n').split('\t')
            if len(fields)!=5:p.error(f'events line {line_number}: expected five TSV columns')
            t=float(fields[0]);sid,fold,length=map(int,fields[1:4]);ids=[int(b) for b in fields[4].split(',') if b]
            if not(math.isfinite(t) and t>=last and 0<=sid<a.sessions and 0<=fold<a.model_count and length>=0 and len(ids)<=length//1024 and all(0<=b<a.blocks for b in ids)):
                p.error(f'invalid event at line {line_number}')
            last=t;count+=1
    subprocess.run(['make','-C',str(ROOT),'all'],check=True,stdout=subprocess.DEVNULL)
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'build/policy_runner'
    env={k:v for k,v in os.environ.items() if not k.startswith('HOHLRAUM_')}
    settings=dict(HOHLRAUM_MODEL_COUNT=str(a.model_count),HOHLRAUM_JOINT_MODELS=str(a.models),HOHLRAUM_ONLINE_PRIOR='20',HOHLRAUM_BRANCH_PRIOR=str(a.models/'branch-prior.txt'),HOHLRAUM_LENGTH_CURVES=str(a.models/'length_curves.txt'),HOHLRAUM_SIGNAL_ABLATION=a.signal)
    env.update(settings)
    command=[str(binary),a.mode,str(a.threshold),str(a.events),str(a.blocks),str(a.sessions),str(a.read_blocks),str(a.staging_blocks),str(a.models),str(a.legacy_models),str(a.end_seconds)]
    source_files=sorted((ROOT/'src').glob('*.cpp'))+sorted((ROOT/'include/hohlraum').glob('*.hpp'))+[Path(__file__)]
    manifest=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},command=command,environment=settings,source_sha256={str(f.relative_to(ROOT)):digest(f) for f in source_files},binary_sha256=digest(binary),events_sha256=digest(a.events),event_rows=count,model_sha256={str(f):digest(f) for f in models},timing='none',write_budget='unlimited',ssd_capacity_blocks=a.blocks)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=subprocess.run(command,env=env,capture_output=True,text=True)
    (a.output/'stderr.log').write_text(result.stderr)
    (a.output/'stdout.txt').write_text(result.stdout)
    (a.output/'status.json').write_text(json.dumps(dict(state='complete' if result.returncode==0 else 'failed',returncode=result.returncode)))
    result.check_returncode()
    metrics=json.loads(result.stdout)
    (a.output/'result.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics))


if __name__=='__main__':main()
