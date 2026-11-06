"""Report freshly computed metrics, separately compare historical reference counts."""
import csv,json,statistics
from pathlib import Path

def report(out,root,reference=None):
    expected=json.loads(Path(reference).read_text()) if reference else {};diff=[];checked=0
    for key,counts in expected.items():
        p=out/(key+'/result.json' if key.startswith('budget/') else key+'.json')
        if not p.is_file():diff.append(dict(case=key,error='missing result'));continue
        got=json.loads(p.read_text());checked+=1
        for k,v in counts.items():
            if got.get(k)!=v:diff.append(dict(case=key,metric=k,expected=v,actual=got.get(k)))
    comparison=dict(reference_checked=reference is not None,exact_match=(not diff) if reference else None,checked_cases=checked,differences=diff)
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    rows=json.loads((out/'budget/results.json').read_text());summary=[]
    for d in (1,3):
        for t in (.1,.5,.7,.9,1):
            row=dict(dwpd=d,start=t)
            for mode,label in [('fixed','fixed'),('dynamic','feedback')]:
                x=[r for r in rows if r['dwpd']==d and r['initial_threshold']==t and r['mode']==mode];assert len(x)==3
                row[label+'_budget_pct']=statistics.mean(100*r['writes']/r['budget_blocks'] for r in x)
                row[label+'_hit_pct']=statistics.mean(100*r['global_hits']/r['refs'] for r in x)
            summary.append(row)
    (out/'budget/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'table3.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(r for r in summary if r['dwpd']==3)
    tex=['% Computed from this run; three paired seed means.','\\begin{tabular}{rrrrr}','Start & Fixed budget (\\%) & Feedback budget (\\%) & Fixed hit (\\%) & Feedback hit (\\%) \\\\']
    for r in summary:
        if r['dwpd']==3:tex.append(f"{r['start']:g} & {r['fixed_budget_pct']:.1f} & {r['feedback_budget_pct']:.1f} & {r['fixed_hit_pct']:.3f} & {r['feedback_hit_pct']:.3f} \\\\")
    tex.append('\\end{tabular}');(out/'table3.tex').write_text('\n'.join(tex)+'\n')
    records=[]
    for f in sorted((out/'policy').glob('seed*/*/coverage*/*.json')):
        x=json.loads(f.read_text());records.append(dict(case=str(f.relative_to(out)),seed=x['seed'],workload=x['workload'],capacity=x['total_blocks'],staging=x['staging_blocks'],hit_pct=100*x['global_hits']/x['refs'],writes=x['written_blocks']))
    with (out/'policy_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (out/'README.txt').write_text('All measurements were recomputed from inputs. comparison.json records a historical comparison only when --reference is supplied; references are never used as measured outputs.\nFigures 8-10: figures/. Table 3: table3.csv and table3.tex. Raw policy measurements: policy_results.csv and policy/. Budget trajectories and 1-DWPD supplementary results: budget/.\nReference match: '+str(comparison['exact_match'])+'\n')
    return comparison
