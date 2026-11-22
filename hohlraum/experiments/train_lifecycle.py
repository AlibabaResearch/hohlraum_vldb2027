"""Train type-independent lifecycle priors on held-out whole-root splits.

Observation endpoints must be supplied explicitly. A last request timestamp is
not an observation endpoint or a terminal label. Runtime learns no terminal or
main/child labels from these offline annotations.
"""
import argparse, collections, hashlib, json, math
from pathlib import Path
QG_SECONDS=60.
HORIZON_SECONDS=3600.  # Existing protocol; not a paper-specified numeric constant.
def grid():
    edges=[];t=QG_SECONDS
    while t<HORIZON_SECONDS:edges.append(t);t*=1.15
    return edges+[HORIZON_SECONDS]
def add(risk,event,edges,duration,returned):
    k=0
    while k<len(risk) and edges[k+1]<=duration:
        risk[k]+=1;k+=1
    if returned and duration>=edges[0] and k<len(risk):risk[k]+=1;event[k]+=1

def observation(task,owner,last_end,annotations):
    identity=task['split_group']
    external=annotations.get(identity,{})
    meta=external.get('owners',{}).get(str(owner),external)
    internal=task.get('owner_observations',{}).get(str(owner),task)
    ended=meta.get('terminal_observed',internal.get('terminal_observed',False))
    if type(ended) is not bool:raise ValueError('terminal_observed must be boolean')
    if ended:return HORIZON_SECONDS,True
    end=meta.get('observation_end',internal.get('observation_end'))
    if end is None:
        raise ValueError(f'{identity} owner {owner}: missing observation_end or confirmed terminal_observed; terminal_time/last request cannot substitute')
    if not math.isfinite(end) or end<last_end:raise ValueError('observation_end precedes final completion')
    return end-last_end,False

def extract(task,annotations,dataset="agentx"):
    terminal_policy="child-with-followup" if dataset=="agentx" else "observed-termination"
    owners=collections.defaultdict(list)
    for i,chain in enumerate(task['chains']):owners[task.get('chain_owner_ids',list(range(len(task['chains']))))[i]].extend(chain)
    result=[]
    roles=collections.defaultdict(set)
    if terminal_policy=="child-with-followup":
        if len(task.get("chain_roles",[]))!=len(task["chains"]):
            raise ValueError("offline terminal labeling requires verified chain_roles")
        for i,role in enumerate(task["chain_roles"]):
            if role not in (0,1):raise ValueError("unknown offline role")
            roles[task["chain_owner_ids"][i]].add(role)
        root_end=max((row["t"]+row.get("api_time",0.) for rows in owners.values() for row in rows),default=0.)
    for owner,rows in owners.items():
        if not rows:continue
        rows.sort(key=lambda r:r['t']);end=None;clock=0.;groups=[];branches=[];pending={};current=set();current_tail=None
        risk_bins=max(65,max(r['raw_input_tokens'] for r in rows)//4096+1)
        risk=[0]*risk_bins;yes=[0]*risk_bins;episode_start=episode_end=None;credited=set()
        for row in rows:
            # Use the same post-completion delay as the external client.
            # Timestamp/api_time differences need not equal trace think_time.
            gap=float(row['think_time'])
            if not math.isfinite(gap) or gap<0:raise ValueError('invalid think_time')
            if end is None:gap=0.
            clock+=gap
            if not groups or gap>=QG_SECONDS:groups.append([])
            groups[-1].append((row,gap))
            blocks=set(row['blocks']);removed=current-blocks
            # Match native old-suffix transitions: any reappearance closes its
            # shared sample once; subsequent reappearances do not count twice.
            for block in blocks:
                g=pending.pop(block,None)
                if g is not None and branches[g]['duration'] is None:
                    branches[g]['duration']=clock-branches[g]['start'];branches[g]['returned']=True
            if removed:
                g=len(branches);branches.append(dict(start=clock,duration=None,returned=False))
                for block in removed:pending[block]=g
            length=row['raw_input_tokens'];changed=any(block!=current_tail for block in removed)
            if episode_start is None:episode_start=episode_end=length
            else:
                if not changed:episode_end=max(episode_end,length)
                for i in range(risk_bins):
                    pos=i*4096
                    if i in credited or episode_start>pos or episode_end<=pos:continue
                    if changed or episode_end>pos+8192:
                        risk[i]+=1;yes[i]+=changed and episode_end<=pos+8192;credited.add(i)
                if changed:episode_start=episode_end=length;credited=set()
            current_tail=row["blocks"][-1] if row["blocks"] else None
            current=blocks;end=max(end or 0.,row['t']+row.get('api_time',0.))
        tail,ended=observation(task,owner,end,annotations)
        branch_complete=ended or bool(task.get('branch_observation_complete',False))
        samples=[]
        for q,group in enumerate(groups):
            returned=q+1<len(groups)
            d=groups[q+1][0][1] if returned else tail
            if not returned and terminal_policy=="child-with-followup":
                # Role is an offline label source, never a runtime feature.
                negative=0 not in roles[owner] and root_end>end+1e-6
                if not negative:returned=True;d=None
            samples.append((min(q,3),0,d,returned))
        branch_samples=[]
        for g in branches:
            d=g['duration'] if g['duration'] is not None else (HORIZON_SECONDS if branch_complete else clock+tail-g['start'])
            branch_samples.append((d,g['returned']))
        result.append(dict(samples=samples,branches=branch_samples,risk=risk,yes=yes,terminal=ended,branch_terminal=branch_complete,tail=tail))
    return result

def unified_splits(report):
    # Legacy 0/2 and 1/3 partitions refer to the same held-out roots. Union the
    # training sets, preserving root isolation; never inspect chain_roles.
    parts=collections.defaultdict(list)
    for f in report['folds']:parts[int(f['fold'])].append(f)
    if set(parts)!={0,1}:raise ValueError('Expected two whole-root cross-validation folds')
    result=[]
    for f,items in sorted(parts.items()):
        train=set().union(*(set(x['train']) for x in items));test=set().union(*(set(x['test']) for x in items))
        if train&test:raise ValueError('Cross-validation root leakage')
        result.append(dict(index=f,fold=f,train=sorted(train),test=sorted(test)))
    return result

def curve(risk,event):
    out=[0.]*len(risk);survive=1.
    for k in reversed(range(len(risk))):survive*=1-(event[k]/risk[k] if risk[k] else 0.);out[k]=1-survive
    return out

def train(source,folds_directory,out,annotations=None,complete_trajectories=False,dataset="agentx"):
    terminal_policy="child-with-followup" if dataset=="agentx" else "observed-termination"
    annotations=annotations or {};report=json.loads((folds_directory/'training.json').read_text())
    if dataset=="agentx":folds=unified_splits(report)
    else:
        records=report if isinstance(report,list) else report['folds']
        folds=[dict(index=i,fold=int(f['fold']),provider=f.get('provider','single-provider'),train=f['train'],test=f['test']) for i,f in enumerate(records)]
        if len(folds) not in (2,6):raise ValueError('Playing requires two root-isolated folds per provider')
        if any(set(f['train'])&set(f['test']) for f in folds):raise ValueError('Playing root leakage')
    tasks={}
    for line in source.open():
        task=json.loads(line);identity=task['split_group']
        if identity in tasks:
            raise ValueError('Each training split group must appear in one source record: '+identity)
        if complete_trajectories:
            if dataset=='playing' and task.get('observation_slice',False):
                raise ValueError('Serving slices cannot be labeled complete trajectories; rebuild uncropped source')
            # A complete recorded trajectory does not establish session death.
            # Keep the existing AgentX labeling convention; for Playing only
            # branch observation is complete unless termination is explicit.
            task=dict(task,**({'branch_observation_complete':True}
                             if dataset=='playing' else {'terminal_observed':True}))
        tasks[identity]=extract(task,annotations,dataset)
    risk_bins=max((len(s['risk']) for task in tasks.values() for s in task),default=65)
    edges=grid();bedges=[0.]+edges;nb=len(edges)-1;all_hazards=[];branches=[];eventual_branches=[];return_curves=[];files={};manifests=[];return_counts=[];branch_counts=[];context_counts=[]
    for item in folds:
        pr=[[0]*nb for _ in range(4)];pe=[[0]*nb for _ in range(4)];counts=[[0]*8 for _ in range(4)]
        br=[0]*(len(bedges)-1);be=br.copy();cr=[0]*risk_bins;cy=cr.copy();audit=collections.Counter();gaps=[[] for _ in range(4)]
        known=[[0]*8 for _ in range(4)];returned_counts=[[0]*8 for _ in range(4)];branch_known=0;branch_returns=[]
        for identity in item['train']:
            if identity not in tasks:raise ValueError('missing training root '+identity)
            for s in tasks[identity]:
                audit['sessions']+=1;audit['confirmed_terminal']+=s['terminal'] and (terminal_policy=='observed-termination' or not s['samples'][-1][3]);audit['branch_terminal_observation']+=s.get('branch_terminal',s['terminal']);audit['zero_observed_tail']+=not s['terminal'] and s['tail']==0
                for q,b,d,returned in s['samples']:
                    if d is not None:
                        add(pr[q],pe[q],edges,d,returned)
                    counts[q][b]+=1
                    audit['samples']+=1;audit['returned']+=returned
                    audit['positive_unknown_interval']+=returned and d is None
                    audit['horizon_nonreturn']+=not returned and d>=HORIZON_SECONDS
                    audit['censored']+=not returned and d<HORIZON_SECONDS
                    if returned and d is not None:gaps[q].append(d)
                    if returned or s['terminal'] or terminal_policy=="child-with-followup":
                        known[q][b]+=1;returned_counts[q][b]+=returned
                    audit['returned_after_one_hour']+=returned and d is not None and d>=HORIZON_SECONDS
                for d,returned in s['branches']:
                    add(br,be,bedges,d,returned)
                    if returned or s.get('branch_terminal',s['terminal']):branch_known+=1
                    if returned:branch_returns.append(d)
                for i,(n,y) in enumerate(zip(s['risk'],s['yes'])):cr[i]+=n;cy[i]+=y
        lines=[]
        for q in range(4):
            pooled=curve(pr[q],pe[q]);lines.extend([str(nb+2),f'0 {pooled[0]:.17g}']);lines.extend(f'{t:.17g} {v:.17g}' for t,v in zip(edges[:-1],pooled));lines.append(f'{HORIZON_SECONDS} 0')
            probs=[];base=sum(returned_counts[q])/sum(known[q]) if sum(known[q]) else 0.
            for b in range(8):
                rr,ee=pr[q],pe[q]
                all_hazards.append([e/r if r else 0. for r,e in zip(rr,ee)]);probs.append(base)
            return_curves.append(' '.join(map(str,[base,*probs,len(gaps[q]),*sorted(gaps[q])])))
        lines.extend([str(risk_bins),' '.join(str(y/n if n else 0.) for n,y in zip(cr,cy))]);files[f"{item['index']}.txt"]='\n'.join(lines)+'\n'
        branches.append([e/r if r else 0. for r,e in zip(br,be)])
        context_counts.append(list(zip(cr,cy)))
        eventual_branches.append(' '.join(map(str,[len(branch_returns)/branch_known if branch_known else 0.,len(branch_returns),*sorted(branch_returns)])))
        branch_counts.append((branch_known,len(branch_returns)))
        return_counts.extend((sum(known[q]),sum(returned_counts[q]),len(gaps[q])) for q in range(4))
        manifests.append(dict(**item,**audit,length_counts=counts,branch_return_counts=dict(total=branch_known,returned=len(branch_returns)),return_counts=[dict(total=sum(known[q]),returned=sum(returned_counts[q]),observed_return_intervals=len(gaps[q])) for q in range(4)]))
    files['length-hazards.txt']='\n'.join([f'{QG_SECONDS} {HORIZON_SECONDS} {len(folds)} {nb}',' '.join(map(str,edges)),*(' '.join(map(str,h)) for h in all_hazards)])+'\n'
    files['branch-prior.txt']='\n'.join([str(len(bedges)),' '.join(map(str,bedges)),*(' '.join(map(str,h)) for h in branches)])+'\n'
    files['length_curves.txt']='\n'.join(return_curves)+'\n'
    files['return-counts.txt']=f'qg-return-counts-v2 {len(folds)} 4'+chr(10)+chr(10).join(
        f'{n} {yes} {observed}' for n,yes,observed in return_counts)+chr(10)
    files['branch-eventual.txt']='\n'.join(eventual_branches)+'\n'
    files['branch-counts.txt']=f'branch-return-counts-v1 {len(folds)}'+chr(10)+chr(10).join(f'{n} {yes}' for n,yes in branch_counts)+chr(10)
    files['context-risk-counts.txt']=f'context-risk-counts-v1 {len(folds)} {risk_bins}'+chr(10)+chr(10).join(f'{n} {yes}' for row in context_counts for n,yes in row)+chr(10)
    files['prior-target.txt']='eventual-return-v1\n'
    meta=dict(session_unit='(root,owner), session QG cohorts; no main/child feature',model_schema='session-qg-only-v1',model_count=len(folds),split_unit='whole root',prior_target='eventual return; all observed next QGs positive regardless of delay',complete_trajectories=complete_trajectories,return_grouping='QG only; serialized legacy slots repeat the same prior',qg_gap_seconds=QG_SECONDS,horizon_seconds=HORIZON_SECONDS,folds=manifests,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),observation_annotations=annotations,terminal_policy=terminal_policy,unknown_positive_intervals='estimated from observed returns only; no fabricated times',clock='trace think_time; queue and service excluded')
    meta['context_risk_bins']=risk_bins;meta['context_risk_bin_tokens']=4096
    meta["context_change_scope"]="interior-prefix changes; terminal-page-only changes remain branch and positional observations"
    meta['session_boundary']='input (root,owner) sessions; trainer does not join distinct roots' if dataset=='playing' else 'root-owner'
    meta['eof_label']='right censored unless explicitly terminated' if dataset=='playing' else terminal_policy
    files['training.json']=json.dumps(meta,indent=2)+'\n';files['model-schema.txt']='session-qg-only-v1\n'
    out.mkdir(parents=True,exist_ok=False)
    for n,s in files.items():(out/n).write_text(s)
    print(json.dumps(dict(output=str(out),folds=[{k:v for k,v in f.items() if k not in ['train','test','length_counts']} for f in manifests])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--folds',type=Path,required=True,help='Directory containing training.json with whole-root train and test splits');p.add_argument('--output',type=Path,required=True);p.add_argument('--observations',type=Path,help='Verified root/owner observation_end or terminal_observed annotations')
    p.add_argument('--complete-trajectories',action='store_true',help='Complete trajectory observation for branch priors; session terminal labels follow the dataset convention')
    p.add_argument('--dataset',choices=['agentx','playing'],default='agentx')
    a=p.parse_args();train(a.source,a.folds,a.output,json.loads(a.observations.read_text()) if a.observations else None,a.complete_trajectories,a.dataset)
