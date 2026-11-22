"""Restore observed policy history at the measurement boundary.

Replays client-ready/completion notifications only; never loads KV or reads
future request timings. Native hashes are bound lazily by the regular SLI path.
"""
import json
from pathlib import Path
from types import SimpleNamespace as NS


def load_checkpoint(path):
    path = Path(path)
    checkpoint = json.loads(path.read_text())
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('lifecycle_history'), list):
        raise ValueError(f'Checkpoint lacks recorded lifecycle history: {path}')
    return checkpoint


def restore_sli(sli,checkpoint,rows,dataset,formal_start=0.):
    from hohlraum.experiments.common.workload import reference_model_fold
    if sli.observed_requests or sli.sessions:
        raise ValueError('SLI history must be restored before formal requests')
    by_key={(r['root'],r['chain'],r['index']):r for r in rows}
    cut=float(checkpoint['clock']);events=checkpoint['lifecycle_history'];active={};seen=set();last=float('-inf')
    for event in events:
        t=float(event['t']);key=tuple(event['key'])
        if t<last or t>cut:raise ValueError('Invalid or future lifecycle event')
        last=t;now=t-cut+formal_start
        if event['kind']=='ready':
            if key in seen:raise ValueError('Duplicate historical request')
            seen.add(key);row=by_key[key]
            meta=dict(root=key[0],chain=key[1],reference_fold=reference_model_fold(row,dataset),prefix_blocks=row['blocks'],prefix_page_size=1024)
            req=NS(rid='warmup-history:'+':'.join(map(str,key)),input_length=row['raw_input_tokens'],sampling_params=NS(custom_params={'hohlraum':meta}))
            sli.client_ready(req,now);active[key]=req
        elif event['kind']=='complete':
            req=active.pop(key);sli.check(sli.lib.hh_clock(sli.ptr,sli.sid(sli.metadata(req)),now,1))
        else:raise ValueError('Unknown lifecycle event')
    # In-flight requests restart on the fresh engine. Keep their known context,
    # close the old busy interval at the cut, and let normal readiness reopen it.
    for req in active.values():
        sli.check(sli.lib.hh_clock(sli.ptr,sli.sid(sli.metadata(req)),formal_start,1))
    sli.advance(max(sli.now,formal_start))
    sli.restored_history=dict(requests=len(seen),sessions=len(sli.sessions),cut=cut,kv_restored=False)
    return sli.restored_history


def restore_resident_history(host, events, origin, now, engine_requests):
    """Restore only observed warmup events while native KV stays resident."""
    if host.sli.observed_requests or host.sli.sessions:
        raise ValueError('Resident warmup restoration requires a fresh SLI')

    class FullPages:
        def __init__(self, meta, length):
            self.meta, self.length = meta, length
        def __len__(self):
            return self.length
        def __getitem__(self, part):
            if not isinstance(part, slice) or part.step is not None:
                raise ValueError('History hashing requires one complete input page')
            start=0 if part.start is None else part.start
            stop=self.length if part.stop is None else part.stop
            if stop-start!=1024:
                raise ValueError('History hashing requires one complete input page')
            i=start//1024
            marker=int(self.meta['root']).to_bytes(4,'big')+int(self.meta['prefix_blocks'][i]).to_bytes(8,'big')
            return [16+b for b in marker]+[7]*(1024-len(marker))

    active={};seen=set()
    for event in sorted(events,key=lambda e:e['monotonic']):
        stamp=event['monotonic']-origin;rid=event['rid']
        if stamp > now+1e-6:
            raise ValueError('Warmup history contains a future event')
        if event['kind']=='ready':
            if rid in seen:
                raise ValueError('Duplicate warmup readiness: '+rid)
            seen.add(rid);meta=event['meta'];length=event['input_length']
            req=NS(rid=rid,input_length=length,input_ids=FullPages(meta,length),
                   sampling_params=NS(custom_params={'hohlraum':meta}))
            host.sli.client_ready(req,stamp)
            hashes=host.sli.hashes_for_request(req,host.cache.page_size)
            host.sli.bind_input_prefixes(req,hashes)
            host.staging_policy.observe((meta['root'],meta['owner']),hashes)
            active[rid]=req
        elif event['kind']=='complete':
            req=active.pop(rid)
            host.sli.check(host.sli.lib.hh_clock(host.sli.ptr,host.sli.sid(host.sli.metadata(req)),stamp,1))
        else:
            raise ValueError('Unknown warmup history event')
    admitted=0
    for rid in active:
        if rid in engine_requests:
            host.observe_request(engine_requests[rid],now);admitted+=1
    host.sli.advance(now)
    resident={key for node in host._nodes() for key in node.hash_value or ()}
    missing=resident-set(host.sli.ids)
    missing_hts=resident-set(host.staging_policy.index.parents)
    if missing or missing_hts:
        raise RuntimeError(f'Warmup KV missing SLI/HTS identities: {len(missing)}/{len(missing_hts)}')
    return dict(requests=len(seen),active=len(active),engine_admitted=admitted,
                resident_pages=len(resident),kv_preserved=True)
