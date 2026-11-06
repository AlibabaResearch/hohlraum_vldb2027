"""Report measured results without equating prefetch completion with KV reuse."""
import json
from pathlib import Path


def report(directory):
    root = Path(directory)
    def read(name):
        p=root/name
        return json.loads(p.read_text()) if p.exists() else {}
    result=read('result.json'); horizon=read('horizon.json'); exit_state=read('exit.json')
    errors=[];notes=[]
    if exit_state.get('returncode',0):errors.append('Execution failed; this is not a performance result.')
    if not result:errors.append('No completed result available.')
    requests=horizon.get('requests',[])
    total=sum(x.get('input_length',0) for x in requests)
    audit=[x for x in horizon.get('completed_audit',[]) if 'storage_hit_tokens' in x]
    # Match completed request set; do not divide counters from different windows.
    ids={x['rid'] for x in requests}
    audit=[x for x in audit if x.get('rid') in ids]
    ratio=None
    if total and len(audit)==len(requests):
        ratio=sum(x['storage_hit_tokens'] for x in audit)/total
        if ratio<.01:notes.append('SSD supplied less than 1% of input tokens (diagnostic heuristic, not a paper target). Check memory coverage, reuse gaps and actual request progress; this alone is not a reproduction error.')
    else:notes.append('Actual SSD-hit ratio unavailable for the complete window; missing data is not zero.')
    outcomes=horizon.get('prefetch_outcomes',{})
    for key in ['declined_rate_limited','declined_anchor_lost','revoked_insufficient','revoked_full_miss']:
        if outcomes.get(key,0):notes.append(f'Prefetch counter {key}={outcomes[key]}; inspect receive capacity and queueing before attributing loss to bandwidth.')
    notes.append('Prefetch realization rate and theoretical reuse are not inferred from lookup/completion counters; request-level matching is required.')
    exhausted=result.get('budget_exhausted_at_s')
    if exhausted is None:notes.append('No budget-exhaustion time reported. Check budget consumption; write savings need not improve throughput when the budget does not bind.')
    metrics=dict(throughput=result.get('throughput',result.get('total_input_tokens_s')),
        goodput=result.get('goodput',result.get('slo10_input_tokens_s')),
        ttft_p90_s=result.get('ttft_p90_s'),write_bytes=result.get('write_bytes'),
        budget_exhausted_at_s=exhausted,ssd_hit_fraction=ratio,prefetch_counters=outcomes)
    value=dict(metrics=metrics,errors=errors,observations=notes,
               interpretation='Conditions and instrumentation diagnostics, not a verdict that low gain is incorrect.')
    (root/'analysis.json').write_text(json.dumps(value,indent=2)+'\n')
    text='Results\n'+ '\n'.join(f'{k}: {v if v is not None else "unavailable"}' for k,v in metrics.items())
    text+='\n\n'+ '\n'.join(['ERROR: '+x for x in errors]+notes)+'\n'
    (root/'analysis.txt').write_text(text)
    print(text)
    return value
