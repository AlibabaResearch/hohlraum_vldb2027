"""Report measured results without equating prefetch completion with KV reuse."""
import gzip,json
from pathlib import Path


def hisim_request_metrics(requests, start, end):
    """Measure completed HiSim requests in the formal window."""
    formal = [
        row for row in requests
        if row.get('ready_at', row.get('created_time', float('-inf'))) >= start
        and row['last_event_time'] <= end
    ]
    if not formal:
        raise RuntimeError('no completed requests inside horizon')
    seconds = end - start
    first_tokens = sorted(row['gen_token_latencies'][0] for row in formal)
    position = .9 * (len(first_tokens) - 1)
    index = int(position)
    upper = first_tokens[min(index + 1, len(first_tokens) - 1)]
    p90 = first_tokens[index] + (position - index) * (upper - first_tokens[index])
    engine = sum(row['input_length'] for row in formal if row['engine_ttft_s'] <= 10) / seconds
    e2e = sum(row['input_length'] for row in formal if row['gen_token_latencies'][0] <= 10) / seconds
    return dict(
        completed=len(formal),
        throughput=sum(row['input_length'] for row in formal) / seconds,
        goodput=e2e,
        engine_goodput10_input_tokens_s=engine,
        e2e_goodput10_input_tokens_s=e2e,
        ttft_p90_s=p90,
    )


def serving_latency_summary(result):
    """Report the available goodput scope alongside end-to-end P90."""
    engine = result.get('engine_goodput10_input_tokens_s')
    e2e = result.get('e2e_goodput10_input_tokens_s',
                     result.get('goodput', result.get('slo10_input_tokens_s')))
    return dict(
        goodput=engine if engine is not None else e2e,
        goodput_scope='engine' if engine is not None else 'end_to_end' if e2e is not None else None,
        engine_goodput10_input_tokens_s=engine,
        e2e_goodput10_input_tokens_s=e2e,
        ttft_p90_s=result.get('ttft_p90_s'),
    )


def report(directory):
    root = Path(directory)
    def read(name):
        p=root/name
        if p.exists():return json.loads(p.read_text())
        compressed=p.with_name(p.name+'.gz')
        return json.loads(gzip.decompress(compressed.read_bytes())) if compressed.exists() else {}
    result=read('result.json'); horizon=read('horizon.json'); exit_state=read('exit.json')
    errors=[];notes=[]
    if exit_state.get('returncode',0):errors.append('Execution failed; this is not a performance result.')
    if not result:errors.append('No completed result available.')
    requests=horizon.get('requests',[])
    manifest=result.get('manifest') or read('manifest.json')
    formal=manifest.get('formal_seconds')
    endpoint=manifest.get('horizon_s',horizon.get('horizon_s'))
    if formal is not None and endpoint is not None:
        start=float(endpoint)-float(formal);end=float(endpoint)
        requests=[x for x in requests
                  if x.get('ready_at',x.get('created_time',float('-inf')))>=start
                  and x.get('last_event_time',float('inf'))<=end]
        request_window=dict(start_s=start,end_s=end,selection='ready_at>=start and completion<=end')
    else:
        request_window=dict(selection='whole available horizon; formal boundaries unavailable')
        notes.append('Formal request boundaries unavailable; SSD-hit diagnostic uses the available horizon.')
    total=sum(x.get('input_length',0) for x in requests)
    audit=[x for x in horizon.get('completed_audit',[]) if 'storage_hit_tokens' in x]
    # Match completed request set; do not divide counters from different windows.
    ids={x['rid'] for x in requests}
    audit=[x for x in audit if x.get('rid') in ids]
    ratio=None
    if total and len(audit)==len(requests):
        ratio=sum(x['storage_hit_tokens'] for x in audit)/total
    else:notes.append('SSD-hit ratio requires storage-hit counts for every completed request in the measurement window.')
    outcomes=horizon.get('prefetch_outcomes',{})
    exhausted=result.get('budget_exhausted_at_s')
    metrics=dict(throughput=result.get('throughput',result.get('total_input_tokens_s')),
        **serving_latency_summary(result),write_bytes=result.get('write_bytes'),
        budget_exhausted_at_s=exhausted,ssd_hit_fraction=ratio,
        ssd_hit_request_window=request_window,ssd_hit_matched_requests=len(audit),
        prefetch_counters=outcomes,
        prefetch_counters_scope='whole available horizon, including warmup when present; not formal-window deltas')
    value=dict(metrics=metrics,errors=errors,observations=notes,
               interpretation='Metrics are reported for the windows identified in the result.')
    (root/'analysis.json').write_text(json.dumps(value,indent=2)+'\n')
    text='Results\n'+ '\n'.join(f'{k}: {v if v is not None else "unavailable"}' for k,v in metrics.items())
    text+='\n\n'+ '\n'.join(['ERROR: '+x for x in errors]+notes)+'\n'
    (root/'analysis.txt').write_text(text)
    print(text)
    return value
