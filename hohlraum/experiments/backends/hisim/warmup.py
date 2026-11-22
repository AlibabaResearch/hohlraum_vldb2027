"""Activate the policy after native warmup, preserving KV and observed history."""
import json


def activate(tree, source, now, output, rows, dataset):
    from hohlraum.core.staging_storage import (
        BudgetStorage, HiSimHost, create_host, migrate_prefetch_leases,
        _install_scheduler_hooks,
    )
    from hohlraum.experiments.common.history_restore import restore_resident_history
    from hohlraum.experiments.common.workload import reference_model_fold

    cfg = json.loads((output / 'formal_policy.json').read_text())
    backend = tree.cache_controller.storage_backend
    assert isinstance(backend, BudgetStorage) and tree.hohlraum_host is None
    assert backend.validation_cap is None
    stored_before = set(backend.backing.storage)
    if not cfg.get('enabled', True):
        return dict(activated=False, native_full_admission=True)
    pool = tree.cache_controller.mem_pool_host.anchor_entry.host_pool
    physical_before = pool.available_size()
    cfg['threshold'] = float(cfg.get('threshold', .7))
    host = create_host(tree, cfg, HiSimHost)
    tree.hohlraum_host = host
    migrated = migrate_prefetch_leases(host)
    _install_scheduler_hooks()
    assert host.ssd_backend is backend and host.storage.backing is backend
    assert host._physical_available() == physical_before and not host.stage

    bykey = {(r['root'], r['chain'], r['index']): r for r in rows}
    events, admitted = [], {}
    records = source.request_admission.records
    for event in source.lifecycle_history:
        key = tuple(event['key'])
        actual = source.requests.get(key)
        rid = actual.rid if actual is not None else 'warm:' + ':'.join(map(str, key))
        converted = dict(kind=event['kind'], rid=rid, monotonic=event['t'])
        if event['kind'] == 'ready':
            row = bykey[key]
            converted.update(input_length=row['raw_input_tokens'], meta=dict(
                root=row['root'], chain=row['chain'], owner=row.get('owner', row['chain']),
                index=row['index'], reference_fold=reference_model_fold(row, dataset),
                prefix_blocks=row['blocks'], prefix_page_size=1024))
        events.append(converted)
        if actual is not None:
            record = records.get(rid, {})
            if record.get('engine_admitted_at') is not None:
                admitted[rid] = actual
    restored = restore_resident_history(host, events, 0., now, admitted)
    assert set(backend.backing.storage) == stored_before
    assert host._physical_available() == physical_before
    assert abs(host.admission_threshold - cfg['threshold']) < 1e-12
    return dict(activated=True, native_full_admission=True,
                migrated_native_ssd_reads=migrated, warm_stage_pages=0,
                restored_requests=restored['requests'],
                restored_active_requests=restored['active'],
                restored_native_prefixes=len(host.sli.ids),
                resident_hashes_missing_sli=0, resident_hashes_missing_hts=0,
                resident_hashes=restored['resident_pages'],
                physical_free_before=physical_before,
                physical_free_after=host._physical_available(),
                same_ssd_backend=host.ssd_backend is backend,
                stage_limit_tokens=host.staging_limit, clock_s=now,
                admission_threshold=host.admission_threshold,
                preserved_ssd_keys=len(stored_before))
