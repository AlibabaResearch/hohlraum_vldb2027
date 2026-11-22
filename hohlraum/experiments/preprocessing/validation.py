"""Validate immutable request inputs at the serving entry point."""
import json
import math
from pathlib import Path

from hohlraum.experiments.common.workload import normalize
from hohlraum.experiments.resources import digest


def validate(dataset, selection):
    rows_path = Path(selection['rows'])
    rows_sha256 = digest(rows_path)
    if selection.get('rows_sha256') and rows_sha256 != selection['rows_sha256']:
        raise ValueError('request rows checksum mismatch')
    rows = json.loads(rows_path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('empty or invalid request rows')
    fields = ('root', 'chain', 'index', 'owner', 'reference_fold',
              'root_offer', 'first_delay', 'think_time', 'raw_input_tokens', 'blocks')
    for row in rows:
        for key in fields:
            if key not in row:
                raise ValueError('missing request field ' + key)
        n = row['raw_input_tokens']
        if not isinstance(n, int) or n < 1 or len(row['blocks']) != n // 1024:
            raise ValueError('request length/block count mismatch')
        if not all(isinstance(b, int) and 0 <= b < 2**64 for b in row['blocks']):
            raise ValueError('invalid block ID')
        for key in ('root_offer', 'first_delay', 'think_time'):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid timing field ' + key)
    deps = dep_hash = None
    dependency_path = selection.get('dependencies')
    if dependency_path:
        path = Path(dependency_path)
        dep_hash = digest(path)
        if selection.get('dependencies_sha256') and dep_hash != selection['dependencies_sha256']:
            raise ValueError('dependency checksum mismatch')
        deps = json.loads(path.read_text())
    elif dataset == 'agentx' or selection.get('dependencies_sha256'):
        raise ValueError('Dependency DAG missing')
    normalize(rows, deps)
    return dict(requests=len(rows), roots=len({r['root'] for r in rows}),
                rows_sha256=rows_sha256, dependencies_sha256=dep_hash,
                checks=['schema', 'page_lengths', 'timing', 'dependency_graph'],
                expected_checksum_checked=bool(selection.get('rows_sha256')))


def ensure_inputs(dataset, selection, seed):
    if selection.get('seed', seed) != seed:
        raise ValueError('selection seed mismatch')
    return selection, dict(action='reused', validation=validate(dataset, selection))
