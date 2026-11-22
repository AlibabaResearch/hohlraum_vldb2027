"""Build page-aligned hourly Playing sessions, isolated replicas and priors.

Original trajectory identities define cross-validation folds. Hours have separate
cache namespaces and retain their predecessor dependencies. Formal independent
roots follow the Poisson arrival plan for the bundled Playing workloads.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import random
import shutil

from hohlraum.experiments.resources import digest
from .rebuild_playing_trace import (
    CONTRACT, PAGE, build_trajectory, cache_read_boundary,
    page_align_hourly_requests, provider_cache_reads, shared_pages,
)

WARMUP_SECONDS = 3600


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def validate_aligned(rows, identity, hours):
    if not rows or len({r['hour'] for r in rows}) != hours:
        raise ValueError('Incomplete original trajectory')
    provider = identity.split('/')[0]
    previous = None
    seen = set()
    seen_prior = set()
    for row in rows:
        unpadded = row['pre_alignment_input_tokens']
        pages = (unpadded + PAGE - 1) // PAGE
        blocks = row['blocks']
        prior = row.get('prior_blocks')
        append = row.get('complete_append')
        if (row['source_trajectory'] != identity or row['shared_padding_tokens'] != 0
                or row['raw_input_tokens'] != PAGE * len(blocks)
                or row['raw_input_tokens'] != PAGE * pages
                or not isinstance(prior, list) or len(prior) != pages
                or not isinstance(append, bool)
                or not math.isfinite(row['think_time']) or row['think_time'] < 0):
            raise ValueError('Aligned input does not satisfy the reconstruction contract')
        same_hour = previous is not None and previous['hour'] == row['hour']
        common = shared_pages(previous['blocks'], blocks) if same_hour else 0
        common_prior = shared_pages(previous['prior_blocks'], prior) if same_hour else 0
        boundary = cache_read_boundary(provider, row['provider_cache_read_tokens'],
                                       row['logged_input_tokens'], unpadded)
        if provider.startswith(('gpt', 'kimi')) and row.get(
                'provider_cache_read_qwen_estimate_tokens') != boundary:
            raise ValueError('Aligned input is missing its Qwen cache-read boundary')
        if append:
            if (not same_hour or previous['segment'] != row['segment']
                    or previous['pre_alignment_input_tokens'] > unpadded
                    or common != previous['pre_alignment_input_tokens'] // PAGE):
                raise ValueError('Invalid append-only prefix')
        elif common != common_prior:
            raise ValueError('Branching input did not honor the provider boundary')
        if (common * PAGE != row['expected_previous_hit_tokens']
                or common_prior > min(boundary // PAGE, unpadded // PAGE)
                or common > unpadded // PAGE or blocks[common:] != prior[common:]
                or seen.intersection(blocks[common:]) or len(set(blocks)) != pages
                or seen_prior.intersection(prior[common_prior:]) or len(set(prior)) != pages):
            raise ValueError('Input block prefix does not match its cache-read boundary')
        seen.update(blocks)
        seen_prior.update(prior)
        previous = row


def reconstruct(job):
    raw, tokenizer, output, identity, hours = job
    visible, aligned = output / 'visible', output / 'aligned'
    report = build_trajectory(raw, tokenizer, visible, identity, hours=hours)
    provider = identity.split('/')[0]
    cache = {}
    for path in sorted((raw / identity).glob('hour_*/session.jsonl'))[:hours]:
        hour = int(path.parent.name.split('_')[1])
        cache.update({(hour, line): n for line, n in provider_cache_reads(path, provider).items()})
    source = visible / (identity.replace('/', '__') + '.jsonl')
    target = aligned / (identity.replace('/', '__') + '.jsonl.gz')
    counts = Counter()
    with source.open() as src, gzip.open(target, 'wt') as dst:
        for row in page_align_hourly_requests((json.loads(line) for line in src), cache,
                                              provider, audit=counts):
            row.update(source_trajectory=identity, provider=provider)
            dst.write(json.dumps(row, separators=(',', ':')) + '\n')
    return dict(**report, source_sha256=digest(source), counts=dict(counts))


def make_folds(identities):
    folds = []
    for provider in ('claude', 'gpt', 'kimi'):
        group = sorted(x for x in identities if x.startswith(provider + '_'))
        for fold in range(2):
            test = [x for x in group if int.from_bytes(hashlib.sha256(
                ('fig8-crossfit:' + x).encode()).digest()[:8], 'big') % 2 == fold]
            train = sorted(set(group) - set(test))
            if not train or not test:
                raise ValueError('Each provider requires two nonempty trajectory folds')
            folds.append(dict(index=len(folds), provider=provider, fold=fold,
                              train=train, test=test))
    return folds


def export(data, folds, output, seed, concurrencies, *, published=None):
    """Sample 65 copies from a provider-balanced195 pool, using no runtime labels."""
    pool = [(identity, replica) for identity in sorted(data)
            for replica in range(5 if identity.startswith('claude') else 13)]
    if Counter(identity.split('_')[0] for identity, _ in pool) != dict(
            claude=65, gpt=65, kimi=65):
        raise ValueError('Expected 13 Claude, 5 GPT and 5 Kimi original trajectories')
    order = random.Random(seed).sample(range(len(pool)), 65)
    rows, dependencies, selection = [], {}, []
    for rank, index in enumerate(order):
        identity, replica = pool[index]
        fold = [f['fold'] for f in folds if identity in f['test']]
        if len(fold) != 1:
            raise ValueError('Missing or ambiguous held-out trajectory fold')
        hours = defaultdict(list)
        for row in data[identity]:
            hours[row['hour']].append(row)
        previous = None
        for local_hour, (hour, requests) in enumerate(sorted(hours.items())):
            root = rank * len(hours) + local_hour
            plan = dict(deps={}, first_delays={'0': 0.})
            if previous is not None:
                gap = requests[0]['observed_at'] - previous[-1]['observed_at']
                if gap < 0:
                    raise ValueError('Cross-hour observation times reversed')
                plan['after_root'] = [root - 1, gap]
            first = requests[0]['t']
            for i, raw in enumerate(requests):
                lag = float(raw['think_time']) if i else 0.
                if lag < 0 or raw['raw_input_tokens'] != len(raw['blocks']) * PAGE:
                    raise ValueError('Invalid delay or unaligned input')
                row = {k: raw[k] for k in (
                    'blocks', 'raw_input_tokens', 'source_trajectory', 'hour',
                    'pre_alignment_input_tokens', 'provider_cache_read_tokens',
                    'expected_previous_hit_tokens')}
                row.update(root=root, chain=0, index=i, owner=root,
                    reference_fold=fold[0], admission_fold=fold[0],
                    root_offer=0.,
                    first_delay=0., think_time=lag, replica=replica, t=raw['t'] - first,
                    session_id=f'{identity}/copy{replica:02d}/hour{hour:02d}')
                rows.append(row)
                plan['deps'][f'0:{i}'] = [[0, i - 1, lag]] if i else []
            dependencies[str(root)] = plan
            previous = requests
        selection.append(dict(rank=rank, trajectory=identity, replica=replica))
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'selection.json', selection)
    dependencies_path = output / 'dependencies.json'
    with dependencies_path.open('w') as stream:
        json.dump(dependencies, stream)
    deps_sha256 = digest(dependencies_path)
    if published is not None:
        source, reference = published
        if deps_sha256 != reference['dependencies_sha256']:
            raise ValueError(f'Reconstructed DAG differs from the published seed {seed}')
    independent = sorted(int(root) for root, plan in dependencies.items()
                         if 'after_root' not in plan)
    for concurrency in concurrencies:
        target = output / f'c{concurrency}'
        target.mkdir()
        rows_path = target / 'rows.json'
        if published is None:
            offers = make_offers(independent, seed, concurrency)
        else:
            point = reference['load_points'][str(concurrency)]
            offers = read_published_offers(source / point['rows'], point['rows_sha256'], independent)
        with rows_path.open('w') as stream:
            stream.write('[')
            for index, row in enumerate(rows):
                row['root_offer'] = offers.get(row['root'], 0.)
                if index:
                    stream.write(',')
                stream.write(json.dumps(row, separators=(',', ':')))
            stream.write(']\n')
        rows_sha256 = digest(rows_path)
        if published is not None and rows_sha256 != point['rows_sha256']:
            raise ValueError(f'Reconstructed rows differ from the published seed {seed}, concurrency {concurrency}')
        dump(target / 'manifest.json', dict(data_contract=CONTRACT,
            seed=seed, concurrency=concurrency, requests=len(rows),
            hourly_cache_sessions=len(dependencies), trajectory_copies=len(selection),
            warmup_seconds=WARMUP_SECONDS, mean_formal_offer_gap_s=WARMUP_SECONDS / concurrency,
            rows_sha256=rows_sha256, dependencies_sha256=deps_sha256))


def make_offers(independent, seed, concurrency):
    rng = random.Random(seed)
    mean_gap = WARMUP_SECONDS / concurrency
    formal_time = WARMUP_SECONDS
    offers = {}
    for rank, root in enumerate(independent):
        if rank < concurrency:
            offers[root] = rank * float(WARMUP_SECONDS) / concurrency
        else:
            formal_time += rng.expovariate(1 / mean_gap)
            offers[root] = formal_time
    return offers


def read_published_offers(path, expected_sha256, independent):
    if digest(path) != expected_sha256:
        raise ValueError(f'Published Playing row checksum mismatch: {path}')
    roots = set(independent)
    rows = json.loads(path.read_text())
    offers = {row['root']: row['root_offer'] for row in rows if row['root'] in roots}
    if offers.keys() != roots:
        raise ValueError(f'Published Playing rows are missing independent roots: {path}')
    return offers


def train(data, folds, output):
    from ..train_lifecycle import train as fit
    splits = output / 'prior-splits'
    splits.mkdir()
    dump(splits / 'training.json', dict(folds=folds))
    source = output / 'training-source.jsonl'
    with source.open('w') as stream:
        for identity, requests in sorted(data.items()):
            hours = defaultdict(list)
            for row in requests:
                hours[row['hour']].append(row)
            chains = []
            for _, rows in sorted(hours.items()):
                first = rows[0]['t']
                chains.append([dict(t=r['t'] - first, api_time=0.,
                    think_time=r['think_time'] if i else 0.,
                    raw_input_tokens=r['raw_input_tokens'], blocks=r['prior_blocks'])
                    for i, r in enumerate(rows)])
            stream.write(json.dumps(dict(split_group=identity, chains=chains,
                chain_owner_ids=list(range(len(chains))), terminal_observed=True,
                observation_slice=False), separators=(',', ':')) + '\n')
    priors = output / 'runtime/priors/playing'
    fit(source, splits, priors, dataset='playing')
    path = priors / 'training.json'
    meta = json.loads(path.read_text())
    meta['session_boundary'] = ('independent hourly cache sessions; original '
        'trajectories kept together for fold isolation')
    meta['data_contract'] = CONTRACT
    dump(path, meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--aligned', type=Path,
                        help='Reuse checked aligned JSONL.gz files instead of tokenizing')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--concurrency', type=int, nargs='+',
                        help='Load points, defaults to the published plan for each seed')
    parser.add_argument('--published-data', type=Path,
                        help='Reuse hash-verified root offers and priors from the extracted release bundle')
    parser.add_argument('--hours', type=int, default=5)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if min([args.hours, args.workers] + (args.concurrency or [])) < 1:
        parser.error('hours, concurrency and workers must be positive')
    manifests = Path(__file__).resolve().parents[1]
    protocol_path = manifests / 'protocol.json'
    published = json.loads(protocol_path.read_text())['playing']['concurrency']
    released = json.loads((manifests / 'inputs.json').read_text()) if args.published_data else None
    load_points = {}
    for seed in dict.fromkeys(args.seeds):
        concurrencies = args.concurrency or published.get(str(seed))
        if not concurrencies:
            parser.error(f'Seed {seed} requires explicit --concurrency')
        load_points[seed] = list(dict.fromkeys(concurrencies))
        if released is not None:
            reference = released['datasets']['playing'].get(str(seed))
            if reference is None or any(str(c) not in reference['load_points'] for c in concurrencies):
                parser.error(f'--published-data requires published load points for seed {seed}')
    if args.aligned is None and (args.raw is None or args.tokenizer is None):
        parser.error('Supply --aligned, or both --raw and --tokenizer')
    args.output.mkdir(parents=True, exist_ok=False)
    aligned = args.aligned
    if aligned is None:
        for name in ('visible', 'aligned'):
            (args.output / name).mkdir()
        identities = [str(p.relative_to(args.raw)) for provider in (
            'claude_13agents', 'gpt_5agents', 'kimi_5agents')
            for p in sorted((args.raw / provider).glob('agent*')) if p.is_dir()]
        with ProcessPoolExecutor(args.workers) as executor:
            checks = list(executor.map(reconstruct, [(args.raw, args.tokenizer,
                args.output, identity, args.hours) for identity in identities]))
        dump(args.output / 'reconstruction.json', checks)
        aligned = args.output / 'aligned'
    data = {}
    for path in sorted(aligned.glob('*.jsonl.gz')):
        rows = load_rows(path)
        identity = rows[0]['source_trajectory']
        if identity in data:
            raise ValueError('Incomplete or repeated original trajectory')
        validate_aligned(rows, identity, args.hours)
        data[identity] = rows
    folds = make_folds(data)
    if released is None:
        train(data, folds, args.output)
    else:
        expected = released['priors']['playing']
        source = args.published_data / 'runtime/priors/playing'
        target = args.output / 'runtime/priors/playing'
        target.mkdir(parents=True)
        for name, checksum in expected.items():
            if digest(source / name) != checksum:
                raise ValueError(f'Published Playing prior checksum mismatch: {name}')
            shutil.copyfile(source / name, target / name)
    for seed, concurrencies in load_points.items():
        reference = (args.published_data, released['datasets']['playing'][str(seed)]) if released else None
        export(data, folds, args.output / f'processed/playing/s{seed}', seed, concurrencies,
               published=reference)


if __name__ == '__main__':
    main()
