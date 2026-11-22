"""Exercise the serving runner across warmup and formal policy activation."""

import json
from pathlib import Path
import subprocess


def validate(data, output, python, *, profile, env):
    root = Path(__file__).resolve().parents[4]
    output = Path(output).resolve()
    inputs = output / 'input'
    inputs.mkdir(parents=True)

    # One resident page, revisited every two simulated minutes. Keep issuing
    # requests beyond the measurement horizon so the scheduler reaches it.
    rows = [dict(root=0, chain=0, index=i, owner=0, reference_fold=0,
                 root_offer=0, first_delay=0, think_time=120 if i else 0,
                 blocks=[11], raw_input_tokens=1024) for i in range(167)]
    dependencies = {'0': dict(
        deps={f'0:{i}': [[0, i - 1, 120]] if i else [] for i in range(len(rows))},
        first_delays={'0': 0},
    )}
    assets = dict(
        hisim_profile=str(Path(profile).resolve()),
        priors=str((Path(data) / 'runtime/priors/agentx').resolve()),
        datasets={'agentx': {'0': dict(seed=0, rows=str(inputs / 'rows.json'),
                                     dependencies=str(inputs / 'dependencies.json'))}},
    )
    for name, value in [('rows', rows), ('dependencies', dependencies), ('assets', assets)]:
        (inputs / f'{name}.json').write_text(json.dumps(value) + '\n')

    run = output / 'run'
    with (output / 'verify.log').open('w') as log:
        subprocess.run([
            str(python), '-m', 'hohlraum.experiments.run', '--backend', 'hisim',
            '--assets', str(inputs / 'assets.json'), '--dataset', 'agentx',
            '--seed', '0', '--concurrency', '1', '--policies', 'staging', '--output', str(run),
        ], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    case = run / 'staging'

    def read(name):
        return json.loads((case / f'{name}.json').read_text())

    cold = read('initial-cache-check')
    warm = read('warm_boundary')
    activation = read('activation-checks')
    horizon = read('horizon')
    result = read('result')
    manifest = result['manifest']
    budget = warm['formal_budget_bytes_per_rank']
    clock = horizon['observed_clock_s']
    checks = [
        (cold['cold_start'] and cold['initial_ssd_bytes'] == 0
         and Path(cold['ssd_index']).resolve() == case / 'keys.txt', 'empty case-local SSD index'),
        (warm['cache_preserved'] and warm['completed'] > 0 and activation['native_full_admission'],
         'resident warmup with completed requests'),
        (activation['activated'] and activation['same_ssd_backend']
         and activation['stage_limit_tokens'] > 0 and activation['restored_requests'] > 0
         and activation['resident_hashes'] > 0 and activation['resident_hashes_missing_sli'] == 0
         and activation['resident_hashes_missing_hts'] == 0, 'staging activation and resident history'),
        (manifest['formal_budget_reset'] and budget > 0
         and horizon['progress']['budget_bytes'] == budget, 'formal budget reset'),
        (manifest['warmup_seconds'] == 3600 and manifest['formal_seconds'] == 14400
         and horizon['horizon_s'] == 18000 and clock >= 18000, 'full simulated measurement window'),
        (result['completed'] > 0 and result['throughput'] > 0
         and horizon['progress']['completed'] > warm['completed'], 'formal request completion'),
    ]
    for passed, label in checks:
        if not passed:
            raise RuntimeError(f'HiSim validation failed at {label}. See {case}')

    summary = dict(status='passed', backend='hisim', checks=[label for _, label in checks],
                   warmup_completed_requests=warm['completed'], formal_completed_requests=result['completed'],
                   observed_clock_s=clock, formal_budget_bytes_per_rank=budget)
    report = output / 'validation.json'
    report.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'HiSim end-to-end validation passed. See {report}')
