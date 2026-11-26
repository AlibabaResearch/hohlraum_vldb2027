# Hohlraum

This artifact accompanies the Hohlraum paper. It provides the Hohlraum
implementation, prepared reference workloads, and commands for evaluating
serving performance and cache policy. Hohlraum divides host KV memory between
a read cache and a staging region that can avoid SSD writes.

## Run a reference serving workload

Use Linux, Python 3.10 or newer, a C++17 compiler, Ninja, and OpenSSL development
headers. The default configuration targets Qwen3-8B on two NVIDIA H20 GPUs with
about 300 GB of host KV memory. Point `--ssd` at a directory on the SSD used for
the experiments. The runner installs dependencies and downloads model weights
on first use when needed.

```bash
./run.sh --ssd /path/on/ssd
```

To run one reference load with local model weights, select a dataset, seed,
and load preset.

```bash
./run.sh --model /path/to/Qwen3-8B --ssd /path/on/ssd \
  --datasets playing --seeds 94266 --loads medium
```

The runner checks and extracts the serving bundle from `data/bundles/`. It
contains processed requests, dependencies, arrival plans, and scoring priors.
Use `--data` to select another prepared input directory. The three supplied
seeds select reference workloads. The concurrency values below are starting
points for the low, medium, and high load presets. Use `--concurrency` to set
the session limit for your hardware. For an override outside the listed
concurrencies, the runner uses that seed's medium arrival plan.

| Dataset | Reference seed | Low | Medium | High |
|---|---:|---:|---:|---:|
| AgentX | 94266 | 6 | 28 | 36 |
| AgentX | 94267 | 3 | 30 | 38 |
| AgentX | 94268 | 13 | 32 | 36 |
| Playing | 94266 | 19 | 36 | 38 |
| Playing | 94267 | 19 | 34 | 38 |
| Playing | 94268 | 19 | 34 | 36 |

Each load point compares unlimited writes, a 640 GB baseline, admission-only,
and fixed 30 GB staging. To run dynamic capacity from 30 GB, add
`--policies staging --staging-gb 30 --dynamic-capacity` (this uses the unified
marginal-throughput controller by default; pass `--capacity-controller tiered`
to select the legacy tiered controller). See `./run.sh --help` for workload and
runtime options.

Each run starts with empty caches and uses full admission during a one-hour
warmup. The four-hour measurement retains the warmup KV and resets the write
budget. Writes during the budget cutover are reported separately from budgeted
writes and included in total physical writes. Staging writes from the head and
discards from the tail. Client backpressure uses a 10-second median engine
queue wait and a 16-request engine waiting-queue limit.

Results and resolved settings are saved under `results/`. Use `summary.csv` to
compare input throughput, 10-second goodput, and P90 TTFT. P90 TTFT includes
client-side queueing. Goodput uses engine timing when available and records its
timing scope. Native SSD write counters are in each run's `client/server-final.json`.
Performance depends on how much reuse falls outside DRAM, when the SSD budget
runs out, and how quickly SSD reads can be served.

## Verify with HiSim

```bash
./run.sh --verify
```

The CPU-only check uses the bundled simulator profile and predictor with a
fixed synthetic workload. It checks the empty SSD index, resident warmup,
staging activation, formal budget reset, and request completion through the
measurement window. It saves `validation.json` and logs under the results
directory. Initial dependency setup may need network access. The performance
measurements above require real GPUs and model weights.

## Generate the policy replay figures

```bash
./run.sh --policy-replay
```

This uses a C++17 compiler and Make. It builds fixed request streams from the
bundled seeds and their medium-load presets. The replay applies the reference
arrival plans and dependency delays with instantaneous request completion.
For each stream, it computes the smallest DRAM capacities at which LRU serves
at least 80%, 90%, or 99% of theoretical prefix reuse (C80, C90, and C99).
All policies use the same requests and timestamps. It writes hit–write frontiers, policy
ablations, and capacity-sensitivity plots in PNG and PDF under `figures/`,
following the evaluation procedures of Figures 8–10 in the paper. The replay
measures logical SSD writes and prefix hit rates.

The first run extracts the serving bundle and installs NumPy and Matplotlib
if needed. Use `--jobs` to adjust CPU parallelism (default four) and `--output`
to set the results directory. Use `--data` to point to an extracted copy of
the bundled workloads, containing `processed/` and `runtime/priors/`.

## Rebuild the Playing inputs

Input provenance and checksums are recorded in
[the input manifest](hohlraum/experiments/inputs.json). Serving uses synthetic
token IDs with trace-derived request lengths and prefix sharing. Playing uses
independent hourly cache sessions with dependencies across hours. Its
reconstruction maps GPT and Kimi cache counts into Qwen token units and uses
Claude's original cache counts. Verified appends preserve matching complete
pages. The last padded page is specific to its request. The source inputs come
from [Playing](https://huggingface.co/datasets/nacloos/machine-playing-paper) and
[AgentX](https://inferencex.semianalysis.com/agentx).

To rebuild Playing from its original logs with the bundled arrival plans and
priors, first run the verification command to extract the bundle. Use a Python
environment with the [preprocessing dependencies](hohlraum/experiments/preprocessing/requirements.txt)
and a local copy of the Qwen3 tokenizer.

```bash
python3 -m hohlraum.experiments.preprocessing.prepare_hourly_playing \
  --raw /path/to/playing-logs --tokenizer /path/to/qwen3-tokenizer \
  --output /path/to/rebuilt-playing --seeds 94266 94267 94268 \
  --published-data data/serving-94266-94268
```

Use the generated request rows, dependency graphs, and priors together.
To serve the rebuilt inputs, pass `--datasets playing --data /path/to/rebuilt-playing`.

Hohlraum is licensed under Apache-2.0. SGLang and HiSim are bundled with their
original licenses and notices. [third_party/UPSTREAM.json](third_party/UPSTREAM.json)
records their source revisions and integration changes.
