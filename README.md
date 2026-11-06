# Hohlraum

## One-command reproduction (HiSim)

Extract the complete reproduction package with `hohlraum-paper/` and
`hohlraum-data/` beside each other. On Linux with Python 3.10+ and a C++17
compiler, run from the source directory:

```bash
./reproduce.sh
```

This prepares an isolated Python environment if needed, uses the bundled model
configuration, predictor and dataset-specific priors, validates inputs, generates
all local paths automatically, and runs both datasets, three seeds and three load
levels serially. Each point includes unlimited-budget, 640 GB, admission-only and
fixed 30 GB staging policies, with one-hour lifecycle warmup and four-hour formal
measurement. No `assets.local.json` editing or recalibration is required.
First-time dependency installation requires network access. HiSim uses dummy
model loading; full inference model weights are not required for this workflow.

For one representative point, or dynamic capacity:

```bash
./reproduce.sh --datasets playing --seeds 272016 --loads medium
./reproduce.sh --datasets playing --seeds 272016 --loads medium --dynamic-capacity
```

Outputs go to `../hohlraum-results/<timestamp>/`, including `summary.csv`,
`summary.json`, the resolved configuration, and per-case logs/results. Existing
output directories are never overwritten. `--check-only` validates data and
configuration without installing or launching the runtime. Set `PYTHON` to reuse
an existing compatible interpreter; otherwise dependencies are prepared in
`.venv-reproduce`. `--data` is optional when the bundle is beside the repository.

A Git clone contains source only: it cannot run without the separately distributed
data/runtime bundle. The complete reproduction package is the one-command entry;
no public download URL is assumed or fabricated. Native GPU experiments and custom
profiles remain available through the advanced entry below.


Hohlraum manages DRAM residency to trade additional SSD reads against avoidable
SSD writes under an endurance budget. This repository contains the shared
staging implementation, bundled SGLang/HiSim integration, dataset reconstruction,
and a separate CPU-only policy replay.

## Entry points

| Path | Purpose |
| --- | --- |
| `hohlraum/core/staging_policy.py` | Candidate selection, HTS constraints, ranking, admission and dynamic threshold |
| `hohlraum/core/capacity_controller.py` | Departure distributions, read protection and capacity decisions |
| `hohlraum/core/sli.py`, `hohlraum/core/cpp/` | Session history and lifecycle scoring |
| `hohlraum/core/staging_storage.py` | Data movement, asynchronous writes and staging-plus-SSD interface |
| `hohlraum/experiments/preprocessing/` | Request reconstruction, hashes and input validation |
| `hohlraum/experiments/run.py` | Three public experiment entries |
| `hohlraum/experiments/backends/hisim/`, `hohlraum/experiments/backends/sglang/` | HiSim and native execution adapters; shared client protocol |
| `hohlraum/experiments/backends/policy_replay/reproduce.sh` | Fixed-stream policy sweeps, counters and plots |

`core/cpp/` contains `session_level_index.hpp` (Session-Level Index, SLI),
`lifecycle_history.hpp` (lifecycle history and scoring), and `sli_bridge.cpp`
(the Python/C++ interface).

SGLang/HiSim source is bundled in `third_party/sglang`; no patch step is needed.
`third_party/UPSTREAM.json` records its base revision. Integration hooks attach
Hohlraum, intercept host leaf eviction and report mixed staging/SSD read sources
to the native transfer estimator. They do not introduce a separate timing model.
Without `HOHLRAUM_CONFIG`, the optional backend is not attached; `enabled=false`
uses only the common SSD write-budget wrapper. Upstream documentation and licenses
remain in the bundled source.

Experiment internals are grouped by responsibility:

```text
experiments/
  run.py                 # public --backend entry
  common/                # workload, backpressure, checkpoints and diagnostics
  backends/
    hisim/               # simulator case preparation and execution
    sglang/              # real engine launcher, client and queue feedback
    policy_replay/       # fixed-stream policy experiments
  preprocessing/         # dataset reconstruction and validation
  protocol.json          # supplied experimental defaults
```

## Install and check

From the repository root, with Python 3.10+, GNU Make and a C++17 compiler:

```sh
python3 -m venv .venv
source .venv/bin/activate
bash install.sh minimal
# Install one serving backend after satisfying its native dependencies:
bash install.sh hisim
# Or: bash install.sh sglang
python3 -m hohlraum.experiments.run --help
```

Developer correctness checks live under `tests/` (`smoke.py`, `check_sli.py`,
`check_experiment_protocol.py`, `check_capacity.py`); they are not additional experiment entrances
or performance benchmarks.
Backend installation retains the upstream CUDA/compiler/Rust requirements; see
[HiSim installation](third_party/sglang/tools/sglang-simulator/README.md) and
[SGLang installation](third_party/sglang/README.md). `install.sh MODE --dry-run`
shows installation commands.

## Run an experiment

Three public entries use the same defaults and supporting code:

```sh
cp assets.example.json assets.local.json
# Edit local data/model/SSD paths once; seed, load and client defaults are built in.
python3 -m hohlraum.experiments.run --backend sglang --dataset playing --load medium --output /path/to/new-run --check-only
python3 -m hohlraum.experiments.run --backend sglang --dataset playing --load medium --output /path/to/new-run
python3 -m hohlraum.experiments.run --backend hisim --dataset playing --load medium --output /path/to/sim-run
python3 -m hohlraum.experiments.run --backend policy --output /path/to/policy-run
```

For native runs, SGLang discovers the GPU devices exposed to it. The entry also
records NVIDIA card names, UUIDs and memory in `hardware.json`, together with
`CUDA_VISIBLE_DEVICES`. This physical inventory does not override visibility or
select GPUs. It does not automatically tune TP, DRAM/cache sizes, concurrency,
write mode or prefetch settings. These remain explicit defaults/configuration;
changing them changes the experimental conditions. If `nvidia-smi` is unavailable,
the inventory says so rather than guessing a card.

HiSim uses the target hardware and predictor specified by its profile, not the
GPU installed on the machine running the simulator. Card detection cannot replace
that profile or select a latency model automatically.

Before either serving backend starts, the entry validates request fields, page
lengths, timing and the dependency graph, plus supplied reference checksums.
Valid data is reused. Missing or invalid data is rebuilt using the selection's
recorded `rebuild` recipe and validated again; only then is it published and the
experiment launched. `data_preparation.json` records reuse/rebuild and validation.
No separate preprocessing command is required for normal experiment execution.

The asset manifest may supply a `prepared-source` recipe (an immutable rows file,
its `rows_sha256`, and for AgentX the DAG and `dependencies_sha256`), or an explicit
`playing-v7` raw reconstruction recipe (`raw`, `tokenizer`, `copies`, `seed`,
`mean_arrival`). The latter must describe that exact selection, not silently
replace the balanced serving pool with uniform replicas. If raw inputs or the
required converter are unavailable, the entry fails explicitly. The AgentX raw
exporter is still not bundled; an immutable prepared source bundle supports
automatic cache restoration for that dataset. Without a reference checksum,
structural validation is reported as such, not as a byte-identical data match.

The default seed is 272015; `--seed`, `--load` and `--concurrency` override the
protocol defaults. Serving runs automatically warm up, save client progress,
then compare `unlimited`, `budget640` and `admission-dynamic` from the same
checkpoint. Add `--policies unlimited budget640 admission-dynamic staging` for
fixed staging (default 30 GB; `--staging-gb` overrides it). Add `--dynamic-capacity` to enable capacity adjustment for the `staging` policy.
The initial size is `--staging-gb`; fixed capacity remains the default. Both
backends use `capacity_control` in `protocol.json`.

Capacity feedback runs every 5 minutes and retains up to 8 windows at the same
allocation. The controller uses byte-weighted departure TTD and rejection-slack
distributions. For an additional 10-minute wait, an avoidable-write fraction of
0%, >0–<1%, 1–5%, 5–10%, or >=10% selects 0, 1, 10, 15, or 30 GB expansion, respectively.
GB values are decimal and aggregate across TP ranks. The read minimum remains
25% of total DRAM. Upper bounds and page rounding can reduce the applied step.

This version uses **capacity failures**, not read-blocking latency, as its
pressure signal: native `host_capacity` prefetch failures and the existing
write-back receiver-space failure callback. Other prefetch failure reasons are
logged but are not automatically attributed to capacity. Failures must occur
in two consecutive complete windows before contraction. The fraction of rejected
bytes with slack >=10 minutes uses the same 0%/1%/5%/10% boundaries and
0/1/10/15/30 GB contraction tiers. No slack samples initially selects zero;
continued unimproved pressure after two observation windows escalates this to
1 GB, so a zero tier cannot indefinitely suppress read protection. After two observation windows,
persistent failures without improvement increase the previous contraction by
one tier, capped at 30 GB. Improvement means fewer mean failed-prefetch requests
or fewer mean failed-write-back tokens per window than before the last shrink.
A clean window breaks persistence. No-pressure contraction is disabled.

Every actual capacity change clears distributions, rejection-episode statistics,
window history and pressure streaks; a zero step or failed resize does not.
Only the last shrink's tier and pressure summary survive for comparison.
Late write acknowledgements from an older allocation cannot replenish the new
history. Expansion requires at least four complete windows (20 minutes) at the
current allocation, including the initial allocation; up to eight windows smooth
the distributions when capacity stays unchanged. Every shrink
restarts a four-window expansion cooldown, while another shrink remains possible
after two full observation windows. Budget exhaustion sets staging to zero.
SLI training, admission statistics, outstanding read handles and write-budget
accounting are not reset by capacity changes.

Five-minute feedback and eight-window history are stated in the paper; tier
boundaries, reset semantics and escalation are the explicitly configured
implementation choices for this version. They are not claimed as paper-specified
constants. `*.capacity.jsonl` records decisions, byte histograms and allocation
epochs. Pending SSD wait is diagnostic only and must not be interpreted as
unoverlapped read latency. Slack starts at the first observed rejection in the
current allocation/episode (window resolution plus entry/exit observations),
not an inferred exact threshold-crossing time. Legacy proportional-step configs
are rejected rather than silently reinterpreted.

Resizing first updates both quotas, then evicts from the shrinking region through
existing policy/native paths. SSD-backed read victims are released; other read
victims enter the newly enlarged staging region. Locked pages and in-flight
writes may delay physical reclamation. This eviction-based resizing is an
explicit simplification of the paper's metadata-transfer description. The
existing budget-exhaustion return path remains separate. Control does not alter
native scheduling or simulation time. Dynamic-capacity performance has not yet
been validated against the paper.

The SLI library is
built automatically when admission is enabled. `--check-only` validates assets
and the dependency graph; it is not a GPU smoke test. The policy entry's check
also builds and validates its frozen numeric inputs.

`assets.local.json` contains machine-specific paths, not a request to retune
concurrency or calibrate hardware. Prepared selections must be supplied for each
requested seed. Native server settings are SGLang `ServerArgs` in JSON, including
model, TP, DRAM, write/prefetch mode and transfer settings. The native paired
runner currently supports the file SSD backend and a local client/server only.
It launches a fresh engine and a unique empty SSD directory for each phase,
stops that engine before the next phase, and removes only its own disposable KV
files. Each formal phase receives a fresh budget; warmup KV is never reused.

HiSim is an optional trend-checking path. Its profile contains native simulator
settings and a fitted predictor database. This package does not currently bundle
the measured profile, predictor or all prepared inputs; missing assets cause an
explicit error rather than silently substituting a model. Native runs do not
need HiSim calibration. `hohlraum/experiments/backends/hisim/profile.example.json`
is a profile format reference, not a calibrated default. A short run checks
execution only; it may miss budget exhaustion or long-gap reuse.

Serving cases produce `analysis.json` and `analysis.txt` after completion, along
with raw results. Re-run diagnostics with:

```sh
python3 -m hohlraum.experiments.run --analyze --output /path/to/case
# Native result/analysis files are inside each phase's client/ directory.
```

Diagnostics report observed SSD usage and available prefetch counters, and
separate runtime errors from departures from reference conditions. Missing
measurements are reported as unavailable, never as zero or inferred success.
Native request-level SSD attribution is not yet complete; the report does not
claim an exact prefetch realization rate from completion counters alone.

### Protocol and comparison

`hohlraum/experiments/protocol.json` records current evaluation settings:
seeds 272015–272017; AgentX concurrency 6/13/22; Playing 20/32/40, with 34/35
as comparison points. These are evaluation points, not universal 3/5/10-second
TTFT operating points. Both serving backends use the same policy configuration builder and explicit
client settings from this protocol, including admission threshold/update periods,
queue limits and idle timeout. Server settings such as DRAM capacity, write-back versus
write-through, prefetch mode and asynchronous transfers come from the profile;
record them explicitly for each comparison.

Each case uses a one-hour warmup followed by four measured hours. Formal runs
start a fresh engine with empty HBM, DRAM and SSD caches, preserving client
session progress and giving each constrained case a fresh 640 GB decimal write
budget. Use the same checkpoint for paired cases. With a 3.84 TB SSD this budget
corresponds to 1 DWPD over four hours; capacity for DWPD accounting is distinct
from a runtime storage-capacity constraint.

The external client uses a 10-second engine waiting-age threshold and at most
eight waiting/reserved requests; running requests do not occupy those eight slots.
Session dependencies and recorded delays are retained, with a 20-minute idle
slot-release rule. External waiting is included in TTFT. The policy scoring
clock excludes client-ready-to-completion time without changing native LRU or
simulator time. Do not compare runs with different client rules as a policy-only
comparison.

Check `exit.json`, the full horizon in `horizon.json`, and `result.json` before
comparing runs. `progress.jsonl` and `budget_exhausted.json` (when exhausted)
locate changes over time. Throughput counts completed input tokens per second;
goodput counts those satisfying the 10-second end-to-end TTFT threshold. Report
both, P90 TTFT, writes and budget exhaustion time, using the same measurement
window. Preserve the input manifest, source revision and generated configurations.

### When to expect a benefit

Hohlraum's benefit depends on a binding endurance budget and reuse that cannot
be served entirely by memory. Avoiding writes must preserve useful future KV
writes sufficiently to offset the additional SSD-read cost. More concurrency
alone does not guarantee this condition, and staging can hurt when memory hits
are lost without enough write savings.

When results differ, compare these quantities on the same inputs and time window:

- **Reuse and memory pressure:** theoretical contiguous-prefix reuse, actual
  HBM/DRAM hits and SSD hits. Session concurrency is not the number of active
  requests; think time, context resets and compression affect the working set.
- **Write budget:** when it exhausts and whether subsequent reusable KV cannot
  be persisted. Exhaustion alone does not imply a throughput loss: useful data
  may already be in SSD or still fit in memory.
- **Prefetch realization:** distinguish SSD lookup hits from data actually used
  at execution. Admission limits, receive-space shortage or eviction before
  execution can turn potential SSD hits into recomputation.
- **Read service and queueing:** report SSD read volume, effective bandwidth,
  transfer overlap and waiting. Successful prefetch does not make reads free;
  reads can still stall execution. Excessive pressure can also hurt the
  unlimited-budget baseline.

The current reproduction target uses **8 GB/s aggregate SSD reads across TP=2**.
Do not enter this aggregate as a per-rank value. Record the units and TP scope
of each profile field. This is a target-platform setting, not a guarantee that
any SSD, container or filesystem sustains that rate. Validate measured profiles
separately; vendor sequential bandwidth is not an engine-level measurement.

Use paired policies with identical bandwidth, transfer modes, client settings,
inputs and cache initialization. Publish per-seed results, including regressions;
matching a gain after changing these settings is not by itself numerical
reproduction of the paper. A low-gain run may reflect a different operating point
or an implementation issue; the measurements above help distinguish them.

## Dataset reconstruction

Playing source: [machine-playing-paper](https://huggingface.co/datasets/nacloos/machine-playing-paper).
Use one data preparation entry; each file owns one complete responsibility:

```text
preprocessing/
  run.py                       # validation, reuse and rebuild orchestration
  rebuild_playing_trace.py      # Playing parsing, tokenization and reconstruction
  tool_definitions.json         # small reconstruction schema resource
  requirements.txt

../hohlraum-data/               # external data bundle, not part of Git
  raw/
  tokenizer/
  processed/
  inventory.json
```

The external data bundle includes original logs in `../hohlraum-data/raw/playing/` and
`../hohlraum-data/raw/agentx/traces.jsonl`, the pinned tokenizer in `../hohlraum-data/tokenizer/`,
and prepared inputs in `../hohlraum-data/processed/`. Playing includes selections for seeds 272015–272017; its balanced 195-session
candidate pool is generated in memory when rebuilding. AgentX includes the same three
seeds, each with 150 root tasks and its dependency DAG. `assets.example.json`
points to these inputs using repository-relative paths; run commands from the
repository root. Model, SSD and simulator-profile paths still require local setup.

Payloads live outside this repository to keep experiment data out of source control.
**A Git clone alone does not include them:** distribute the entire
`hohlraum-data/` directory alongside the source repository. Set `HOHLRAUM_DATA_DIR`
to override the preprocessing data root, or pass explicit input/output arguments.
`../hohlraum-data/inventory.json` lists payload sizes and SHA-256 hashes. The tool definitions
in `preprocessing/tool_definitions.json` are a verified partial tool set, not full hidden
request recovery. Original ordering and reconstruction assumptions are unchanged.

```sh
pip install -r hohlraum/experiments/preprocessing/requirements.txt
python3 -m hohlraum.experiments.preprocessing.run build --help
python3 -m hohlraum.experiments.preprocessing.run build --dataset playing --seeds 272015 272016 272017 --output /path/to/rebuilt-playing
python3 -m hohlraum.experiments.preprocessing.run validate --data ../hohlraum-data/processed/playing
python3 -m hohlraum.experiments.preprocessing.run validate --data ../hohlraum-data/processed/agentx
```

The default output is `../hohlraum-data/processed/playing/`; use a fresh output directory to
avoid replacing supplied inputs. The default `balanced` layout reconstructs each
original trajectory once, retaining requests through relative time 14,400 seconds
plus the first successor. Shared-padding estimation still uses first requests
from all hours. It then creates 5 replicas of each Claude trajectory and 13 of
each GPT/Kimi trajectory, giving 65 candidates per provider. Canonical order is
lexicographic provider/trajectory followed by ascending replica number.
`random.Random(seed).sample(range(195), 65)` selects the final arrival order.
No candidate-pool payload is required. Per-trajectory reconstruction intermediates
are stored under `_trajectories/`; `--resume` can reuse matching intermediates.

The historical deterministic block-ID namespace and arrival RNG (20261013) are
preserved for input compatibility. The selected replica's runtime root identity
isolates cache pages; `source_blocks` is not used or emitted. Initial arrival slots
use the original schedule transformation (t*6 for t<=600, otherwise t+3000).
The three development seeds were screened for at least 25 high-theoretical-hit-rate
sessions among the first 30. New seeds are sampled directly, without this filter.
These are conditional workloads, not an unconditioned random sample.

To generate a different seed, pass `--seed 123` instead of `--seeds ...`.
`--layout uniform --copies 10` retains the older full-trajectory export mode;
it is not the current balanced experiment input. The experiment entry consumes
prepared selections and their rebuild recipes from `assets.local.json`.
AgentX raw conversion remains unpackaged; its supplied rows and dependency DAG
allow direct replay but do not constitute an end-to-end raw reconstruction tool.

Visible messages retain their original order and use the pinned Qwen3-32B
tokenizer (`9216db5781bf21249d130ec9da846c4624c16137`). Each trajectory is treated
as one cache session across hourly restarts; replicas have independent namespaces.
For each trajectory, the mean positive first-request length gap across hours
becomes a fixed shared padding prefix. Within an hour, unchanged history retains
its padding; new gaps are appended. Compression retains padding only in the
unchanged visible prefix. Hourly restart drops history padding but retains the
fixed prefix. Lengths are not truncated if reconstruction exceeds logged lengths.
Cumulative prefix hashes use 1024-token blocks.

These are explicit reconstruction assumptions, not recovered hidden text.
In particular, missing Claude startup content may include changing information;
shared padding can overestimate cross-hour reuse. Length gaps also include
tokenizer differences. Recorded inter-observation delays are replayed after
completion and are not verified pure think time; they may contain original
service time. Preserve these assumptions when comparing results.

The current serving pool has 65 copies per provider (13 Claude trajectories ×5;
5 GPT and 5 Kimi trajectories ×13 each); a run samples 65 entries. The builder's
uniform `--copies` option alone does not create this balanced selection. Use the
prepared pool/selection and its manifest. Development seeds were selected using
workload characteristics, not an unconditional IID draw. AgentX samples 150
roots with their child DAGs; its raw-to-DAG exporter is not yet packaged, so supply
prepared rows and dependencies. Do not mix AgentX requests into Playing runs.
Converted inputs are Git-ignored assets, distributed separately where available.

## Other validation paths

CPU-only policy replay uses frozen numeric inputs and priors, separate from the
serving workload. It measures contiguous prefix hits and writes, not TTFT or
throughput:

```sh
bash hohlraum/experiments/backends/policy_replay/reproduce.sh \
  --data /path/to/data/paper --output /path/to/results --check-only
bash hohlraum/experiments/backends/policy_replay/reproduce.sh \
  --data /path/to/data/paper --output /path/to/results --jobs 2
```

The wrapper builds/tests and installs plotting dependencies. `--no-plots` skips
plots; `--resume` requires identical inputs/configuration. Outputs include
`policy_results.csv`, `table3.csv`, figures and manifests. Optional
`--reference /path/to/counters.json` checks counters; without it, successful
execution is not a numerical match check.

The native client now uses the shared dependency source, checkpoint restore and
10-second/8-waiting-request gate. A local experiment adapter reads actual engine
queue state, acknowledges received requests and sends early client-ready events
to SLI. It does not modify the native scheduling or timing model. The adapter
requires the experiment server entry; arbitrary HTTP endpoints are not treated
as equivalent. CPU protocol and HTTP transport checks pass; this adapter still
requires GPU validation before claiming numerical agreement with HiSim.

```sh
python3 tests/check_experiment_protocol.py
```

The SLI implementation omits CLOCK reclamation and persistent history snapshots.

The capacity controller also reports a coarse, observation-only SSD-wait idle
signal: native scheduling selected no batch, no submitted batch result remains
unconsumed, and this scheduling pass encountered a request waiting on an SSD
read. Overlapping waiting requests count time only once. The observer uses the
existing runtime clock, does not synchronize CUDA, and does not change capacity
decisions. An unconsumed result conservatively counts as in-flight; the metric
is therefore not a complete measurement of GPU idle time or a causal proof that
SSD alone prevented execution. Per-window idle seconds, fraction and longest
interval are recorded in `*.capacity.jsonl`; cumulative seconds appear in
capacity statistics. This signal has no 2-second threshold.
