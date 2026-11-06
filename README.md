# Hohlraum

Hohlraum manages competing DRAM residency roles to trade SSD read costs against
avoidable SSD writes under an endurance budget. The implementation integrates
with SGLang; a CPU-only policy replay is also provided.

## Run

Use Linux, Python 3.10+, a C++17 compiler, and a CUDA environment supported by
[the bundled SGLang](third_party/sglang/README.md). The native defaults target
Qwen3-8B on two H20 GPUs, approximately 300 GB of host KV memory and an SSD.
The serving data package contains processed requests, dependency graphs and
scoring priors. The entry downloads it to `data/`; select an existing directory
with `--data`. Original logs and preprocessing tokenizers are not required.

```bash
./run.sh
```

The entry prepares Python dependencies, downloads model weights if necessary,
and runs native SGLang experiments. To use existing weights or an SSD mounted
elsewhere:

```bash
./run.sh --model /path/to/Qwen3-8B --ssd /path/on/ssd
```

Defaults use both datasets, seeds 272015–272017, three load levels, and four
policies: unlimited writes, budget-constrained baseline, admission-only and
30 GB staging. Native and simulated serving share the same experiment protocol:
Playing concurrency 20/32/40 and AgentX 6/13/22; one-hour lifecycle warmup followed
by four measured hours. Each policy starts with empty KV caches and a fresh
640 GB write budget, except the unlimited baseline. External waiting counts
in TTFT. Results and resolved configurations are saved under `results/`.

## Other experiments

```bash
./run.sh --datasets playing --seeds 272016 --loads medium --concurrency 35
./run.sh --policies staging --staging-gb 30 --dynamic-capacity
./run.sh --backend policy
./run.sh --backend hisim
```

Policy replay uses separate fixed event streams (seeds 20261012–20261014),
needs no GPU, and measures cache behavior rather than inference throughput. HiSim is an optional simulator and
requires its own calibrated profile and predictor. Neither is needed for native
execution. `./run.sh --help` lists options; `--check-only` validates inputs and
configuration without launching serving.

Inputs derive from [Playing](https://huggingface.co/datasets/nacloos/machine-playing-paper)
and [AgentX](https://inferencex.semianalysis.com/agentx). Input checksums and source
references are recorded in `hohlraum/experiments/inputs.json`; preprocessing code
is in `hohlraum/experiments/preprocessing/`. Serving replays preserve trace lengths
and prefix sharing using synthetic token IDs; they do not evaluate model quality.

Other seeds and workloads can be used with their prepared inputs. Report the
chosen workload and configuration: gains depend on a binding write budget,
reuse that exceeds memory coverage, and SSD reads that can be served effectively.
A workload that fits in memory or never exhausts its budget need not benefit;
SSD bandwidth, prefetch behavior and queueing affect the trade-off. Serving seeds were selected
for Playing early-session reuse: at least 25 of the
first 30 sessions exceed 98.5% theoretical reuse. They are not unconditional
random samples.

SGLang/HiSim source and its licenses are in `third_party/sglang`; the imported
revision is recorded in `third_party/UPSTREAM.json`. Hohlraum is Apache-2.0 licensed.
