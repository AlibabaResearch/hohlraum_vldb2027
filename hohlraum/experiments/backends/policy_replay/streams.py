"""Produce fixed-timestamp policy streams from the published serving inputs."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

from hohlraum.experiments.common.workload import (
    PAGE,
    TraceSource,
    normalize,
    reference_model_fold,
    uniform_warmup_offers,
)
from hohlraum.experiments.preprocessing.validation import ensure_inputs
from hohlraum.experiments.resources import digest
from hohlraum.experiments.run import read_serving_protocol, session_concurrency


INPUT_MANIFEST = Path(__file__).resolve().parents[2] / "inputs.json"


def _verify(data, name, checksum, checked):
    if name not in checked:
        path = data / name
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Published serving input missing or checksum mismatched: {path}")
        checked[name] = checksum
    return data / name


def _write_stream(rows, plans, *, dataset, concurrency, idle_timeout, training, output):
    queued = []
    root_offers = {}
    for serial, row in enumerate(rows):
        root = row["root"]
        expected_offer = root_offers.setdefault(root, row["root_offer"])
        if expected_offer != row["root_offer"] or not math.isfinite(expected_offer) or expected_offer < 0:
            raise ValueError(f"Invalid root offer for {dataset} root {root}")
        meta = dict(row, serial=serial, reference_fold=reference_model_fold(row, dataset, training))
        request = SimpleNamespace(
            rid=serial,
            sampling_params=SimpleNamespace(custom_params={"simulation": {"agentx": meta}}),
        )
        queued.append((0, serial, request))
    source = TraceSource(queued, concurrency, plans)
    source.idle_timeout = idle_timeout

    session_ids, block_ids = {}, {}
    parents = []
    count = references = 0
    last = 0.
    output.mkdir(parents=True)
    events_path, baseline_path = output / "events.tsv", output / "baseline.tsv"
    with events_path.open("w") as events, baseline_path.open("w") as baseline:
        while count < len(rows):
            when = source.next_time()
            if when is None:
                raise ValueError(f"{dataset} dependencies left {len(rows) - count} requests unscheduled")
            for request in source.pop_due(when):
                ready = source.arrivals[request.rid]
                if ready < last:
                    raise ValueError(f"{dataset} request times are out of order")
                row = request.sampling_params.custom_params["simulation"]["agentx"]
                root, chain = row["root"], row["chain"]
                session = session_ids.setdefault((root, chain), len(session_ids))
                length = row["raw_input_tokens"]
                if length < 1 or len(row["blocks"]) != length // PAGE:
                    raise ValueError(f"Invalid complete pages for {dataset} request {request.rid}")
                path = []
                parent = -1
                for raw in row["blocks"]:
                    # Serving token IDs and SLI both isolate input blocks by root.
                    key = (root, raw)
                    block = block_ids.setdefault(key, len(parents))
                    if block == len(parents):
                        parents.append(parent)
                    elif parents[block] != parent:
                        raise ValueError(f"Prefix identity {raw} has two parents in {dataset} root {root}")
                    path.append(str(block))
                    parent = block
                stamp = format(ready, ".17g")
                events.write(f"{stamp}\t{session}\t{row['reference_fold']}\t{length}\t{','.join(path)}\n")
                # Retain the baseline reader's compaction column, which this
                # replay ignores. Both readers consume the same complete pages.
                baseline.write(f"{stamp}\t{session}\t0\t{length}\t{len(path)}")
                if path:
                    baseline.write("\t" + "\t".join(path))
                baseline.write("\n")
                count += 1
                references += len(path)
                last = ready
                # Use this same causal stream for every policy. Finish at the
                # ready timestamp, since this replay omits inference and queues.
                source.finish(request, ready)

    if len(block_ids) >= 2**32 or len(session_ids) >= 2**32:
        raise ValueError(f"{dataset} replay exceeds the block or session ID range")
    return {
        "blocks": len(block_ids), "sessions": len(session_ids), "requests": count, "refs": references,
        "end_seconds": max(1., math.nextafter(last, math.inf)), "last_ready_seconds": last,
        "active_root_limit": concurrency, "idle_release_s": idle_timeout,
        "roots": len(root_offers), "independent_roots": sum("after_root" not in p for p in plans.values()),
        "stream_sha256": {path.name: digest(path) for path in (events_path, baseline_path)},
    }


def prepare_streams(data: Path, output: Path):
    """Build complete medium-load replay streams for all published serving seeds."""
    manifest = json.loads(INPUT_MANIFEST.read_text())
    protocol = read_serving_protocol()
    checked, cases = {}, []
    for dataset in ("agentx", "playing"):
        priors_name = f"runtime/priors/{dataset}"
        for name, checksum in manifest["priors"][dataset].items():
            _verify(data, f"{priors_name}/{name}", checksum, checked)
        training = (json.loads((data / priors_name / "training.json").read_text())
                    if dataset == "playing" else None)
        for seed in protocol["seeds"]:
            source = manifest["datasets"][dataset][str(seed)]
            concurrency = session_concurrency(protocol, dataset, seed, "medium")
            selected = source["load_points"][str(concurrency)]
            rows_path, deps_path = data / selected["rows"], data / source["dependencies"]
            _, preparation = ensure_inputs(dataset, {
                "seed": seed, "rows": rows_path, "rows_sha256": selected["rows_sha256"],
                "dependencies": deps_path, "dependencies_sha256": source["dependencies_sha256"],
            }, seed)
            validation = preparation["validation"]
            checked[selected["rows"]] = validation["rows_sha256"]
            checked[source["dependencies"]] = validation["dependencies_sha256"]
            rows = json.loads(rows_path.read_text())
            plans = normalize(rows, json.loads(deps_path.read_text()))
            if protocol.get("uniform_warmup_arrivals", False):
                rows = uniform_warmup_offers(rows, plans, concurrency, protocol["warmup_seconds"])
            case = _write_stream(
                rows, plans, dataset=dataset, concurrency=concurrency,
                idle_timeout=protocol["idle_timeout_seconds"], training=training,
                output=output / "streams" / f"seed{seed}" / dataset,
            )
            cases.append(dict(case, seed=seed, workload=dataset,
                              rows=str(rows_path.relative_to(data)), dependencies=str(deps_path.relative_to(data))))
    return {
        "cases": cases,
        "input_sha256": checked,
        "timing": "Complete medium-load trace; fixed client-ready timestamps with immediate completion.",
    }
