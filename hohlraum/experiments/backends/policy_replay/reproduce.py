"""Evaluate cache policies on reference workloads and generate plots."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from hohlraum.experiments.resources import digest, lifecycle_environment

from .streams import prepare_streams


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
BINARIES = ("policy_runner", "baseline_runner")
FIGURE_SWEEP = {
    "thresholds": [0., .1, .3, .5, .7, .9, 1.],
    "staging_fractions": [0., .1, .25, .4, .5, .6, .8],
    "larc_ratios": [.1, .2, .4, .8, 1.6, 3.2, 6.4, 16, 64, 256],
    "access_counts": [1, 2, 3, 4, 5, 8, 12, 20],
    "coverage_levels": ["80", "90", "99"],
}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Point:
    case: dict
    level: str
    capacity: int
    mode: str = "lru_nocopy"
    threshold: float = .7
    fraction: float = 0.
    signal: str = "full"
    family: str = "policy"

    def path(self, output):
        case_dir = output / "policy" / f"seed{self.case['seed']}" / self.case["workload"]
        mode = self.mode.replace(":", "-")
        name = f"{self.family}-{mode}-t{self.threshold:.2f}-r{self.fraction:g}-{self.signal}-c{self.capacity}.json"
        return case_dir / f"coverage{self.level}" / name


def calibrate_case(case, output, binaries):
    workload, seed = case["workload"], case["seed"]
    stream = output / "streams" / f"seed{seed}" / workload / "baseline.tsv"
    folder = output / "policy" / f"seed{seed}" / workload
    folder.mkdir(parents=True)
    log_path = folder / "calibration.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            [str(binaries / "baseline_runner"), str(stream), "coverage", str(case["refs"])],
            text=True, stdout=subprocess.PIPE, stderr=log,
        )
    if result.returncode:
        raise RuntimeError(f"DRAM calibration failed for {workload} seed {seed}. See {log_path}")
    row = json.loads(result.stdout)
    levels = row["levels"]
    theory = row["theory_hits"]
    if (row["requests"] != case["requests"] or row["refs"] != case["refs"]
            or not 0 < theory <= row["refs"] or set(levels) != set(FIGURE_SWEEP["coverage_levels"])):
        raise ValueError(f"Invalid DRAM calibration for {workload} seed {seed}")
    for level, sample in levels.items():
        if not (0 <= sample["previous_hits"] * 100 < int(level) * theory
                <= sample["hits"] * 100 <= theory * 100) or sample["capacity_blocks"] < 1:
            raise ValueError(f"DRAM calibration does not bracket c{level} for {workload} seed {seed}")
        sample["coverage"] = sample["hits"] / theory
        sample["previous_capacity_coverage"] = sample["previous_hits"] / theory
    write_json(folder / "capacities.json", levels)
    return row


def figure_points(config):
    for case in config["cases"]:
        for level, capacity in case["capacities"].items():
            settings = [
                ("policy", "full_nocopy", .7, fraction, "full")
                for fraction in config["staging_fractions"] if fraction
            ]
            settings.append(("policy", "lru_nocopy", .7, 0., "full"))
            settings.append(("baseline", "all", 0., 0., "full"))
            settings.extend(("baseline", f"larc:{max(1, round(capacity * ratio))}", 0., 0., "full")
                            for ratio in config["larc_ratios"])
            if level == "80":
                for threshold in config["thresholds"]:
                    if threshold != .7:
                        settings.extend([
                            ("policy", "full_nocopy", threshold, .5, "full"),
                            ("policy", "lru_nocopy", threshold, 0., "full"),
                        ])
                settings.extend(("baseline", f"access:{count}", 0., 0., "full")
                                for count in config["access_counts"])
                if case["workload"] == "playing":
                    settings.extend(("policy", mode, .7, .5, "full")
                                    for mode in ("lru_nocopy", "full_copy", "lru_copy"))
                    settings.extend(("policy", "full_nocopy", .7, .5, signal)
                                    for signal in ("no_return", "no_branch"))
            # Rounded ghost capacities can coincide. Each distinct policy point
            # has a single owner and produces a single result file.
            for family, mode, threshold, fraction, signal in dict.fromkeys(settings):
                yield Point(case, level, capacity, mode, threshold, fraction, signal, family)


def build_runners(output):
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    binaries = output / "bin"
    binaries.mkdir()
    # Concurrent entrypoints may share the source checkout. Pin each run's
    # compiled binaries before giving another build access to the directory.
    with (build / ".lock").open("a") as lock, (output / "build.log").open("w") as log:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            subprocess.run(["make", f"PYTHON={sys.executable}", *(f"build/{name}" for name in BINARIES)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Could not build the policy replay. See {output / 'build.log'}") from error
        for name in BINARIES:
            shutil.copy2(build / name, binaries / name)
    return binaries


def run_point(point, data, output, binaries, environments, expected_streams):
    case = point.case
    dataset, seed = case["workload"], case["seed"]
    stream = output / "streams" / f"seed{seed}" / dataset
    stage = round(point.capacity * point.fraction)
    env = environments[dataset]
    if point.family == "baseline":
        command = [binaries / "baseline_runner", stream / "baseline.tsv", point.mode, point.capacity]
    else:
        command = [
            binaries / "policy_runner", point.mode, point.threshold, stream / "events.tsv",
            case["blocks"], case["sessions"], point.capacity - stage, stage,
            data / "runtime" / "priors" / dataset, case["end_seconds"],
        ]
        env = {**env, "HOHLRAUM_SIGNAL_ABLATION": point.signal}
    command = [str(value) for value in command]
    path = point.path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".log").open("w") as log:
        process = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=log)
    if process.returncode:
        raise RuntimeError(f"Replay failed for {path.stem}. See {path.with_suffix('.log')}")
    row = json.loads(process.stdout)
    if not 0 <= row["global_hits"] <= row["theory_hits"] <= row["refs"] or row["written_blocks"] < 0:
        raise ValueError(f"Invalid replay counters for {path.stem}")
    expected = expected_streams.get((dataset, seed))
    if expected is not None and (row["refs"], row["theory_hits"]) != expected:
        raise ValueError(f"Replay stream counters differ for {path.stem}")
    if point.family == "policy" and (
        row["requests"] != case["requests"] or row["global_hits"] != row["dram_hits"] + row["ssd_reads"]
        or row["fallbacks"] or row["ssd_evictions"]
    ):
        raise ValueError(f"Invalid unlimited policy replay counters for {path.stem}")
    row.update(seed=seed, workload=dataset, coverage=int(point.level),
               total_blocks=point.capacity, staging_blocks=stage,
               staging_fraction=point.fraction, variant=point.signal, family=point.family, command=command)
    write_json(path, row)
    return row


def reproduce(data: Path, output: Path, *, jobs: int):
    """Run the medium reference loads and write the plots in PNG and PDF."""
    if jobs < 1:
        raise ValueError("Replay jobs must be positive")
    import matplotlib
    import numpy

    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Replay output already exists: {output}")
    output.mkdir(parents=True)
    status = output / "status.json"
    expected_streams = {}

    def run_points(points, phase):
        plan = list(points)
        write_json(status, dict(state="running", phase=phase, done=0, jobs=len(plan)))
        results = {}
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            pending = {
                executor.submit(run_point, point, data, output, binaries, environments, expected_streams): point
                for point in plan
            }
            try:
                for done, task in enumerate(as_completed(pending), 1):
                    point = pending[task]
                    path = point.path(output)
                    results[path] = task.result()
                    write_json(status, dict(state="running", phase=phase, done=done, jobs=len(plan)))
                    print(f"{phase} {done}/{len(plan)} {path.relative_to(output)}", flush=True)
            except Exception:
                for task in pending:
                    task.cancel()
                raise
        return results

    try:
        write_json(status, dict(state="running", phase="streams"))
        streams = prepare_streams(data, output)
        cases = streams["cases"]
        config = {**FIGURE_SWEEP, "cases": cases}
        inherited = {name: value for name, value in os.environ.items() if not name.startswith("HOHLRAUM_")}
        environments = {
            dataset: {
                **inherited, **lifecycle_environment(data / "runtime" / "priors" / dataset),
                "HOHLRAUM_STATISTICS_PERIOD_S": "3600", "HOHLRAUM_SIGNAL_ABLATION": "full",
            }
            for dataset in {case["workload"] for case in cases}
        }
        write_json(status, dict(state="running", phase="build"))
        binaries = build_runners(output)
        write_json(status, dict(state="running", phase="calibration", jobs=len(cases), done=0))
        calibrations = {}
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            pending = {executor.submit(calibrate_case, case, output, binaries): case for case in cases}
            for done, task in enumerate(as_completed(pending), 1):
                case = pending[task]
                row = task.result()
                key = (case["workload"], case["seed"])
                calibrations[key] = row["levels"]
                expected_streams[key] = (row["refs"], row["theory_hits"])
                case["capacities"] = {level: sample["capacity_blocks"] for level, sample in row["levels"].items()}
                write_json(status, dict(state="running", phase="calibration", jobs=len(cases), done=done))
                print(f"calibration {done}/{len(cases)} {key}: {case['capacities']}", flush=True)
        write_json(output / "manifest.json", {
            "input_sha256": streams["input_sha256"], "config": config, "timing": streams["timing"],
            "binary_sha256": {name: digest(binaries / name) for name in BINARIES},
            "plot_dependencies": {"matplotlib": matplotlib.__version__, "numpy": numpy.__version__},
            "statistics_period_seconds": 3600,
        })
        points = run_points(figure_points(config), "replay")
        for case in cases:
            key = (case["workload"], case["seed"])
            for level, capacity in case["capacities"].items():
                actual = points[Point(case, level, capacity).path(output)]["dram_hits"]
                if actual != calibrations[key][level]["hits"]:
                    raise ValueError(f"DRAM calibration disagrees with zero-staging replay at c{level} for {key}")
        write_json(output / "policy" / "completed.json", dict(jobs=len(points) + len(calibrations)))
        write_json(status, dict(state="running", phase="plot"))
        with (output / "plot.log").open("w") as log:
            try:
                subprocess.run([sys.executable, str(PAPER / "plot.py"), str(output)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"Could not plot the policy replay. See {output / 'plot.log'}") from error
        figures = output / "figures"
        pngs = {file.stem for file in figures.glob("*.png") if file.stat().st_size}
        pdfs = {file.stem for file in figures.glob("*.pdf") if file.stat().st_size}
        if len(pngs) != 4 or pngs != pdfs:
            raise RuntimeError(f"Expected four PNG/PDF figure pairs in {figures}. See {output / 'plot.log'}")
        write_json(status, dict(state="complete", figures=sorted(pngs), jobs=len(points) + len(calibrations)))
    except Exception as error:
        write_json(status, dict(state="failed", error=str(error)))
        raise
    print(f"Policy replay figures: {figures}", flush=True)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=int)
    args = parser.parse_args()
    reproduce(args.data, args.output, jobs=args.jobs)


if __name__ == "__main__":
    main()
