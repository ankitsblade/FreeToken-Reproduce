"""Sample GPU and whole-host CPU state while an experiment process runs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import time
from pathlib import Path


GPU_FIELDS = (
    "timestamp", "utilization.gpu", "utilization.memory", "power.draw",
    "clocks.current.sm", "temperature.gpu", "pcie.link.gen.current",
    "pcie.link.width.current",
)


def cpu_counters() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    return sum(values), values[3] + values[4]


def cpu_mhz() -> float | None:
    frequencies = []
    for path in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"):
        try:
            frequencies.append(int(Path(path).read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return sum(frequencies) / len(frequencies) if frequencies else None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.interval <= 0 or args.out.exists():
        parser.error("interval must be positive and output path must not exist")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    previous = cpu_counters()
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*GPU_FIELDS, "cpu_utilization_pct",
                                                 "cpu_mhz_mean", "watch_pid"])
        writer.writeheader()
        while alive(args.watch_pid):
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
                "--format=csv,noheader,nounits",
            ], check=True, capture_output=True, text=True)
            gpu = next(csv.reader(result.stdout.splitlines()))
            current = cpu_counters()
            elapsed, idle = current[0] - previous[0], current[1] - previous[1]
            previous = current
            writer.writerow({
                **dict(zip(GPU_FIELDS, (value.strip() for value in gpu))),
                "cpu_utilization_pct": 100 * (elapsed - idle) / elapsed if elapsed else "",
                "cpu_mhz_mean": cpu_mhz() or "", "watch_pid": args.watch_pid,
            })
            f.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
