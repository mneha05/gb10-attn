#!/usr/bin/env python3
"""Turn ncu's raw CSV into the three numbers that explain a bandwidth result.

A roofline percentage on its own says a kernel is at 78% of peak but not why.
These do:

  dram__throughput...pct_of_peak   what the memory system actually delivered,
                                   measured rather than inferred from wall time
  lts__t_sector_hit_rate.pct       how much of that never reached DRAM. A
                                   decode kernel streaming a KV cache larger
                                   than L2 should show a LOW hit rate; a high
                                   one means the benchmark is measuring cache.
  sm__warps_active...pct           occupancy. If DRAM throughput is low AND
                                   occupancy is low, the kernel is latency-
                                   bound, not bandwidth-bound, and the fix is
                                   more parallelism (which is exactly what v3's
                                   context split was for).

Usage: summarize_ncu.py <results_dir>
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

KEYS = {
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM %peak",
    "dram__bytes_read.sum": "DRAM read",
    "dram__bytes_write.sum": "DRAM write",
    "lts__t_sector_hit_rate.pct": "L2 hit %",
    "l1tex__t_sector_hit_rate.pct": "L1 hit %",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy %",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM %peak",
    "launch__waves_per_multiprocessor": "waves/SM",
    "gpu__time_duration.sum": "duration",
}


def parse(path: Path):
    """ncu --csv emits some banner lines before the header; skip to it."""
    rows = []
    with path.open(newline="", errors="ignore") as fh:
        lines = [l for l in fh if l.strip()]
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"') or
                  ("Kernel Name" in l and "Metric Name" in l)), None)
    if start is None:
        return rows
    for r in csv.DictReader(lines[start:]):
        rows.append(r)
    return rows


def short(kernel: str) -> str:
    m = re.search(r"(paged_attention\w*)", kernel or "")
    return m.group(1) if m else (kernel or "?")[:40]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__); return 2
    out = Path(sys.argv[1])
    files = sorted(out.glob("ncu_*.csv"))
    if not files:
        print("no ncu_*.csv found in", out); return 1

    for f in files:
        rows = parse(f)
        if not rows:
            print(f"\n{f.name}: no parseable rows (profiling likely failed)")
            continue
        # (kernel, metric) -> [values]
        agg: dict[tuple[str, str], list[float]] = defaultdict(list)
        units: dict[str, str] = {}
        for r in rows:
            k = short(r.get("Kernel Name", ""))
            metric = r.get("Metric Name", "")
            if metric not in KEYS:
                continue
            try:
                v = float(str(r.get("Metric Value", "")).replace(",", ""))
            except ValueError:
                continue
            agg[(k, metric)].append(v)
            units[metric] = r.get("Metric Unit", "")

        print(f"\n=== {f.name} ===")
        kernels = sorted({k for k, _ in agg})
        for k in kernels:
            print(f"  {k}")
            for metric, label in KEYS.items():
                vals = agg.get((k, metric))
                if not vals:
                    continue
                mean = statistics.fmean(vals)
                unit = units.get(metric, "")
                if "bytes" in metric:
                    print(f"    {label:12s} {mean/1e6:12.2f} MB")
                else:
                    print(f"    {label:12s} {mean:12.2f} {unit}")
            # the interpretive line -- what the numbers together mean
            dram = agg.get((k, "dram__throughput.avg.pct_of_peak_sustained_elapsed"))
            occ = agg.get((k, "sm__warps_active.avg.pct_of_peak_sustained_active"))
            l2 = agg.get((k, "lts__t_sector_hit_rate.pct"))
            if dram and occ:
                d, o = statistics.fmean(dram), statistics.fmean(occ)
                h = statistics.fmean(l2) if l2 else None
                if d > 70:
                    verdict = "bandwidth-bound: near the bus, little left to win"
                elif o < 30:
                    verdict = "latency-bound: too few warps to cover DRAM latency"
                elif h is not None and h > 60:
                    verdict = f"served from L2 ({h:.0f}% hits) -- not a DRAM measurement"
                else:
                    verdict = "neither saturated: check launch config / divergence"
                print(f"    -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
