#!/usr/bin/env python3
"""Aggregate the runtime expert-prediction stats from a llama-server log.

The engine prints one expert-tier report per stats interval. The pin-hot-experts
section lists, for every enabled layer, how many predicted experts the decode
step actually routed (hits/predicted); the l2-cache section reports the
read-ahead work. This tool sums those interval lines into global figures, which
is what tells whether a layer or a threshold is worth it.

  python predict_report.py C:/models/logs/llama-server.log
  python predict_report.py run1.log run2.log --json report.json --top 8

Per-layer hit rate is the prediction precision; a low rate means the layer
reads experts it never uses. The prefetch block separates reads that paid off
(useful) from reads evicted unused (wasted) and from reads that finished after
their layer was already filled (late).
"""

import argparse
import json
import re
from collections import defaultdict

# one per-layer entry of a predict line: L<layer>=<routed>/<predicted>
LAYER_TOKEN = re.compile(r"L(\d+)=(\d+)/(\d+)")
# a continuation line holds only layer tokens
LAYER_LINE = re.compile(r"(?:L\d+=\d+/\d+\s*)+")
PREDICT_SUMMARY = re.compile(r"predict\s*:\s*(\d+)\s+of\s+(\d+)\s+steps", re.IGNORECASE)
VOLUME = re.compile(r"([\d.]+)\s*(GiB|MiB|KiB)")

SCALE = {"KiB": 1024.0, "MiB": 1024.0 ** 2, "GiB": 1024.0 ** 3}


def first_volume(text):
    match = VOLUME.search(text)
    if not match:
        return 0.0
    return float(match.group(1)) * SCALE[match.group(2)]


def number(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0


def parse_prefetch(line, acc):
    """Parse one prefetch report line into acc (handles the old single line)."""
    if "experts read ahead" in line:  # legacy: one line, no outcome split
        acc["read"] += number(r":\s*(\d+)\s+experts read ahead", line)
        match = re.search(r"read ahead[^(]*\(([^)]*)\)", line)
        if match:
            acc["read_bytes"] += first_volume(match.group(1))
        acc["resident"] += number(r"\|\s*(\d+)\s+resident", line)
        acc["noslot"] += number(r"(\d+)\s+no free slot", line)
        return
    # new: first line carries the outcomes, the second the skip reasons
    acc["offered"] += number(r":\s*(\d+)\s+predicted", line)
    if re.search(r"(\d+)\s+read\s*\(", line):
        acc["read"] += number(r"(\d+)\s+read\s*\(", line)
        match = re.search(r"(\d+)\s+read\s*\(([^)]*)\)", line)
        if match:
            acc["read_bytes"] += first_volume(match.group(2))
    if re.search(r"(\d+)\s+useful\s*\(", line):
        acc["useful"] += number(r"(\d+)\s+useful\s*\(", line)
        match = re.search(r"(\d+)\s+useful\s*\(([^)]*)\)", line)
        if match:
            acc["useful_bytes"] += first_volume(match.group(2))
    if re.search(r"(\d+)\s+wasted\s*\(", line):
        acc["wasted"] += number(r"(\d+)\s+wasted\s*\(", line)
        match = re.search(r"(\d+)\s+wasted\s*\(([^)]*)\)", line)
        if match:
            acc["wasted_bytes"] += first_volume(match.group(2))
    if re.search(r"(\d+)\s+late\s*\(", line):
        acc["late"] += number(r"(\d+)\s+late\s*\(", line)
        match = re.search(r"(\d+)\s+late\s*\(([^)]*)\)", line)
        if match:
            acc["late_bytes"] += first_volume(match.group(2))
    acc["resident"] += number(r"(\d+)\s+already resident", line)
    acc["passed"] += number(r"(\d+)\s+layer passed", line)
    acc["noslot"] += number(r"(\d+)\s+no free slot", line)


def parse_logs(paths):
    per_layer = defaultdict(lambda: {"predicted": 0, "routed": 0})
    reports = 0
    summary_predicted = 0
    summary_routed = 0
    disk_read_bytes = 0.0
    l2_served_bytes = 0.0
    prefetch = {k: 0 for k in
                ("offered", "read", "useful", "wasted", "late", "resident", "passed", "noslot")}
    for k in ("read_bytes", "useful_bytes", "wasted_bytes", "late_bytes"):
        prefetch[k] = 0.0

    for path in paths:
        in_layer_block = False
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                stripped = raw.strip()
                if "[expert tiers]" in stripped:
                    reports += 1
                    continue

                if "predict/L:" in stripped:
                    in_layer_block = True
                    for layer, hit, pred in LAYER_TOKEN.findall(stripped.split("predict/L:", 1)[1]):
                        per_layer[int(layer)]["routed"] += int(hit)
                        per_layer[int(layer)]["predicted"] += int(pred)
                    continue
                if in_layer_block:
                    if LAYER_LINE.fullmatch(stripped):
                        for layer, hit, pred in LAYER_TOKEN.findall(stripped):
                            per_layer[int(layer)]["routed"] += int(hit)
                            per_layer[int(layer)]["predicted"] += int(pred)
                        continue
                    in_layer_block = False

                if stripped.startswith("predict"):
                    pair = re.search(r"(\d+)\s+predicted,\s+(\d+)\s+routed", stripped)
                    if pair:
                        summary_predicted += int(pair.group(1))
                        summary_routed += int(pair.group(2))
                    continue

                if stripped.startswith("prefetch"):
                    parse_prefetch(stripped, prefetch)
                    continue

                if "read over" in stripped:
                    match = re.search(r":\s*([\d.]+)\s*(GiB|MiB|KiB)\s+read over", stripped)
                    if match:  # cumulative: keep the largest
                        disk_read_bytes = max(disk_read_bytes, float(match.group(1)) * SCALE[match.group(2)])
                elif stripped.startswith("l2 served"):
                    l2_served_bytes += first_volume(stripped.split("|", 1)[0])

    return {
        "per_layer": {il: per_layer[il] for il in sorted(per_layer)},
        "reports": reports,
        "summary_predicted": summary_predicted,
        "summary_routed": summary_routed,
        "prefetch": prefetch,
        "disk_read_bytes": disk_read_bytes,
        "l2_served_bytes": l2_served_bytes,
    }


def rate(hit, pred):
    return 100.0 * hit / pred if pred else 0.0


def gib(n):
    return n / (1024.0 ** 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="llama-server log file(s)")
    parser.add_argument("--min-pred", type=int, default=0, help="hide layers with fewer predicted experts")
    parser.add_argument("--json", help="also write the aggregate as JSON")
    parser.add_argument("--top", type=int, default=0,
                        help="show only the N best and N worst layers by hit rate (0 = all)")
    args = parser.parse_args()

    data = parse_logs(args.logs)
    per_layer = data["per_layer"]
    prefetch = data["prefetch"]

    total_pred = sum(v["predicted"] for v in per_layer.values())
    total_hit = sum(v["routed"] for v in per_layer.values())

    print("reports: %d" % data["reports"])
    print("prediction: %d predicted, %d routed (%.1f%% global)" % (
        total_pred, total_hit, rate(total_hit, total_pred)))
    if prefetch["offered"] or prefetch["read"]:
        print("read-ahead: %d predicted | %d read (%.2f GiB) | %d useful (%.2f GiB, %.1f%% of reads)" % (
            prefetch["offered"], prefetch["read"], gib(prefetch["read_bytes"]),
            prefetch["useful"], gib(prefetch["useful_bytes"]),
            rate(prefetch["useful"], prefetch["read"])))
        print("            %d wasted (%.2f GiB) | %d late (%.2f GiB) | %d already resident | %d layer passed | %d no slot" % (
            prefetch["wasted"], gib(prefetch["wasted_bytes"]),
            prefetch["late"], gib(prefetch["late_bytes"]),
            prefetch["resident"], prefetch["passed"], prefetch["noslot"]))
    if data["disk_read_bytes"]:
        print("disk: %.2f GiB demand reads | %.2f GiB L2-avoided" % (
            gib(data["disk_read_bytes"]), gib(data["l2_served_bytes"])))

    layers = [il for il in per_layer if per_layer[il]["predicted"] >= args.min_pred]
    if not layers:
        print("no layers match --min-pred")
        return

    print()
    print("per-layer prediction (predicted = experts offered, routed = actually selected)")
    print("%6s %10s %10s %7s" % ("layer", "predicted", "routed", "hit%"))
    for il in layers:
        v = per_layer[il]
        print("%6d %10d %10d %6.1f%%" % (il, v["predicted"], v["routed"], rate(v["routed"], v["predicted"])))

    ranked = sorted(layers, key=lambda il: rate(per_layer[il]["routed"], per_layer[il]["predicted"]))
    if args.top and len(ranked) > 2 * args.top:
        ranked = ranked[:args.top] + ranked[-args.top:]
        print("\nbest/worst layers by hit rate")
    else:
        print("\nlayers by hit rate (worst first)")
    print("%6s %10s %10s %7s" % ("layer", "predicted", "routed", "hit%"))
    for il in ranked:
        v = per_layer[il]
        print("%6d %10d %10d %6.1f%%" % (il, v["predicted"], v["routed"], rate(v["routed"], v["predicted"])))

    if args.json:
        out = dict(data)
        out["global"] = {"predicted": total_pred, "routed": total_hit, "hit_rate": rate(total_hit, total_pred)}
        out["per_layer_rate"] = {str(il): {
            "predicted": v["predicted"], "routed": v["routed"],
            "hit_rate": rate(v["routed"], v["predicted"])} for il, v in per_layer.items()}
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2, sort_keys=True)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
