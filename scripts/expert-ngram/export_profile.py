#!/usr/bin/env python3
"""Build a compact expert-prediction profile from --expert-ngram-record files.

A profile is a per-workflow table: n-gram hash -> per-layer top-M experts, plus
the per-layer allowlist of layers worth prefetching. It is meant to be mmap'd and
read in place by the runtime with one hash and one probe per decode step, and
iterated in layer order.

Layer selection matters because prefetching a noisy layer reads experts that are
not routed. The allowlist is chosen from an internal two-fold split of the turns
(train half A, measure on half B, and back), never from the evaluation split, and
`--sweep` re-checks each threshold on a separate held-out turn split.

  python export_profile.py coding.prof session1.rec session2.rec --sweep
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study import (  # noqa: E402
    build_model, eval_split, key_ids_for_order, load,
)

MAGIC = b"LENGPROF"
VERSION = 1
# magic, version, n_layer, n_expert, n_expert_used, max_order, top_m, table_size,
# n_enabled, layer_mask_bytes, model_fp, tokenizer_fp, n_entries
HEADER = struct.Struct("<8sIIIIIIIIIQQQ")

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1
ORDER_SALT = 0x9E3779B97F4A7C15


def hash_key(values, order):
    """Hash an n-gram (oldest first). Must match the runtime exactly."""
    h = FNV_OFFSET
    for value in values:
        h ^= int(value) & 0xFFFFFFFF
        h = (h * FNV_PRIME) & MASK64
    h ^= (order * ORDER_SALT) & MASK64
    h = (h * FNV_PRIME) & MASK64
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & MASK64
    h ^= h >> 33
    return h if h != 0 else 1


def select_layers(stats, threshold, limit, max_reads_per_step):
    """Greedy selection by per-read efficiency, optionally capped."""
    picked = [s for s in stats if s["efficiency"] >= threshold]
    picked.sort(key=lambda s: -s["efficiency"])
    if limit > 0:
        picked = picked[:limit]
    if max_reads_per_step > 0:
        capped = []
        total = 0.0
        for s in picked:
            if capped and total + s["reads_per_step"] > max_reads_per_step:
                break
            capped.append(s)
            total += s["reads_per_step"]
        picked = capped
    return sorted(s["layer"] for s in picked)


def selection_folds(d, orders, ref_budget, min_support, resident_c):
    """Per-layer estimates for the allowlist. With two or more sessions these are
    leave-one-session-out folds (the transfer a profile has to survive); a single
    session falls back to a turn-parity split within it."""
    sessions = sorted(set(int(s) for s in d["session"]))
    folds = []
    if len(sessions) >= 2:
        for s in sessions:
            folds.append(eval_split(d, d["session"] != s, np.flatnonzero(d["session"] == s),
                                    orders, [ref_budget], min_support, resident_c))
    else:
        for parity in (0, 1):
            tr = (d["turn"] % 2) == parity
            ev = np.flatnonzero((d["turn"] % 2) == (1 - parity))
            folds.append(eval_split(d, tr, ev, orders, [ref_budget], min_support, resident_c))
    return folds


def layer_stats(folds, n_layer):
    steps = sum(f["n_eval"] for f in folds) or 1
    out = []
    for il in range(n_layer):
        reads = sum(f["per_layer_rec"][il]["pred_nonres"] for f in folds)
        help_ = sum(f["per_layer_rec"][il]["help"] for f in folds)
        miss = sum(f["per_layer_rec"][il]["miss"] for f in folds)
        out.append({
            "layer": il,
            "efficiency": help_ / reads if reads else 0.0,
            "reads_per_step": reads / steps,
            "help_per_step": help_ / steps,
            "miss_per_step": miss / steps,
        })
    return out


def build_entries(d, orders, top_m, min_support, selected):
    """(hash, per-enabled-layer ids) for every trained key, in ascending layer order."""
    routes = d["routes"]
    train = np.ones(len(routes), dtype=bool)
    n_layer = d["header"]["n_layer"]
    n_expert = d["header"]["n_expert"]
    enabled = sorted(selected)
    entries = []
    keys_per_order = {}
    for order in orders:
        ids, table = key_ids_for_order(d["tokens"], d["seg"], order)
        model, _ = build_model(routes, ids, train, n_layer, n_expert, top_m, min_support)
        lookup = [None] * len(table)
        for tup, idx in table.items():
            lookup[idx] = tup
        for key_id, arr in model.items():
            row = np.full((len(enabled), top_m), 0xFFFF, dtype=np.uint16)
            for slot, il in enumerate(enabled):
                vals = arr[il]
                row[slot, :] = np.where(vals < 0, 0xFFFF, vals).astype(np.uint16)
            entries.append((hash_key(lookup[key_id], order), order, row))
        keys_per_order[order] = len(model)
        del model
    return entries, keys_per_order


def write_profile(path, d, meta, entries):
    n_layer = d["header"]["n_layer"]
    n_enabled = len(meta["selected"])
    top_m = meta["top_m"]
    layer_mask_bytes = (n_layer + 7) // 8
    mask = bytearray(layer_mask_bytes)
    for il in meta["selected"]:
        mask[il // 8] |= 1 << (il % 8)
    budgets = np.zeros(n_layer, dtype="<u4")
    for il in meta["selected"]:
        budgets[il] = top_m

    n_entries = len(entries)
    table_size = 8
    while table_size < n_entries * 2:
        table_size <<= 1
    keys = np.zeros(table_size, dtype=np.uint64)
    payload = np.full((table_size, max(n_enabled * top_m, 1)), 0xFFFF, dtype=np.uint16)
    for h, _order, row in entries:
        idx = int(h) & (table_size - 1)
        while keys[idx] != 0:
            idx = (idx + 1) & (table_size - 1)
        keys[idx] = h
        if n_enabled:
            payload[idx] = row.ravel()
    slot = np.dtype([("key", "<u8"), ("payload", "<u2", (max(n_enabled * top_m, 1),))])
    slots = np.zeros(table_size, dtype=slot)
    slots["key"] = keys
    slots["payload"] = payload

    with open(path, "wb") as handle:
        handle.write(HEADER.pack(
            MAGIC, VERSION, n_layer, d["header"]["n_expert"], d["header"]["n_expert_used"],
            meta["max_order"], top_m, table_size, n_enabled, layer_mask_bytes,
            d["header"]["model_fp"], d["header"]["tokenizer_fp"], n_entries))
        handle.write(bytes(mask))
        handle.write(budgets.tobytes())
        handle.write(slots.tobytes())
    return table_size, layer_mask_bytes


def read_profile(path):
    """Reference reader; the C++ runtime must match HEADER and the slot layout."""
    with open(path, "rb") as handle:
        blob = handle.read()
    fields = HEADER.unpack_from(blob, 0)
    (magic, version, n_layer, n_expert, n_expert_used, max_order, top_m, table_size,
     n_enabled, layer_mask_bytes, model_fp, tokenizer_fp, n_entries) = fields
    if magic != MAGIC or version != VERSION:
        raise ValueError("not a LENGPROF v1 profile")
    off = HEADER.size
    mask = np.frombuffer(blob, dtype=np.uint8, count=layer_mask_bytes, offset=off)
    off += layer_mask_bytes
    budgets = np.frombuffer(blob, dtype="<u4", count=n_layer, offset=off)
    off += 4 * n_layer
    slot = np.dtype([("key", "<u8"), ("payload", "<u2", (max(n_enabled * top_m, 1),))])
    slots = np.frombuffer(blob, dtype=slot, count=table_size, offset=off)
    return {
        "n_layer": n_layer, "n_expert": n_expert, "n_expert_used": n_expert_used,
        "max_order": max_order, "top_m": top_m, "table_size": table_size,
        "n_enabled": n_enabled, "mask": mask, "budgets": budgets, "slots": slots,
        "model_fp": model_fp, "tokenizer_fp": tokenizer_fp, "n_entries": n_entries,
        "enabled": [il for il in range(n_layer) if mask[il // 8] & (1 << (il % 8))],
    }


def lookup(prof, window):
    """Backoff lookup: list of (layer, ids) for the largest matching order, or []."""
    for order in range(prof["max_order"], 0, -1):
        if len(window) < order:
            continue
        h = hash_key(window[-order:], order)
        idx = h & (prof["table_size"] - 1)
        for _ in range(prof["table_size"]):
            key = int(prof["slots"]["key"][idx])
            if key == 0:
                break
            if key == h:
                row = prof["slots"]["payload"][idx]
                out = []
                for slot, il in enumerate(prof["enabled"]):
                    vals = [int(v) for v in row[slot * prof["top_m"]:(slot + 1) * prof["top_m"]] if int(v) != 0xFFFF]
                    if vals:
                        out.append((il, vals))
                return out
            idx = (idx + 1) & (prof["table_size"] - 1)
    return []



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="profile file to write")
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--top-m", type=int, default=6)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--resident-c", type=int, default=8)
    parser.add_argument("--layer-threshold", type=float, default=0.45,
                        help="minimum per-read efficiency (reads that hit / reads) for a "
                             "layer to be prefetched (default 0.45)")
    parser.add_argument("--layer-limit", type=int, default=0,
                        help="cap on the number of enabled layers (0 = no cap)")
    parser.add_argument("--max-reads-per-step", type=float, default=0.0,
                        help="cap on expected prefetch reads per decode step (0 = no cap)")
    parser.add_argument("--sweep", action="store_true", help="print a threshold sweep and stop")
    parser.add_argument("--verify", action="store_true", help="re-read and round-trip check")
    args = parser.parse_args()

    orders = sorted(set(o for o in args.orders if o >= 1))
    d = load(args.recordings)
    n_layer = d["header"]["n_layer"]
    ref_budget = min(d["header"]["n_expert_used"], args.top_m)

    # selection: leave-one-session-out per-layer efficiency, averaged
    folds = selection_folds(d, orders, ref_budget, args.min_support, args.resident_c)
    stats = layer_stats(folds, n_layer)

    total_miss = sum(s["miss_per_step"] for s in stats) or 1.0
    if args.sweep:
        # the same held-out folds realize each selection (cross-session when the
        # inputs are multiple sessions)
        print("per-layer cross-session efficiency and realized value at budget %d" % ref_budget)
        print("| layer | eff (select) | reads/step | help/step |")
        print("| --- | --- | --- | --- |")
        for s in stats:
            print("| %d | %.3f | %.1f | %.1f |" % (s["layer"], s["efficiency"], s["reads_per_step"], s["help_per_step"]))
        print()
        print("threshold sweep (selected and realized on the held-out folds)")
        print("| threshold | layers | reads/step | covered | reads useful |")
        print("| --- | --- | --- | --- | --- |")
        for threshold in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            sel = select_layers(stats, threshold, args.layer_limit, args.max_reads_per_step)
            sel_set = set(sel)
            chosen = [s for s in stats if s["layer"] in sel_set]
            reads = sum(s["reads_per_step"] for s in chosen)
            help_ = sum(s["help_per_step"] for s in chosen)
            print("| %.2f | %d | %.1f | %.1f%% | %.1f%% |" % (
                threshold, len(sel), reads, 100 * help_ / total_miss,
                100 * help_ / reads if reads else 0))
        return

    selected = select_layers(stats, args.layer_threshold, args.layer_limit, args.max_reads_per_step)
    if not selected:
        print("no layer passed the threshold; nothing written")
        return
    print("enabled layers (%d): %s" % (len(selected), ", ".join(str(il) for il in selected)))
    print("| layer | eff | reads/step | help/step |")
    print("| --- | --- | --- | --- |")
    for s in sorted(stats, key=lambda s: s["layer"]):
        if s["layer"] in set(selected):
            print("| %d | %.3f | %.1f | %.1f |" % (s["layer"], s["efficiency"], s["reads_per_step"], s["help_per_step"]))

    entries, keys_per_order = build_entries(d, orders, args.top_m, args.min_support, selected)
    meta = {"selected": selected, "top_m": args.top_m, "max_order": max(orders)}
    table_size, mask_bytes = write_profile(args.output, d, meta, entries)
    print("wrote %s: %d entries, %d slots, %d enabled layers, %.1f MiB" % (
        args.output, len(entries), table_size, len(selected), os.path.getsize(args.output) / 1048576.0))

    if args.verify:
        prof = read_profile(args.output)
        assert prof["enabled"] == selected, (prof["enabled"], selected)
        # round-trip: every trained key must be found and return the same ids
        checked = 0
        for order in orders:
            ids, table = key_ids_for_order(d["tokens"], d["seg"], order)
            lookup_table = [None] * len(table)
            for tup, idx in table.items():
                lookup_table[idx] = tup
            model, _ = build_model(d["routes"], ids, np.ones(len(ids), dtype=bool),
                                   n_layer, d["header"]["n_expert"], args.top_m, args.min_support)
            for key_id in list(model)[:200]:
                window = lookup_table[key_id]
                found = dict(lookup(prof, list(window)))
                for slot, il in enumerate(selected):
                    expected = [int(v) for v in model[key_id][il] if v >= 0]
                    if expected and found.get(il) != expected:
                        raise AssertionError("mismatch order %d key %s layer %d: %s vs %s" % (
                            order, window, il, expected, found.get(il)))
                checked += 1
        print("verify ok: %d sampled keys round-tripped" % checked)


if __name__ == "__main__":
    main()
