#!/usr/bin/env python3
"""Vectorized expert-prediction study over --expert-ngram-record files.

Deep, repeatable analysis: build an n-gram -> expert model on a training part of
the recording(s) and score prediction quality on a held-out part, against the
baselines that matter.

Two splits are reported:

- `turns`: hold out every K-th server request across all files. For one file this
  is within-session transfer.
- `cross`: each input file is a session; leave-one-session-out (train on the
  other sessions, evaluate on the held-out one). This is the cross-session
  transfer a per-workflow profile has to survive.

The metric that decides whether prefetch can help is `useful`: of the routed
experts that were NOT already resident (a static top-C set plus the previous
step), how many a predictor covered before the read. `waste` reports the wrong
fraction of the prefetched non-resident experts. The whole answer depends on the
cache size, so a residency sweep is included.

The splits are deterministic, so the same inputs always yield the same report.
Requires numpy.

  python study.py run1.rec run2.rec --report report.md
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import read_recording  # noqa: E402


def load(paths):
    """Read recordings into dense arrays; each file is one session."""
    tokens = []
    seg = []
    turn = []
    session = []
    rows = []
    header = None
    per_file = []
    gseg = -1
    for fi, path in enumerate(paths):
        local_turn = -1
        steps = 0
        truncated = None
        with open(path, "rb") as handle:
            size = os.fstat(handle.fileno()).st_size
        for kind, payload in read_recording(path):
            if kind == "header":
                if header is None:
                    header = payload
                elif payload != header:
                    raise ValueError("%s: header differs from %s" % (path, paths[0]))
                continue
            if kind == "prompt":
                gseg += 1
                local_turn += 1
                continue
            if kind == "truncated":
                truncated = payload
                continue
            token, layers = payload
            row = np.full((header["n_layer"], header["n_expert_used"]), -1, dtype=np.int16)
            for il, ids in layers.items():
                if 0 <= il < header["n_layer"]:
                    ids = ids[: header["n_expert_used"]]
                    row[il, : len(ids)] = ids
            rows.append(row)
            tokens.append(token)
            seg.append(gseg)
            turn.append(local_turn)
            session.append(fi)
            steps += 1
        per_file.append({"path": path, "name": os.path.basename(path), "size": size,
                         "steps": steps, "turns": local_turn + 1, "truncated": truncated})
    routes = np.stack(rows) if rows else np.zeros((0, header["n_layer"], header["n_expert_used"]), np.int16)
    return {
        "paths": paths,
        "header": header,
        "per_file": per_file,
        "tokens": np.asarray(tokens, dtype=np.int32),
        "routes": routes,
        "seg": np.asarray(seg, dtype=np.int64),
        "turn": np.asarray(turn, dtype=np.int64),
        "session": np.asarray(session, dtype=np.int64),
    }


def previous_routes(routes, seg):
    """routes[t-1] with the first step of every turn/session blanked to -1."""
    prev = np.full_like(routes, -1)
    if len(routes) > 1:
        prev[1:] = routes[:-1]
    first = np.ones(len(routes), dtype=bool)
    first[1:] = seg[1:] != seg[:-1]
    prev[first] = -1
    return prev


def key_ids_for_order(tokens, seg, order):
    """Exact token n-gram id per step; -1 where the key crosses a turn or is
    shorter than the order. Returns (ids, table) with table tuple -> id."""
    n = len(tokens)
    ids = np.full(n, -1, dtype=np.int64)
    if order <= 0 or n == 0:
        return ids, {}
    first = np.zeros(n, dtype=np.int64)
    if n > 1:
        for s in np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]]):
            first[s:] = s
    valid = np.arange(n) - first >= order - 1
    table = {}
    for t in np.flatnonzero(valid):
        key = tuple(int(x) for x in tokens[t - order + 1 : t + 1])
        idx = table.get(key)
        if idx is None:
            idx = len(table)
            table[key] = idx
        ids[t] = idx
    return ids, table


def build_model(routes, key_ids, train_mask, n_layer, n_expert, max_m, min_support):
    """Top-`max_m` experts per (key, layer); model[key] is int16 [n_layer, max_m]."""
    sel = np.flatnonzero(train_mask & (key_ids >= 0))
    if len(sel) == 0:
        return {}, 0
    keys = key_ids[sel]
    rts = routes[sel].astype(np.int64)
    layer_idx = np.broadcast_to(np.arange(n_layer)[None, :, None], rts.shape)
    valid = rts >= 0
    combined = (keys[:, None, None] * (n_layer * n_expert) + layer_idx * n_expert + rts)[valid]
    uniq, counts = np.unique(combined, return_counts=True)
    key_support = np.bincount(keys)
    key_of = uniq // (n_layer * n_expert)
    layer_of = (uniq // n_expert) % n_layer
    expert_of = uniq % n_expert
    order = np.lexsort((-counts, layer_of, key_of))
    key_of, layer_of, expert_of = key_of[order], layer_of[order], expert_of[order]
    new_group = np.r_[True, (key_of[1:] != key_of[:-1]) | (layer_of[1:] != layer_of[:-1])]
    group_start = np.flatnonzero(new_group)
    group_len = np.diff(np.r_[group_start, len(uniq)])
    rank = np.arange(len(uniq)) - np.repeat(group_start, group_len)
    model = {}
    for pos in np.flatnonzero(rank < max_m):
        kk = int(key_of[pos])
        if key_support[kk] < min_support:
            continue
        arr = model.get(kk)
        if arr is None:
            arr = np.full((n_layer, max_m), -1, dtype=np.int16)
            model[kk] = arr
        arr[int(layer_of[pos]), int(rank[pos])] = int(expert_of[pos])
    return model, len(model)


def pred_array_from_model(model, key_ids, idx, n_layer, max_m):
    out = np.full((len(idx), n_layer, max_m), -1, dtype=np.int16)
    for row, t in enumerate(idx):
        arr = model.get(int(key_ids[t]))
        if arr is not None:
            out[row] = arr
    return out


def pad_to(ids, max_m):
    e, layer, k = ids.shape
    out = np.full((e, layer, max_m), -1, dtype=np.int16)
    out[..., : min(k, max_m)] = ids[..., : min(k, max_m)]
    return out


def tally(actual, act_valid, resident, pred, budgets):
    """Cumulative top-m tallies, one per budget, for a single predictor."""
    act_res = np.take_along_axis(resident, np.where(act_valid, actual, 0).astype(np.int64), axis=2)
    miss = act_valid & ~act_res
    out = {budget: {"pred": 0, "hit": 0, "actual": int(act_valid.sum()),
                    "miss": int(miss.sum()), "help": 0, "pred_nonres": 0, "wasted": 0}
           for budget in budgets}
    for m in range(1, max(budgets) + 1):
        if m > pred.shape[2]:
            break
        pid = pred[..., m - 1]
        p_valid = pid >= 0
        in_act = ((pid[..., None] == actual) & act_valid).any(axis=2)
        p_res = np.take_along_axis(resident, np.where(p_valid, pid, 0).astype(np.int64)[..., None], axis=2)[..., 0]
        slot = {
            "pred": int(p_valid.sum()),
            "hit": int((p_valid & in_act).sum()),
            "help": int((p_valid & in_act & ~p_res).sum()),
            "pred_nonres": int((p_valid & ~p_res).sum()),
            "wasted": int((p_valid & ~in_act & ~p_res).sum()),
        }
        for budget in budgets:
            if m <= budget:
                rec = out[budget]
                for key in slot:
                    rec[key] += slot[key]
    return out


def ratios(rec):
    pred = rec["pred"]
    actual = rec["actual"]
    miss = rec["miss"]
    nonres = rec["pred_nonres"]
    return {
        "precision": rec["hit"] / pred if pred else 0.0,
        "recall": rec["hit"] / actual if actual else 0.0,
        "useful": rec["help"] / miss if miss else 0.0,
        "waste": rec["wasted"] / nonres if nonres else 0.0,
    }


def merge(dst, src):
    for key in dst:
        dst[key] += src[key]


def resident_mask(static_res, prev_ids, n_layer, n_expert):
    res = np.broadcast_to(static_res[None, :, :], (len(prev_ids), n_layer, n_expert)).copy()
    np.put_along_axis(res, np.where(prev_ids >= 0, prev_ids, 0).astype(np.int64), True, axis=2)
    return res


def eval_split(d, train_mask, eval_idx, orders, budgets, min_support, resident_c, lead=0):
    """Evaluate every predictor on eval_idx with a model built on train_mask.

    `lead` shifts the target: the key ending at token t-K predicts the routing
    of step t, so the prediction is made K decode steps before it is needed."""
    routes = d["routes"]
    seg = d["seg"]
    tokens = d["tokens"]
    n_layer, n_expert = d["header"]["n_layer"], d["header"]["n_expert"]
    max_m = max(budgets)
    ref_budget = min(int(d["header"]["n_expert_used"]), max_m)

    # rows whose key is K steps back and still inside the same turn
    ok = np.ones(len(tokens), dtype=bool)
    if lead > 0:
        ok[:lead] = False
        ok[lead:] = seg[lead:] == seg[:-lead]
        eval_idx = eval_idx[ok[eval_idx]]

    static_counts = np.zeros((n_layer, n_expert), dtype=np.int64)
    for il in range(n_layer):
        v = routes[train_mask, il, :].astype(np.int64)
        v = v[v >= 0]
        if v.size:
            static_counts[il] = np.bincount(v, minlength=n_expert)
    static_rank = {il: np.argsort(-static_counts[il]) for il in range(n_layer)}
    static_res = np.zeros((n_layer, n_expert), dtype=bool)
    for il in range(n_layer):
        static_res[il, static_rank[il][:resident_c]] = True

    prev = previous_routes(routes, seg)
    eval_prev = prev[eval_idx]
    actual = routes[eval_idx]
    act_valid = actual >= 0
    res = resident_mask(static_res, eval_prev, n_layer, n_expert)

    preds = {}
    coverage = {}
    for o in orders:
        ids, table = key_ids_for_order(tokens, seg, o)
        if lead > 0:
            shifted = np.full_like(ids, -1)
            shifted[lead:] = ids[:-lead]
            shifted[~ok] = -1
            ids = shifted
        model, kept = build_model(routes, ids, train_mask, n_layer, n_expert, max_m, min_support)
        preds[o] = pred_array_from_model(model, ids, eval_idx, n_layer, max_m)
        covered = sum(1 for t in eval_idx if ids[t] >= 0 and int(ids[t]) in model)
        coverage[o] = {"distinct_keys": len(table), "kept_keys": kept,
                       "coverage": covered / len(eval_idx) if len(eval_idx) else 0.0}
        del model

    static_pred = np.full((len(eval_idx), n_layer, max_m), -1, dtype=np.int16)
    for il in range(n_layer):
        static_pred[:, il, : min(max_m, len(static_rank[il]))] = static_rank[il][:max_m]

    predictors = {"oracle": pad_to(actual, max_m), "static": static_pred,
                  "recency": pad_to(eval_prev, max_m)}
    for o in orders:
        predictors["ngram%d" % o] = preds[o]
        mix = static_pred.copy()
        has = (preds[o] >= 0).any(axis=2)
        mix[has] = preds[o][has]
        predictors["ngram%d+static" % o] = mix
    backoff = static_pred.copy()
    for o in orders:
        has = (preds[o] >= 0).any(axis=2)
        backoff[has] = preds[o][has]
    predictors["backoff"] = backoff

    # the runtime predictor: the largest matching order, and nothing when no key
    # matches. Unlike `backoff` there is no static fallback, so the precision is
    # over the matched steps only, exactly like the engine's predict/L counter
    backoff_ngram = np.full_like(static_pred, -1)
    for o in orders:
        has = (preds[o] >= 0).any(axis=2)
        backoff_ngram[has] = preds[o][has]
    predictors["ngram_backoff"] = backoff_ngram

    results = {name: tally(actual, act_valid, res, pred, budgets) for name, pred in predictors.items()}
    per_layer = []
    per_layer_rec = []
    per_layer_rec_ngram = []
    for il in range(n_layer):
        rec = tally(actual[:, il : il + 1, :], act_valid[:, il : il + 1, :],
                    res[:, il : il + 1, :], backoff[:, il : il + 1, :], [ref_budget])[ref_budget]
        per_layer.append(ratios(rec))
        per_layer_rec.append(rec)
        rec_ng = tally(actual[:, il : il + 1, :], act_valid[:, il : il + 1, :],
                       res[:, il : il + 1, :], backoff_ngram[:, il : il + 1, :], [ref_budget])[ref_budget]
        per_layer_rec_ngram.append(rec_ng)
    return {"results": results, "coverage": coverage, "n_eval": len(eval_idx),
            "ref_budget": ref_budget, "per_layer": per_layer, "per_layer_rec": per_layer_rec,
            "per_layer_rec_ngram": per_layer_rec_ngram,
            "static_rank": static_rank, "budgets": budgets, "pred_backoff": backoff}


def fmt_ratio(r):
    return "%.3f" % r


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--top-m", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12, 16])
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--resident-c", type=int, default=8)
    parser.add_argument("--report", help="write a markdown report")
    parser.add_argument("--json", help="write raw numbers")
    args = parser.parse_args()

    orders = sorted(set(o for o in args.orders if o >= 1))
    budgets = sorted(set(m for m in args.top_m if m >= 1))
    d = load(args.recordings)
    h = d["header"]
    n_layer, n_expert = h["n_layer"], h["n_expert"]
    n = len(d["routes"])
    n_sessions = len(args.recordings)
    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    emit("# Expert prediction study")
    emit()
    emit("## Dataset")
    emit()
    emit("| file | size MiB | decode steps | turns | distinct tokens | max turn | truncated |")
    emit("| --- | --- | --- | --- | --- | --- | --- |")
    for fi, pf in enumerate(d["per_file"]):
        s = d["session"] == fi
        toks = d["tokens"][s]
        st = np.bincount(d["turn"][s][d["turn"][s] >= 0])
        emit("| %s | %.1f | %d | %d | %d | %d | %s |" % (
            pf["name"], pf["size"] / 1048576.0, pf["steps"], pf["turns"], len(np.unique(toks)),
            int(st.max()) if len(st) else 0,
            ("byte %d" % pf["truncated"]) if pf["truncated"] is not None else "-"))
    emit("| **total** | **%.1f** | **%d** | **%d** | **%d** | - | - |" % (
        sum(pf["size"] for pf in d["per_file"]) / 1048576.0, n,
        sum(pf["turns"] for pf in d["per_file"]), len(np.unique(d["tokens"]))))
    emit()
    emit("Model: %d layers, %d experts, top-%d. Fingerprint %d." % (
        n_layer, n_expert, h["n_expert_used"], h["model_fp"]))
    emit()

    # ---- turns split ----
    holdout = (d["turn"] % args.holdout_every) == 0 if args.holdout_every > 0 else np.zeros(n, dtype=bool)
    turns = eval_split(d, ~holdout, np.flatnonzero(holdout), orders, budgets,
                       args.min_support, args.resident_c)
    emit("## Turn split (hold out every %d-th turn)" % args.holdout_every)
    emit()
    emit("%d train steps, %d held-out steps. one marker is one server request (a turn);" % (
        int((~holdout).sum()), turns["n_eval"]))
    emit("turns of one session share context, so one file measures within-session transfer.")
    emit()

    emit("### Prediction quality (turn split)")
    emit()
    emit("| predictor | budget | precision | recall | useful | waste |")
    emit("| --- | --- | --- | --- | --- | --- |")
    for name in (["oracle", "static", "recency"] + [n for o in orders for n in
                 ("ngram%d" % o, "ngram%d+static" % o)] + ["backoff"]):
        for budget in budgets:
            r = ratios(turns["results"][name][budget])
            emit("| %s | %d | %.3f | %.3f | %.3f | %.3f |" % (
                name, budget, r["precision"], r["recall"], r["useful"], r["waste"]))
    emit()

    # ---- residency sweep ----
    rb = turns["ref_budget"]
    eval_idx = np.flatnonzero(holdout)
    prev = previous_routes(d["routes"], d["seg"])
    emit("### Prefetch benefit vs cache residency (backoff, budget %d)" % rb)
    emit()
    emit("The previous step is always resident. As the static set grows the demand misses shrink.")
    emit()
    emit("| static experts/layer | demand misses/step | useful | reads/step | reads useful |")
    emit("| --- | --- | --- | --- | --- |")
    for c in [2, 4, 8, 16, 32, 64, 128]:
        if c > n_expert:
            continue
        sr = np.zeros((n_layer, n_expert), dtype=bool)
        for il in range(n_layer):
            sr[il, turns["static_rank"][il][:c]] = True
        res = resident_mask(sr, prev[eval_idx], n_layer, n_expert)
        rec = tally(d["routes"][eval_idx], d["routes"][eval_idx] >= 0, res,
                    turns["pred_backoff"], [rb])[rb]
        r = ratios(rec)
        emit("| %d | %.1f | %.1f%% | %.1f | %.1f%% |" % (
            c, rec["miss"] / max(len(eval_idx), 1), 100 * r["useful"],
            rec["pred_nonres"] / max(len(eval_idx), 1), 100 * (1 - r["waste"])))
    emit()

    # ---- per-layer ----
    emit("### Per-layer usefulness (backoff, budget %d)" % rb)
    emit()
    emit("| layer | precision | recall | useful |")
    emit("| --- | --- | --- | --- |")
    for il, r in enumerate(turns["per_layer"]):
        emit("| %d | %.3f | %.3f | %.3f |" % (il, r["precision"], r["recall"], r["useful"]))
    emit()

    # ---- cross session ----
    if n_sessions > 1:
        emit("## Cross-session transfer (leave-one-session-out)")
        emit()
        emit("Each row trains on the other sessions and evaluates on the held-out one. "
             "This is the transfer a per-workflow profile must survive.")
        emit()
        emit("| held-out session | steps | coverage o1 | precision | recall | useful | waste |")
        emit("| --- | --- | --- | --- | --- | --- | --- |")
        agg = {b: {"pred": 0, "hit": 0, "actual": 0, "miss": 0, "help": 0,
                   "pred_nonres": 0, "wasted": 0} for b in budgets}
        per_layer_cross = {}
        for s in range(n_sessions):
            si = np.flatnonzero(d["session"] == s)
            tr = d["session"] != s
            fold = eval_split(d, tr, si, orders, budgets, args.min_support, args.resident_c)
            per_layer_cross[s] = fold["per_layer"]
            for b in budgets:
                merge(agg[b], fold["results"]["backoff"][b])
            r = ratios(fold["results"]["backoff"][rb])
            emit("| %s | %d | %.2f | %.3f | %.3f | %.3f | %.3f |" % (
                d["per_file"][s]["name"], len(si), fold["coverage"][orders[0]]["coverage"],
                r["precision"], r["recall"], r["useful"], r["waste"]))
        rm = ratios(agg[rb])
        emit("| **mean** | - | - | %.3f | %.3f | %.3f | %.3f |" % (
            rm["precision"], rm["recall"], rm["useful"], rm["waste"]))
        emit()
        emit("Mean cross-session prefetch efficiency: **%.1f%%** of the reads hit a routed expert." % (
            100 * (1 - rm["waste"])))
        emit()

        # does the per-layer ranking transfer?
        ph = np.array([r["useful"] for r in turns["per_layer"]])
        corrs = []
        for s in range(n_sessions):
            pc = np.array([r["useful"] for r in per_layer_cross[s]])
            corrs.append(float(np.corrcoef(ph, pc)[0, 1]))
        emit("Per-layer usefulness correlation (turn-split ranking vs cross-session): %s" % (
            ", ".join("session %d: %.3f" % (s, c) for s, c in enumerate(corrs))))
        emit()

    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        print("wrote %s" % args.report)
    if args.json:
        payload = {
            "dataset": {"steps": n, "sessions": n_sessions, "budgets": budgets},
            "turn_split": {"coverage": {str(o): turns["coverage"][o] for o in orders},
                           "results": {name: {str(b): ratios(turns["results"][name][b]) for b in budgets}
                                       for name in turns["results"]}},
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("wrote %s" % args.json)


if __name__ == "__main__":
    main()
