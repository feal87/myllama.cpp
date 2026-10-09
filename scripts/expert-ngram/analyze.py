#!/usr/bin/env python3
"""Offline expert-prediction study over --expert-ngram-record files.

The recording (see README.md) is a stream of decode steps: the token a graph
consumed and the experts each MoE layer routed for it. This tool answers one
question: given the last N decode tokens, how well can we predict the experts
the next graph will route? It builds an n-gram -> expert model on a training
split of the prompts and reports precision/recall on a held-out split, next to
the baselines that matter (a static top-M set, and the previous step's experts).

The metric that decides whether prefetch can help is `useful`: of the routed
experts that were NOT already resident (static set + previous step), how many
did the predictor cover before the read. Precision says how much of the
prediction is wasted; recall says how much of the demand it covers.

Stdlib only: the model is built in a temporary SQLite database so the tool does
not need to hold every (key, layer, expert) count in RAM.
"""

import argparse
import os
import sqlite3
import struct
import sys
import tempfile

HEADER = struct.Struct("<8sIIIIQQ")
REC_MAGIC = b"LENGREC1"
REC_VERSION = 1
REC_PROMPT_BEGIN = 1
REC_DECODE_STEP = 2
STEP_HEAD = struct.Struct("<iH")   # token, n_layer_obs
LAYER_HEAD = struct.Struct("<HH")  # layer, k


def read_recording(path):
    """Yield ('header', dict) then ('prompt',) / ('step', token, {layer: ids}).

    A recording can end mid-record if the engine was killed while appending, so a
    truncated tail yields a final ('truncated', offset) event and stops instead
    of raising. Every complete record before it is still usable.
    """
    with open(path, "rb") as f:
        raw = f.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("%s: short header" % path)
        magic, version, n_layer, n_expert, n_used, model_fp, tok_fp = HEADER.unpack(raw)
        if magic != REC_MAGIC:
            raise ValueError("%s: not a llama expert n-gram recording" % path)
        if version != REC_VERSION:
            raise ValueError("%s: recording version %d, expected %d" % (path, version, REC_VERSION))
        yield "header", {
            "n_layer": n_layer,
            "n_expert": n_expert,
            "n_expert_used": n_used,
            "model_fp": model_fp,
            "tokenizer_fp": tok_fp,
        }
        while True:
            offset = f.tell()
            b = f.read(1)
            if not b:
                return
            rectype = b[0]
            if rectype == REC_PROMPT_BEGIN:
                yield "prompt", None
            elif rectype == REC_DECODE_STEP:
                head = f.read(STEP_HEAD.size)
                if len(head) != STEP_HEAD.size:
                    yield "truncated", offset
                    return
                token, n_obs = STEP_HEAD.unpack(head)
                layers = {}
                for _ in range(n_obs):
                    layer_head = f.read(LAYER_HEAD.size)
                    if len(layer_head) != LAYER_HEAD.size:
                        yield "truncated", offset
                        return
                    il, k = LAYER_HEAD.unpack(layer_head)
                    blob = f.read(4 * k)
                    if len(blob) != 4 * k:
                        yield "truncated", offset
                        return
                    layers[il] = struct.unpack("<%di" % k, blob)
                yield "step", (token, layers)
            else:
                raise ValueError("%s: unknown record type %d at offset %d" % (path, rectype, offset))


def key_blob(window, order):
    return struct.pack("<%di" % order, *window[-order:])


def open_db():
    fd, path = tempfile.mkstemp(prefix="expert-ngram-", suffix=".sqlite")
    os.close(fd)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=MEMORY")
    db.executescript(
        "CREATE TABLE keys (ord INTEGER, key BLOB, support INTEGER, PRIMARY KEY (ord, key)) WITHOUT ROWID;"
        "CREATE TABLE counts (ord INTEGER, key BLOB, layer INTEGER, expert INTEGER, n INTEGER,"
        " PRIMARY KEY (ord, key, layer, expert)) WITHOUT ROWID;"
    )
    return db, path


def build(db, paths, orders, holdout_every):
    """Aggregate the training prompts into db. Returns dataset stats."""
    upsert_key = "INSERT INTO keys VALUES (?,?,?) ON CONFLICT(ord,key) DO UPDATE SET support=support+excluded.support"
    upsert_count = ("INSERT INTO counts VALUES (?,?,?,?,?) "
                    "ON CONFLICT(ord,key,layer,expert) DO UPDATE SET n=n+excluded.n")
    stats = {"prompts": 1, "train_prompts": 0, "steps": 0, "train_steps": 0, "layers": set(),
             "expert_max": 0, "max_prompt_steps": 0, "truncated": None}
    prompt_index = 0
    cur_train = holdout_every <= 0 or (prompt_index % holdout_every) != 0
    if cur_train:
        stats["train_prompts"] = 1
    window = []
    prompt_steps = 0
    pending_keys = []
    pending_counts = []

    def flush():
        if pending_keys:
            db.executemany(upsert_key, pending_keys)
            db.executemany(upsert_count, pending_counts)
            pending_keys.clear()
            pending_counts.clear()

    for path in paths:
        for kind, payload in read_recording(path):
            if kind == "header":
                stats["layers"].update(range(payload["n_layer"]))
                stats["expert_max"] = max(stats["expert_max"], payload["n_expert"])
                continue
            if kind == "prompt":
                prompt_index += 1
                stats["prompts"] = max(stats["prompts"], prompt_index + 1)
                window = []
                prompt_steps = 0
                cur_train = holdout_every <= 0 or (prompt_index % holdout_every) != 0
                if cur_train:
                    stats["train_prompts"] += 1
                continue
            if kind == "truncated":
                stats["truncated"] = payload
                continue
            token, layers = payload
            stats["steps"] += 1
            prompt_steps += 1
            stats["max_prompt_steps"] = max(stats["max_prompt_steps"], prompt_steps)
            window.append(token)
            if not cur_train:
                continue
            stats["train_steps"] += 1
            local_keys = {}
            local_counts = {}
            for order in orders:
                if len(window) < order:
                    continue
                kb = key_blob(window, order)
                local_keys[(order, kb)] = local_keys.get((order, kb), 0) + 1
                for il, ids in layers.items():
                    for e in ids:
                        ck = (order, kb, il, e)
                        local_counts[ck] = local_counts.get(ck, 0) + 1
            pending_keys.extend((o, k, n) for (o, k), n in local_keys.items())
            pending_counts.extend((o, k, il, e, n) for (o, k, il, e), n in local_counts.items())
            if len(pending_counts) >= 50000:
                flush()
    flush()
    db.commit()
    return stats


class Model:
    """Top-M experts per (order, key, layer), with a bounded lookup cache."""

    def __init__(self, db, top_m, min_support, cache_limit=500000):
        self.db = db
        self.top_m = top_m
        self.min_support = min_support
        self.cache = {}
        self.cache_limit = cache_limit

    def lookup(self, order, kb):
        ck = (order, kb)
        if ck in self.cache:
            return self.cache[ck]
        cur = self.db.execute(
            "SELECT c.layer, c.expert, c.n FROM counts c JOIN keys ks ON ks.ord=c.ord AND ks.key=c.key "
            "WHERE c.ord=? AND c.key=? AND ks.support>=? ORDER BY c.layer, c.n DESC, c.expert",
            (order, kb, self.min_support))
        model = {}
        for layer, expert, _n in cur:
            lst = model.setdefault(layer, [])
            if len(lst) < self.top_m:
                lst.append(expert)
        if len(self.cache) >= self.cache_limit:
            self.cache.clear()
        self.cache[ck] = model
        return model

    def static_top(self, budget):
        """Per-layer top-`budget` experts over every key (the static baseline)."""
        cur = self.db.execute(
            "SELECT layer, expert, SUM(n) AS total FROM counts GROUP BY layer, expert "
            "ORDER BY layer, total DESC, expert")
        out = {}
        for layer, expert, _t in cur:
            lst = out.setdefault(layer, [])
            if len(lst) < budget:
                lst.append(expert)
        return out


class Score:
    def __init__(self):
        self.pred = 0
        self.hit = 0
        self.actual = 0
        self.miss = 0
        self.help = 0
        self.steps = 0
        self.covered_steps = 0

    def add(self, predicted, actual, resident):
        self.steps += 1
        pred = set(predicted)
        act = set(actual)
        self.pred += len(pred)
        self.actual += len(act)
        self.hit += len(pred & act)
        miss = act - resident
        self.miss += len(miss)
        self.help += len(pred & miss)
        if pred:
            self.covered_steps += 1

    def fmt(self):
        precision = self.hit / self.pred if self.pred else 0.0
        recall = self.hit / self.actual if self.actual else 0.0
        useful = self.help / self.miss if self.miss else 0.0
        return "prec %.3f  rec %.3f  useful %.3f" % (precision, recall, useful)


def evaluate(db, paths, orders, holdout_every, top_ms, min_support, resident_c):
    model = Model(db, max(top_ms), min_support)
    static_all = model.static_top(max(top_ms))
    resident_static = {il: set(es[:resident_c]) for il, es in model.static_top(resident_c).items()}

    # one score bucket per (predictor, budget)
    names = ["oracle", "static", "recency"]
    for o in orders:
        names.append("ngram%d" % o)
        names.append("ngram%d+static" % o)
    scores = {(n, m): Score() for n in names for m in top_ms}

    prompt_index = 0
    window = []
    prev = {}
    prompt_steps = 0
    for path in paths:
        for kind, payload in read_recording(path):
            if kind == "header":
                continue
            if kind == "prompt":
                prompt_index += 1
                window = []
                prev = {}
                prompt_steps = 0
                continue
            if kind == "truncated":
                continue
            token, layers = payload
            prompt_steps += 1
            window.append(token)
            if holdout_every > 0 and (prompt_index % holdout_every) == 0:
                prev = layers
                continue
            key_cache = {}
            for o in orders:
                key_cache[o] = key_blob(window, o) if len(window) >= o else None
            for il, ids in layers.items():
                actual = set(ids)
                resident = set(resident_static.get(il, ())) | set(prev.get(il, ()))
                # predictors keyed by budget
                for m in top_ms:
                    scores[("oracle", m)].add(actual, actual, resident)
                    scores[("static", m)].add(static_all.get(il, ())[:m], actual, resident)
                    scores[("recency", m)].add(list(prev.get(il, ()))[:m], actual, resident)
                    for o in orders:
                        pred = None
                        fallback = static_all.get(il, ())[:m]
                        if key_cache[o] is not None:
                            hit = model.lookup(o, key_cache[o]).get(il, [])[:m]
                            if hit:
                                pred = hit
                        scores[("ngram%d" % o, m)].add(pred if pred is not None else (), actual, resident)
                        scores[("ngram%d+static" % o, m)].add(pred if pred is not None else fallback, actual, resident)
            prev = layers

    return scores, static_all


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+", help="--expert-ngram-record files")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3],
                        help="n-gram orders to build and evaluate (default: 1 2 3)")
    parser.add_argument("--top-m", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                        help="per-layer prediction budgets to sweep (default: 1 2 4 8 16)")
    parser.add_argument("--min-support", type=int, default=2,
                        help="minimum times a key must be seen on the train split to be used "
                             "(default: 2)")
    parser.add_argument("--holdout-every", type=int, default=5,
                        help="hold out every K-th prompt for evaluation, 0 = evaluate on the "
                             "train prompts too (default: 5)")
    parser.add_argument("--resident-c", type=int, default=0,
                        help="per-layer size of the simulated static resident set; 0 = use the "
                             "largest budget. The previous step's experts are always resident")
    args = parser.parse_args()

    orders = sorted(set(o for o in args.orders if o >= 1))
    top_ms = sorted(set(m for m in args.top_m if m >= 1))
    if not orders or not top_ms:
        parser.error("orders and top-m must be positive")
    resident_c = args.resident_c if args.resident_c > 0 else max(top_ms)

    db, db_path = open_db()
    try:
        stats = build(db, args.recordings, orders, args.holdout_every)
        print("dataset")
        print("  recordings      : %s" % ", ".join(args.recordings))
        print("  prompts         : %d (%d train)" % (stats["prompts"], stats["train_prompts"]))
        print("  decode steps    : %d" % stats["steps"])
        print("  train steps     : %d" % stats["train_steps"])
        print("  moe layers      : %d" % len(stats["layers"]))
        print("  experts (max)   : %d" % stats["expert_max"])
        print("  max prompt steps: %d" % stats["max_prompt_steps"])
        if stats["truncated"] is not None:
            print("  truncated tail  : last record cut at byte %d, ignored" % stats["truncated"])
        print("  holdout every   : %d prompt(s)" % args.holdout_every)

        scores, static_all = evaluate(db, args.recordings, orders, args.holdout_every,
                                      top_ms, args.min_support, resident_c)
        print("")
        print("prediction quality (held-out prompts; resident = static top-%d + previous step)"
              % resident_c)
        for m in top_ms:
            print("  budget M=%d" % m)
            for name in ["oracle", "static", "recency"] + [n for o in orders for n in
                                                           ("ngram%d" % o, "ngram%d+static" % o)]:
                print("    %-14s %s" % (name, scores[(name, m)].fmt()))
    finally:
        db.close()
        os.unlink(db_path)


if __name__ == "__main__":
    sys.exit(main())
