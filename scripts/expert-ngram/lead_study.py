#!/usr/bin/env python3
"""Measure how much the n-gram prediction degrades when it is made K decode
steps before the routing it predicts (the lead, or lookahead, in decode steps).

The runtime predictor can only help if it predicts far enough ahead that the
read lands before the layer needs it. This evaluates lead 0 (predict the current
step, what the engine does today), lead 1 and lead 2 on the same cross-session
folds as the exporter and prints the per-layer useful rate at each lead.

  python lead_study.py session1.rec session2.rec --leads 0 1 2
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study import eval_split, load  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--leads", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--top-m", type=int, default=6)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--resident-c", type=int, default=8)
    args = parser.parse_args()

    d = load(args.recordings)
    n_layer = d["header"]["n_layer"]
    sessions = sorted(set(int(s) for s in d["session"]))

    by_lead = {}
    by_lead_static = {}
    for lead in args.leads:
        agg = [{"pred": 0, "hit": 0, "actual": 0, "miss": 0, "help": 0,
                "pred_nonres": 0, "wasted": 0} for _ in range(n_layer)]
        agg_static = [{"pred": 0, "hit": 0, "actual": 0, "miss": 0, "help": 0,
                       "pred_nonres": 0, "wasted": 0} for _ in range(n_layer)]
        for s in sessions:
            tr = d["session"] != s
            ev = np.flatnonzero(d["session"] == s)
            result = eval_split(d, tr, ev, args.orders, [args.top_m], args.min_support,
                                args.resident_c, lead=lead)
            for il, rec in enumerate(result["per_layer_rec_ngram"]):
                for key in agg[il]:
                    agg[il][key] += int(rec[key])
            for il, rec in enumerate(result["per_layer_rec"]):
                for key in agg_static[il]:
                    agg_static[il][key] += int(rec[key])
        by_lead[lead] = agg
        by_lead_static[lead] = agg_static

    print("top_m=%d, orders=%s, min_support=%d, resident_c=%d" % (
        args.top_m, args.orders, args.min_support, args.resident_c))
    print()
    print("global precision (routed / predicted), n-gram only, no static fallback")
    print("%6s %12s %12s %12s" % ("lead", "precision", "predicted", "routed"))
    for lead in args.leads:
        agg = by_lead[lead]
        pred = sum(r["pred"] for r in agg)
        hit = sum(r["hit"] for r in agg)
        print("%6d %11.2f%% %12d %12d" % (
            lead, 100.0 * hit / pred if pred else 0.0, pred, hit))

    print()
    print("global precision with a static fallback (the old study predictor)")
    print("%6s %12s %12s %12s" % ("lead", "precision", "predicted", "routed"))
    for lead in args.leads:
        agg = by_lead_static[lead]
        pred = sum(r["pred"] for r in agg)
        hit = sum(r["hit"] for r in agg)
        print("%6d %11.2f%% %12d %12d" % (
            lead, 100.0 * hit / pred if pred else 0.0, pred, hit))

    print()
    print("per-layer precision (routed / predicted), n-gram only, by lead")
    header = "%6s" % "layer" + "".join("%10s" % ("lead %d" % lead) for lead in args.leads)
    print(header)
    for il in range(n_layer):
        row = "%6d" % il
        for lead in args.leads:
            rec = by_lead[lead][il]
            row += "%9.1f%%" % (100.0 * rec["hit"] / rec["pred"] if rec["pred"] else 0.0)
        print(row)


if __name__ == "__main__":
    main()
