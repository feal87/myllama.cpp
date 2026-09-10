#!/usr/bin/env python3
# Local benchmark driver for ggml_cuda_op_top_k implementations.
# Requires the GGML_CUDA_TOPK_IMPL / GGML_TOPK_BENCH / GGML_BACKEND_OPS_MAX_RUNS changes.
import ast
import os
import re
import subprocess
import sys

EXE = os.path.join("build", "bin", "test-backend-ops.exe")
BACKEND = "CUDA0"
MAX_RUNS = "16"
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

IMPLS = ["auto", "devicetok", "argsort", "bitonic", "radix"]


def build_shapes():
    shapes = []
    for nrows in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        shapes.append((512, nrows, 10))
    for ncols in [128, 256, 1024, 2048, 4096, 8192]:
        shapes.append((ncols, 1, 10))
        shapes.append((ncols, 512, 10))
    for k in [1, 2, 4, 8, 16, 32, 64]:
        shapes.append((512, 1, k))
        shapes.append((512, 512, k))
    # dedup, keep order
    seen = set()
    out = []
    for s in shapes:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def variant(ncols, nrows, k):
    return f"TOP_K(type=f32,ne=[{ncols},{nrows},1,1],k={k},ties=0)"


INSERT_RE = re.compile(r"INSERT INTO test_backend_ops .* VALUES \((.*)\);")


def run_impl(impl, shapes):
    flt = ",".join(variant(*s) for s in shapes)
    env = dict(os.environ)
    env["GGML_TOPK_BENCH"] = "1"
    env["GGML_BACKEND_OPS_MAX_RUNS"] = MAX_RUNS
    if impl == "auto":
        env.pop("GGML_CUDA_TOPK_IMPL", None)
    else:
        env["GGML_CUDA_TOPK_IMPL"] = impl
    proc = subprocess.run(
        [EXE, "perf", "-b", BACKEND, "--output", "sql", "-o", flt],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    out = {}
    for line in proc.stdout.splitlines():
        m = INSERT_RE.search(line)
        if not m:
            continue
        try:
            vals = ast.literal_eval("(" + m.group(1) + ")")
        except Exception:
            continue
        # fields: test_time, build_commit, backend_name, op_name, op_params, test_mode,
        #         supported, passed, error_message, time_us, flops, bw, mem_kb, n_runs, ...
        if vals[3] != "TOP_K" or vals[5] != "perf":
            continue
        op_params = vals[4]
        mm = re.search(r"ne=\[(\d+),(\d+),1,1\],k=(\d+)", op_params)
        if not mm:
            continue
        key = (int(mm.group(1)), int(mm.group(2)), int(mm.group(3)))
        out[key] = float(vals[9])
    return out


def main():
    shapes = build_shapes()
    results = {}  # (impl, shape) -> us (min over repeats)
    for impl in IMPLS:
        # bitonic argsort is only valid up to 1024 columns (one thread per column)
        use = [s for s in shapes if impl != "bitonic" or s[0] <= 1024]
        for rep in range(REPEATS):
            print(f"running {impl} (repeat {rep + 1}/{REPEATS}) ...", file=sys.stderr, flush=True)
            r = run_impl(impl, use)
            for s, t in r.items():
                key = (impl, s)
                results[key] = t if key not in results else min(results[key], t)

    # header
    print()
    print(f"{'ncols':>6} {'nrows':>6} {'k':>5} | " +
          " ".join(f"{i:>12}" for i in IMPLS) + " | best      vs devicetok")
    print("-" * (24 + 16 * len(IMPLS) + 24))
    for s in shapes:
        ncols, nrows, k = s
        row = []
        for impl in IMPLS:
            t = results.get((impl, s))
            row.append(f"{t:12.2f}" if t is not None else f"{'-':>12}")
        avail = {impl: results[(impl, s)] for impl in IMPLS if (impl, s) in results}
        if avail:
            best = min(avail, key=avail.get)
            dev = avail.get("devicetok")
            ratio = f"{dev / avail[best]:.2f}x" if dev else "-"
            beststr = f"{best:>9} {ratio}"
        else:
            beststr = "-"
        print(f"{ncols:>6} {nrows:>6} {k:>5} | " + " ".join(row) + f" | {beststr}")


if __name__ == "__main__":
    main()
