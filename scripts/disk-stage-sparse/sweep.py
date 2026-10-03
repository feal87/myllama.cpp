#!/usr/bin/env python3
"""Measure prompt eval time for growing real-text prompt prefixes.

Builds one token pool from the given text files, then sends pool[:n] for each
size with cache_prompt=false and n_predict=1. Run once with the sparse cut at 0
(full slab) and once above the ubatch size (sparse), both on a warm cache.

Example:
  python sweep.py 64 128 200 300 430 700 1200 2000 \
      --files ../../README.md ../../docs/build.md ../../docs/function-calling.md
"""

import argparse
import time

from _common import DEFAULT_BASE, post, read_prompt, tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="+", type=int, help="prompt prefix lengths in tokens")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--files", nargs="+", required=True, help="real text files for the pool")
    args = parser.parse_args()

    text = "\n\n".join(read_prompt(path) for path in args.files)
    pool = tokenize(args.base, text)
    print(f"pool {len(pool)} tokens", flush=True)

    for n in args.sizes:
        if n > len(pool):
            print(f"n={n:6d} skipped, pool is only {len(pool)} tokens", flush=True)
            continue
        started = time.time()
        result = post(
            args.base,
            "/completion",
            {
                "prompt": pool[:n],
                "n_predict": 1,
                "temperature": 0.0,
                "cache_prompt": False,
                "stream": False,
            },
            timeout=7200,
        )
        timings = result.get("timings", {})
        print(
            f"n={n:6d} eval={result.get('tokens_evaluated'):6d} "
            f"prompt_ms={timings.get('prompt_ms', 0):9.1f} "
            f"tok/s={timings.get('prompt_per_second', 0):8.2f} "
            f"wall={time.time() - started:6.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
