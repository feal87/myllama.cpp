#!/usr/bin/env python3
"""Warm the RAM disk decode cache with one real open-ended decode.

Random-token prompts make the model collapse into a repetition loop, which
routes a handful of experts and never fills the cache. Always warm with real
text like this before measuring prefill.

Check the RAM tier report afterwards: it must show `layers N/N` and a large
resident count.
"""

import argparse
import time

from _common import DEFAULT_BASE, post

PROMPT = (
    "Write me a long, detailed story about a tree and a house. Include dialogue, "
    "descriptions of the place, and a surprising twist at the end. Do not stop "
    "early, keep writing until the token limit."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--tokens", type=int, default=500, help="decode tokens to generate")
    args = parser.parse_args()

    started = time.time()
    result = post(
        args.base,
        "/completion",
        {
            "prompt": PROMPT,
            "n_predict": args.tokens,
            "temperature": 1.0,
            "top_k": 20,
            "top_p": 0.95,
            "seed": 1,
            "cache_prompt": False,
            "stream": False,
        },
        timeout=7200,
    )
    timings = result.get("timings", {})
    print(
        f"warm: gen={result.get('tokens_predicted')} "
        f"tok/s={timings.get('predicted_per_second', 0):.2f} "
        f"wall={time.time() - started:.0f}s"
    )


if __name__ == "__main__":
    main()
