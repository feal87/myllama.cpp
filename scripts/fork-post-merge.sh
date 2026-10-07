#!/usr/bin/env bash
# Run after an upstream merge. Drops the upstream features that duplicate fork ones.
# See "Fork merge policy" in AGENTS.md.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# upstream GPU MoE expert cache (the fork uses src/llama-moecache.*)
git rm -q -f src/llama-moe-cache.cpp src/llama-moe-cache.h 2>/dev/null || true

if git grep -n -E "moe_cache_size|llama-moe-cache" -- common src include tests; then
    echo "error: upstream MoE cache leftovers found, strip them before committing" >&2
    exit 1
fi

echo "ok: no upstream MoE cache leftovers"
