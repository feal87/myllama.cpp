#!/usr/bin/env bash
# Run after an upstream merge. Drops the upstream features that duplicate fork ones.
# See "Fork merge policy" in AGENTS.md.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# upstream GPU MoE expert cache (the fork uses src/llama-moecache.*)
git rm -q -f src/llama-moe-cache.cpp src/llama-moe-cache.h 2>/dev/null || true

# both classes are named llama_moe_cache, so also match the upstream-only members
leftovers="moe_cache_size|moe_cache_slru|llama_moe_cache_ptr|get_slot_map"
leftovers+="|llama_moe_cache_layer_experts|llama_moe_cache_same_layout|llama_moe_cache_is_host_weight"
leftovers+="|llama-moe-cache|build_moe_cache_slots|moe-cache-mib|MOE_CACHE_MIB"

if git grep -n -E "$leftovers" -- common src include tests tools; then
    echo "error: upstream MoE cache leftovers found, strip them before committing" >&2
    exit 1
fi

echo "ok: no upstream MoE cache leftovers"
