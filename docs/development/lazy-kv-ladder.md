# Lazy KV cache ladder

Status: implemented in `src/llama-kv-cache.cpp`. Same-role and per-role ladders.

```
LLAMA_KV_CACHE_LAZY_QUANT=1                            # legacy: one step per role, derived from its type
LLAMA_KV_CACHE_LAZY_QUANT=f16,q8_0,q4_0               # one list for both roles
LLAMA_KV_CACHE_LAZY_QUANT=k=f16,q8_0,q4_0;v=q4_0      # one list per role
LLAMA_KV_CACHE_LAZY_QUANT=f16/f16,q8_0/q8_0,q8_0/q4_0 # explicit (k, v) stages
```

K and V must be non-increasing across stages, and the last stage must equal the
configured `-ctk` / `-ctv`. A stage whose shared capacity does not grow is
skipped with a warning (with the default per-role sizing this is common for a
stage that only drops the non-binding role). A role that is not listed stays at
its configured type. On restore the cache advances by K first, then by V, so K
may repeat while only V drops.

## Motivation

The KV cache can start in a higher-precision type and be quantized in flight
when it fills. Today the ladder has exactly two rungs and both roles (K and V)
must use the same type:

- target `Q8_0` starts as `F16`
- target `Q4_0` starts as `Q8_0`

Users want:

- more than two steps, so precision is traded away gradually
- a threshold per step instead of "only when full"
- independent K and V ladders, e.g. K `q8_0` and V `q4_0`

## Current design

Where it lives:

- `src/llama-kv-cache.cpp` constructor around the `LLAMA_KV_CACHE_LAZY_QUANT`
  env read.
- `src/llama-kv-cache.h`: `cache_format`, `current`, `target`, `overlay`,
  `has_lazy_ladder`, `lazy_quant_pending`, and the per layer views in
  `kv_layer`.
- `try_lazy_quantize()`: converts `current` to `target` in one step and grows
  the cells.
- `get_has_lazy_quant()`: `current.size != target.size`.
- `get_needs_lazy_quant()`: `get_has_lazy_quant() && lazy_quant_pending`.
- `llama_kv_cache_read_conversion()`: allows `F16 -> Q8_0` and `Q8_0 -> Q4_0`
  on checkpoint restore.
- Wrappers (`iswa`, `dsa`, `dsa_iswa`, `msa`, `dsv4`, `memory_hybrid`,
  `memory_hybrid_iswa`, `memory_hybrid_idx`) delegate `try_lazy_quantize`,
  `get_has_lazy_quant`, and now `can_reset_lazy_quant` / `reset_lazy_quant`.

Sizing: the backing buffer is one pool per layer, allocated for the final types
and `N = target.size` cells:

```
pool = align_up(N * row(type_k), 256) + N * row(type_v)
```

Every step is a view over that same pool. K always starts at offset 0 and V
starts at `align_up(size * row(step_k), 256)`, so an asymmetric step spends the
bytes the other role leaves unused. The shared cell count of step `i` is the
largest value whose two aligned regions fit the pool of every layer:

```
cap(i) = max c such that align_up(c * row(step_k), 256) + c * row(step_v) <= pool
```

This is at least the old K/V minimum, so no bytes are wasted.

## Design

### Ladder model

An ordered list of steps, each with a K and a V type:

```cpp
struct cache_format {
    uint32_t size;
    ggml_type type_k;
    ggml_type type_v;
    uint32_t n_rot_k;
    uint32_t n_rot_v;
};

std::vector<cache_format> lazy_ladder; // [0] = overlay, back() = target
```

Rules:

- at least 2 stages
- K and V do not increase in bytes per element across stages
- the last stage must equal the configured `-ctk` / `-ctv`
- every type must be a supported KV cache type
- a stage whose shared capacity does not grow is skipped

K drives the restore advance first (the serialized K type), then V refines it
(the serialized V type), so K may repeat as long as the `(k, v)` pairs are
ordered. This is what lets an asymmetric target such as `-ctk q8_0 -ctv q4_0`
use `f16/f16, q8_0/q8_0, q8_0/q4_0`.

### Capacity and thresholds

For step `i`, K and V share one cell count out of one pool:

```
k_bytes(i) = cap(i) * row(ladder[i].type_k)
v_bytes(i) = cap(i) * row(ladder[i].type_v)
k_bytes(i) + v_bytes(i) <= pool
```

The shared cell count is what the single attention index requires. Filling the
pool instead of the per-role minimum lets an asymmetric step borrow from the
other role. If a role has no cache (MLA / indexer), the pool holds K only.

By default, a step advances when `find_slot` cannot place the batch, which is
exactly when `cap(i)` is reached. An explicit earlier trigger can be added as
an optional per-step fraction `trigger(i) <= cap(i)`, evaluated in `prepare`
or before `decode`. Converting earlier than capacity only loses precision
sooner; it does not save memory, so the default should stay capacity based and
the explicit trigger should be opt-in.

### Per-role ladders

Implemented. Each rung carries both types, and both roles use the shared cell
count from the pool. A role with no cache (MLA / indexer) leaves the pool to K.
Both roles advance together, so `current.size` stays consistent.

### Config surface

Implemented as the existing env var, so no CLI plumbing is needed:

```
LLAMA_KV_CACHE_LAZY_QUANT=1                          # legacy: one step per role
LLAMA_KV_CACHE_LAZY_QUANT=f16,q8_0,q4_0             # one list for both roles
LLAMA_KV_CACHE_LAZY_QUANT=k=f16,q8_0,q4_0;v=q4_0    # one list per role
```

A missing role stays at its configured type. A bad ladder disables lazy quant
with a warning instead of aborting.

### Conversion

`try_lazy_quantize()` advances `current` to the next step instead of jumping to
`target`. The existing `llama_kv_cache_convert` path through float is unchanged;
the destination is the next step's view.

The steps share the pool, so a new K region can overlap the current V region.
When K grows on a transition, old V is staged to the host first. K converts in
place, then V is written from the stage. Host memory is bounded by one layer's V
and no device scratch is needed, so the pool size never changes. When K does not
grow, the transition is safe in place and nothing is staged. This is what makes
non-monotone ladders such as `f16/f16, q8_0/q8_0, q8_0/q4_0, q4_0/q4_0` work.

The per layer struct gets one view per step:

```cpp
std::vector<ggml_tensor *> k_step;              // one per step, k_step.back() == k_target
std::vector<ggml_tensor *> v_step;
std::vector<std::vector<ggml_tensor *>> k_stream_step;
std::vector<std::vector<ggml_tensor *>> v_stream_step;
```

`layer.k` / `layer.k_stream` point into the active step. Advancing is a pointer
update plus the data conversion. `k_target` / `v_target` are the final step
views, so the worst-case graph reserve and the reported size stay the pool.

The decode retry loop already calls `try_lazy_quantize()` once per
`FAILED_PREPARE` retry, so a batch that needs more cells than the next step
provides advances again on the next iteration. No new loop is required.

Checkpoint restore needs the same treatment:

- `state_read_sinfo()`: advance until `cell_count` fits
- `state_read_data()`: advance while the serialized type is further down the
  ladder than `current`
- `llama_kv_cache_read_conversion()`: accept any earlier-to-later pair, since
  the conversion is float mediated. Keep the current rotation rule
  (`src_rot == 0 || src_rot == dst_rot`).

### Reset

`reset_lazy_quant()` already exists. With a ladder it should set `current` to
`ladder[0]`, shrink the cells to `ladder[0]` capacity, and restore the step 0
views. It still requires an empty cache.

### Reserve and memory reporting

`get_size_target()` stays the final cell count, so the worst-case graph reserve
does not change. `sched_need_reserve` after a step is redundant once the
reserve is final sized, but can be kept for safety. `memory_breakdown()` should
report the backing (final) allocation.

## Invariants

- `ladder` is strictly decreasing in bytes per element and ends at the
  configured cache types.
- `current.size == cap(step)` and `target.size == cap(last)`.
- K and V always share `current.size`; both roles advance together.
- A checkpoint with a type earlier than `current` converts forward; a type
  equal to a later step advances first.
- `reset_lazy_quant()` never runs while any stream has data.

## Risks and open questions

1. Per-role ladders and no-V caches. MLA has no V cache, so the pool holds K
   only and an asymmetric stage is a no-op. The qwen4exp indexer also has no V
   and mirrors the attention layout cell for cell, so it must not run its own
   ladder: it is only created when QSA is on (`LLAMA_QSA_ALLOW`), and with QSA
   off the indexer cache is not created at all.
2. Rotation per step. The `n_rot` decision is currently "quantized type gets a
   rotation, F16 does not". A step with an unusual type (for example `Q6_K`)
   needs an explicit rule.
3. Checkpoint compatibility across a config change. The store key does not
   include the lazy ladder, only the configured types. Conversion handles
   mixed types, but adding the ladder to `server_prompt_cache_key` avoids
   surprises.
4. More tensors per layer. N step views per role per layer. For N of 3 to 4
   this is small, but it grows the tensor count.
5. Threshold semantics. Capacity based thresholds need no new state. Explicit
   per-step triggers need a decision about what happens when a trigger is
   larger than the step capacity (reject) or smaller (convert early).
6. `test-llama-archs --lazy-kv` is the existing lazy regression. It should
   gain a ladder case once the same-role ladder lands.

## Phased plan

Implemented:

1. Same-role multi-step ladder.
2. Per-role ladders with one pool per layer and the staged transition.
3. Explicit stages and docs.

Validated with `tests/test-llama-archs --lazy-kv` on CPU and CUDA, and end to
end on the qwen4exp hybrid model with the ladder
`f16/f16, q8_0/q8_0, q8_0/q4_0, q4_0/q4_0`, which crosses the staged K-growth
transition.

## Test plan

- Same-role ladder: start `f16`, advance to `q8_0`, then `q4_0`, checking
  context length grows at each step and generation stays correct.
- Per-role: K `q8_0` / V `q4_0`, verify the cell count fills the pool and the
  checkpoint round trip converts each role independently.
- Checkpoint restore: save at each step, restart, restore into the overlay and
  into a later step, verify tokens and logits.
- Reset: fill, downshift, clear, reset, confirm the overlay and the original
  context capacity are back.
- Existing `test-lazy-kv` must still pass.
