# Disk staging of MoE experts (Windows, experimental)

> Fork-private design/TODO record. Not for upstream submission. Delete this file
> before any PR to ggml-org.

Status: prefill and decode both work. Prefill streams the whole layer slab at
7.4 GB/s; decode serves the routed experts from a per-layer RAM cache filled by
the same unbuffered reader, with a per-layer ranking that keeps the hottest
experts resident. Under greedy decoding the generated text is byte-identical to
the earlier cache path and to `--load-mode mmap+pin` on the shorter prompts.
What remains is performance and cleanup below.

## Current state

Branch `disk-moe-staging`, three commits, not pushed:

```
82a8c5836 llama : rank the disk decode cache per layer and drop its env vars
781fcf8ae llama : size the disk decode cache from --pin-hot-experts
fe37bf459 llama : stream MoE experts from disk with unbuffered reads
```

Configuration (no environment variables any more):

```
llama-server -m <model> --load-mode dio -ngl 99 -c <ctx> \
  --pin-hot-experts-budget-mib <MiB> [--pin-hot-experts <N> ...]
```

`--load-mode dio` alone enables the streaming and no longer faults on decode:
the stage is created only when the model really has Disk-buffer weights, so
`mmap` mode is untouched, and the minimal decode cache is built even with no
budget flag. The next measurement to take is a real `llama-server` baseline in a
clean environment (see the note under item 13 about short-run numbers).

## Goal

Make an SSD first-class for MoE expert reads. The stock path mmaps the expert
weights and lets page faults pull them in; that path is capped for reasons below.
Direct I/O (FILE_FLAG_NO_BUFFERING) reaches full drive speed, so expert reads
should go through DIO instead of the mapping.

## Root cause: why mmap caps unbuffered reads

Measured with `F:\LLM\iodiag\iodiag.exe` (Windows only). On this machine
(`F:\`, NVMe):

- Unbuffered read of a file with **no** live data section: 7.26-7.29 GB/s.
- Unbuffered read of the **same** file while a `CreateFileMapping` section is
  live: 3.1-3.4 GB/s.

It is per file and independent of:

- view size (whole file or 1 MiB), destination buffer, buffer reaping
- queue depth (8/16/32/48), IOCP vs event completion, pinned destination
- `CreateFileMapping` with no view at all (still caps)
- reading a different file while another file is mapped (does not cap)
- releasing the mapping (restores 7.29 GB/s)
- buffered reads (never cap)

So one live section on a shard halves that shard's unbuffered read bandwidth.
The fix is to not map the expert shards at all.

`diskspd` on the raw file reads 6983 MiB/s, min latency 0.313 ms. No elevation
needed for reads (the earlier admin prompt was from the destructive `-c` variant).

## Architecture

### Disk buffer type (`src/llama-disk-buft.{h,cpp}`)

A dedicated `ggml_backend_buffer_type` for tensors that will never be read at
load time:

- `is_host = true`, `get_base = nullptr`, no-op set/get/memset/clear,
  `set_tensor` aborts.
- `llama_disk_buft_reserve(size)` reserves address space only:
  `VirtualAlloc(MEM_RESERVE)` on Windows, `mmap(PROT_NONE)` on POSIX. Do not
  commit; the pages are never touched by a correct graph.
- Tensors routed to it get a distinct `t->data` inside the reserved range so the
  graph allocator does not try to allocate them (a null `data` trips
  `GGML_ASSERT(tensor->buffer == NULL)` in `ggml_backend_tensor_alloc`).

### Loader routing (`src/llama-model-loader.{h,cpp}`)

- `use_direct_io` is set from `--load-mode dio`; `disk_stream = use_direct_io`.
- `buft_for_tensor` routes `FFN_GATE_EXPS` / `FFN_UP_EXPS` / `FFN_DOWN_EXPS`
  to `llama_disk_buft()` when `disk_stream`.
- `load_all_data` skips disk-buft tensors.
- `init_mappings` maps only the files that need it:
  `need_map = use_mmap || !lazy.for_file(idx).empty()`. `llama_mmap`'s ctor
  takes a `bool map`; with `map = false` it stores `size = 0, addr = nullptr`
  so the per-file `mappings` vector still has an entry (callers do
  `mappings.at(idx)`).

`src/llama-model.cpp` records name -> (path, offset) in
`pimpl->tensor_regions` from `ml.weights_map`, and `tensor_file_region` falls
back to it when `t->data` is not inside a mapping. This is how the staging
reader finds a disk tensor's bytes.

### Staging (`src/llama-disk-stage.{h,cpp}`)

Per-layer staging tensors alias one pinned pool, one region per role (max size
across layers, so about one layer's worth, not the sum over all layers):

- `llama_disk_stage_layer { gate, up, down }`
- `layer(il)` returns the layer's staging tensors, `fill(il)` reads the whole
  layer unbuffered.
- Rows are read with a `head` slack so the 4096-aligned read lands exactly where
  the tensor expects it (no bounce copy).
- Reader: single-thread windowed IOCP, queue depth 32.

Decode cache (always built when disk streaming is active; sized by
`--pin-hot-experts` / `--pin-hot-experts-budget-mib`):

- `llama_disk_stage_cache_layer { gate, up, down, table, n_slots }`
- One compact slot array per layer plus an I32 `table[n_expert]` mapping expert
  id -> slot. `n_trans = min(n_expert_used, n_expert)` slots are transient, the
  rest (`n_resident`) are hot slots.
- Geometry: `slots_per_layer` comes from `--pin-hot-experts-budget-mib` and/or
  `--pin-hot-experts`, capped by `n_expert`; see the cache sizing under Key
  facts. With neither flag it falls back to `n_trans + 1` and warns.
- Residency is ranking-driven per layer (see the hot-expert wiring below).
  `resident_add(il, id)` only RESERVES a slot and marks it unfilled;
  `resident_remove(il, id)` frees it. No I/O happens on promotion.
- `fill_cache(il, ids, n_ids)` is the only reader in decode. For each routed
  expert: if it is resident and filled, reuse the slot; if it is resident but
  not yet filled, read it into the slot (this is the read-once promotion read);
  otherwise read it into a transient slot. All of the layer's reads go in one
  batch, and the table is updated before the graph reads it.
- The slot stride inside each cache tensor is padded by one alignment unit
  (`ct->nb[2] = stride + 4096`). An expert read starts `head` bytes before its
  slot to stay sector-aligned, so without the pad a batch of adjacent slots
  would clobber the tail of the previous slot's expert (see the bugs below).
- All cache state is under `cache_mu`, but every caller is on the decode thread
  now; `io_mu` just serializes read batches as a guard.
- `is_active()`, `supported()`, `resident_capacity()`.

### Graph wiring (`src/llama-graph.cpp`, `src/llama-hot-experts.*`)

- `llm_graph_context` holds `const llama_disk_stage * disk_stage`.
- `build_moe_ffn` substitutes `gate_exps/up_exps/down_exps`: with the layer's
  staging tensors on a multi-token ubatch (so the host->VRAM offload copy sources
  resident memory instead of faulting the mapping), and with the compact cache
  tensors on a single-token ubatch (with the ids remapped through the layer's
  table).
- The fill happens mid-graph through the hot-expert eval callback:
  `eval_callback` watches tensors named `ffn_moe_topk-<il>`, `wants_observe(il)`
  decides whether to break the graph there, and `observe(il, t)` copies the ids
  and calls `disk_stage->fill(il)` (multi-token) or `fill_cache(il, ids, n)`
  (single-token).
- `llama_context` installs the callback for multi-token ubatches when
  `--hot-experts-prefetch` is on, and for every ubatch when the disk stage is
  active (single-token decode needs the fill too). Single-token ubatches also
  feed the post-compute ranking path (`observe_decode_*`) from top-k tensors
  registered as graph outputs.

## Verified

- `--load-mode dio` loads with no expert shard mapped. Disk buffer usage prints
  0.00 MiB, the PLE stays in CPU_Mapped (36.6 GiB, lazy), and 85.55 GiB of
  address space is reserved.
- Prefill fills run at **7.4 GB/s for layers 3-46** (was 3.1-3.4 through the
  mapping). Layers 0-1 are still 3.5-3.9 GB/s because shard 1 stays mapped for
  the lazy PLE (per-file cap, see above).
- All 48 layers fill and prefill completes.
- Single-token decode completes through the RAM cache: 16 GiB gives 89 slots per
  layer (79 resident, 10 transient), and the greedy output matches the older
  cache path byte for byte. Throughput on this machine is about 3-7 t/s
  depending on how warm the resident set is and on which policy filled it.
- The resident set is ranking-driven per layer (`--pin-hot-experts N` /
  `--pin-hot-experts-budget-mib`): over 96 decode tokens the set filled to
  3160/3792 slots with a 48% routed-expert hit rate, and the greedy output was
  still byte-identical to the earlier cache path. The mmap path still reports its
  global mlock set (`N x 48` with skew, real lock calls).

### Bugs found and fixed on the way

1. Disk tensors with `data == nullptr` hit `GGML_ASSERT(tensor->buffer == NULL)`
   in `ggml_backend_tensor_alloc`. Fixed by reserving address space and giving
   every disk tensor a distinct `data`.
2. The fill/substitution gate used `cur->ne[1]`, which is the per-layer token
   count, not the ubatch size. On this model layer 47 is an attention layer fed
   a 1-token slice inside a multi-token ubatch, so it read 1, was skipped, and
   the CPU `mul_mat_id` read `blk.47.ffn_gate_exps` from the uncommitted
   reserved range (access violation). Fixed by gating on the ubatch size
   (`this->n_tokens` in `build_moe_ffn`, `n_tokens_cur` in `observe`).
3. The decode down projection still passed the raw expert ids while its weight
   tensor was the compact cache, so `mul_mat_id` indexed past the cache. Fixed
   by remapping gate, up and down together (`selected_experts_c`).
4. The per-expert read starts `head` bytes before the destination slot to keep
   the unbuffered read sector-aligned. Reading several adjacent slots in one
   batch meant each read overwrote the last `head` bytes of the previous slot's
   expert (visible as a cache mismatch exactly `stride - head` bytes in). Fixed
   by padding the slot stride in the cache tensor.
5. All reads go through one I/O completion port. When the promotion worker and
   the decode fill both reaped it, they stole each other's completions
   (`fails=36`, then a fatal `unbuffered read failed`). Fixed by serializing the
   batches with a mutex; the read-once promotion (item 13) then removed the
   worker's reads entirely, so only the decode thread reads now and the guard is
   no longer contended. Anyone who reintroduces a second reader must bring its
   own completion port and handle set.
6. `--moe-expert-cache*` (the VRAM tier) segfaulted in dio mode, at exactly 512
   decode tokens. The Disk buffer type reports `is_host = true` (the scheduler
   needs that), so `llama_moe_cache::reserve()` accepted the Disk expert tensors
   as host-resident, reserved a device pool, and once `maybe_activate()` fired
   the upload worker read `src->data` from the reserved-but-never-committed
   address range. The graph-side use was already dead: `build_moe_ffn` replaces
   `gate_exps` with the disk cache tensor before `llama_moe_cache::lookup()`
   compares pointers, so the VRAM tier never served a token. `llama_context` now
   warns and drops the tier when disk streaming is active, and the launcher does
   not pass the flags in dio. Feeding VRAM from the disk cache slots (a real
   VRAM > RAM > disk hierarchy) is future work, not a bug fix: in dio the graph
   already remaps ids through the disk table (`selected_experts_c`), so the
   VRAM skip cannot use `src[3]`/host_table (the kernel indexes it by the id in
   the ids tensor, ggml-cpu.c:1659), and the upload source must become a disk or
   RAM-slot read instead of `src->data`. The shape that fits the existing code:
   give the disk cache a zeroed dummy slot, map VRAM residents to it in the disk
   table (the disk chain then emits zeros for them, no src[3] needed), keep VRAM
   residents a subset of the disk residents, and upload from the resident RAM
   slot so there is no extra disk read and no second reader on the IOCP.

## Remaining work

### A. Wire the decode path (done)

Kept here as a record. The changes that made single-token generation work:

1. `src/llama-context.cpp`: the eval callback is installed whenever the disk
   stage is active, not only for multi-token ubatches.
2. `wants_observe()` in `src/llama-hot-experts.cpp`: returns true for
   single-token ubatches when the layer has a decode cache.
3. `observe()`: single-token calls `disk_stage->fill_cache(il, ids, n_ids)`,
   multi-token still calls `fill(il)`, both gated on the ubatch size.
4. `build_moe_ffn`: single-token and a cache layer substitutes the compact
   tensors and remaps the ids with
   `selected_experts_c = reshape(get_rows(table, selected_experts))`, used by all
   three projections. The top-k weights keep the original ids, so the pairing is
   preserved.

Open follow-up: none; the configuration is covered by B.

### B. Cache budget as a real flag (done)

5. Done. `--pin-hot-experts-budget-mib` is the decode cache budget and
   `--pin-hot-experts N` its resident experts per layer. When disk streaming is
   active the hot-expert engine is built per-layer and the copied disk cache is
   the RAM tier; nothing is mlock'd in place, and its counts still drive the
   VRAM tier and the prefetch. With neither flag given the cache falls back to a
   minimal `n_expert_used + 1` slots per layer and warns, so single-token decode
   works instead of faulting on the unmapped tensors.

### C. Performance

6. Slab fill: skip resident experts (memcpy from the cache) instead of re-reading
   them from disk.
7. Double buffering: read layer `i+1` while layer `i` computes and copies.
8. Decode already reads only the experts missing from the cache: resident slots
   are kept across tokens and only new ids are read into transient slots. A
   promotion is mapping-only - `fill_cache` reads the expert when it is next
   routed - so an expert is read from disk once, never twice (see item 13).
9. Layers 0-1 at 3.5-3.9 GB/s. The first hypothesis - that the shard-1 mapping
   (kept for the lazy PLE) caps that file's unbuffered reads - was WRONG. The
   loader now skips mapping a file whose only lazy tensor is served by a direct
   reader (qwen4exp PLE calls `llama_model_loader::lazy_read::set_direct`),
   which also removes a spurious 37 GiB host-pointer buffer over a null address
   (`get_mapping_range` returns first=0 from the zero-size mapping and last=tensor
   end, so the host-ptr branch used to build it anyway). With that in place
   shard 1 has no section, yet the fills are unchanged: 3.9 and 3.4 GB/s for
   layers 0-1, and layer 2 (already never mapped, it is shard 2) is also slow at
   4.8, while layer 3+ AND layer 25 (first read of the never-mapped shard 3) run
   at 7.4. iodiag on the same file measures 7.47 GB/s without a mapping and 3.07
   with one, so the mapping really is gone. The PLE gather is 0.2-2.0 ms, so it
   is not the cause either. What remains is a START-OF-UBATCH effect on the
   first ~3 layer reads; the likely causes are the reader thread being
   CPU-starved while the main thread runs ubatch setup, or the drive queue not
   yet ramped. Not yet isolated.
9b. The ranking can be slower than the old first-come fill on SHORT runs (the
    first-come set is already close to hot, and the policy pays a warm-up plus
    churn). A long baseline is needed to see whether it wins; see item 13. The
    obvious knobs are `--pin-hot-experts-min-count` and the decay window.

### D. Cleanup

10. Trim the temporary logging (per-layer `fill: ... GB/s` line, reserved-address
    line) once stable, or gate it behind an env flag.
11. Keep the `n_tokens >= 1` experiment reverted (it inflated scheduler buffers
    and drove RAM up).
12. Remove `LLAMA_PREFETCH_LAYER_AHEAD` (regressed; `--hot-experts-prefetch`
    covers it).

### E. Correctness and robustness

13. Done. The disk decode cache's resident set is ranking-driven, per layer.
    `llama_context` builds the hot-expert engine even in dio mode, with a
    per-layer capacity equal to the cache's resident slots. When the disk stage is
    attached, `try_promote` takes a per-layer path (`try_promote_layer`): fill up
    to `n_pin` residents for the layer, then take over that layer's coldest
    resident, with the same min-count floor, `takeover_min_lead` and
    `evict_grace_tokens` as the mlock tier.

    Promotion is mapping-only: `resident_add` reserves a slot and marks it
    unfilled, and `fill_cache` reads the expert into it the next time that expert
    is routed, in the same batch as the transients. So a promoted expert is read
    from disk exactly once - the earlier design read it as a transient and then
    read it AGAIN in the promotion worker, which also meant a shared IOCP needed
    its reapers serialized. There is no promotion read and no worker in the disk
    path now; all disk reads happen on the decode thread (`fill` in prefill,
    `fill_cache` in decode), so the `io_mu` guard is no longer contended. The mlock
    worker still starts in dio mode (`n_pin > 0`) but receives no jobs.

    The mlock path keeps its global ranking, so mmap mode is unchanged (verified:
    `pinned=573/768 (N=16 x 48)` with skew and real lock calls).

    Performance note: on SHORT runs the per-layer ranking measured slower than the
    old first-come fill (about 3.3 vs 4.4 t/s here, and 2.9 t/s once in a 32-token
    run with `--pin-hot-experts-min-count 4`). The first-come set happened to sit
    close to the hot set, while the policy pays a warm-up (min-count) plus churn.
    These numbers are noisy and from a machine with other load; a long
    `llama-server` baseline is the real measurement. Knobs to try:
    `--pin-hot-experts-min-count`, `--pin-hot-experts-decay-tokens`, and a larger
    budget.
14. Confirm the cache `table` tensor always stays host-allocated (written by the
    callback, read by the graph right after the chunk boundary).
15. Done: greedy output over 32 tokens is byte-identical within dio to the
    earlier cache path, and matched `mmap+pin` on the prompts where the prefill
    happened to batch the same way (see the cross-load-mode gotcha). Worth
    repeating on a longer generation and on a second model.
16. Decide how the reserved address space relates to the scheduler: it exists so
    the graph allocator ignores disk tensors. Revisit if a cleaner marker appears.

### F. Platform (phase 2)

17. Linux: `O_DIRECT` instead of `FILE_FLAG_NO_BUFFERING`; `llama_disk_buft`
    reserve via `mmap(PROT_NONE)`; the reader is Windows-only IOCP and needs a
    POSIX equivalent (`pread`, or io_uring). `llama_mmap` already takes the
    `map` flag, so the loader side is portable.

## VRAM MoE tier on the disk cache

Status: implemented and validated in dio. `--moe-expert-cache-budget-mib` now
works with `--load-mode dio`, so both load modes have the same three tiers
(VRAM > RAM > disk).

Measured A/B on a 900-token greedy run (40000 MiB RAM budget, 2048 MiB VRAM
budget):

| dio, 900 tokens        | RAM hit | VRAM hit | combined | decode  |
| ---------------------- | ------- | -------- | -------- | ------- |
| VRAM tier off          | 76.9%   | -        | 76.9%    | 6.05 t/s |
| VRAM tier on (2048 MiB)| 75.6%   | 17.0%    | 92.7%    | 6.27 t/s |

The two tiers add up rather than overlap: the VRAM residents left the RAM set
and the RAM tier refilled those slots with the next-hottest experts. Coverage
rose ~16 points; decode rose only ~3.6%, so decode is not purely expert-read
bound. A larger VRAM budget (the card had ~7.7 GB free) should raise the hit
rate and the gap.

Why it was not a small change (both facts confirmed in the code):

- The tier's skip is `src[3]`, and `ggml-cpu.c:1659` indexes that table by the
  value in the ids tensor (`moe_tbl[i02]`, `i02` from `ids->data`). In dio
  `build_moe_ffn` already remapped the ids through the disk table
  (`selected_experts_c = get_rows(dc->table, selected_experts)`), so `i02` is a
  DISK SLOT, not an expert id. The expert-indexed `host_table` is the wrong table
  and would be read out of range.
- The upload worker does `ggml_backend_tensor_set(dst, src->data + expert*sz, ...)`
  with `src` = the model expert tensor. In dio `src->data` is the reserved,
  never-committed range, so the first upload is an access violation (reproduced:
  segfault at exactly 512 decode tokens, `maybe_activate`).

How it works now:

1. **Sentinel slot.** The disk cache tensors are `ne[2] = n_slots + 1`; index
   `n_slots` is never filled and never reused. An expert served by VRAM has
   `table[e] = sentinel`.
2. **Slot-indexed skip table.** `slot_skip` (I32 `[n_slots + 1]`) is 1 at the
   sentinel, else 0. The graph sets `src[3] = dc->slot_skip` and
   `op_params[0] = 0`, so the host `mul_mat_id` SKIPS a VRAM expert (no read)
   instead of reading a zero slot. A single sentinel covers every VRAM expert of
   the layer, so `src[3]` must be attached on the disk chain, not `host_table`.
3. **Replace, not overlap.** When an upload completes, `tick` publishes the
   device table (`sync_tables`) and then calls `disk_stage->vram_commit(il, e)`:
   the host table entry becomes the sentinel and the RAM slot is FREED for the
   next promotion. The RAM tier refills that slot with the next-hottest expert,
   so the two tiers add up instead of covering the same experts twice. This is
   why the measured RAM hit stays high while the VRAM hit is added on top.
4. **Upload source is the RAM slot.** The worker copies from
   `resident_data(il, e, role)` (`c.data[role] + slot*slot_stride`) for the three
   roles, not from the model tensor or a second disk read. VRAM candidates are
   gated on `disk_stage->resident_filled(il, e)`.
5. **Private snapshot.** The hot tier could evict the source slot as a takeover
   victim mid-copy, so the queue step copies the three rows out under the cache
   lock (`resident_copy`) into a private buffer and the worker uploads from that.
   The hot tier may then reuse the slot at any time, and there is nothing to
   unwind if an upload fails or a layout rebuild drops the job.
6. **Per-layer capacity** stays the existing top-heavy
   `assign_global_capacity(budget)`, clamped per layer to
   `disk_stage->resident_capacity()` (the RAM set is uniform, so VRAM cannot
   exceed it). Eviction from VRAM calls `disk_stage->vram_release` and hands the
   expert back to the RAM tier via `hot->promote_expert`. The per-prompt layout
   rebuild must `vram_release` the old residents too, or the rebuilt device
   table would not serve them while the host chain still skipped them.

API added to `llama_disk_stage` (all under `cache_mu`, all no-ops off Windows):
`resident_filled`, `resident_copy(il, id, sz[3], dst, cap)`, `vram_commit`,
`vram_release`, `is_vram`; `llama_disk_stage_cache_layer` gained `slot_skip` and
`sentinel`. `llama_moe_cache` gained `set_disk_stage` (call before `reserve()`);
its `reserve()` records the disk cache tensors as `lookup` keys, so
`build_moe_ffn` finds the layer by the tensor it actually passes.

Cost: one extra untouched slot per layer (~190 MB for 48 layers) plus the tiny
skip table. The `host_table` device copy is still maintained (unused in dio).

Not done yet / tuning:

- The VRAM tier is decode-only (n_tokens == 1) and uses the existing separate
  gate/up/down + plain swiglu guard; fused and clamp layers keep the stock disk
  chain.
- The 17% VRAM hit was with a 2048 MiB budget; the top-heavy layout is very
  uneven (L1 got 1 slot, L32 got 22). To tune the split, sweep
  `--moe-expert-cache-budget-mib` against the RAM budget.
- Still to measure: decode t/s with the tier on vs off, and a greedy output
  comparison (GPU vs CPU rounding means exact equality is not guaranteed after
  activation, but the text should track closely).

## Key facts

Model: `Qwen3.8-Flash-Next-Uncensored-Q5_K_M` (arch `qwen4exp`), 48 layers,
512 experts, top-10, n_embd 2560, n_ff 640.

- Shard map: layers 0-1 -> shard 1, layers 2-24 -> shard 2, layers 25-47 -> shard 3.
- Expert total 85.55 GiB, 24576 experts, average 3.564 MiB/expert.
- Per layer: gate Q5_K 576.7 MB, up Q5_K 576.7 MB, down Q8_0 891.3 MB or Q5_1
  629.1 MB (pool ~2.04 GB/layer).
- `per_layer_token_embd.weight` = 35.76 GiB (shard 1), lazy.
- Non-expert, non-PLE total 3.58 GiB.
- Expert slices are sector-aligned in this model (e.g. gate stride 1,126,400 =
  275 x 4096), but do not rely on it; use the alignment slack.

Cache sizing (`--pin-hot-experts-budget-mib`):

`slots_per_layer = budget / (n_layer * ref_bytes_per_expert)`, where
`ref_bytes_per_expert` is taken from the first stageable layer and is the largest
variant (Q8_0 down, 3,993,600 bytes). Resident slots = slots - `n_expert_used`
(the transient slots). Because most layers in this model have a smaller Q5_1
down, the ACTUAL allocation lands below the budget (measured 8192 MiB -> 7.38 GiB
/ 44 slots, 16384 MiB -> 14.92 GiB / 89 slots); 40960 MiB would give 224 slots
and about 37 GiB actual. Each slot carries a 4096-byte alignment pad on top of the
expert stride (see the padded slot stride in the bugs list).

Total RAM for streaming:

- decode cache: at most the budget, about 93% of it here
- staging slab pool: 1950 MiB, one layer's worth (3 regions aliased across all
  layers), not per layer
- so about `1.9 GiB + cache`, plus 1-2 GB for the context and non-expert tensors

Machine: 63.6 GB RAM (~45-57 GB free depending on other load); RTX 4060 Ti 8 GB
(~4.7-7.1 GB free depending on other load).

Prefill budget scaling: IQ4_XS about 54 GB per 4096 tokens (200 t/s),
Q4_K_M about 82 GB (110 t/s), Q5_K_M about 138 GB (72 t/s) at ~2.4 GB/s cold.
`prefill_time = max(compute + PCIe, cold_read)`; at 7 GB/s the read term drops
about 3x.

Offload: `ggml_backend_cuda_device_offload_op` returns true when
`get_op_batch_size(op) >= GGML_OP_OFFLOAD_MIN_BATCH` (default 32, env
`GGML_OP_OFFLOAD_MIN_BATCH`). For `MUL_MAT_ID` the batch is `op->ne[2]`, i.e.
`n_tokens`. So prefill offloads the MoE to the GPU and copies the staged layer
host->VRAM; decode runs it on the CPU.

GPU handling: KV cache, compute buffers, attention and routers are on the GPU.
The CPU handles decode and ultra-small prefill (< 32 tokens). So the CPU RAM
budget is essentially the expert cache plus the staging ring.

## Knobs

Disk streaming is enabled by `--load-mode dio` (Windows only) and configured by
the existing hot-expert flags; it has no environment variables of its own. It is
created only when the model actually has Disk-buffer weights, so `--load-mode
mmap` is unaffected. The staging slab always uses the device's pinned host buffer
when one exists (plain CPU memory otherwise).

- `--load-mode dio` - route the MoE expert tensors to the Disk buffer and stream
  them. The model tensors are never mapped.
- `--pin-hot-experts N` - resident experts per layer for the decode cache.
- `--pin-hot-experts-budget-mib N` - hard cap, in MiB, on the decode cache
  across all layers. With neither flag the cache falls back to a minimal
  `n_expert_used + 1` slots per layer and warns.
- `--hot-experts-prefetch` - no longer required by disk streaming (the stage
  installs the callback itself); still useful in `mmap` mode.

Other knobs:

- `LLAMA_MMAP_CACHE_HINT` (default `random`).
- `LLAMA_PREFETCH_LAYER_AHEAD` (regressed, to remove).
- `--pin-hot-experts-decay-tokens`, `--pin-hot-experts-min-count` - ranking
  aging and promotion floor, shared with the mlock tier.
- `--moe-expert-cache-budget-mib N` - VRAM expert cache (separate feature).

## Files touched

New:

- `src/llama-disk-buft.{h,cpp}` - Disk buffer type and address reservation.
- `src/llama-disk-stage.{h,cpp}` - staging pool, unbuffered reader, decode cache.

Modified:

- `src/CMakeLists.txt` - add the new sources; add `src/../ggml/src` to includes.
- `src/llama-mmap.{h,cpp}` - `map` flag on `llama_mmap`.
- `src/llama-model-loader.{h,cpp}` - `disk_stream`, routing, skip, per-file map.
- `src/llama-model.{h,cpp}` - disk buffer branch, `tensor_regions`,
  `tensor_file_region` fallback.
- `src/llama-graph.{h,cpp}` - `disk_stage` member and the substitution.
- `src/llama-context.{h,cpp}` - create the stage, callback gating.
- `src/llama-hot-experts.{h,cpp}` - `set_disk_stage`, fill from `observe`.

## Repro and debugging

Run:

```
./build/bin/llama-cli.exe ^
  -m c:\models\Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf ^
  --load-mode dio -ngl 99 -c 2048 --no-warmup -st -n 64 ^
  --pin-hot-experts-budget-mib 16384 ^
  -p "Hello" -lv 4
```

`--load-mode dio` is the only switch needed; `--pin-hot-experts-budget-mib`
sizes the decode cache. `-st` / `--single-turn` makes `llama-cli` exit after one
turn. Without it the
interactive loop spins on EOF (an empty turn per iteration, `> ` forever); that
is unrelated to disk staging but makes any run look like it never finishes.

Symbolicated crash (a `build-dbg` with `RelWithDebInfo` exists and has PDBs):

```
cdbX64.exe -y "F:\LLM\llama.cpp\build-dbg\bin" -c "g; kb 6; q" ^
  ./build-dbg/bin/llama-cli.exe -m ... --load-mode dio --hot-experts-prefetch ...
```

`kb` gives the true frames (`ggml_compute_forward_mul_mat_id` etc.); a Release
build with no PDBs only shows nearest exports.

Bandwidth harness: `F:\LLM\iodiag\` (`iodiag.cpp`, `build.cmd`, `runmatrix.cmd`).
Never pass `-c` to `diskspd` on a real file; it overwrites (this destroyed
shard 1 once; it was restored).

## Gotchas

- Rebuilding the CUDA objects needs the MSVC environment on PATH. Run
  `cmd //c build-eval-ninja.bat build <target>` (it calls `vcvars64.bat` then
  ninja); a plain `cmake --build` fails with `nvcc fatal : Cannot find compiler
  'cl.exe' in PATH` as soon as a `.cu` object is out of date.
- `bin/llama.dll` fails to link while `llama-server` is running. Stop it first.
- If other software holds VRAM, the `--fit` probe can fail with
  `CUDA error: out of memory ... ggml_backend_cuda_buffer_clear` during load, at
  ~3 seconds, before any tensor is read. `-fit off` skips the probe; on a clean
  machine it is not needed.
- Comparing `dio` against `mmap+pin` byte-for-byte is not guaranteed: the two
  load modes can split the prefill ubatches differently, and a different
  numerical reduction order can flip an early greedy token even though the
  weights are identical. Compare within one load mode (or accept that the
  divergence is in the first token).
- The in-dio A/B switch (`LLAMA_DISK_STAGE_DECODE_FULL`, which ran single-token
  decode through the full slab) was removed with the other env vars. To re-check
  the cache against the slab, temporarily restore the `decode_full()` branch in
  `build_moe_ffn`, `wants_observe` and `observe`.
- `init_mappings` / `get_mapping_range` dereference `mappings.at(idx)`, so a
  "not mapped" file must still have an entry (hence the `map` flag, not a
  skipped vector slot).
- Disk streaming installs its own eval callback, so it no longer needs
  `--hot-experts-prefetch`; it only disables itself if a custom `cb_eval` is
  already in use and no hot-expert engine can be built.
- The reserved region must never be touched by a correct graph; doing so is an
  access violation, and that is the intended failure mode for a missed
  substitution.
- No small MoE model is available locally for fast iteration; only the 97/117/134
  GiB Qwen3.8-Flash-Next quants.
