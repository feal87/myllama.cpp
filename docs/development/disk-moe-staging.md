# Disk staging of MoE experts (Windows, experimental)

> Fork-private design/TODO record. Not for upstream submission. Delete this file
> before any PR to ggml-org.

Status: prefill and decode both work. Prefill streams the whole layer slab at
7.4 GB/s; decode serves the routed experts from a per-layer RAM cache filled by
the same unbuffered reader. Under greedy decoding the generated text is
byte-identical to the proven `--load-mode mmap+pin` path. What remains is the
budget flag, performance and cleanup below. Everything here is the working tree,
not a merged feature.

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
  id -> slot.
- Geometry: `slots_per_layer = (cache_mib << 20) / (n_layer * per_expert)`,
  `n_trans = min(n_expert_used, n_expert)`, `n_resident = slots - n_trans`.
  The first `n_resident` distinct experts seen become resident (first-come,
  filled once); later non-resident experts use the `n_trans` transient slots.
- `fill_cache(il, ids, n_ids)` reads only the routed experts into slots (same
  aligned-slack read as `fill`) and updates the table. Weight bytes are read in
  place from the slot; there is no copy.
- The slot stride inside each cache tensor is padded by one alignment unit
  (`ct->nb[2] = stride + 4096`). An expert read starts `head` bytes before its
  slot to stay sector-aligned, so without the pad a batch of adjacent slots
  would clobber the tail of the previous slot's expert (see the bugs below).
- `is_active()`, `supported()`.

### Graph wiring (`src/llama-graph.cpp`, `src/llama-hot-experts.*`)

- `llm_graph_context` holds `const llama_disk_stage * disk_stage`.
- `build_moe_ffn` substitutes `gate_exps/up_exps/down_exps` with the layer's
  staging tensors for multi-token ubatches, so the host->VRAM offload copy
  sources resident memory instead of faulting the mapping.
- The fill happens mid-graph through the hot-expert eval callback:
  `eval_callback` watches tensors named `ffn_moe_topk-<il>`, `wants_observe(il)`
  decides whether to break the graph there, and `observe(il, t)` copies the ids
  and calls `disk_stage->fill(il)`.
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
  layer (79 resident, 10 transient), and greedy output matches `mmap+pin` byte
  for byte. Throughput on this machine is about 4.5-7 t/s depending on how warm
  the resident set is.
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

   Still open: the resident set is first-come, so `--pin-hot-experts` sizes the
   set but does not yet choose the hottest experts in dio mode. See item 13.

### C. Performance

6. Slab fill: skip resident experts (memcpy from the cache) instead of re-reading
   them from disk.
7. Double buffering: read layer `i+1` while layer `i` computes and copies.
8. Decode already reads only the experts missing from the cache: resident slots
   are kept across tokens and only new ids are read into transient slots. The
   remaining decode cost is the warm-up, when most ids are still new.
9. Layers 0-1 at 3.5-3.9 GB/s: shard 1 stays mapped for the lazy PLE
   (`per_layer_token_embd`, 35.76 GiB, buffered ~170-byte rows via
   `ple_direct_reader` / `qwen4exp.cpp`). Either unmap shard 1 after the PLE
   table is no longer needed, or read PLE rows unbuffered, to lift them to
   7.4 GB/s.

### D. Cleanup

10. Trim the temporary logging (per-layer `fill: ... GB/s` line, reserved-address
    line) once stable, or gate it behind an env flag.
11. Keep the `n_tokens >= 1` experiment reverted (it inflated scheduler buffers
    and drove RAM up).
12. Remove `LLAMA_PREFETCH_LAYER_AHEAD` (regressed; `--hot-experts-prefetch`
    covers it).

### E. Correctness and robustness

13. Done. The disk decode cache's resident set is now ranking-driven, per layer:
    `llama_context` builds the hot-expert engine even in dio mode, with a per-layer
    capacity equal to the cache's resident slots. When the disk stage is attached,
    `try_promote` takes a per-layer path (`try_promote_layer`): fill up to `n_pin`
    residents for the layer, then take over that layer's coldest resident, with the
    same min-count floor, `takeover_min_lead` and `evict_grace_tokens` as the mlock
    tier. A "pin" is a copy into a resident slot (`llama_disk_stage::pin`), an
    eviction frees the slot (`unpin`); the slot is published only after the read
    completes, so the graph can never read a half-filled slot. The mlock path keeps
    its global ranking, so mmap mode is unchanged.

    The promotion worker and the decode fill share one I/O completion port, so
    their reads are serialized by a mutex (two concurrent reapers corrupt each
    other). On short runs this makes the ranking-driven cache slower than the old
    first-come fill (about 3.3 vs 4.4 t/s here), because the first-come set was
    already close to the hot set and the per-layer policy pays a warm-up plus the
    promotion reads. Over long sessions the ranking should win, since it does not
    keep stale first-seen experts. A dedicated completion port and handle set for
    the worker would remove the serialization.
14. Confirm the cache `table` tensor always stays host-allocated (written by the
    callback, read by the graph right after the chunk boundary).
15. Done: greedy output over 32 tokens is byte-identical to `--load-mode
    mmap+pin` for the same prompt. Worth repeating on a longer generation and on
    a second model before trusting it broadly.
16. Decide how the reserved address space relates to the scheduler: it exists so
    the graph allocator ignores disk tensors. Revisit if a cleaner marker appears.

### F. Platform (phase 2)

17. Linux: `O_DIRECT` instead of `FILE_FLAG_NO_BUFFERING`; `llama_disk_buft`
    reserve via `mmap(PROT_NONE)`; the reader is Windows-only IOCP and needs a
    POSIX equivalent (`pread`, or io_uring). `llama_mmap` already takes the
    `map` flag, so the loader side is portable.

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

Cache capacity (`--pin-hot-experts-budget-mib` targets):

| budget | experts | share |
|--------|---------|-------|
| 40 GB  | 11491   | 46.8% |
| 45 GB  | 12928   | 52.6% |
| 50 GB  | 14364   | 58.4% |
| 58 GB  | 16662   | 67.8% |

Machine: 63.6 GB RAM, ~56.7 GB free; RTX 4060 Ti 8 GB (7.07 GB free).

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
- `init_mappings` / `get_mapping_range` dereference `mappings.at(idx)`, so a
  "not mapped" file must still have an entry (hence the `map` flag, not a
  skipped vector slot).
- `llama_disk_stage` requires the eval callback, which requires
  `--hot-experts-prefetch`, or it disables itself with a warning.
- The reserved region must never be touched by a correct graph; doing so is an
  access violation, and that is the intended failure mode for a missed
  substitution.
- No small MoE model is available locally for fast iteration; only the 97/117/134
  GiB Qwen3.8-Flash-Next quants.
