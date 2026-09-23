#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

// Synchronous direct-read staging for MoE expert weights (Windows, experimental).
//
// Prefill offloads the expert mul_mat_id to the GPU (ggml_backend_cuda_device_offload_op,
// MUL_MAT_ID batch >= op_offload_min_batch_size). The weights live in a pageable
// mmap, so the scheduler's host->VRAM copy faults every cold expert in one page
// at a time. This stage substitutes a per-layer host tensor that is filled with
// unbuffered (FILE_FLAG_NO_BUFFERING) reads immediately before the layer computes,
// so the offload copy sources resident memory instead of faulting the mapping.
//
// The staging tensors for every layer alias one pooled buffer (one region per
// gate/up/down tensor): the scheduler keys copies by tensor, so the layers need
// distinct tensor objects or a single copy would be reused across all of them.
// The staging buffer holds TWO such sets and the stageable layers alternate
// between them: while layer i computes on the GPU, a reader thread fills layer
// i+1 into the other set, so the prefill read overlaps the compute instead of
// preceding it.
struct llama_disk_stage_layer {
    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;
    ggml_tensor * down = nullptr;
};

// Persistent decode cache view of one MoE layer: an I32 table mapping expert id
// -> slot, plus the gate/up/down slot tensors it reads through. Layers of one
// pool share those tensors, so a hot layer can own more resident slots than a
// cold one; without aligned GGUF data every layer has its own pool.
//
// The graph remaps selected_experts through `table` and runs mul_mat_id on these
// tensors, so the weights are read in place from the slot, no copy. A routed
// expert that is not resident is read into one of the layer's transient slots at
// fill time; the pool's resident slots hold the hot set and are filled once.
//
// Slot `sentinel` is a never-filled spare: an expert served by the VRAM cache
// has its table entry pointed there, and `slot_skip` marks it, so the host
// mul_mat_id skips it instead of reading a slot.
struct llama_disk_stage_cache_layer {
    ggml_tensor * gate  = nullptr; // [n_ff, n_embd, n_slots + 1]
    ggml_tensor * up    = nullptr;
    ggml_tensor * down  = nullptr;
    ggml_tensor * table = nullptr; // I32 [n_expert], expert id -> slot
    ggml_tensor * slot_skip = nullptr; // I32 [n_slots + 1], 1 at the sentinel
    // split-hot: two static slot-indexed tables that partition the slots, so the
    // host MoE can run a hot pass (residents) and a cold pass (transients) that
    // overlap the read of the cold experts with the compute of the hot ones.
    // skip_hot is 1 on transient + sentinel slots, skip_cold is 1 on resident +
    // sentinel slots. Null when the split is off
    ggml_tensor * slot_skip_hot  = nullptr;
    ggml_tensor * slot_skip_cold = nullptr;
};

class llama_disk_stage {
public:
    // staging is Windows-only and requires the unbuffered read path
    static bool supported();

    // dev selects the host buffer type the staging pool is allocated from
    // (the device's pinned host buffer when it has one, CPU otherwise).
    // n_pin_experts is the decode cache's resident experts per layer
    // (--pin-hot-experts), cache_budget_bytes its hard cap across all layers
    // (--pin-hot-experts-budget-mib); 0 means no explicit budget
    // pool_layers_max caps how many layers share one decode-cache pool (0 = no
    // limit, 1 = one pool per layer). Layers in a pool share resident slots
    // base_experts_path: optional "llama-expert-base v1" set of experts to read
    // into the decode cache at load and keep permanently resident (nullptr =
    // none). The cache must have room for the whole set or the constructor throws
    // warm_experts_path: optional set in the same format, read into the slots
    // left free by the base set. These are evictable count-0 residents; base
    // experts are skipped. Best effort: it fills up to capacity
    llama_disk_stage(const llama_model & model, ggml_backend_dev_t dev,
                     int32_t n_pin_experts, uint64_t cache_budget_bytes,
                     int32_t pool_layers_max, const char * base_experts_path,
                     const char * warm_experts_path);
    ~llama_disk_stage();

    // staging tensors of MoE layer il, or null when the layer is not stageable
    const llama_disk_stage_layer * layer(int il) const;

    // ensure layer il is staged, then start reading the next stageable layer
    // into the other buffer. Blocks only if layer il's own read is still in
    // flight, i.e. when this layer's compute was shorter than its read
    void fill(int il);

    // fill only the routed experts of layer il into the layer slab, synchronously.
    // Used for small ubatches, where the routed set is a small part of the slab;
    // residents are copied from the decode cache and the rest is read from disk
    void fill_selected(int il, const int32_t * ids, int64_t n_ids);

    // whether a ubatch of n_tokens should use fill_selected() instead of fill():
    // small ubatches route few experts, large ones touch almost every expert
    bool sparse_ubatch(int64_t n_tokens) const;

    // persistent decode cache of layer il, or null when the cache is off
    const llama_disk_stage_cache_layer * cache_layer(int il) const;

    // ensure the routed experts of layer il are in the cache and update the
    // layer's id table; reads the non-resident ones into transient slots
    void fill_cache(int il, const int32_t * ids, int64_t n_ids);

    // split-hot variant: fill_cache_begin() sets the tables and hands the disk
    // batch to a worker, then returns so the hot pass computes; fill_cache_wait()
    // joins the worker and runs the miss copies. Only used when split_hot() is on,
    // i.e. Windows + LLAMA_DISK_STAGE_SPLIT_HOT=1. The cache then carries two
    // static skip tables that split its resident slots from the transient ones,
    // so the decoder can run the hot (resident) experts and the cold (disk)
    // experts as two host passes: the disk read overlaps the hot compute
    void fill_cache_begin(int il, const int32_t * ids, int64_t n_ids);
    void fill_cache_wait(int il);

    // true when the split-hot decode path is active (Windows + env opt-in)
    bool split_hot() const;

    // number of independent decode-cache pools (0 when the cache is off)
    int n_pools() const;

    // pool id of layer il, or -1 when the layer has no cache. Layers in one pool
    // share their expert tensors and their resident slots, so a hot layer can
    // hold more of them than a cold one; without aligned GGUF data every layer
    // is its own pool
    int pool_id(int il) const;

    // resident slots of the pool serving layer il (0 when the layer has no cache)
    int32_t resident_capacity(int il) const;

    // largest pool resident capacity, for logs and the hot-expert engine gate
    int32_t resident_capacity() const;

    // base-expert set parsed from the profile: [layer] -> expert ids. Empty when
    // no set was given. The vectors are immutable once the constructor returns
    const std::vector<std::vector<int32_t>> & base_experts() const;

    // warm experts actually read into the cache (base entries excluded), for the
    // hot-expert engine to register as evictable count-0 residents
    const std::vector<std::vector<int32_t>> & warm_experts() const;

    // total base experts loaded, for stats/logs
    int32_t base_count() const;

    // true when (il, id) is a base expert: a permanent decode-cache resident that
    // the promotion policy must never evict. Lock-free (immutable after load)
    bool is_base(int il, int32_t id) const;

    // make expert id of layer il resident: reserve a slot for it and mark it
    // unfilled. No I/O here - fill_cache() reads the expert into the slot the next
    // time it is routed, so a promoted expert is read from disk exactly once
    bool resident_add(int il, int32_t id);

    // drop a resident expert, freeing its slot for the next promotion
    void resident_remove(int il, int32_t id);

    // true when (il, id) is a resident whose bytes are present in its slot, so
    // it is a candidate for a VRAM upload
    bool resident_filled(int il, int32_t id) const;

    // copy the resident expert's three rows (gate, up, down, in that order) into
    // dst, which must have room for gate_sz + up_sz + down_sz bytes. The copy
    // runs under the cache lock, so the source slot cannot be freed mid-copy;
    // the caller then uploads from this private copy and the hot tier may reuse
    // the slot at any time. false when (il, id) is not a filled resident
    bool resident_copy(int il, int32_t id, const size_t sz[3], void * dst, size_t dst_cap) const;

    // publish (il, id) as VRAM-served: its table entry becomes the sentinel, so
    // the host mul_mat_id skips it, and its RAM slot (when still held) is freed
    // for the next promotion. Only call between graphs, after the device upload
    void vram_commit(int il, int32_t id);

    // stop serving (il, id) from VRAM: the next fill_cache() reads it again
    void vram_release(int il, int32_t id);

    bool is_vram(int il, int32_t id) const;

    bool is_active() const;

    // print the L2 eviction-pool stats (warm hit rate, disk bytes saved, and the
    // per-interval delta). The pool reuses the prefill staging slabs during
    // decode; no-op when the pool is disabled. Called from the shared
    // --experts-stats-interval report.
    void print_stats();

    // true when decode is served by the internal scheduler hook instead of the
    // mid-graph eval callback (the decode cache maps each layer's expert ids)
    bool internal_decode_fill() const;

    // ggml_backend_sched_node_prepare_callback: stage the routed experts of a
    // layer whose slot-id remap is about to run. Fires on the layer's cache
    // table lookup, so the routed ids are already host-side and the graph is not
    // chunked. user_data is the llama_disk_stage
    static void node_prepare_callback(struct ggml_tensor * node, void * user_data);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;

    // build and run the blocking read batch for one layer; called by the
    // staging reader thread, and directly by fill() when there is one buffer.
    // `used` selects sparse fill: only those experts are read, the rest of the
    // slab keeps stale data and is never touched by the reader
    void fill_run(int il, const uint8_t * used = nullptr);

    // read every base expert into its reserved resident slot. Called once from
    // the constructor, before any graph is served
    void preload_base();

    // read warm experts into the slots left free by the base set, spread as
    // evenly as possible across the layers of each pool
    void preload_warm();

    // slot assignment shared by fill_cache() and the split-hot pair: writes the
    // layer's id table and collects the disk jobs, the independent copies and
    // the miss copies. false when the layer has no cache or no routed ids
    bool fill_cache_plan(int il, const int32_t * ids, int64_t n_ids);
};
