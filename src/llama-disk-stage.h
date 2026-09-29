#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-ext.h"

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
    // cache-aware opportunistic dropping (--disk-stage-drop-fraction): F32
    // [n_expert], 1 for an expert kept this token, 0 for a cold expert dropped.
    // The graph multiplies the expert weights by get_rows(keep, selected_experts)
    // before the norm_w pass, so the kept experts are renormalized. Null when
    // dropping is off (probe-only keeps the force below but not this tensor)
    ggml_tensor * keep = nullptr;
    // probe or drop: force the ffn_moe_weights get_rows onto the host so the
    // disk stage can read the router scores in its node-prepare callback
    bool force_weights_host = false;
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
    // base_template_path: optional expert-set template (llama-expert-base.h)
    // that names the base set, the per-mode sets and the tool conditions that
    // select between them. It replaces base_experts_path (which is then
    // ignored) and the mode is switched at runtime through select_base_set()
    // split_hot: split the host decode MoE into a hot (resident) and a cold
    // (disk) pass on a second CPU backend, so the cold read overlaps the hot
    // compute
    llama_disk_stage(const llama_model & model, ggml_backend_dev_t dev,
                     int32_t n_pin_experts, uint64_t cache_budget_bytes,
                     int32_t pool_layers_max, const char * base_experts_path,
                     const char * warm_experts_path, const char * base_template_path,
                     bool split_hot, float drop_fraction, float drop_below_rel,
                     float drop_max_mass, float drop_max_mass_token, float substitute_rel, int32_t substitute_pool,
                     bool drop_probe);
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
    // layer's id table; reads the non-resident ones into transient slots.
    // probs, when non-null, is the raw router score of each routed id (same
    // order), used by cache-aware dropping (--disk-stage-drop-fraction)
    void fill_cache(int il, const int32_t * ids, int64_t n_ids, const float * probs = nullptr);

    // split-hot variant: fill_cache_begin() sets the tables and hands the disk
    // batch to a worker, then returns so the hot pass computes; fill_cache_wait()
    // waits for the read phase, which the cold pass needs. Only used when
    // split_hot() is on (--disk-stage-split-hot, Windows). The cache
    // then carries two static skip tables that split its resident slots from the
    // transient ones, so the decoder can run the hot (resident) experts and the
    // cold (disk) experts as two host passes: the disk read overlaps the hot
    // compute. The worker keeps storing the transient slots into the L2 pool
    // while the cold pass computes, and the next fill drains it before the
    // window is reused.
    void fill_cache_begin(int il, const int32_t * ids, int64_t n_ids, const float * probs = nullptr);
    void fill_cache_wait(int il);

    // end of a decode step, from the graph compute: closes the layer whose cold
    // pass no later split reported and prints the token totals
    void split_token_end();

    // true when the split-hot decode path is active (Windows + env opt-in)
    bool split_hot() const;

    // the graph records, per layer, whether it emitted a cold pass for it. The
    // decode fill consults this to pick the split or the synchronous path, so a
    // layer without a cold pass never hands a batch to the worker
    void set_split_cold(int il, bool cold) const;
    bool split_cold(int il) const;

    // true once per process, and only with LLAMA_DISK_STAGE_TRACE, so the graph
    // can report why a layer got no cold pass without flooding the log
    static bool split_trace_once();

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

    // currently selected base-expert target: [layer] -> expert ids. The set
    // vectors are immutable after construction; the active view changes on a switch
    const std::vector<std::vector<int32_t>> & base_experts() const;

    // ----- mode switching (--pin-experts-template) ---------------------------
    // The template's sets are parsed and merged with the base set at load, so a
    // switch is a lookup: base_set_by_name() returns the resident target of a
    // set (base + that set merged per layer), or null when the name is unknown.
    // The empty name is the base set alone and always resolves
    const std::vector<std::vector<int32_t>> * base_set_by_name(const std::string & name) const;
    std::string select_base_set(const std::vector<std::string> & tools) const;
    std::string fence_base_set() const;
    bool set_base_target(const std::string & name);

    // make `id` of layer `il` a base resident: marks it unevictable and reserves
    // its slot (no I/O: the bytes are read when the expert is next routed). An
    // expert that already holds a slot or is served from VRAM only needs the
    // mark. false when the pool has no slot to spare, in which case nothing
    // changed. Call from the decode thread only
    bool base_add(int il, int32_t id);

    // drop the base mark of `id`: it keeps its slot and becomes an ordinary
    // resident, so the hot-expert policy decides whether it stays
    void base_remove(int il, int32_t id);
    bool resident_slot_held(int il, int32_t id) const;

    // set selected by the last base_set_by_name() lookup, for logs and stats
    const std::string & active_base_set() const;

    // read the base experts whose bytes are not in the cache yet. Called once at
    // load, before any graph runs; blocks for the full batched read
    size_t fill_base_experts();

    // warm experts actually read into the cache (base entries excluded), for the
    // hot-expert engine to register as evictable count-0 residents
    const std::vector<std::vector<int32_t>> & warm_experts() const;

    // make expert id of layer il resident: reserve a slot for it and mark it
    // unfilled. No I/O here - fill_cache() reads the expert into the slot the next
    // time it is routed, so a promoted expert is read from disk exactly once
    bool resident_add(int il, int32_t id);

    // drop a resident expert, freeing its slot for the next promotion. `keep_l2`
    // controls whether its filled data is copied to the L2 pool first
    void resident_remove(int il, int32_t id, bool keep_l2 = true);

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

    // Aggregate DIO cache counters and current state for the server metrics exporter.
    void stats_snapshot(llama_expert_stats & out) const;

    // print the stats report via LLAMA_LOG_INFO (verbosity 4): a header line plus
    // one themed line each for the layout, the decode fill cost, the decode cache
    // hit rate, the L2 pool (hit rate, disk bytes served, admission policy), the
    // split-hot overlap and the per-pool breakdown. decode_tokens is the number of
    // single-token decode ubatches seen so far (the hot cache's content_tokens()),
    // so the fill and split costs can be reported per decode token. No-op when the
    // decode cache is off. Called from the shared --experts-stats-interval report.
    void print_stats(uint64_t decode_tokens);

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
    bool fill_cache_plan(int il, const int32_t * ids, int64_t n_ids, const float * probs);

    // wait out the decode I/O worker and publish the L2 slots its stores
    // filled, so the next plan can hit them. Also called before a prefill
    // clobbers the staging slabs the L2 pool lives on
    void dec_io_drain();

    // split-hot trace hooks, called from the node-prepare callback
    void split_cold_end();  // the cold pass of the pending layer computed
    void split_report();    // print and account the pending layer
};
