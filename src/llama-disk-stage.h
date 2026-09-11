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
    llama_disk_stage(const llama_model & model, ggml_backend_dev_t dev,
                     int32_t n_pin_experts, uint64_t cache_budget_bytes,
                     int32_t pool_layers_max);
    ~llama_disk_stage();

    // staging tensors of MoE layer il, or null when the layer is not stageable
    const llama_disk_stage_layer * layer(int il) const;

    // ensure layer il is staged, then start reading the next stageable layer
    // into the other buffer. Blocks only if layer il's own read is still in
    // flight, i.e. when this layer's compute was shorter than its read
    void fill(int il);

    // persistent decode cache of layer il, or null when the cache is off
    const llama_disk_stage_cache_layer * cache_layer(int il) const;

    // ensure the routed experts of layer il are in the cache and update the
    // layer's id table; reads the non-resident ones into transient slots
    void fill_cache(int il, const int32_t * ids, int64_t n_ids);

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

private:
    struct impl;
    std::unique_ptr<impl> pimpl;

    // build and run the blocking read batch for one layer; called by the
    // staging reader thread, and directly by fill() when there is one buffer
    void fill_run(int il);
};
