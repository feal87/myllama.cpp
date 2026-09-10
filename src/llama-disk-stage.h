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
// The staging tensors for every layer alias one pinned pool (one region per
// gate/up/down tensor): the scheduler keys copies by tensor, so the layers need
// distinct tensor objects or a single copy would be reused across all of them.
struct llama_disk_stage_layer {
    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;
    ggml_tensor * down = nullptr;
};

// Persistent decode cache of one MoE layer: n_slots experts in one compact
// tensor per role, plus an I32 table mapping expert id -> slot. The graph
// remaps selected_experts through `table` and runs mul_mat_id on these tensors,
// so the weights are read in place from the slot, no copy. The first
// n_resident slots hold the resident (hot) set and are filled once; a routed
// expert that is not resident is read into a transient slot at fill time.
struct llama_disk_stage_cache_layer {
    ggml_tensor * gate  = nullptr; // [n_ff, n_embd, n_slots]
    ggml_tensor * up    = nullptr;
    ggml_tensor * down  = nullptr;
    ggml_tensor * table = nullptr; // I32 [n_expert], expert id -> slot
    int32_t       n_slots = 0;
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
    llama_disk_stage(const llama_model & model, ggml_backend_dev_t dev,
                     int32_t n_pin_experts, uint64_t cache_budget_bytes);
    ~llama_disk_stage();

    // staging tensors of MoE layer il, or null when the layer is not stageable
    const llama_disk_stage_layer * layer(int il) const;

    // synchronously fill layer il's staging tensors from the model file
    void fill(int il);

    // persistent decode cache of layer il, or null when the cache is off
    const llama_disk_stage_cache_layer * cache_layer(int il) const;

    // ensure the routed experts of layer il are in the cache and update the
    // layer's id table; reads the non-resident ones into transient slots
    void fill_cache(int il, const int32_t * ids, int64_t n_ids);

    // resident (hot) experts the decode cache can hold per layer, uniform across
    // layers; this is the per-layer pin capacity of the hot-expert ranking
    int32_t resident_capacity() const;

    // make expert id of layer il resident: reserve a slot for it and mark it
    // unfilled. No I/O here - fill_cache() reads the expert into the slot the next
    // time it is routed, so a promoted expert is read from disk exactly once
    bool resident_add(int il, int32_t id);

    // drop a resident expert, freeing its slot for the next promotion
    void resident_remove(int il, int32_t id);

    bool is_active() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
