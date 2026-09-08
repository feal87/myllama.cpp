#pragma once

// GPU-resident cache for the HOTTEST MoE experts, layered on top of the host
// hot-expert cache (see llama-hot-experts.h). Both tiers are driven by ONE
// global ranked set: the hot-expert cache's decayed per-(layer, expert) routing
// counts. The VRAM tier copies the very top of that ranking into device memory
// so their host reads are skipped entirely during decode; the hot-expert cache
// mlock's the next ranks in place and prefetches the rest. The VRAM tier only
// needs the ranking, not the pinning: with --pin-hot-experts off, llama_context
// still spins up the observation engine (n_pin_experts = 0) so nothing is
// mlock'd but the counts above are maintained.
//
// Lifecycle:
//  - requested via llama_context_params (CLI --moe-expert-cache-budget-mib +
//    --moe-expert-cache-inserts). Requires the hot-expert ranking engine (the
//    counts live there).
//  - activated lazily on the first single-token decode ubatch once enough
//    routing has been observed (the prefill of the current request). At that
//    point the per-layer VRAM slot counts are sized from the actual routing
//    profile: a global budget is handed to the experts with the highest counts,
//    so hot layers get many slots and cold layers get none (top-heavy, not
//    uniform).
//  - content is maintained in batches at rebalance boundaries (every
//    rebalance interval of content tokens): each cached layer holds the top
//    C_l of the CURRENT ranking; losers are evicted back to the RAM tier (the
//    hot cache re-mlock's them) and winners are uploaded asynchronously and
//    published between graphs. The LRU-free design means a super-expert that
//    keeps being routed is never displaced by a merely-recent one.
//
// Mechanism (no custom kernels):
//  - per cached layer, companion tensors in the device buffer type of that
//    layer's router (ffn_gate_inp), with ne[2] == slots + 1:
//      fused gate_up layout : c_gate (one tensor covering gate+up)
//      separate gate/up      : c_gate + c_up
//    plus c_down for the down projection. Slot `slots` is permanently zero
//    (the "dummy" slot).
//  - an I32 table[1, n_expert] maps expert id -> slot, or `slots` when not
//    resident. Device copy (ggml_get_rows remaps ids for the cache-side
//    mul_mat_id chain) + host copy (CPU mul_mat_id skip via src[3], zeroing the
//    dst rows of residents).
//  - host and device down projections are summed: residents contribute 0 via
//    the CPU chain (skip) and non-residents contribute 0 via the cache chain
//    (dummy slot), so the merge is exact by construction.
//  - decode-only (n_tokens == 1); prefill/batch ubatches build the stock graph.

#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;
struct ggml_tensor;
class llama_hot_expert_cache;

// per-layer view read by the graph builder (build_moe_ffn)
struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0; // capacity; the dummy slot index is n_slots

    // layout: fused = one gate+up tensor (gate_src == up_src), else separate
    bool fused = false;

    const ggml_tensor * gate_src = nullptr; // authoritative host tensor (ffn_gate_up_exps or ffn_gate_exps)
    const ggml_tensor * up_src   = nullptr; // ffn_up_exps, or the fused tensor again
    const ggml_tensor * d_src    = nullptr; // host authoritative down weights

    // device cache tensors, ne[2] == n_slots + 1 (last slot all zeros)
    ggml_tensor * c_gate  = nullptr; // fused: the gate+up tensor; separate: gate tensor
    ggml_tensor * c_up    = nullptr; // separate layout only (null when fused)
    ggml_tensor * c_down  = nullptr;

    ggml_tensor * dev_table  = nullptr; // I32 [1, n_expert] on the device: expert id -> slot or n_slots
    ggml_tensor * host_table = nullptr; // same content in host memory (CPU mul_mat_id skip via src[3])
};

class llama_moe_cache {
  public:
    // hot:      ranking source (must outlive this object; the hot cache itself is
    //           kept by llama_context, destroyed after this)
    // slots:    per-layer uniform capacity override (0 = derive per-layer slots
    //           from budget_bytes and the observed routing profile)
    // budget_bytes: global device-memory cap across all cached layers (used when slots == 0)
    // max_inserts: max expert uploads per decode step, GLOBAL across all cached layers
    llama_moe_cache(const llama_model & model, llama_hot_expert_cache * hot,
                    int32_t slots, uint64_t budget_bytes, int32_t max_inserts);
    ~llama_moe_cache();

    llama_moe_cache(const llama_moe_cache &) = delete;
    llama_moe_cache & operator=(const llama_moe_cache &) = delete;

    bool is_active() const;

    // called before every decode ubatch until activated: allocates the device
    // buffers once enough routing has been observed to size them
    void maybe_activate();

    // cache layer owning `gate` (the ffn_gate_up_exps or ffn_gate_exps tensor),
    // or nullptr when the cache is inactive or the layer has no slots
    const llama_moe_cache_layer * lookup(const ggml_tensor * gate) const;

    // called between graph executions (after each ubatch): publishes completed
    // uploads and periodically rebalances content against the current ranking
    void tick(int64_t n_content_tokens);

    // print the periodic stats report (resident vs capacity, hit rate, list
    // churn since the previous report, per-layer breakdown) via LLAMA_LOG_INFO
    // (verbosity 4). Called by llama_context at the shared
    // --experts-stats-interval cadence; also refreshes the churn snapshot.
    void print_stats();

    // llama_hot_expert_cache::vram_query_fn-compatible: fills `flags` with the
    // 0/1 residency of every expert of layer `il` (empty when the layer has no
    // device cache). The RAM tier uses it to skip double-covering residents and
    // to feed this cache's decode-time hit/miss stats (see vram_stats).
    static void vram_resident_cb(void * ud, int il, std::vector<uint8_t> & flags);

  private:
    // reconcile the residents with the current ranking (see llama-moecache.cpp)
    void rebalance();

    struct impl;
    std::unique_ptr<impl> pimpl;
};
