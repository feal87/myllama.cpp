#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;
    bool training;           // set by llama_opt_init()

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    // --pin-hot-experts N: number of hottest MoE experts to mlock per layer, ranked
    // GLOBALLY across all layers (total slots = N x num_moe_layers, 0 = disabled)
    int32_t n_pin_hot_experts;
    // hard cap in bytes on total memory locked by n_pin_hot_experts, across all layers (0 = unlimited)
    uint64_t n_pin_hot_experts_budget_bytes;
    // max layers sharing one disk decode-cache pool (0 = no limit, 1 = no pooling)
    int32_t n_pin_hot_experts_pool_layers;
    // print the periodic expert-tier stats report (RAM pin + VRAM MoE) every N
    // seconds of evaluation (0 = only at context teardown)
    uint64_t n_experts_stats_interval;
    // halve all usage counts every N tokens (0 = disabled, lifetime counts)
    uint64_t n_pin_hot_experts_decay_tokens;
    // divide all usage counts by N at every prompt start (0 or 1 = keep them)
    uint64_t n_pin_hot_experts_prompt_decay;
    // minimum usage count an expert must reach before it is pinned (0 = any routed
    // expert); with aging the counts shrink, so scale this with the decay window
    uint64_t n_pin_hot_experts_min_count;
    const char * expert_profile_path;
    // raw binary decode token + routed-expert recording for offline expert-prediction
    // experiments (nullptr = off)
    const char * expert_ngram_record_path;
    // base set of experts to keep resident in the disk decode cache (nullptr = off)
    const char * pin_experts_from_profile_path;
    // warm set of experts to prefill the remaining decode-cache slots (nullptr = off)
    const char * warm_experts_from_profile_path;
    // tool-conditioned base-expert set template (nullptr = off)
    const char * pin_experts_template_path;

    // GPU-resident MoE expert cache, VRAM tier on top of the hot-expert cache
    // (see llama-moecache.h). The shared routing ranking is observed by the
    // hot-expert engine, which llama_context spins up automatically when this
    // tier is requested even if pinning (n_pin_hot_experts) is off.
    uint64_t n_moe_cache_budget_bytes; // total device memory reserved up-front for the whole cache (dummy slots and
                                       // device tables included; the final layout always fits within it) (0 = no cap)
    int32_t  n_moe_cache_inserts;      // max expert uploads per decode step, across all cached layers (global)
    float    n_moe_cache_drift_percent;// rebuild the per-layer layout when it drifts more than this percent from
                                       // the ideal composition for the current ranking (0 = disabled)
    uint64_t n_moe_cache_decay_tokens; // rank the VRAM tier by recent decode routing counts halved every N tokens
                                       // (0 = share the hot-expert cache's long-term ranking)
    uint64_t n_moe_cache_rebalance_tokens; // content tokens between VRAM content rebalances (0 = default 256)

    // batch/prefill read-ahead of the routed-but-unpinned MoE expert rows
    // (--hot-experts-prefetch; multi-token ubatches only). Starts the hot-expert
    // engine on its own, so it works standalone (no pin, no MoE tier)
    bool hot_experts_prefetch;

    // disk stage: multi-token ubatches smaller than this read only their routed
    // experts instead of the whole slab (--disk-stage-sparse-max). 0 always
    // streams the slab; resolved from the flag, then the environment, then 32
    int32_t disk_stage_sparse_max;

    // disk stage: drop the lowest fraction of the routed experts of a decode
    // token when they would need a disk read (--disk-stage-drop-fraction, 0 = off)
    float   disk_stage_drop_fraction;
    // disk stage: only drop an expert whose score is below this times the
    // layer's highest routed score (--disk-stage-drop-below-rel)
    float   disk_stage_drop_below_rel;
    // disk stage: hard ceiling on the score mass a single decode layer may
    // perturb, as a fraction of that layer's routed score mass
    // (--disk-stage-drop-max-mass, 0 = off)
    float   disk_stage_drop_max_mass;
    // disk stage: hard ceiling on the score mass a single decode token may
    // perturb, as a fraction of the token's routed score mass
    // (--disk-stage-drop-max-mass-token, 0 = off)
    float   disk_stage_drop_max_mass_token;
    // disk stage: substitution of a cold routed expert by a nearby resident one
    // (--disk-stage-drop-substitute-rel/pool, pool = 0 = off)
    float   disk_stage_drop_substitute_rel;
    int32_t disk_stage_drop_substitute_pool;
    // disk stage: measure and report the routed score distribution without
    // dropping anything (--disk-stage-drop-probe)
    bool    disk_stage_drop_probe;
    // disk stage: dedicated mlocked L2 pool that survives prefill
    // (--disk-stage-l2-permanent-mib, 0 = off)
    uint64_t disk_stage_l2_permanent_bytes;

    llama_context * ctx_other;
};
