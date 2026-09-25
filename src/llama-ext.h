#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// --moe-expert-cache*: release / re-reserve the MoE expert cache's device pool
// so its VRAM can serve another consumer (e.g. the mmproj) and be restored
// afterwards. suspend() must be called with no graph in flight. Both are no-ops
// that return false when the cache is disabled. budget_bytes() and device()
// describe the pool (0 / null when disabled) for the caller's fit check.
LLAMA_API bool         llama_moe_cache_suspend      (struct llama_context * ctx);
LLAMA_API bool         llama_moe_cache_resume       (struct llama_context * ctx);
LLAMA_API uint64_t     llama_moe_cache_budget_bytes (const struct llama_context * ctx);
LLAMA_API ggml_backend_dev_t llama_moe_cache_device (const struct llama_context * ctx);

struct llama_expert_cache_stats {
    bool     enabled = false;
    bool     active  = false;
    uint64_t residents        = 0;
    uint64_t capacity         = 0;
    uint64_t resident_bytes   = 0;
    uint64_t locked_bytes     = 0;
    uint64_t pool_bytes       = 0;
    uint64_t budget_bytes     = 0;
    uint64_t route_hits       = 0;
    uint64_t route_misses     = 0;
    uint64_t assigned_routes  = 0;
    uint64_t unassigned_routes = 0;
    uint64_t fills            = 0;
    uint64_t base_routes      = 0;
    uint64_t base_experts_used = 0;
    uint64_t resident_changes = 0;
    uint64_t uploads_queued   = 0;
    uint64_t uploads_succeeded = 0;
    uint64_t uploads_failed   = 0;
    uint64_t rebalances       = 0;
};

struct llama_expert_l2_stats {
    bool     enabled = false;
    bool     warm    = false;
    uint64_t entries          = 0;
    uint64_t capacity         = 0;
    uint64_t hits             = 0;
    uint64_t misses           = 0;
    uint64_t cold_lookups     = 0;
    uint64_t hit_bytes        = 0;
    uint64_t promotions       = 0;
    uint64_t promotion_bytes  = 0;
    uint64_t evictions        = 0;
    uint64_t demotions        = 0;
    uint64_t decode_fill_calls       = 0;
    uint64_t decode_fill_bytes       = 0;
    uint64_t decode_fill_microseconds = 0;
};

struct llama_expert_stats {
    bool     dio_active = false;
    uint64_t routed_experts = 0;
    uint64_t decode_tokens  = 0;
    uint64_t experts_seen   = 0;

    llama_expert_cache_stats decode_cache;
    llama_expert_l2_stats    disk_l2;
    llama_expert_cache_stats vram_cache;
};

LLAMA_API llama_expert_stats llama_get_expert_stats(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
