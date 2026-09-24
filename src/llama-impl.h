#pragma once

#include "ggml.h" // for ggml_log_level
#include "ggml-backend.h"

#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

LLAMA_ATTRIBUTE_FORMAT(2, 3)
void llama_log_internal        (ggml_log_level level, const char * format, ...);
void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLAMA_LOG(...)       llama_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) llama_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LLAMA_LOG_CONT(...)  llama_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

//
// memory accounting
//

// attribute every backend buffer allocated on this thread until the guard is
// destroyed to `tag` (diagnostics, see ggml_backend_mem_foreach)
struct llama_mem_tag_scope {
    explicit llama_mem_tag_scope(const char * tag) { ggml_backend_mem_push(tag); }
    ~llama_mem_tag_scope() { ggml_backend_mem_pop(); }

    llama_mem_tag_scope(const llama_mem_tag_scope &) = delete;
    llama_mem_tag_scope & operator=(const llama_mem_tag_scope &) = delete;
};

//
// helpers
//

template <typename T>
struct no_init {
    T value;
    no_init() = default;
};

// sparse attention is opt-in: a model can carry the indexer metadata and still be served better
// by full attention, because the indexer keeps only a small budget of cells per token and drops
// long-range detail. the metadata alone must never switch it on
static inline bool llama_qsa_allowed() {
    static const bool allowed = [] {
        const char * e = getenv("LLAMA_QSA_ALLOW");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();

    return allowed;
}

// GLM5-Next: serve the MLA layers with plain dense attention over the whole MLA
// cache instead of the k-pool DSA indexer (which caps a query at
// indexer_top_k/kpool tokens -- 2048 of them -- however long the context is).
// opt-in via LLAMA_GLM_FULL_ATTN=1. When on, the indexer weights are not loaded,
// its KV cache is not allocated, and the prefill graph loses the indexer
// workspace (measured on GLM-5.3-Flash: -1090 MiB of compute buffer at ub 512).
static inline bool llama_glm_full_attn() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_GLM_FULL_ATTN");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();

    return on;
}

template <typename dst_t, typename src_t>
static inline dst_t llama_cast(src_t v) {
    if constexpr (std::is_same_v<src_t, dst_t>) {
        return v;
    } else if constexpr (std::is_same_v<src_t, ggml_fp16_t> && std::is_same_v<dst_t, float>) {
        return ggml_fp16_to_fp32(v);
    } else if constexpr (std::is_same_v<src_t, float> && std::is_same_v<dst_t, ggml_fp16_t>) {
        return ggml_fp32_to_fp16(v);
    } else {
        static_assert(std::is_same_v<dst_t, void>, "unsupported type combination");
    }
}

static inline ggml_tensor * llama_mul_mat_hadamard(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d(ctx, cur, n, ggml_nelements(cur)/n);
    } else {
        res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    }
    res = ggml_mul_mat(ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

template <typename T>
struct buffer_view {
    T * data;
    size_t size = 0;

    bool has_data() const {
        return data && size > 0;
    }
};

void replace_all(std::string & s, const std::string & search, const std::string & replace);

// TODO: rename to llama_format ?
LLAMA_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * fmt, ...);

std::string llama_format_tensor_shape(const std::vector<int64_t> & ne);
std::string llama_format_tensor_shape(const struct ggml_tensor * t);

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);
