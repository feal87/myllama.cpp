#include "llama-moecache.h"

#include "llama-hot-experts.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// decode tokens of routing to collect before the VRAM tier sizes itself from
// the profile (the first decode tokens of the first request: the shared ranking
// is fed by single-token decode ubatches only, see llama-hot-experts.h)
static constexpr uint64_t kMinProfileContentTokens = 512;

// content tokens between content rebalances (each rebalance reconciles the
// residents with the current global ranking)
static constexpr uint64_t kRebalanceContentTokens = 256;

struct llama_moe_cache::impl {
    // one layer the cache can serve (static once the model is loaded: expert
    // weights host-resident, router on the device)
    struct candidate {
        int      il      = -1;
        bool     fused   = false; // single fused gate+up tensor
        // authoritative host source tensors (== the llama_moe_cache_layer views)
        const ggml_tensor * gate_src = nullptr; // fused: ffn_gate_up_exps, else ffn_gate_exps
        const ggml_tensor * up_src   = nullptr; // fused: gate_src again, else ffn_up_exps
        const ggml_tensor * d_src    = nullptr; // ffn_down_exps
        const ggml_tensor * router   = nullptr; // ffn_gate_inp: gives the device pool's buft
        int64_t n_expert = 0;
        size_t  nbytes_1slot = 0;   // one expert across all of the layer's cache tensors
    };

    struct layer_state {
        llama_moe_cache_layer pub;

        // (cache tensor, source tensor) pairs copied on upload
        std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> uploads;

        // bookkeeping
        std::vector<int32_t> slot_expert;   // slots -> published resident expert (-1 = empty)
        std::vector<int32_t> slot_target;   // slots -> expert whose upload is in flight (-1 = none)
        std::vector<uint8_t> resident;      // n_expert -> currently served from VRAM (hot query / stats)
        std::deque<int32_t>  pending_q;     // desired ids not yet resident/in flight, hottest first
    };

    struct upload_job {
        size_t  layer_idx;
        int32_t expert;
        int32_t slot;
    };

    const llama_model & model;
    llama_hot_expert_cache * const hot;
    const uint64_t budget_bytes;   // total device footprint of the cache, reserved up-front as one pool
    const int32_t  max_inserts;

    bool activated = false;
    bool failed    = false;

    // reserved at context creation (reserve()): the cacheable layers and the
    // single device pool all of their cache tensors (slots + dummy + tables)
    // are carved from at activation
    std::vector<candidate> cands;
    ggml_backend_buffer_t  pool = nullptr;

    std::vector<layer_state> layers;

    // tensor metadata (per-layer device cache tensors + one shared host context
    // for the tables). The device tensors are bound into `pool` at activation
    std::vector<ggml_context *> ctxs_dev;
    ggml_context *        ctx_host = nullptr;
    ggml_backend_buffer_t buf_host = nullptr;

    // async upload worker: slices are copied off the decode thread; the new
    // mapping is only published at the next tick(), after the copy completed
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;

    // guards the bookkeeping above. The hot cache calls vram_resident_cb() (holding
    // its own mutex) from the router observation, which runs during graph compute,
    // while tick()/rebalance() run between graphs - so contention is negligible.
    // hot.mu -> this mutex is the only lock order (never take hot's mutex while
    // holding this one).
    std::mutex mtx;

    uint64_t n_content      = 0;
    uint64_t last_rebalance = 0;
    uint64_t n_ticks        = 0;

    // (layer, expert) residents at the previous stats report; diffed against the
    // current residents to measure list churn (key = (uint32 layer << 32) | expert)
    std::unordered_set<uint64_t> prev_resident;

    // per-prompt epoch: the layout is rebuilt once per new prompt, after its
    // first kMinProfileContentTokens decode tokens refreshed the ranking.
    // epoch_content is the hot cache's decode-token count at the prompt start
    // (set by on_prompt_begin()); the initial activation is the same decision
    // with the epoch anchored at context start, so no extra flags are needed.
    uint64_t epoch_content   = 0;   // hot->content_tokens() at the prompt boundary
    bool     epoch_rebuilt   = false; // layout (re)built for the current prompt

    // bumped on every layout (re)build: decode graphs embed the cache tensors,
    // so the graph-reuse check compares this to force a rebuild (see
    // llama_context::graph_params / llm_graph_params::allow_reuse)
    uint32_t layout_gen = 0;

    // contexts of a layout that was just replaced. The decode graph that still
    // referenced them is reset on the same ubatch as the rebuild, so they are
    // freed at the next graph boundary (tick) rather than immediately (the
    // rebuild itself runs before the graph reset). ctxs_retired_gen records the
    // layout generation they were built at: they are only freed once a strictly
    // newer layout is live, so a failed rebuild (which retires the still-current
    // tensors without bumping the generation) can never free contexts a cached
    // decode graph still references.
    std::vector<ggml_context *> ctxs_retired;
    ggml_context        * ctx_host_retired = nullptr;
    ggml_backend_buffer_t buf_host_retired = nullptr;
    uint32_t              ctxs_retired_gen = 0;

    void free_retired() {
        for (auto * ctx : ctxs_retired) {
            ggml_free(ctx);
        }
        ctxs_retired.clear();
        if (buf_host_retired) {
            ggml_backend_buffer_free(buf_host_retired);
            buf_host_retired = nullptr;
        }
        if (ctx_host_retired) {
            ggml_free(ctx_host_retired);
            ctx_host_retired = nullptr;
        }
    }

    // free the retired contexts when no live decode graph can still reference
    // them (only once a newer layout generation has been built and executed)
    void free_retired_if_safe() {
        if (!ctxs_retired.empty() && ctxs_retired_gen < layout_gen) {
            free_retired();
        }
    }

    // move the current layout's tensor contexts into the retired set (the
    // caller has stopped the upload worker and is about to carve the pool anew)
    void retire_layout() {
        ctxs_retired_gen = layout_gen;
        for (auto * ctx : ctxs_dev) {
            ctxs_retired.push_back(ctx);
        }
        ctxs_dev.clear();
        ctx_host_retired = ctx_host;
        ctx_host         = nullptr;
        buf_host_retired = buf_host;
        buf_host         = nullptr;
    }

    impl(const llama_model & model_, llama_hot_expert_cache * hot_,
         uint64_t budget_, int32_t inserts_) :
        model(model_), hot(hot_), budget_bytes(budget_), max_inserts(inserts_) {}

    layer_state * find_layer(int il) {
        for (auto & ls : layers) {
            if (ls.pub.il == il) {
                return &ls;
            }
        }
        return nullptr;
    }

    ~impl() {
        {
            std::lock_guard<std::mutex> lock(wmtx);
            stop = true;
        }
        wcv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        if (pool) {
            ggml_backend_buffer_free(pool);
        }
        for (auto * ctx : ctxs_dev) {
            ggml_free(ctx);
        }
        if (buf_host) {
            ggml_backend_buffer_free(buf_host);
        }
        if (ctx_host) {
            ggml_free(ctx_host);
        }
        // leftover contexts of a layout that was rebuilt but never ticked
        free_retired();
    }
};

static size_t expert_slice_bytes(const ggml_tensor * w) {
    return w->nb[2];
}

// write one layer's full expert->slot table to both copies (device + host).
// Batching every boundary change into one per-layer refresh keeps the device
// traffic at a single small copy per changed layer instead of one 4-byte set
// per expert. Caller holds the impl mutex; no graph is running.
static void sync_tables(llama_moe_cache_layer & pub, const std::vector<int32_t> & slot_expert) {
    const int32_t n_expert = (int32_t) pub.gate_src->ne[2];
    const int32_t dummy    = pub.n_slots;

    std::vector<int32_t> tbl(n_expert, dummy);
    for (int32_t s = 0; s < pub.n_slots; ++s) {
        const int32_t e = slot_expert[s];
        if (e >= 0 && e < n_expert) {
            tbl[e] = s;
        }
    }
    ggml_backend_tensor_set(pub.dev_table,  tbl.data(), 0, n_expert*sizeof(int32_t));
    ggml_backend_tensor_set(pub.host_table, tbl.data(), 0, n_expert*sizeof(int32_t));
}

// move up to `max_inserts` pending experts to the upload worker's queue, one per
// layer per pass so a hot layer cannot starve the others behind a long pending
// list. The worker tops its own queue up whenever it runs dry (see below), so
// this only matters right after a rebalance filled pending_q; the uploads then
// proceed at the worker's pace instead of a fixed number per decode step, which
// is what kept the cache from filling for many steps. Caller holds p->wmtx;
// takes p->mtx (wmtx -> mtx is the lock order used everywhere).
void llama_moe_cache::fill_upload_queue(llama_moe_cache::impl * p) {
    std::lock_guard<std::mutex> lk(p->mtx);

    while ((int32_t) p->todo.size() < p->max_inserts) {
        bool pushed = false;
        for (size_t li = 0; li < p->layers.size(); ++li) {
            auto & ls = p->layers[li];
            const int32_t n_slots = ls.pub.n_slots;
            if (ls.pending_q.empty()) {
                continue;
            }
            // find a free slot (published-empty and not uploading)
            int32_t slot = -1;
            for (int32_t s = 0; s < n_slots; ++s) {
                if (ls.slot_expert[s] < 0 && ls.slot_target[s] < 0) {
                    slot = s;
                    break;
                }
            }
            if (slot < 0) {
                continue; // no free slot right now; retried once a slot frees
            }

            const int32_t e = ls.pending_q.front();
            ls.pending_q.pop_front();
            if (ls.resident[e] || e < 0) {
                continue; // published meanwhile (or bad id)
            }

            ls.slot_target[slot] = e;
            p->todo.push_back({ li, e, slot });
            pushed = true;
            if ((int32_t) p->todo.size() >= p->max_inserts) {
                break;
            }
        }
        if (!pushed) {
            break; // nothing uploadable left in any layer
        }
    }
}

llama_moe_cache::llama_moe_cache(const llama_model & model, llama_hot_expert_cache * hot,
                                 uint64_t budget_bytes, int32_t max_inserts) {
    if (hot == nullptr || budget_bytes == 0) {
        return; // disabled (missing ranking source or no capacity requested)
    }
    if (max_inserts <= 0) {
        max_inserts = 2;
    }

    pimpl = std::make_unique<impl>(model, hot, budget_bytes, max_inserts);
}

llama_moe_cache::~llama_moe_cache() {
    if (pimpl) {
        if (pimpl->hot) {
            pimpl->hot->set_vram_query(nullptr, nullptr);
        }
    }
    // pimpl releases the worker, buffers and contexts
}

bool llama_moe_cache::is_active() const {
    return pimpl != nullptr && pimpl->activated;
}

// llama_hot_expert_cache::vram_query_fn: copy this layer's residency flags (one
// 0/1 byte per expert) for the RAM tier's observation, which snapshots each layer
// once per ubatch instead of querying per routed expert.
void llama_moe_cache::vram_resident_cb(void * ud, int il, std::vector<uint8_t> & flags) {
    auto * self = static_cast<llama_moe_cache *>(ud);
    if (!self || !self->pimpl || !self->pimpl->activated) {
        return;
    }
    std::lock_guard<std::mutex> lock(self->pimpl->mtx);
    auto * ls = self->pimpl->find_layer(il);
    if (!ls) {
        return;  // not a cached layer: leave `flags` empty
    }
    flags.assign(ls->resident.begin(), ls->resident.end());
}

void llama_moe_cache::reserve() {
    auto * p = pimpl.get();
    if (!p || p->failed || p->pool) {
        return;
    }

    // collect the host-resident MoE layers that have a device home for the
    // cache. Both the fused gate_up layout and the separate gate/up layout work.
    bool saw_moe_with_data = false;
    for (size_t il = 0; il < p->model.layers.size(); ++il) {
        const auto & l = p->model.layers[il];
        if (!l.ffn_down_exps || !l.ffn_gate_inp) {
            continue;
        }
        if (l.ffn_down_exps->ne[2] == 0) {
            continue;
        }
        if (!l.ffn_down_exps->data) {
            continue; // dry-run / memory-estimation model: weights not loaded
        }
        saw_moe_with_data = true;
        if (ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
            continue; // no device home for the cache tensors
        }

        const ggml_tensor * gate = nullptr;
        const ggml_tensor * up   = nullptr;
        bool fused = false;

        if (l.ffn_gate_up_exps && l.ffn_gate_up_exps->data &&
                ggml_backend_buffer_is_host(l.ffn_gate_up_exps->buffer)) {
            // fused gate+up tensor
            if (l.ffn_gate_up_exps->ne[2] != l.ffn_down_exps->ne[2]) {
                continue;
            }
            gate  = l.ffn_gate_up_exps;
            up    = l.ffn_gate_up_exps;
            fused = true;
        } else if (l.ffn_gate_exps && l.ffn_up_exps &&
                l.ffn_gate_exps->data && l.ffn_up_exps->data &&
                ggml_backend_buffer_is_host(l.ffn_gate_exps->buffer) &&
                ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
            // separate gate + up tensors
            if (l.ffn_gate_exps->ne[2] != l.ffn_up_exps->ne[2] ||
                l.ffn_gate_exps->ne[2] != l.ffn_down_exps->ne[2]) {
                continue;
            }
            gate = l.ffn_gate_exps;
            up   = l.ffn_up_exps;
        } else {
            continue; // experts already on a device, or an unsupported layout
        }

        if (!ggml_backend_buffer_is_host(l.ffn_down_exps->buffer)) {
            continue; // down already on a device
        }

        impl::candidate c;
        c.il           = (int) il;
        c.fused        = fused;
        c.gate_src     = gate;
        c.up_src       = up;
        c.d_src        = l.ffn_down_exps;
        c.router       = l.ffn_gate_inp;
        c.n_expert     = gate->ne[2];
        c.nbytes_1slot = expert_slice_bytes(l.ffn_down_exps);
        c.nbytes_1slot += fused ? expert_slice_bytes(gate) : expert_slice_bytes(gate) + expert_slice_bytes(up);
        p->cands.push_back(c);
    }

    if (p->cands.empty()) {
        if (saw_moe_with_data) {
            LLAMA_LOG_WARN("%s: no host-resident MoE layer with a device router was found - MoE expert cache stays disabled\n", __func__);
        }
        p->failed = true;
        return;
    }

    // all candidates share ONE device pool (single-GPU model), allocated in the
    // same buffer type as the layer routers
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(p->cands[0].router->buffer);
    for (const auto & c : p->cands) {
        if (ggml_backend_buffer_get_type(c.router->buffer) != buft) {
            LLAMA_LOG_WARN("%s: cacheable MoE layers span multiple devices - MoE expert cache stays disabled\n", __func__);
            p->failed = true;
            p->cands.clear();
            return;
        }
    }

    p->pool = ggml_backend_buft_alloc_buffer(buft, p->budget_bytes);
    if (!p->pool) {
        throw std::runtime_error("failed to reserve " + std::to_string(p->budget_bytes/(1024*1024)) +
                " MiB of device memory for the MoE expert cache (--moe-expert-cache-budget-mib): the device must have this much free VRAM on top of the model");
    }

    LLAMA_LOG_INFO("%s: MoE expert cache: reserved %.1f MiB of device memory over %zu cacheable layer(s)\n",
            __func__, p->budget_bytes/(1024.0*1024.0), p->cands.size());
}

void llama_moe_cache::maybe_activate() {
    if (!pimpl || pimpl->failed) {
        return;
    }
    auto * p = pimpl.get();

    // the device pool was reserved at context creation (reserve() at the end of
    // the llama_context constructor); if that did not happen (no cacheable
    // layer, or not enough free device memory for the budget) the cache is
    // either failed or has no pool and stays off
    if (p->failed || !p->pool) {
        return;
    }

    // wait until enough decode routing has been observed to size the layers
    // from a real profile (the shared ranking is decode-only)
    if (p->activated) {
        // already active: rebuild the layout once per new prompt, after the
        // first kMinProfileContentTokens decode tokens of that prompt refreshed
        // the ranking. on_prompt_begin() re-arms the epoch at every prompt
        // boundary (see llama-context.cpp), and the old layout keeps serving
        // the first 512 tokens of the new prompt untouched. A rebuild can shrink
        // or grow the per-layer slot counts, so the cache tensors are re-carved
        // below exactly like the initial activation.
        if (p->epoch_rebuilt) {
            return;
        }
        if (p->hot->content_tokens() - p->epoch_content < kMinProfileContentTokens) {
            return;
        }
    } else if (p->hot->content_tokens() < kMinProfileContentTokens) {
        return;
    }
    const bool relayout = p->activated;

    // Per-layer VRAM slot capacities, derived from the observed routing profile:
    // hand the GLOBAL byte budget to the globally hottest experts (top of the
    // shared ranking first). A layer earns a slot only when one of its experts
    // is actually among the hottest model-wide, so hot layers end up with many
    // slots and cold layers with none. Each layer also pays a one-time fixed
    // cost when it earns its first slot (the dummy slot, the device table and
    // the tensor alignment padding), so the layout carved into the pool below
    // always fits inside the reserved budget.
    const size_t align = ggml_backend_buffer_get_alignment(p->pool);
    std::vector<int32_t> caps(p->model.layers.size(), 0);
    std::vector<size_t> bytes_per_layer(p->model.layers.size(), 0);
    std::vector<size_t> fixed_bytes(p->model.layers.size(), 0);
    for (const auto & c : p->cands) {
        bytes_per_layer[c.il] = c.nbytes_1slot;
        fixed_bytes[c.il]    = c.nbytes_1slot + (size_t) c.n_expert*sizeof(int32_t) + 4*align;
    }
    if (p->hot->assign_global_capacity(p->budget_bytes, bytes_per_layer, fixed_bytes, caps) <= 0) {
        if (relayout) {
            // keep the old layout serving rather than tearing it down for a
            // profile that gives no layer any slot
            LLAMA_LOG_WARN("%s: relayout skipped - the new prompt profile gives no layer any VRAM slot\n", __func__);
            p->epoch_rebuilt = true;
        } else {
            LLAMA_LOG_WARN("%s: no routing observed yet - MoE expert cache stays disabled\n", __func__);
            p->failed = true;
        }
        return;
    }

    // full per-prompt layout rebuild: pull every published VRAM resident back
    // into the RAM pin tier (their mlock was dropped when the VRAM copy took
    // over), then retire the old layout's tensor contexts. The decode graph that
    // still references those tensors is reset on this same ubatch - the layout
    // generation bump below forces the graph rebuild - so the retired contexts
    // are freed at the next tick() instead of here. The pool itself is kept and
    // re-carved below (it is a fixed reservation, not tied to the old tensors).
    if (relayout) {
        {
            std::lock_guard<std::mutex> wlk(p->wmtx);
            p->stop = true;
            p->todo.clear();
            p->done.clear();
        }
        p->wcv.notify_all();
        if (p->worker.joinable()) {
            p->worker.join();
        }

        // retire the old layout's tensor contexts. The decode graph that still
        // references them is reset on this same ubatch - the layout generation
        // bump below forces the graph rebuild - so they are freed once a newer
        // layout is live (free_retired_if_safe at the tick boundary) or by the
        // destructor. The pool itself is kept and re-carved below: it is a fixed
        // reservation, not tied to the old tensors.
        std::vector<std::pair<int, int32_t>> evicted;
        {
            std::lock_guard<std::mutex> lock(p->mtx);
            for (auto & ls : p->layers) {
                for (int32_t s = 0; s < (int32_t) ls.slot_expert.size(); ++s) {
                    const int32_t e = ls.slot_expert[s];
                    if (e < 0) {
                        continue;
                    }
                    ls.slot_expert[s] = -1;
                    ls.resident[(size_t) e] = 0;
                    evicted.emplace_back(ls.pub.il, e);
                }
                std::fill(ls.slot_target.begin(), ls.slot_target.end(), -1);
                ls.pending_q.clear();
            }
            p->layers.clear();
            p->retire_layout();
        }
        // re-admit the evicted experts to the RAM tier: their mlock was dropped
        // when the VRAM copy took over, so they must be mlock'd again before the
        // VRAM copies below are rebuilt. No impl lock held here - the hot cache
        // locks its own mutex and re-enters this one through the VRAM residency
        // query (hot.mu -> impl.mtx is the documented lock order).
        for (const auto & [il, e] : evicted) {
            p->hot->promote_expert(il, e);
        }
    }

    // shared host context for the CPU-side tables
    {
        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (p->cands.size() + 4),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        p->ctx_host = ggml_init(ip);
        if (!p->ctx_host) {
            // a failed relayout has already retired the old layout: fall back to
            // host-only decoding (exact) rather than serving a broken layout
            p->activated = false;
            p->failed    = true;
            return;
        }
    }

    p->layers.reserve(p->cands.size());
    int n_cached = 0;

    // bind every cached layer's device tensors into the reserved pool, one
    // aligned slice after the other (the same layout the per-layer buffers
    // used, so the assignment overhead charges match the real footprint)
    char * const pool_base = (char *) ggml_backend_buffer_get_base(p->pool);
    const size_t pool_size = ggml_backend_buffer_get_size(p->pool);
    size_t off = 0;

    for (const auto & c : p->cands) {
        const int32_t n_slots = caps[c.il];
        if (n_slots <= 0) {
            continue; // this layer earned no VRAM slots
        }

        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx_dev = ggml_init(ip);
        if (!ctx_dev) {
            p->activated = false;
            p->failed    = true;
            return;
        }
        p->ctxs_dev.push_back(ctx_dev);

        p->layers.emplace_back();
        auto & ls = p->layers.back();
        auto & pub = ls.pub;
        pub.il       = c.il;
        pub.n_slots  = n_slots;
        pub.fused    = c.fused;
        pub.gate_src = c.gate_src;
        pub.up_src   = c.up_src;
        pub.d_src    = c.d_src;

        pub.c_gate = ggml_new_tensor_3d(ctx_dev, c.gate_src->type, c.gate_src->ne[0], c.gate_src->ne[1], n_slots + 1);
        ggml_format_name(pub.c_gate, "moe_cache_gate.%d", c.il);
        ls.uploads.emplace_back(pub.c_gate, c.gate_src);

        if (!c.fused) {
            pub.c_up = ggml_new_tensor_3d(ctx_dev, c.up_src->type, c.up_src->ne[0], c.up_src->ne[1], n_slots + 1);
            ggml_format_name(pub.c_up, "moe_cache_up.%d", c.il);
            ls.uploads.emplace_back(pub.c_up, c.up_src);
        }
        pub.c_down = ggml_new_tensor_3d(ctx_dev, c.d_src->type, c.d_src->ne[0], c.d_src->ne[1], n_slots + 1);
        ggml_format_name(pub.c_down, "moe_cache_down.%d", c.il);
        ls.uploads.emplace_back(pub.c_down, c.d_src);

        pub.dev_table = ggml_new_tensor_2d(ctx_dev, GGML_TYPE_I32, 1, c.n_expert);
        ggml_format_name(pub.dev_table, "moe_cache_tbl.%d", c.il);

        // bind this layer's cache tensors into the shared pool
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_dev); t != nullptr; t = ggml_get_next_tensor(ctx_dev, t)) {
            size_t sz = ggml_backend_buffer_get_alloc_size(p->pool, t);
            sz = (sz + align - 1) & ~(align - 1);
            if (off + sz > pool_size) {
                LLAMA_LOG_ERROR("%s: MoE cache layout exceeds the reserved pool (%zu > %zu bytes) - MoE expert cache disabled\n",
                        __func__, off + sz, pool_size);
                p->activated = false;
                p->failed    = true;
                return;
            }
            ggml_backend_tensor_alloc(p->pool, t, pool_base + off);
            off += sz;
        }

        pub.host_table = ggml_new_tensor_2d(p->ctx_host, GGML_TYPE_I32, 1, c.n_expert);
        ggml_format_name(pub.host_table, "moe_cache_htbl.%d", c.il);

        ls.slot_expert.assign(n_slots, -1);
        ls.slot_target.assign(n_slots, -1);
        ls.resident.assign(c.n_expert, 0);
        n_cached++;
    }

    if (n_cached == 0) {
        LLAMA_LOG_WARN("%s: no layer earned any VRAM slot from the routing profile - MoE expert cache disabled\n", __func__);
        p->activated = false;
        p->failed    = true;
        return;
    }

    // zero the whole pool once: the dummy slot (n_slots) of every cached layer
    // must stay zeros for the cache-side mul_mat chain to contribute nothing
    ggml_backend_buffer_clear(p->pool, 0);

    // host buffer for the CPU-side tables
    p->buf_host = ggml_backend_alloc_ctx_tensors_from_buft(p->ctx_host, ggml_backend_cpu_buffer_type());
    if (!p->buf_host) {
        LLAMA_LOG_WARN("%s: failed to allocate the CPU-side MoE cache tables - disabled\n", __func__);
        p->activated = false;
        p->failed    = true;
        return;
    }

    // everything uncached -> dummy slot n_slots, in both the device and host tables
    for (auto & ls : p->layers) {
        const int32_t n_expert = (int32_t) ls.pub.gate_src->ne[2];
        std::vector<int32_t> dummy(n_expert, ls.pub.n_slots);
        ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
        ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));
    }

    // (re)start the upload worker: a per-prompt relayout stopped it above, so
    // clear the stop flag before the new thread starts (activation leaves it
    // cleared from the constructor)
    {
        std::lock_guard<std::mutex> wlk(p->wmtx);
        p->stop = false;
    }

    // async upload worker
    p->worker = std::thread([p]() {
        for (;;) {
            impl::upload_job j;
            {
                std::unique_lock<std::mutex> lk(p->wmtx);
                // self-sustain: refill the queue from pending_q before waiting, so
                // a burst of additions (activation, content rebalance) uploads at
                // the worker's pace instead of one tick() batch per decode step
                if (!p->stop && p->todo.empty()) {
                    llama_moe_cache::fill_upload_queue(p);
                }
                p->wcv.wait(lk, [p]() { return p->stop || !p->todo.empty(); });
                if (p->stop && p->todo.empty()) {
                    return;
                }
                j = p->todo.front();
                p->todo.pop_front();
            }

            auto & ls = p->layers[j.layer_idx];
            for (const auto & [dst_c, src] : ls.uploads) {
                const size_t sz = src->nb[2];
                if ((size_t) j.expert*sz + sz <= ggml_nbytes(src) &&
                        (size_t) j.slot*sz + sz <= ggml_nbytes(dst_c)) {
                    ggml_backend_tensor_set(dst_c,
                            (const char *) src->data + (size_t) j.expert*sz,
                            (size_t) j.slot*sz, sz);
                }
            }
            {
                std::lock_guard<std::mutex> lk(p->wmtx);
                p->done.push_back(j);
            }
        }
    });

    p->hot->set_vram_query(&llama_moe_cache::vram_resident_cb, this);

    p->activated = true;
    // decode graphs embed the cache tensors and the per-layer slot counts, so
    // the graph-reuse check compares this generation: every (re)build forces a
    // decode graph rebuild on the same ubatch (the initial activation also
    // flips the moe_cache pointer in the graph params from null to non-null)
    p->layout_gen++;

    // one (re)build per prompt: anchor the epoch so this prompt does not rebuild
    // again; the next on_prompt_begin() re-arms it at the following boundary
    p->epoch_content = p->hot->content_tokens();
    p->epoch_rebuilt = true;

    {
        std::string layers_str;
        size_t n_slots_total = 0;
        for (auto & ls : p->layers) {
            n_slots_total += (size_t) ls.pub.n_slots;
            if (!layers_str.empty()) {
                layers_str += ",";
            }
            layers_str += "L" + std::to_string(ls.pub.il) + "=" + std::to_string(ls.pub.n_slots);
        }
        LLAMA_LOG_INFO("%s: MoE expert cache %s: %d layer(s), %zu VRAM slots total (%.1f of the reserved %.1f MiB pool used), up to %d uploads queued (global); profile: %s\n",
                __func__, relayout ? "layout rebuilt for the new prompt" : "enabled",
                n_cached, n_slots_total, off/(1024.0*1024.0), p->budget_bytes/(1024.0*1024.0), p->max_inserts, layers_str.c_str());
    }

    // seed the content from the current ranking immediately, then kick the
    // upload worker (tick() keeps the queue topped up between graphs)
    rebalance();
    {
        std::lock_guard<std::mutex> wlk(p->wmtx);
        fill_upload_queue(p);
    }
    p->wcv.notify_one();
}

// a new prompt has begun (llama_context detects the decode -> prefill
// transition): start a fresh 512-token profile window for this prompt. The
// existing layout keeps serving normally until the window elapses and
// maybe_activate() rebuilds it from the new prompt's routing mix.
void llama_moe_cache::on_prompt_begin() {
    if (!pimpl || pimpl->failed) {
        return;
    }
    auto * p = pimpl.get();
    std::lock_guard<std::mutex> lock(p->mtx);
    p->epoch_content = p->hot->content_tokens();
    p->epoch_rebuilt = false;
}

uint32_t llama_moe_cache::layout_generation() const {
    return pimpl ? pimpl->layout_gen : 0;
}

const llama_moe_cache_layer * llama_moe_cache::lookup(const ggml_tensor * gate) const {
    if (!pimpl || !pimpl->activated) {
        return nullptr;
    }
    for (const auto & ls : pimpl->layers) {
        if (ls.pub.gate_src == gate) {
            return &ls.pub;
        }
    }
    return nullptr;
}

// rebalance the residents against the current ranking. No locks held when
// entering (queries the hot cache, which locks its own mutex). Applies the
// evictions/queues the additions under the local mutex, then re-admits the
// evicted experts to the RAM tier.
void llama_moe_cache::rebalance() {
    auto * p = pimpl.get();

    // snapshot each cached layer's current counts and pick its top n_slots
    struct plan {
        size_t               li;
        std::vector<int32_t> desired; // top of the ranking, hottest first
    };
    std::vector<plan> plans;
    plans.reserve(p->layers.size());

    // index the cached layers by transformer layer id, then walk the shared
    // ranking ONCE: per-layer queries used to rescan the whole table once per
    // cached layer (quadratic in the number of cached layers)
    std::unordered_map<int, size_t> idx;
    idx.reserve(p->layers.size());
    for (size_t li = 0; li < p->layers.size(); ++li) {
        idx.emplace(p->layers[li].pub.il, li);
    }

    std::vector<std::tuple<int, int32_t, uint64_t>> all;
    p->hot->all_counts(all); // locks the hot cache's mutex

    std::vector<std::vector<std::pair<int32_t, uint64_t>>> raw(p->layers.size());
    for (const auto & [layer, expert, count] : all) {
        const auto it = idx.find(layer);
        if (it != idx.end()) {
            raw[it->second].emplace_back(expert, count);
        }
    }

    for (size_t li = 0; li < p->layers.size(); ++li) {
        auto & ls = p->layers[li];

        std::sort(raw[li].begin(), raw[li].end(), [](const auto & a, const auto & b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });

        plan pl;
        pl.li = li;
        pl.desired.reserve(ls.pub.n_slots);
        for (const auto & [id, cnt] : raw[li]) {
            if ((int32_t) pl.desired.size() >= ls.pub.n_slots) {
                break;
            }
            pl.desired.push_back(id);
        }
        plans.push_back(std::move(pl));
    }

    // apply: evict residents that left the top set, queue additions
    std::vector<std::pair<int, int32_t>> evicted;
    {
        std::lock_guard<std::mutex> lock(p->mtx);
        for (auto & pl : plans) {
            auto & ls = p->layers[pl.li];
            const int32_t n_slots = ls.pub.n_slots;

            auto in_desired = [&](int32_t e) {
                return std::find(pl.desired.begin(), pl.desired.end(), e) != pl.desired.end();
            };

            // evict published residents that are no longer in the top set
            bool changed = false;
            for (int32_t s = 0; s < n_slots; ++s) {
                const int32_t e = ls.slot_expert[s];
                if (e < 0) {
                    continue;
                }
                if (in_desired(e)) {
                    continue;
                }
                ls.slot_expert[s] = -1;
                ls.resident[e]    = 0;
                evicted.emplace_back(ls.pub.il, e);
                changed = true;
            }
            if (changed) {
                sync_tables(ls.pub, ls.slot_expert);
            }

            // queue the additions (in ranking order) for the ticks to drain
            ls.pending_q.clear();
            for (int32_t e : pl.desired) {
                if (ls.resident[e]) {
                    continue;
                }
                // skip ids whose upload is already in flight
                bool inflight = false;
                for (int32_t s = 0; s < n_slots; ++s) {
                    if (ls.slot_target[s] == e) {
                        inflight = true;
                        break;
                    }
                }
                if (!inflight) {
                    ls.pending_q.push_back(e);
                }
            }
        }
    }

    // the evicted experts are still recent-hot: hand them back to the RAM tier
    // so they stay mlock'd while VRAM no longer serves them
    for (const auto & [il, e] : evicted) {
        p->hot->promote_expert(il, e);
    }
}

void llama_moe_cache::tick(int64_t n_content_tokens) {
    if (!pimpl || !pimpl->activated) {
        return;
    }
    auto * p = pimpl.get();

    // free the tensor contexts of a layout a per-prompt rebuild replaced, now
    // that the decode graph referencing them was reset on that same ubatch
    p->free_retired_if_safe();

    // 1) publish completed uploads (sync point: no graph is executing)
    std::vector<std::pair<int, int32_t>> became_resident;
    {
        std::lock_guard<std::mutex> wlk(p->wmtx);
        std::lock_guard<std::mutex> lk(p->mtx);
        p->n_content += (uint64_t) std::max<int64_t>(0, n_content_tokens);
        p->n_ticks++;

        std::vector<char> dirty(p->layers.size(), 0);
        for (const auto & j : p->done) {
            auto & ls = p->layers[j.layer_idx];
            ls.slot_expert[j.slot] = j.expert;
            ls.slot_target[j.slot] = -1;
            ls.resident[j.expert]  = 1;
            dirty[j.layer_idx]     = 1;
            became_resident.emplace_back(ls.pub.il, j.expert);
        }
        // publish the new mappings in one table refresh per changed layer
        for (size_t li = 0; li < p->layers.size(); ++li) {
            if (dirty[li]) {
                sync_tables(p->layers[li].pub, p->layers[li].slot_expert);
            }
        }
        p->done.clear();
    }

    // residents are served from VRAM now: drop their RAM-tier mlock
    for (const auto & [il, e] : became_resident) {
        p->hot->vram_takeover(il, e);
    }

    // 2) periodic content rebalance against the shared ranking
    if (p->n_content - p->last_rebalance >= kRebalanceContentTokens) {
        p->last_rebalance = p->n_content;
        rebalance();
    }

    // 3) top the upload queue up again: the worker keeps its own queue full from
    //    pending_q between calls, so this only matters right after a rebalance
    //    filled pending_q (and it is what wakes the worker from idle)
    {
        std::lock_guard<std::mutex> wlk(p->wmtx);
        fill_upload_queue(p);
    }
    p->wcv.notify_one();
}

// periodic stats report, driven by llama_context at the shared
// --experts-stats-interval cadence. Data is gathered under the same locks as
// tick(), then logged outside them.
void llama_moe_cache::print_stats() {
    auto * p = pimpl.get();

    struct layer_report {
        int      il;
        int32_t  n_slots;
        int32_t  n_res;
        uint64_t n_hit;
        uint64_t n_miss;
    };

    std::vector<layer_report> report;
    uint64_t n_hit_total = 0;
    uint64_t n_miss_total = 0;
    size_t   n_slots_total = 0;
    size_t   n_res_total   = 0;
    size_t   n_new         = 0;  // residents that arrived since the previous report
    uint64_t n_ticks       = 0;

    {
        std::lock_guard<std::mutex> wlk(p->wmtx);
        std::lock_guard<std::mutex> lk(p->mtx);

        n_ticks = p->n_ticks;
        report.reserve(p->layers.size());
        for (auto & ls : p->layers) {
            layer_report r = { ls.pub.il, ls.pub.n_slots, 0, 0, 0 };
            for (int32_t e : ls.slot_expert) {
                if (e < 0) {
                    continue;
                }
                r.n_res++;
                const uint64_t k = ((uint64_t) (uint32_t) ls.pub.il << 32) | (uint32_t) e;
                if (p->prev_resident.find(k) == p->prev_resident.end()) {
                    n_new++;
                }
            }
            n_slots_total += (size_t) r.n_slots;
            n_res_total   += (size_t) r.n_res;
            report.push_back(r);
        }

        // refresh the churn snapshot for the next report
        p->prev_resident.clear();
        p->prev_resident.reserve(n_res_total);
        for (auto & ls : p->layers) {
            for (int32_t e : ls.slot_expert) {
                if (e >= 0) {
                    p->prev_resident.insert(((uint64_t) (uint32_t) ls.pub.il << 32) | (uint32_t) e);
                }
            }
        }
    }

    // decode-time VRAM-tier hit/miss telemetry: the hot cache's router
    // observation replaced this cache's own decode hook, so read the counters
    // back here (after the locks, keeping the hot.mu -> impl.mtx lock order)
    for (auto & r : report) {
        p->hot->vram_stats(r.il, r.n_hit, r.n_miss);
        n_hit_total  += r.n_hit;
        n_miss_total += r.n_miss;
    }

    const uint64_t t_total = n_hit_total + n_miss_total;
    LLAMA_LOG_INFO("[moe-cache] VRAM tier: resident=%zu/%zu slots (%zu layer(s), queue cap=%d)"
                   " | hit=%.1f%% (%" PRIu64 "/%" PRIu64 " routed)"
                   " | churn=%.1f%% (%zu/%zu changed since last report)"
                   " | ticks=%" PRIu64 " | per-layer: {",
                   n_res_total, n_slots_total, report.size(), p->max_inserts,
                   t_total ? 100.0 * n_hit_total / t_total : 0.0, n_hit_total, t_total,
                   n_res_total ? 100.0 * n_new / n_res_total : 0.0, n_new, n_res_total,
                   n_ticks);
    for (size_t li = 0; li < report.size(); ++li) {
        const auto & r   = report[li];
        const uint64_t t = r.n_hit + r.n_miss;
        LLAMA_LOG_CONT("L%d:slots=%d res=%d hit=%.1f%%%s", r.il, r.n_slots, r.n_res,
                t ? 100.0 * r.n_hit / t : 0.0, (li + 1 < report.size()) ? ", " : "");
    }
    LLAMA_LOG_CONT("}\n");
}
