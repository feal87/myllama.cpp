#pragma once

// --pin-hot-experts N
//
// Keeps the N most frequently routed MoE experts per layer locked in RAM
// (mlock()/VirtualLock()) IN PLACE inside the model's own weight tensors, so the
// OS cannot evict them. Ranking is GLOBAL across all layers: total capacity is
// N x num_moe_layers slots and the set is maintained online from the actual
// router decisions (observed via the ggml_backend_sched eval callback on the
// "ffn_moe_topk-<il>" nodes). Only experts that live in host (CPU) memory can be
// pinned; experts offloaded to a device buffer are skipped entirely.
//
// This is also the ranking engine behind the VRAM MoE tier (llama_moe_cache)
// and the standalone prefetch (--hot-experts-prefetch): the same router
// observation maintains the decayed global counts it sizes itself from. The
// engine therefore also runs with n_pin_experts == 0 (e.g. only
// --moe-expert-cache* or --hot-experts-prefetch given): the counts are kept,
// nothing is mlock'd.
//
// Prefetching (--hot-experts-prefetch / --no-hot-experts-prefetch) is an
// independent knob, gated on prefetch_enabled rather than on pinning. Every
// observation of a topk tensor immediately prefetches (asynchronously:
// PrefetchVirtualMemory on Windows, posix_madvise WILLNEED on POSIX) the rows
// the layer just routed that are neither pinned nor served from VRAM. The calls
// are per-layer and issued while the rest of the ubatch still computes, so the
// reads pipeline in the background: on the next token the same experts are
// routed again with high probability (temporal locality), so their rows are
// already resident when the next FFN wants them. Prompt processing gets the
// same pipelining inside a single ubatch. This is the proven layout;
// bulk-prefetching everything at ubatch start was tried and regressed both
// prefill and generation (the burst serializes ahead of the demand reads).
//
// Optional knobs:
//  - aging (--pin-hot-experts-decay-tokens N): every N tokens of content all
//    usage counts are halved, so the pin set tracks the RECENT routing mix
//    instead of lifetime leaders (a long session otherwise lets first-past-the-
//    post experts occupy slots after they drifted cold).
//
// Pinning bookkeeping runs on the decode thread, but the mlock()/VirtualLock()
// syscalls themselves - the only step that can fault a long-cold expert's pages
// in - run on a dedicated worker thread, so a pin takeover never stalls the
// graph callback. Evictions stay inline: munlock is cheap (no page-in). Syscalls
// slower than lock_slow_us are counted as "slow" in the stats output.

#include "ggml.h"
#include "llama-mmap.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct llama_model;

class llama_hot_expert_cache {
  public:
    // n_pin_experts:     number of hottest experts to keep mlock'd per layer (N in --pin-hot-experts N);
    //                    total global capacity = N * num_moe_layers, ranked globally. 0 = ranking-only
    //                    mode (observe the routing for llama_moe_cache / prefetch, pin nothing)
    // budget_bytes:      hard cap on total bytes locked across ALL layers combined (0 = unlimited, NOT recommended)
    // stats_interval:    print_stats() is called automatically every `stats_interval` router
    //                    observations (0 = disabled, only the destructor prints a final summary)
    // decay_interval:    halve all usage counts every N tokens (0 = disabled, lifetime counts)
    //                    (tokens of content: prefill and generation both count)
    // prefetch_enabled:  read-ahead (madvise WILLNEED / PrefetchVirtualMemory) the rows of the
    //                    experts a layer just routed that are neither pinned nor served from VRAM,
    //                    so their next read does not page-fault (--hot-experts-prefetch)
    llama_hot_expert_cache(const llama_model & model,
                           int32_t             n_pin_experts,
                           uint64_t            budget_bytes,
                           uint64_t            stats_interval,
                           uint64_t            decay_interval,
                           bool                prefetch_enabled);
    ~llama_hot_expert_cache();

    llama_hot_expert_cache(const llama_hot_expert_cache &)             = delete;
    llama_hot_expert_cache & operator=(const llama_hot_expert_cache &) = delete;

    // ggml_backend_sched_eval_callback-compatible entry point.
    // Pass `this` as user_data when installing.
    static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

    // true if `expert_id` in layer `il` is currently mlock'd in place
    bool is_pinned(int il, int32_t expert_id) const;

    // ----- GPU-tier (llama_moe_cache) integration ---------------------------
    // One global ranked set feeds both tiers: the RAM tier mlock's the top of it
    // in place, the VRAM tier copies the very top into VRAM. These methods hand
    // the shared decayed ranking to the VRAM tier and let it report residents so
    // the RAM tier does not waste slots double-covering them.

    // all (expert_id, count) pairs of one layer with count > 0 (unsorted)
    void layer_counts(int il, std::vector<std::pair<int32_t, uint64_t>> & out) const;

    // hand a global BYTE budget to the globally hottest (layer, expert) pairs:
    // out[il] = slots for layer il (0 = none). bytes_per_layer[il] = cost of one
    // expert of layer il; an expert is kept only while its whole cost fits, so
    // hot layers end up with many slots and cold layers with none. Returns the
    // number of slots assigned.
    int32_t assign_global_capacity(uint64_t budget_bytes,
            const std::vector<size_t> & bytes_per_layer, std::vector<int32_t> & out) const;

    // lifetime content tokens observed (all ubatches, prefill + generation)
    uint64_t content_tokens() const;

    // experts currently served from VRAM are reported through this query so the
    // RAM tier skips them (and prefetches nothing for them). Call with null to
    // clear. The callback runs while the cache mutex is held.
    using vram_query_fn = bool (*)(void * ud, int il, int32_t expert_id);
    void set_vram_query(vram_query_fn fn, void * ud);

    // an expert just became VRAM-resident: drop its RAM mlock (the VRAM copy
    // serves it; the mlock would only waste a RAM slot for a deeper expert)
    void vram_takeover(int il, int32_t expert_id);

    // an expert was evicted from VRAM: re-admit it to the RAM tier right away
    // (it is still recent-hot, it must not fall out of mlock protection)
    void promote_expert(int il, int32_t expert_id);

    // Called by llama_context right before every graph compute (one call per
    // ubatch). Advances the ubatch counter and optionally decays the usage
    // counts once every `decay_interval` tokens of content.
    void on_ubatch_begin(int64_t n_tokens);

    // Prints a summary (bytes locked, per-layer breakdown, router observations,
    // prefetch, slow lock syscalls) directly to stderr with fprintf.
    // Deliberately bypasses LLAMA_LOG_* / the ggml log callback: at destruction
    // time (process teardown, or a caller that already tore down its own log
    // sink) those can silently swallow output, so this is a best-effort
    // guaranteed-visible dump.
    void print_stats() const;

  private:
    // Unique key identifying a specific expert in a specific layer
    struct expert_key {
        int     layer;
        int32_t expert_id;

        bool operator==(const expert_key & o) const { return layer == o.layer && expert_id == o.expert_id; }
    };

    struct expert_key_hash {
        std::size_t operator()(const expert_key & k) const {
            return std::hash<int>()(k.layer) ^ (std::hash<int32_t>()(k.expert_id) << 1);
        }
    };

    struct mlock_deleter {
        void operator()(llama_mlock * p) const {
            if (p) {
                p->unlock();
                delete p;
            }
        }
    };

    // holds the mlock guards keeping one expert's rows resident for each
    // relevant weight tensor; unlock() is called via custom deleter on destruction
    struct pinned_expert {
        std::unique_ptr<llama_mlock, mlock_deleter> gate_lock;
        std::unique_ptr<llama_mlock, mlock_deleter> up_lock;
        std::unique_ptr<llama_mlock, mlock_deleter> down_lock;
        std::unique_ptr<llama_mlock, mlock_deleter> gate_up_lock;
        size_t                                      nbytes_locked = 0;
    };

    struct layer_state {
        const ggml_tensor * t_gate    = nullptr;
        const ggml_tensor * t_up      = nullptr;
        const ggml_tensor * t_down    = nullptr;
        const ggml_tensor * t_gate_up = nullptr;

        bool tensors_are_host = false;  // false => experts live on a non-CPU backend, pinning is a no-op
        bool resolved_tensors = false;
    };

    // -- tuning constants ----------------------------------------------------
    // an mlock syscall slower than this (us) is counted as a stall in the stats;
    // the syscalls run on the pin worker thread, so a large n_lock_slow count no
    // longer stalls decode directly but still competes for the disk and CPU
    static constexpr int64_t lock_slow_us = 2000;
    // pin jobs queued ahead of the worker (bounds the in-flight budget reserve
    // and the memory of the queue itself); when full, newcomers are skipped and
    // retried on a later observation instead of blocking the decode thread
    static constexpr size_t pin_queue_max = 1024;

    // -- observation ---------------------------------------------------------
    // ask phase: would the layer's topk data be useful this ubatch?
    bool wants_observe(int il);
    // !ask phase: consume the topk values of layer il
    void observe(int il, const struct ggml_tensor * t);

    void resolve_tensors(int il, layer_state & ls);

    // called once per (layer, selected expert) observation; updates global counts and
    // pins/evicts on the fly against the global top-N set
    void observe_expert(int il, layer_state & ls, int32_t expert_id);

    // pin/evict bookkeeping for one expert at its CURRENT count; shared by
    // observe_expert (after a fresh count++) and promote_expert (VRAM eviction)
    void try_promote(int il, layer_state & ls, int32_t expert_id, uint64_t count);

    // is expert_id currently served by the VRAM tier? (caller holds mu)
    bool is_vram_resident(int il, int32_t expert_id) const;

    // -- pinning -------------------------------------------------------------
    void unpin_expert(int il, layer_state & ls, int32_t expert_id);

    // expected bytes grow_to() would lock for `expert_id` across all of the
    // layer's expert tensors; used to reserve budget BEFORE anything is evicted
    // (so a takeover never needs a rollback) and to preflight pin jobs
    size_t expert_row_bytes(const layer_state & ls, int32_t expert_id) const;

    // -- async pin worker ----------------------------------------------------
    // The decode thread only mutates bookkeeping and enqueues pin jobs. The
    // worker thread performs the actual mlock()/VirtualLock() syscalls - the
    // only step that can fault cold expert pages in (tens of ms) - so a pin
    // takeover never stalls the graph callback. Evictions stay on the decode
    // thread: munlock is cheap (no page-in).
    struct pin_job {
        int     il        = -1;
        int32_t expert_id = -1;

        const ggml_tensor * t_gate    = nullptr;
        const ggml_tensor * t_gate_up = nullptr;
        const ggml_tensor * t_up      = nullptr;
        const ggml_tensor * t_down    = nullptr;

        size_t expected_bytes = 0;  // reserved from the budget at enqueue time
    };

    void pin_worker_main();

    // queue `job` and reserve its expected bytes (caller holds mu). Returns
    // false when the queue is full or the reservation would exceed the budget;
    // the job is dropped and the expert stays unpinned until a later observation
    bool enqueue_pin(const pin_job & job);

    std::thread             pin_worker;
    std::deque<pin_job>     pin_queue;
    std::condition_variable pin_cv;
    // jobs queued but not yet locked by the worker (counted against the slot
    // capacity and the budget together with `pinned`/`n_bytes_locked`)
    std::unordered_set<expert_key, expert_key_hash> pin_inflight;
    uint64_t n_bytes_reserved = 0;  // expected bytes of in-flight pin jobs
    bool     pin_stop         = false;

    // -- prefetch ------------------------------------------------------------
    // appends the host-memory row ranges of `expert_id` within every expert tensor
    // of layer `ls`; returns the total number of bytes appended. Called right after
    // an observation so the routed-but-unpinned rows are read in the background
    // while the rest of the ubatch computes (see the class comment for why)
    size_t add_expert_ranges(const layer_state & ls,
                             int32_t             expert_id,
                             std::vector<std::pair<const void *, size_t>> & out);

    // halve all usage counts (and the rank keys of the pinned entries)
    void decay_counts();

    const llama_model & model;

    const int32_t  n_pin;            // N experts per layer
    int32_t        n_pin_total = 0;  // N * num_moe_layers (global cap)
    const uint64_t budget_bytes;     // 0 = unlimited
    const uint64_t stats_interval;   // 0 = disabled periodic printing
    const uint64_t decay_interval;   // 0 = lifetime counts (no aging)
    const bool     prefetch_enabled; // read-ahead routed-but-unpinned expert rows (--hot-experts-prefetch)
    uint64_t       n_tokens_seen = 0;  // tokens since the last decay

    mutable std::mutex                   mu;
    std::unordered_map<int, layer_state> layers;

    // Global tracking across all layers
    std::unordered_map<expert_key, uint64_t, expert_key_hash> counts;  // (layer, expert_id) -> times selected

    // Global pinned set: (count, layer, expert_id) ordered ascending by count
    // begin() is always the coldest pinned expert globally
    std::set<std::tuple<uint64_t, int, int32_t>>                   pinned_rank;
    std::unordered_map<expert_key, pinned_expert, expert_key_hash> pinned;

    uint64_t n_ubatches     = 0;   // graph computes (ubatch chunks) seen
    uint64_t n_eval_calls   = 0;   // topk tensors actually observed
    uint64_t n_bytes_locked = 0;   // sum of llama_mlock::size() for expert rows
    uint64_t n_lock_calls   = 0;   // mlock syscalls issued (pin worker thread)
    uint64_t n_lock_slow    = 0;   // ... that took longer than lock_slow_us
    uint64_t n_pin_failures = 0;   // async pins that locked nothing (queue/budget/OS)
    uint64_t n_prefetch_calls    = 0;
    uint64_t n_prefetch_bytes    = 0;
    uint64_t n_prefetch_failures = 0;
    uint64_t n_decays            = 0;
    uint64_t n_content_tokens    = 0;  // lifetime content tokens (all ubatches)

    // VRAM-tier residency query (see the public API docs); guarded by mu
    vram_query_fn vram_query = nullptr;
    void *        vram_ud    = nullptr;
};
