#pragma once

// --pin-hot-experts N
//
// Keeps the N most frequently routed MoE experts per layer locked in RAM
// (mlock()/VirtualLock()) IN PLACE inside the model's own weight tensors, so the
// OS cannot evict them. Ranking is GLOBAL across all layers: total capacity is
// N x num_moe_layers slots and the set is maintained online from the actual
// router decisions. Single-token decode graphs are observed AFTER their compute:
// llama_context keeps each layer's "ffn_moe_topk-<il>" tensor alive
// (GGML_TENSOR_FLAG_OUTPUT) and llama_hot_expert_cache::observe_decode reads the
// routed ids once per decode ubatch - no mid-graph eval callback, so the decode
// graph is never chunked at every MoE layer. Only the multi-token prefetch still
// observes mid-graph through the ggml_backend_sched eval callback. Only experts
// that live in host (CPU) memory can be pinned; experts offloaded to a device
// buffer are skipped entirely.
//
// The ranking is fed ONLY by single-token decode ubatches: the tiers exist to
// accelerate generation, and the decode routing mix is what predicts which
// experts the coming decode steps keep re-routing. Multi-token (batch/prefill)
// ubatches never update the counts nor the pinned set - their wide, one-shot
// expert sweep would churn both tiers and leave generation with a stale hot set
// that then has to be re-learned from scratch; the prefetch below is what keeps
// prefill reads fast instead. The decay clock ticks on the same decode ubatches,
// so a prefill neither ages nor decays the ranking.
//
// This is also the ranking engine behind the VRAM MoE tier (llama_moe_cache)
// and the standalone prefetch (--hot-experts-prefetch): the same router
// observation maintains the decayed global counts it sizes itself from. The
// engine therefore also runs with n_pin_experts == 0 (e.g. only
// --moe-expert-cache* or --hot-experts-prefetch given): the counts feed the
// VRAM tier, nothing is mlock'd. When the prefetch is the ONLY feature
// requested the counts have no consumer and are dropped: only multi-token
// (batch/prefill) ubatches are observed, purely to read-ahead the routed rows.
//
// Prefetching (--hot-experts-prefetch / --no-hot-experts-prefetch) is an
// independent knob, gated on prefetch_enabled rather than on pinning. It only
// engages on multi-token (batch/prefill) ubatches: a wide ubatch routes a
// mostly-unpinned expert set per layer that would otherwise be demand-paged one
// fault at a time, so each observation of a topk tensor asynchronously
// prefetches (PrefetchVirtualMemory on Windows, posix_madvise WILLNEED on
// POSIX) the rows the layer just routed that are not pinned, right before that
// layer's FFN reads them. VRAM-served experts are prefetched like any other:
// multi-token graphs never use the VRAM tier (its chain is single-token only),
// so a prefill reads their host rows too. Single-token decode is
// deliberately left alone: it re-reads the same few experts every token, which
// are pinned and/or VRAM-resident after warm-up, so per-token prefetch is pure
// syscall/page-cache churn (it measurably regressed decode). Bulk-prefetching
// everything at ubatch start was also tried and regressed both prefill and
// generation (the burst serializes ahead of the demand reads); issuing the
// prefetch per layer at its topk keeps the reads pipelined inside the ubatch.
//
// Optional knobs:
//  - aging (--pin-hot-experts-decay-tokens N): every N single-token decode
//    tokens all usage counts are halved, so the pin set tracks the RECENT
//    routing mix instead of lifetime leaders (a long session otherwise lets
//    first-past-the-post experts occupy slots after they drifted cold). Prefill
//    tokens never advance the clock: they neither feed nor age the ranking.
//
// Pinning bookkeeping runs on the decode thread, but the mlock()/VirtualLock()
// syscalls themselves - the only step that can fault a long-cold expert's pages
// in - run on a dedicated worker thread, so a pin takeover never stalls the
// graph callback. Evictions stay inline: munlock is cheap (no page-in). Syscalls
// slower than lock_slow_us are counted as "slow" in the stats output.

#include "ggml-backend.h"
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
    // decay_interval:    halve all usage counts every N decode tokens (0 = disabled, lifetime
    //                    counts); prefill tokens neither count nor age the ranking
    // prefetch_enabled:  read-ahead (madvise WILLNEED / PrefetchVirtualMemory) the rows of the
    //                    experts a layer just routed that are not pinned, so their next read does
    //                    not page-fault (--hot-experts-prefetch). Engages on multi-token
    //                    (batch/prefill) ubatches only; VRAM-served experts are still prefetched
    //                    (multi-token graphs never use the VRAM tier and read their host rows)
    // track_rank:        true when pinning or the VRAM MoE tier consumes the usage ranking; false in
    //                    prefetch-only runs, where the count bookkeeping is skipped entirely. The
    //                    ranking is fed by single-token decode ubatches only: batch/prefill ubatches
    //                    are observed for the prefetch alone and never move the counts or the pins
    llama_hot_expert_cache(const llama_model & model,
                           int32_t             n_pin_experts,
                           uint64_t            budget_bytes,
                           uint64_t            decay_interval,
                           bool                prefetch_enabled,
                           bool                track_rank);
    ~llama_hot_expert_cache();

    llama_hot_expert_cache(const llama_hot_expert_cache &)             = delete;
    llama_hot_expert_cache & operator=(const llama_hot_expert_cache &) = delete;

    // ggml_backend_sched_eval_callback-compatible entry point. Only engaged for
    // multi-token (batch/prefill) ubatches with --hot-experts-prefetch: the
    // prefetch must read the routed rows while the ubatch still computes. Decode
    // ubatches are observed post-compute via observe_decode() instead, so no
    // eval callback is installed for them (it would chunk the graph at every MoE
    // layer and probe every node for nothing). Pass `this` as user_data when
    // installing.
    static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

    // Feed the ranking from a single-token decode graph that just finished
    // computing. `topk` maps layer id -> its ffn_moe_topk tensor (null for layers
    // with no registered MoE topk). llama_context registers the tensors at graph
    // build time and keeps them alive (GGML_TENSOR_FLAG_OUTPUT), so their routed
    // ids are read here, after the compute, instead of mid-graph through the eval
    // callback - which chunked and synchronized the decode graph at every MoE
    // layer. The readback is staged in two steps so the whole ubatch costs one
    // device sync instead of one per layer:
    //   observe_decode_begin()  issue one async D2H copy per device top-k tensor
    //                            (queued on its backend stream right behind the
    //                            decode graph); host tensors are copied directly.
    //                            llama_context then calls ggml_backend_sched_synchronize()
    //   observe_decode_finish()  rank the staged ids (one lock for the whole
    //                            ubatch, flat per-layer counters/bitmaps)
    // Runs once per decode ubatch, before the VRAM tier's tick.
    void observe_decode_begin(const std::vector<ggml_tensor *> & topk, ggml_backend_sched_t sched);
    void observe_decode_finish();

    // true if `expert_id` in layer `il` is currently mlock'd in place
    bool is_pinned(int il, int32_t expert_id) const;

    // ----- GPU-tier (llama_moe_cache) integration ---------------------------
    // One global ranked set feeds both tiers: the RAM tier mlock's the top of it
    // in place, the VRAM tier copies the very top into VRAM. These methods hand
    // the shared decayed ranking to the VRAM tier and let it report residents so
    // the RAM tier does not waste slots double-covering them.

    // (layer, expert_id, count) of every routed expert with count > 0, unsorted:
    // one lock and one pass over the whole ranking (the VRAM tier rebalances
    // all of its cached layers from one snapshot instead of asking per layer,
    // which used to rescan the full table once per cached layer)
    void all_counts(std::vector<std::tuple<int, int32_t, uint64_t>> & out) const;

    // hand a global BYTE budget to the globally hottest (layer, expert) pairs:
    // out[il] = slots for layer il (0 = none). bytes_per_layer[il] = cost of one
    // expert of layer il; fixed_bytes[il] = one-time per-layer cost (the dummy
    // slot plus the device tables), charged when the layer's first slot is
    // granted, so the total device memory of the resulting layout (slots +
    // dummy + tables) is bounded by budget_bytes. An expert is kept only while
    // its whole cost fits, so hot layers end up with many slots and cold layers
    // with none. Returns the number of slots assigned.
    int32_t assign_global_capacity(uint64_t budget_bytes,
            const std::vector<size_t> & bytes_per_layer,
            const std::vector<size_t> & fixed_bytes, std::vector<int32_t> & out) const;

    // decode tokens observed (single-token ubatches only; prefill ubatches do
    // not feed the ranking and do not advance its clock)
    uint64_t content_tokens() const;

    // The experts currently served from VRAM are reported through this query so
    // the RAM tier skips them (and prefetches nothing for them). The callback
    // fills `flags` with one 0/1 byte per expert id of layer `il` (empty when
    // the layer has no device cache) and runs while this cache's mutex is held,
    // once per observed layer per ubatch. Call with null to clear.
    using vram_query_fn = void (*)(void * ud, int il, std::vector<uint8_t> & flags);
    void set_vram_query(vram_query_fn fn, void * ud);

    // decode-time VRAM-tier hit/miss counters of one layer: a routed expert is a
    // hit when its device copy served it (the host read was skipped), a miss when
    // it was routed to the host experts. Kept by the shared observation while the
    // VRAM tier is active; read by its stats report (both are 0 while inactive).
    void vram_stats(int il, uint64_t & n_hit, uint64_t & n_miss) const;

    // an expert just became VRAM-resident: drop its RAM mlock (the VRAM copy
    // serves it; the mlock would only waste a RAM slot for a deeper expert)
    void vram_takeover(int il, int32_t expert_id);

    // an expert was evicted from VRAM: re-admit it to the RAM tier right away
    // (it is still recent-hot, it must not fall out of mlock protection)
    void promote_expert(int il, int32_t expert_id);

    // Called by llama_context right before every graph compute (one call per
    // ubatch). Advances the ubatch counter and optionally decays the usage
    // counts once every `decay_interval` decode tokens (single-token ubatches
    // only: prefill ubatches neither count nor age the ranking).
    void on_ubatch_begin(int64_t n_tokens);

    // Prints the periodic stats report (pinned vs capacity, realized RAM-tier hit
    // rate, list churn since the previous report, locked bytes, per-layer pinned
    // breakdown) via LLAMA_LOG_INFO (verbosity 4). Called by llama_context at the
    // shared --experts-stats-interval cadence and once by the destructor.
    void print_stats();

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

        // number of experts (ne[2] of the layer's expert tensors); the flat
        // per-expert tables below are sized to it in resolve_tensors()
        uint32_t n_experts = 0;
        // per-expert usage count, indexed by expert id (hot path of the ranking)
        std::vector<uint64_t> counts;
        // per-expert pin bookkeeping mirror, indexed by expert id: bit 0 =
        // mlock'd resident (pinned map), bit 1 = pin job queued/in flight
        // (pin_inflight set). Mirrors are updated at the same funnel points as
        // their maps (enqueue_pin, pin_worker_main, unpin_expert) so the hot
        // path never touches the hash containers
        std::vector<uint8_t> pin_state;

        // decode-time VRAM-tier hit/miss of this layer (see vram_stats()); updated
        // by observe() only while the VRAM tier serves this layer
        uint64_t n_vram_hit  = 0;
        uint64_t n_vram_miss = 0;
    };

    enum : uint8_t { PIN_RESIDENT = 1, PIN_INFLIGHT = 2 };

    // -- tuning constants ----------------------------------------------------
    // an mlock syscall slower than this (us) is counted as a stall in the stats;
    // the syscalls run on the pin worker thread, so a large n_lock_slow count no
    // longer stalls decode directly but still competes for the disk and CPU
    static constexpr int64_t lock_slow_us = 2000;
    // pin jobs queued ahead of the worker (bounds the in-flight budget reserve
    // and the memory of the queue itself); when full, newcomers are skipped and
    // retried on a later observation instead of blocking the decode thread
    static constexpr size_t pin_queue_max = 1024;
    // byte bound on the same queue: caps how much cold (not yet locked) work can
    // sit ahead of a newer, hotter expert, so a large (or missing) budget cannot
    // be consumed by reservations for a backlog of colder pins
    static constexpr uint64_t pin_queue_max_bytes = 1ULL << 30; // 1 GiB
    // hysteresis for full-set takeovers (see try_promote), so single-token count
    // noise at the cold edge cannot keep swapping two near-equal-heat experts:
    // a routed expert must lead the coldest pinned one by takeover_min_lead
    // counts, and an expert that was itself just evicted stays out until
    // evict_grace_tokens decode tokens have passed
    static constexpr uint64_t takeover_min_lead  = 4;
    static constexpr uint64_t evict_grace_tokens = 256;

    // -- observation ---------------------------------------------------------
    // ask phase: would the layer's topk data be useful this ubatch?
    bool wants_observe(int il);
    // !ask phase: consume the topk values of layer il
    void observe(int il, const struct ggml_tensor * t);

    void resolve_tensors(int il, layer_state & ls);

    // pin/evict bookkeeping for one expert at its CURRENT count; shared by the
    // decode ranking loop (observe_decode_finish, after a fresh count++) and
    // promote_expert (VRAM eviction). vram_resident is the result of this
    // layer's residency snapshot, so the hot path never re-queries the VRAM
    // tier per expert. apply_hysteresis is true on the routing path: full-set
    // takeovers then need takeover_min_lead and the challenger cannot return
    // within evict_grace_tokens of its own eviction, so count noise cannot swap
    // two near-equal experts back and forth. The VRAM eviction handoff
    // (promote_expert) passes false: that decision was already made on a full
    // set recompute and must stay prompt
    void try_promote(int il, layer_state & ls, int32_t expert_id, uint64_t count, bool vram_resident,
                     bool apply_hysteresis);

    // is expert_id currently served by the VRAM tier? (caller holds mu)
    bool is_vram_resident(int il, int32_t expert_id) const;

    // -- pinning -------------------------------------------------------------
    void unpin_expert(int il, layer_state & ls, int32_t expert_id);

    // (mu held) flat per-expert state of a resolved layer. `ls` must be resolved
    // and `expert_id` in [0, n_experts) - all callers guarantee this. These are
    // the only read/write points of the flattened counters/bitmaps, so the rest
    // of the code never indexes the vectors by hand
    static uint64_t & count_at(layer_state & ls, int32_t expert_id) {
        return ls.counts[(size_t) expert_id];
    }
    static uint8_t & pin_state_at(layer_state & ls, int32_t expert_id) {
        return ls.pin_state[(size_t) expert_id];
    }

    // (mu held) usage count of (il, expert_id) for cold paths (rank rebuild,
    // VRAM handoff) that may not hold a resolved layer_state reference; 0 when
    // the layer was never resolved
    uint64_t count_of(int il, int32_t expert_id) const;

    // rebuild pinned_rank from the current counts (caller holds mu). Called at
    // decay boundaries and in the stats report: updating one ordered-set key per
    // routed selection was pure churn on the decode thread, so the keys are
    // refreshed lazily instead; eviction decisions only heal the cold end of
    // the set (see try_promote) rather than rebuilding it
    void rebuild_pinned_rank();

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
    const uint64_t decay_interval;   // 0 = lifetime counts (no aging)
    const bool     prefetch_enabled; // read-ahead routed-but-unpinned expert rows (--hot-experts-prefetch)
    const bool     track_rank;       // rank feeds pinning/VRAM tier; false = prefetch-only mode
    uint64_t       n_tokens_seen = 0;  // tokens since the last decay
    int64_t        n_tokens_cur  = 0;  // tokens of the ubatch being computed (ask-phase gate)
    // per-observation scratch, reused instead of per-call allocation: observation
    // runs once per layer per ubatch on the compute thread, so nothing here is
    // shared across threads (ids are read on the compute thread, the residency
    // flags and prefetch ranges under mu)
    std::vector<int32_t>                                 obs_scratch;      // routed expert ids of the ubatch (pageable; also the multi-token prefetch scratch)
    // decode readback staging: how each layer's ids are packed into the staging
    // buffer by observe_decode_begin() (obs_off[il] = element offset, -1 when the
    // layer is not staged; obs_cnt[il] = staged id count)
    std::vector<int>                                     obs_off;
    std::vector<int>                                     obs_cnt;
    // where observe_decode_finish() reads the staged ids from: obs_stage (pinned
    // host memory, preferred) or obs_scratch (pageable fallback)
    const int32_t *                                      obs_ids = nullptr;
    std::vector<uint8_t>                                 vram_flags;       // per-layer VRAM residency snapshot
    std::vector<std::pair<const void *, size_t>>         prefetch_ranges;  // rows to read ahead

    // pinned host staging buffer for the decode readback. Async D2H copies into
    // ordinary pageable memory block until each copy completes (the CUDA runtime
    // stages pageable transfers on the host side), serializing the whole
    // readback on the decode thread behind the decode graph; pinned memory keeps
    // the copies truly async. Allocated once from the host buffer type of the
    // device that produces the top-k tensors; freed in the destructor
    ggml_backend_buffer_t obs_stage_buf = nullptr;
    void *                obs_stage     = nullptr;
    size_t                obs_stage_cap = 0;
    // ensure obs_stage holds at least `bytes` of pinned host memory usable for
    // async reads from `backend`; returns false when no pinned host buft exists
    // (caller falls back to the pageable obs_scratch)
    bool ensure_obs_stage(ggml_backend_t backend, size_t bytes);

    mutable std::mutex                   mu;
    std::unordered_map<int, layer_state> layers;

    // Global tracking across all layers. The per-expert usage counts and the
    // resident/in-flight pin mirrors live flattened per layer inside layer_state
    // (indexed by expert id) so the per-token routing path is pure array work;
    // the maps below hold the heavyweight state (mlock guards, ordered rank) and
    // are only touched on pins/evictions/decay/stat reports.

    // Global pinned set: (count, layer, expert_id) ordered ascending by count;
    // begin() is the coldest pinned expert. Keys are only kept exact when the
    // set is rebuilt (see rebuild_pinned_rank): between rebuilds the keys of
    // routed experts lag their true counts, which they never exceed
    std::set<std::tuple<uint64_t, int, int32_t>>                   pinned_rank;
    std::unordered_map<expert_key, pinned_expert, expert_key_hash> pinned;
    // pinned set at the previous stats report; diffed against the current one to
    // measure the list churn (expert replaced since the last report)
    std::unordered_set<expert_key, expert_key_hash> pinned_prev;
    // decode-token time of the last takeover eviction of each expert (see the
    // evict_grace_tokens constant): a freshly evicted expert is refused a slot
    // until its grace expires, so it cannot immediately re-take the slot it
    // just lost (the A/B ping-pong behind the steady churn in the reports)
    std::unordered_map<expert_key, uint64_t, expert_key_hash> evicted_at;

    uint64_t n_ubatches     = 0;   // graph computes (ubatch chunks) seen
    uint64_t n_eval_calls   = 0;   // topk tensors actually observed
    uint64_t n_distinct     = 0;   // (layer, expert) pairs ever routed (counts went 0->1)
    uint64_t n_bytes_locked = 0;   // sum of llama_mlock::size() for expert rows
    uint64_t n_lock_calls   = 0;   // mlock syscalls issued (pin worker thread)
    uint64_t n_lock_slow    = 0;   // ... that took longer than lock_slow_us
    uint64_t n_pin_failures = 0;   // async pins that locked nothing (queue/budget/OS)
    uint64_t n_prefetch_calls    = 0;
    uint64_t n_prefetch_bytes    = 0;
    uint64_t n_prefetch_failures = 0;
    uint64_t n_decays            = 0;
    uint64_t n_hysteresis_holds  = 0;  // takeovers refused by the margin/grace guards
    uint64_t n_content_tokens    = 0;  // decode tokens observed (see content_tokens())
    uint64_t n_route_hit         = 0;  // routed expert selections served by the pinned (RAM) tier
    uint64_t n_route_miss        = 0;  // routed selections not pinned (VRAM-served experts are skipped)

    // VRAM-tier residency query (see the public API docs); guarded by mu
    vram_query_fn vram_query = nullptr;
    void *        vram_ud    = nullptr;
};
