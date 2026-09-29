#include "llama-disk-stage.h"

#include "llama-expert-base.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// streaming stores for the L2 copy: the destination is not read again until a
// hit, so keeping its lines out of the CPU cache saves the RFO traffic
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <emmintrin.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// unbuffered reads require offset, length and destination aligned to the volume
// sector; 4096 covers both 512e and 4Kn media
static const size_t disk_stage_align = 4096;
// one request per chunk; a few dozen in flight saturate the drive
static const size_t disk_stage_chunk = 1u << 20;
// below this many bytes a concurrent copy does not pay for a worker thread
static const size_t disk_stage_copy_thread_min = 256 * 1024;
// below this ubatch size the routed experts are few enough that reading only
// them beats streaming the whole slab; the same cut makes the CUDA mul_mat_id
// fall back to the CPU, which reads only the routed rows anyway
static const int64_t disk_stage_sparse_tokens = 32;

static size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

// Parse a base-expert set: a required header, then one "<layer> <expert>" pair
// per line. Blank lines and '#' comments are skipped. Only the layer range is
// checked here; the constructor validates stageability and the expert range
static std::vector<std::vector<int32_t>> disk_stage_parse_expert_set(
        const std::string & path, int n_layer, const char * what) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(std::string("disk stage: cannot open ") + what + " '" + path + "'");
    }

    std::vector<std::vector<int32_t>> base((size_t) n_layer);
    std::string line;
    int  lineno      = 0;
    bool have_header = false;

    const auto fail = [&](const std::string & msg) {
        throw std::runtime_error("disk stage: " + path + ":" + std::to_string(lineno) + ": " + msg);
    };

    while (std::getline(in, line)) {
        lineno++;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const char * s = line.c_str();
        while (*s == ' ' || *s == '\t') {
            s++;
        }
        if (*s == '\0' || *s == '#') {
            continue;
        }

        if (!have_header) {
            if (std::strcmp(s, "llama-expert-base v1") != 0) {
                fail("expected header 'llama-expert-base v1'");
            }
            have_header = true;
            continue;
        }

        char *       end  = nullptr;
        const long   il   = std::strtol(s, &end, 10);
        if (end == s) {
            fail("expected '<layer> <expert>'");
        }
        char *     end2 = nullptr;
        const long id   = std::strtol(end, &end2, 10);
        if (end2 == end) {
            fail("expected an expert id after the layer");
        }
        while (*end2 == ' ' || *end2 == '\t') {
            end2++;
        }
        if (*end2 != '\0') {
            fail("trailing characters after the expert id");
        }
        if (il < 0 || il >= n_layer) {
            fail("layer " + std::to_string(il) + " out of range [0, " + std::to_string(n_layer) + ")");
        }
        if (id < 0 || id > INT32_MAX) {
            fail("expert " + std::to_string(id) + " out of range");
        }
        base[(size_t) il].push_back((int32_t) id);
    }

    if (!have_header) {
        throw std::runtime_error(std::string("disk stage: ") + what + " '" + path + "' has no header");
    }

    for (auto & v : base) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    return base;
}

#if defined(_WIN32)
struct disk_stage_file {
    HANDLE h     = INVALID_HANDLE_VALUE; // prefetch / prefill reads
    HANDLE h_dec = INVALID_HANDLE_VALUE; // split-hot decode reads, own completion port

    ~disk_stage_file() {
        if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
        }
        if (h_dec != INVALID_HANDLE_VALUE) {
            CloseHandle(h_dec);
        }
    }

    void open(const std::string & path) {
        // OVERLAPPED is required for real queue depth: blocking ReadFile calls on
        // a synchronous handle are serialized by the file object, which caps the
        // reads near QD1 no matter how many threads issue them
        h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                        nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("disk stage: cannot open " + path + " for unbuffered reads");
        }
    }

    // a handle belongs to exactly one completion port, so the decode split needs
    // its own handle on the same file to run its own queue
    void open_dec(const std::string & path) {
        h_dec = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h_dec == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("disk stage: cannot open " + path + " for the decode read queue");
        }
    }
};

struct disk_stage_job {
    HANDLE  h    = INVALID_HANDLE_VALUE;
    char *  dst  = nullptr;
    uint64_t off = 0;
    size_t  len  = 0;
};

// Issue up to `queue_depth` unbuffered reads concurrently and reissue as they
// complete, from one thread, reaping through an I/O completion port. This is the
// exact model diskspd uses to saturate the drive; a pool of waiting threads or a
// WaitForMultipleObjects pump does not hold the depth on this machine.
static void disk_stage_run_jobs(const std::vector<disk_stage_job> & jobs, int queue_depth, HANDLE iocp) {
    if (jobs.empty()) {
        return;
    }
    const size_t n_slots = std::min<size_t>(jobs.size(), (size_t) queue_depth);

    std::vector<OVERLAPPED> ovs(n_slots);
    std::vector<size_t>     job_of(n_slots, (size_t) -1);
    size_t                  next = 0;
    bool                    failed = false;

    // fill a slot with jobs until one is actually in flight
    const auto pump = [&](size_t s) -> bool {
        while (next < jobs.size()) {
            const size_t job_idx = next++;
            const disk_stage_job & j = jobs[job_idx];

            OVERLAPPED & ov = ovs[s];
            std::memset(&ov, 0, sizeof(ov));
            ov.Offset     = (DWORD) j.off;
            ov.OffsetHigh = (DWORD) (j.off >> 32);

            DWORD n = 0;
            if (ReadFile(j.h, j.dst, (DWORD) j.len, &n, &ov)) {
                if (n != j.len) {
                    return false;
                }
                continue; // completed inline, take the next job
            }
            if (GetLastError() != ERROR_IO_PENDING) {
                return false;
            }
            job_of[s] = job_idx;
            return true;
        }
        job_of[s] = (size_t) -1;
        return true;
    };

    if (!failed) {
        for (size_t s = 0; s < n_slots && !failed; ++s) {
            failed = !pump(s);
        }
    }

    for (;;) {
        bool any = false;
        for (size_t s = 0; s < n_slots; ++s) {
            if (job_of[s] != (size_t) -1) {
                any = true;
                break;
            }
        }
        if (!any) {
            break;
        }

        DWORD        n   = 0;
        ULONG_PTR    key = 0;
        OVERLAPPED * ov  = nullptr;
        if (!GetQueuedCompletionStatus(iocp, &n, &key, &ov, INFINITE) || ov == nullptr) {
            failed = true;
            break;
        }
        GGML_UNUSED(key);

        const size_t s = (size_t) (ov - ovs.data());
        if (s >= n_slots || job_of[s] == (size_t) -1 || n != jobs[job_of[s]].len) {
            failed = true;
            break;
        }
        job_of[s] = (size_t) -1;
        failed = !pump(s);
    }

    if (failed) {
        throw std::runtime_error("disk stage: unbuffered read failed");
    }
}
#endif

struct llama_disk_stage::impl {
    struct region {
        size_t              pool_off = 0; // aligned offset of the read destination within the pool
        size_t              read_len = 0; // aligned length of the read
        size_t              head     = 0; // bytes between the aligned start and the tensor data
        size_t              stride   = 0; // bytes per expert in the file
        int32_t             n_expert = 0; // experts in the tensor
        disk_stage_file *   file     = nullptr;
        uint64_t            file_off = 0; // aligned-down source offset
    };

    explicit impl(const llama_model & m) : model(m) {}

    const llama_model & model;
    bool active = false;

#if defined(_WIN32)
    HANDLE iocp     = nullptr; // one completion port, every staging file associated with it
    HANDLE iocp_dec = nullptr; // split-hot decode: a second port over a second handle per file
#endif

    ggml_context *       ctx  = nullptr;
    ggml_backend_buffer_t pool = nullptr;
    char *               base = nullptr; // aligned pool base

    std::vector<llama_disk_stage_layer> layers;      // indexed by layer id, null tensors = not staged
    std::vector<std::vector<region>>    layer_regions; // [layer][role], role in {gate, up, down}
    std::map<std::string, std::unique_ptr<disk_stage_file>> files;

    // persistent decode cache: one shared slot array per pool of layers that have
    // identical expert tensors. A resident expert takes a slot from the pool's
    // global free list, so a hot layer can hold more of them than a cold one; the
    // layer's table remaps the graph onto its slot. A pool is shared only when
    // every layer's tensor data is sector-aligned in the file (head == 0), which
    // the realign script guarantees; otherwise each layer is its own pool, which
    // reproduces the old static per-layer layout exactly
    struct cache_pool {
        ggml_tensor * gate = nullptr;
        ggml_tensor * up   = nullptr;
        ggml_tensor * down = nullptr;
        char *        data[3] = { nullptr, nullptr, nullptr };
        size_t        slot_stride[3] = { 0, 0, 0 }; // padded bytes per slot
        int32_t       n_layers = 0;
        int32_t       res_base = 0; // first resident slot == n_trans
        int32_t       res_cap  = 0; // resident slots
        int32_t       sentinel = 0; // == res_base + res_cap: first spare slot
        std::vector<int32_t> free_slots;   // resident slots with no expert (LIFO)
    };
    struct cache_layer {
        int           pool           = -1;
        ggml_tensor * table    = nullptr;  // I32 [n_expert], expert id -> slot
        ggml_tensor * slot_skip = nullptr; // I32 [n_slots + 1], 1 at the sentinel
        ggml_tensor * slot_skip_hot  = nullptr; // split-hot pass 1: 1 on transient + sentinel
        ggml_tensor * slot_skip_cold = nullptr; // split-hot pass 2: 1 on resident + sentinel
        ggml_tensor * keep           = nullptr; // F32 [n_expert], 1 = keep, 0 = dropped this token
        std::vector<int32_t> resident_slot;   // expert id -> slot, -1 when not resident
        std::vector<uint8_t> resident_filled; // expert id -> its slot holds this expert's data
        std::vector<uint8_t> vram;            // expert id -> served by the VRAM cache (host chain skips it)
        std::vector<uint8_t> base;            // expert id -> permanent base resident (never evicted)
        llama_disk_stage_cache_layer pub;     // public view returned by cache_layer()
    };
    ggml_backend_buffer_t cache_buf  = nullptr;
    ggml_context *        cache_ctx  = nullptr;
    std::unique_ptr<llama_mlock> cache_lock; // decode cache held in RAM for the process lifetime
    std::vector<cache_pool>  pools;
    std::vector<cache_layer> cache;
    // base-expert set from --pin-experts-from-profile: [layer] -> expert ids,
    // parsed and validated by the constructor, immutable afterwards
    std::vector<std::vector<int32_t>> base_set;
    // --pin-experts-template: the base set merged with each declared mode set,
    // keyed by set name ("" = the base set alone). Merged at load so a switch
    // is a lookup; only base_target (the resident one) is read afterwards, and
    // the map itself never changes, so the pointer stays valid
    std::map<std::string, std::vector<std::vector<int32_t>>> base_targets;
    llama_expert_base_template base_template;
    bool have_base_template = false;
    const std::vector<std::vector<int32_t>> * base_target = nullptr;
    std::string     base_active;  // set name of the resident target
    // warm-expert set from --warm-experts-from-profile: candidates (base removed)
    // and the subset actually read into the cache
    std::vector<std::vector<int32_t>> warm_set;
    std::vector<std::vector<int32_t>> warm_loaded;
    std::unordered_map<const ggml_tensor *, int32_t> table_layer; // cache id table -> layer id
    std::mutex   cache_mu;       // guards resident_slot / free_slots and table writes
    std::mutex   io_mu;          // one reaper at a time: the IOCP is shared by all reads
    int32_t      n_trans = 0;    // transient slots per layer, 0 when no cache
    // per layer: the graph emitted a cold pass for it, so the split path applies.
    // Written at graph build, read by the decode fill on the compute thread
    mutable std::vector<uint8_t> split_cold;

    // second-level (L2) expert pool. During decode the two prefill staging
    // slabs are idle, so they hold the experts the resident cache missed. A miss
    // is read straight into the transient slot the graph executes from and the
    // worker stores that slot into the pool while the layer computes; a later
    // hit copies from the pool into the transient slot instead of reading the
    // disk. One pool per expert-bundle type on one staging slab (a slot stride
    // is fixed per tensor); spare slabs split a type's layers, so every idle
    // slab holds slots even for a model with a single layout.
    //
    // SLRU: a miss enters the probation segment, a hit promotes it to the
    // protected segment, and evictions come from the probation LRU, so a
    // one-shot miss cannot push out an expert that keeps being re-read.
    // Demotions (recently resident experts) enter protected directly.
    struct evict_pool {
        char *  data[3]   = { nullptr, nullptr, nullptr }; // role -> slab region
        size_t  stride[3] = { 0, 0, 0 };
        int32_t cap       = 0;
        int32_t prot_cap  = 0; // protected cap; probation holds the rest
        std::list<int64_t> prob; // probation, front = MRU
        std::list<int64_t> prot; // protected, front = MRU
        std::unordered_map<int64_t, int32_t>      slot_of;   // key -> slot
        std::vector<std::list<int64_t>::iterator> iter_of;   // slot -> position
        std::vector<int64_t>                      slot_key;  // slot -> key, -1 when empty
        std::vector<uint8_t>                      seg;       // slot -> 0 probation, 1 protected
        std::vector<int32_t>                      free_slots;
        uint64_t hits      = 0; // counted only once the RAM tier is warm
        uint64_t misses    = 0;
        uint64_t evictions = 0;
        // admission filter: a key enters the ghost unseen and is only admitted
        // to the pool on a second sight, so a one-shot miss neither evicts a
        // re-read nor pays a store. The queue is bounded; a stale entry is
        // skipped when popped because its key was removed or re-stamped
        std::deque<std::pair<uint64_t, int64_t>> ghost_q;     // stamp, key (FIFO)
        std::unordered_map<int64_t, uint64_t>    ghost_stamp; // key -> stamp
        uint64_t                                  ghost_next = 1;
        size_t                                    ghost_cap  = 0;
    };
    std::vector<evict_pool> evict_pools;
    std::vector<int>        evict_pool_id;    // layer -> pool, -1 when not pooled
    bool                    evict_populated = false; // pools hold entries a prefill would clobber
    uint64_t                n_l2_filtered   = 0;     // misses the ghost turned away

    uint64_t n_l2_hits        = 0;
    uint64_t n_l2_misses      = 0;
    uint64_t n_l2_cold        = 0; // lookups while the RAM tier was still filling, excluded from the hit rate
    uint64_t n_l2_evictions   = 0;
    uint64_t n_l2_demotions   = 0;
    uint64_t n_l2_hit_bytes   = 0;
    uint64_t n_l2_promo_bytes = 0; // resident fills served from the L2 instead of the disk
    uint64_t n_l2_promotions  = 0;
    uint64_t n_l2_all_hits    = 0;
    uint64_t n_l2_all_misses  = 0;
    bool     l2_warm          = false; // latched when every resident pool is full

    uint64_t n_decode_cache_hits            = 0;
    uint64_t n_decode_cache_misses          = 0;
    // every routed selection, the denominator the cache hit rate is reported
    // against: hits, misses, drops and substitutions all count
    uint64_t n_routed_routes                = 0;
    uint64_t n_decode_cache_fills           = 0;
    uint64_t n_decode_cache_resident_changes = 0;
    // cache-aware opportunistic dropping: routed experts skipped instead of read
    // from disk, and the disk bytes those reads would have cost. The score mass
    // is the router score sum over all routed selections and over the dropped
    // ones, so the report can state the share of the token's score mass removed
    uint64_t n_dropped_routes = 0;
    uint64_t n_dropped_bytes  = 0;
    uint64_t n_subst_routes   = 0;
    uint64_t n_subst_bytes    = 0;
    double   n_routed_mass    = 0.0;
    double   n_cold_mass      = 0.0;
    double   n_dropped_mass   = 0.0;
    double   n_subst_mass     = 0.0; // sum |c - e| of the substitutions
    // worst single-layer perturbation share (dropped score plus substitution
    // |c - e|) since the last report, so an outlier layer is visible even when
    // the interval average looks small
    double   drop_worst_mass  = 0.0;
    // worst whole-token perturbation share since the last report
    double   drop_worst_token_mass = 0.0;

    // fill_cache() scratch: the decode fill runs once per layer per token on the
    // compute thread, so reuse these instead of reallocating per call
    struct pool_copy {
        const char * src;
        char *       dst;
        size_t       len;
        bool         pre = false; // lands in a resident slot, so it must complete before the hot pass
    };

    // store a frozen bundle into the L2 pool. The destination is not read again
    // until a hit, so a streaming store keeps it out of the CPU cache and skips
    // the read-for-ownership; the transient source is read right after by the
    // matmul, so it is loaded normally
    void store_l2(const pool_copy & cp) const {
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
        if (cp.len >= 256) {
            char *       d = cp.dst;
            const char * s = cp.src;
            size_t       i = 0;
            for (; i + 64 <= cp.len; i += 64) {
                const __m128i a = _mm_loadu_si128((const __m128i *) (s + i));
                const __m128i b = _mm_loadu_si128((const __m128i *) (s + i + 16));
                const __m128i c = _mm_loadu_si128((const __m128i *) (s + i + 32));
                const __m128i e = _mm_loadu_si128((const __m128i *) (s + i + 48));
                _mm_stream_si128((__m128i *) (d + i),      a);
                _mm_stream_si128((__m128i *) (d + i + 16), b);
                _mm_stream_si128((__m128i *) (d + i + 32), c);
                _mm_stream_si128((__m128i *) (d + i + 48), e);
            }
            for (; i + 16 <= cp.len; i += 16) {
                _mm_stream_si128((__m128i *) (d + i), _mm_loadu_si128((const __m128i *) (s + i)));
            }
            _mm_sfence();
            if (i < cp.len) {
                std::memcpy(d + i, s + i, cp.len - i);
            }
            return;
        }
#endif
        std::memcpy(cp.dst, cp.src, cp.len);
    }

    std::vector<disk_stage_job>          fs_jobs;
    std::vector<int32_t>                 fs_newly_filled;
    // routed expert scores aligned with the ids, reused by the drop plan
    std::vector<float>                   fs_probs;
    // per-position drop flags of the current fill, reused to avoid a per-layer
    // allocation on the decode hot path
    std::vector<uint8_t>                 fs_drop;
    // cold routed positions of the current fill, ranked by score
    std::vector<int64_t>                 fs_cold;
    // substitution decision of the current fill, per routed position
    std::vector<uint8_t>                 fs_subst;
    std::vector<int32_t>                 fs_subst_slot;
    std::vector<float>                   fs_subst_scale;
    std::vector<int32_t>                 fs_subst_used;
    std::vector<pool_copy>               fs_copies;       // L2 hit -> transient, independent of the disk batch
    std::vector<pool_copy>               fs_copies_pre;   // L2 hit -> resident, before the disk batch
    std::vector<pool_copy>               fs_stores;       // transient -> L2, after the disk batch
    std::vector<pool_copy>               fs_res_fills;    // transient -> resident, after the disk batch
    std::vector<std::pair<int, int64_t>> fs_l2_consume;
    // L2 slots the worker's stores will fill, and residents they will fill: only
    // published once the bytes have landed
    struct l2_pending  { int pid; int64_t key; int32_t slot; };
    struct res_pending { int il; int32_t id; };
    std::vector<l2_pending>              fs_l2_pending;
    std::vector<res_pending>             fs_res_pending;

    // decode I/O worker: the decode fill reads the disk straight into the
    // transient slot the graph computes from, then stores that slot into the L2
    // pool (and a promoted resident slot) on the worker while the layer
    // computes. fill_cache()/fill_cache_wait() wait only for the read phase; the
    // next plan drains the stores. One batch in flight, since layers are
    // sequential
    bool split_hot_active = false;
    // cache-aware opportunistic dropping (--disk-stage-drop-fraction): drop the
    // lowest `drop_fraction` of the routed experts of a token when cold and below
    // `drop_below_rel` times the layer's highest score. `drop_probe` only measures
    float   drop_fraction  = 0.0f;
    float   drop_below_rel = 0.5f;
    // hard ceilings on the combined perturbation (dropped score plus
    // substitution |c - e|): per layer, as a fraction of the layer's routed
    // mass, and per token, as a fraction of the token's routed mass (0 = off)
    float   drop_max_mass = 0.0f;
    float   drop_max_mass_token = 0.0f;
    // substitution: replace a cold routed expert with a nearby resident one
    // instead of reading or dropping it. The spare must score inside
    // [substitute_rel, 1/substitute_rel] of the cold score. pool = 0 = off
    float   substitute_rel  = 0.0f;
    int32_t substitute_pool = 0;
    // per-token accumulator for the token ceiling: reset at the token boundary
    // in split_token_end() and fed by each layer's fill in fill_cache_plan
    double  tok_routed_mass  = 0.0;
    double  tok_dropped_mass = 0.0; // dropped score mass plus substitution |c - e|
    bool    drop_probe     = false;
    // dropping and the probe wait until the resident cache is this full, so the
    // warm-up cold experts (not cached yet, not weak) are not dropped
    static constexpr int drop_min_fill_percent = 90;
    bool drop_warm = false;
    // cleared once the weights get_rows can no longer be intercepted: the
    // callback then falls back to the plain table-triggered fill
    bool drop_active = false;
    // one-shot: the reason the weights intercept was given up, for the log
    bool drop_disable_warned = false;
    // the warm-up check scans every layer, so it runs at most this often
    static constexpr int64_t drop_fill_check_us = 100000;
    int64_t drop_fill_check_at = 0;

    bool substitute_enabled() const {
        return substitute_pool > 0 && substitute_rel > 0.0f && substitute_rel < 1.0f;
    }
    bool drop_enabled() const {
        return drop_fraction > 0.0f || drop_probe || substitute_enabled();
    }

    // share of the resident decode slots that hold an expert's bytes
    double resident_fill() const {
        size_t cap    = 0;
        size_t filled = 0;
        for (const auto & pool : pools) {
            cap += (size_t) pool.res_cap;
        }
        for (const auto & c : cache) {
            for (size_t id = 0; id < c.resident_filled.size(); ++id) {
                if (c.resident_slot[id] >= 0 && c.resident_filled[id] != 0) {
                    filled++;
                }
            }
        }
        return cap ? 100.0 * (double) filled / (double) cap : 0.0;
    }

    // latches once the cache is nearly full; before that a cold expert is one
    // that was not read yet, not one that is weak
    bool resident_warm() {
        if (drop_warm) {
            return true;
        }
        const int64_t now = ggml_time_us();
        if (drop_fill_check_at != 0 && now - drop_fill_check_at < drop_fill_check_us) {
            return false;
        }
        drop_fill_check_at = now;
        if (resident_fill() < (double) drop_min_fill_percent) {
            return false;
        }
        drop_warm = true;
        LLAMA_LOG_INFO("%s: decode cache %.0f%% full, cache-aware dropping enabled\n",
                       __func__, resident_fill());
        return true;
    }

    // stop intercepting the weights get_rows: reset every keep table and let the
    // table get_rows drive the fill again. Called when the scores cannot be read
    void disable_drop() {
        drop_active = false;
        for (auto & c : cache) {
            if (c.keep == nullptr) {
                continue;
            }
            float * k = (float *) c.keep->data;
            for (int64_t e = 0; e < c.keep->ne[1]; ++e) {
                k[e] = 1.0f;
            }
        }
    }

    // drop probe (--disk-stage-drop-probe): the cold score relative to the
    // layer max, so the report can print the quantiles that pick
    // --disk-stage-drop-below-rel and the score mass a given floor would remove.
    // cold_n counts every cold routed selection, rel_n the ones with a valid
    // relative score, routed_n all of them
    uint64_t probe_routed   = 0;
    uint64_t probe_cold_n   = 0;
    uint64_t probe_rel_n    = 0;
    uint64_t probe_cold_max = 0;
    // cold experts per decode layer fill, for the p50/p90 of the cold count
    uint64_t probe_cold_hist[33] = {};
    uint64_t probe_calls          = 0;
    // cold score relative to the layer's max, linear [0,1]: the report reads the
    // cumulative count and mass at a few floors
    static constexpr int   probe_n_rel = 64;
    std::vector<uint64_t>  probe_rel      = std::vector<uint64_t>(probe_n_rel, 0);
    std::vector<double>    probe_rel_mass = std::vector<double>(probe_n_rel, 0.0);

    // substitution simulation (--disk-stage-drop-probe): for each cold routed
    // selection, whether a resident spare just outside the top-k is close enough
    // in score to take its place. P is the number of non-selected experts
    // considered by score, S the spare score floor relative to the cold expert.
    // Accounting only, nothing is substituted
    static constexpr int sim_n_p   = 3; // P = 2, 4, 8
    static constexpr int sim_n_s   = 5; // S = 0.50, 0.70, 0.80, 0.90, 0.95
    static constexpr int sim_ref_p = 1; // report reference P = 4
    static constexpr int sim_ref_s = 3; // report reference S = 0.90
    uint64_t sim_cold      = 0;
    uint64_t sim_fired[sim_n_p][sim_n_s] = {};
    uint64_t sim_top_fired = 0;
    uint64_t sim_ref_n     = 0;
    double   sim_ref_loss  = 0.0;
    uint64_t sim_ref_bytes = 0;
    // gap of the reference firings: (c - e), signed. Positive = the spare is
    // weaker than the cold expert it would replace. The absolute gap is taken
    // against the layer's score mass, so a tiny cold score cannot dominate
    uint64_t sim_ref_weaker   = 0;
    uint64_t sim_ref_stronger = 0;
    double   sim_ref_rel_sum  = 0.0; // signed (c - e) / layer_mass
    double   sim_ref_rel_abs  = 0.0; // |c - e| / layer_mass
    static constexpr double sim_rel_max  = 0.20;
    static constexpr int    sim_rel_bins = 100;
    uint64_t sim_ref_rel_hist[sim_rel_bins] = {};
    // full router distribution of the current decode fill, for the simulation
    std::vector<float>                    fs_all;
    int64_t                               fs_n_all = 0;
    std::vector<std::pair<float,int32_t>> fs_pool;

    static int probe_rel_bin(float r) {
        if (!(r > 0.0f)) {
            return 0;
        }
        int b = (int) (r * (float) probe_n_rel);
        if (b < 0) {
            b = 0;
        }
        if (b >= probe_n_rel) {
            b = probe_n_rel - 1;
        }
        return b;
    }
    uint64_t probe_rel_below(float a) const {
        const int b = probe_rel_bin(a);
        uint64_t acc = 0;
        for (int i = 0; i < b; ++i) {
            acc += probe_rel[(size_t) i];
        }
        return acc;
    }
    double probe_rel_mass_below(float a) const {
        const int b = probe_rel_bin(a);
        double acc = 0.0;
        for (int i = 0; i < b; ++i) {
            acc += probe_rel_mass[(size_t) i];
        }
        return acc;
    }
    // floor below which `frac` of the cold scores fall, as a bin center
    float probe_rel_quantile(float frac) const {
        if (probe_rel_n == 0) {
            return 0.0f;
        }
        const uint64_t target = (uint64_t) (frac * (double) probe_rel_n);
        uint64_t acc = 0;
        for (int i = 0; i < probe_n_rel; ++i) {
            acc += probe_rel[(size_t) i];
            if (acc >= target) {
                return ((float) i + 0.5f) / (float) probe_n_rel;
            }
        }
        return 1.0f;
    }
    std::thread             dec_io;
    std::mutex              dec_io_mu;
    std::condition_variable dec_io_cv;
    std::condition_variable dec_io_done_cv;
    std::condition_variable dec_io_reads_cv;
    bool                    dec_io_active     = false; // worker started
    bool                    dec_io_ready      = false; // a batch was handed over
    bool                    dec_io_running    = false; // worker is running it
    bool                    dec_io_reads_done = false; // the read phase is over
    bool                    dec_io_stop       = false;
    bool                    dec_io_error      = false;
    // the decode reads run on their own completion port, so a batch never
    // queues behind the prefetch reader on io_mu
    std::mutex dec_io_disk_mu;
    // one batch handed to the worker. The copies write transient slots the hot
    // pass skips and overlap the reads; the stores run after the reads, since
    // they read the bytes the reads just landed
    struct dec_batch {
        std::vector<disk_stage_job> jobs;
        std::vector<pool_copy>     copies;     // L2 hit -> transient slot, before the reads
        std::vector<pool_copy>     stores;     // transient slot -> L2 slot, after the reads
        std::vector<pool_copy>     res_fills;  // transient slot -> resident slot, after the reads
    };
    dec_batch dec_io_batch;
    // begin()/wait() handoff state, compute thread only
    int32_t dec_wait_il   = -1;
    size_t  dec_bytes     = 0; // disk bytes handed to the worker, for the stats
    // routed ids served from RAM / cold / skipped (VRAM), set by fill_cache_plan
    int64_t trace_n_res  = 0;
    int64_t trace_n_cold = 0;
    int64_t trace_n_vram = 0;

    // split-hot trace: absolute marks of one layer, reported as deltas between
    // the two calls so the read/hot intersection is visible
    struct split_phase {
        int32_t il      = -1;
        int64_t t_begin = 0; // fill_cache_begin entry, the plan starts
        int64_t t_plan0 = 0; // fill_cache_plan returned
        int64_t t_copy  = 0; // copies_pre memcpy over
        int64_t t_hot   = 0; // the batch is with the worker, the hot pass starts
        int64_t t_cold  = 0; // the cold split was reached, the read must be done
        int64_t t_join  = 0; // the read landed; the worker may still be storing
        int64_t rd0     = 0; // worker entered the read
        int64_t rd1     = 0; // worker left it, 0 when the batch had no read
        bool    started = false;
    };
    split_phase trace_split;
    struct split_acc {
        int32_t layers         = 0;
        int64_t plan_us        = 0; // t_plan0 - t_begin, the plan and its copies
        int64_t copy_us        = 0; // copies_pre memcpy
        int64_t hot_us         = 0; // hot pass compute (t_cold - t_hot)
        int64_t read_us        = 0; // worker read
        int64_t cold_us        = 0; // cold pass compute (t_end - t_join)
        int64_t ovl_us         = 0; // read time hidden behind the hot pass
        int64_t stall_us       = 0; // read time still outstanding when the cold pass needed it
        int64_t wait_us        = 0; // the decode thread blocked in fill_cache_wait
        int64_t span_us        = 0; // wall clock from begin() to the end of the cold pass
        int64_t bytes          = 0;
        int32_t n_read_limited = 0; // layers whose read was not covered by their hot pass

        split_acc operator-(const split_acc & o) const {
            split_acc r;
            r.layers     = layers     - o.layers;
            r.plan_us    = plan_us    - o.plan_us;
            r.copy_us    = copy_us    - o.copy_us;
            r.hot_us     = hot_us     - o.hot_us;
            r.read_us    = read_us    - o.read_us;
            r.cold_us    = cold_us    - o.cold_us;
            r.ovl_us     = ovl_us     - o.ovl_us;
            r.stall_us   = stall_us   - o.stall_us;
            r.wait_us    = wait_us    - o.wait_us;
            r.span_us    = span_us    - o.span_us;
            r.bytes      = bytes      - o.bytes;
            r.n_read_limited = n_read_limited - o.n_read_limited;
            return r;
        }

        split_acc & operator+=(const split_acc & o) {
            layers     += o.layers;
            plan_us    += o.plan_us;
            copy_us    += o.copy_us;
            hot_us     += o.hot_us;
            read_us    += o.read_us;
            cold_us    += o.cold_us;
            ovl_us     += o.ovl_us;
            stall_us   += o.stall_us;
            wait_us    += o.wait_us;
            span_us    += o.span_us;
            bytes      += o.bytes;
            n_read_limited += o.n_read_limited;
            return *this;
        }
    };
    split_acc trace_split_tok;
    // running totals of the same: trace_split_tok is reset every decode token and
    // only the trace prints it, while the report needs the interval
    split_acc split_tot;
    split_acc prev_split_tot;
    uint32_t  n_reports = 0;

    // previous stats report, for the per-interval deltas
    uint64_t prev_l2_hits        = 0;
    uint64_t prev_l2_all_hits    = 0;
    uint64_t prev_l2_misses      = 0;
    uint64_t prev_l2_cold        = 0;
    uint64_t prev_l2_evictions   = 0;
    uint64_t prev_l2_demotions   = 0;
    uint64_t prev_l2_hit_bytes   = 0;
    uint64_t prev_l2_promo_bytes = 0;
    uint64_t prev_l2_filtered    = 0;
    // previous report's decode-cache counters and decode token count, for the
    // interval hit rate and the per-token fill cost
    uint64_t prev_dec_cache_hits    = 0;
    uint64_t prev_dec_cache_misses  = 0;
    uint64_t prev_dec_routed_routes = 0;
    uint64_t prev_dec_cache_fills   = 0;
    uint64_t prev_dec_cache_changes = 0;
    uint64_t prev_dec_dropped       = 0;
    uint64_t prev_dec_substituted   = 0;
    uint64_t prev_dec_tokens        = 0;
    // previous report's funnel bases, for the per-interval tier rates
    uint64_t prev_base_l2           = 0;
    uint64_t prev_base_sub          = 0;
    uint64_t prev_base_drop         = 0;
    double   prev_dec_routed_mass   = 0.0;
    double   prev_dec_dropped_mass  = 0.0;

    // decode fill timing: the blocking read share of a decode step, so it can be
    // compared with the compute. Summed over every fill_cache() call
    uint64_t n_dec_fill_us    = 0;
    uint64_t n_dec_fill_calls = 0;
    uint64_t n_dec_fill_bytes = 0;
    uint64_t prev_dec_fill_us    = 0;
    uint64_t prev_dec_fill_calls = 0;
    uint64_t prev_dec_fill_bytes = 0;
    // trace state (compute thread only): previous fill end and layer, plus the
    // running totals of the decode token being traced
    int64_t  trace_prev_us      = 0;
    int32_t  trace_prev_il      = -1;
    int64_t  trace_tok_start_us = 0;
    uint64_t trace_tok_fill_us  = 0;
    uint64_t trace_tok_gap_us   = 0;
    uint64_t trace_tok_calls    = 0;

    int evict_pool_for(int il) const {
        return (il >= 0 && il < (int) evict_pool_id.size()) ? evict_pool_id[(size_t) il] : -1;
    }

    static void evict_mru(evict_pool & ep, int32_t slot) {
        std::list<int64_t> & l = ep.seg[(size_t) slot] != 0 ? ep.prot : ep.prob;
        if (ep.iter_of[(size_t) slot] != l.begin()) {
            l.splice(l.begin(), l, ep.iter_of[(size_t) slot]);
            ep.iter_of[(size_t) slot] = l.begin();
        }
    }

    static void evict_push(evict_pool & ep, int64_t key, int32_t slot, uint8_t seg) {
        std::list<int64_t> & l = seg != 0 ? ep.prot : ep.prob;
        l.push_front(key);
        ep.iter_of[(size_t) slot] = l.begin();
        ep.seg[(size_t) slot] = seg;
        ep.slot_key[(size_t) slot] = key;
        ep.slot_of[key] = slot;
    }

    // demote the protected LRU into probation so protected stays at its cap
    static void evict_trim(evict_pool & ep) {
        while ((int32_t) ep.prot.size() > ep.prot_cap) {
            const int64_t key = ep.prot.back();
            const int32_t slot = ep.slot_of.at(key);
            ep.prot.pop_back();
            ep.prob.push_front(key);
            ep.iter_of[(size_t) slot] = ep.prob.begin();
            ep.seg[(size_t) slot] = 0;
        }
    }

    static bool ghost_has(const evict_pool & ep, int64_t key) {
        return ep.ghost_stamp.find(key) != ep.ghost_stamp.end();
    }

    // remember a key that was not admitted; the oldest beyond the cap is dropped
    static void ghost_add(evict_pool & ep, int64_t key) {
        if (ep.ghost_cap == 0 || ghost_has(ep, key)) {
            return;
        }
        const uint64_t stamp = ep.ghost_next++;
        ep.ghost_stamp[key] = stamp;
        ep.ghost_q.emplace_back(stamp, key);
        while (ep.ghost_q.size() > ep.ghost_cap) {
            const auto oldest = ep.ghost_q.front();
            ep.ghost_q.pop_front();
            const auto it = ep.ghost_stamp.find(oldest.second);
            if (it != ep.ghost_stamp.end() && it->second == oldest.first) {
                ep.ghost_stamp.erase(it);
            }
        }
    }

    static void ghost_remove(evict_pool & ep, int64_t key) {
        ep.ghost_stamp.erase(key); // the queued entry is skipped when popped
    }

    // a free slot, or the probation LRU's slot; protected is the fallback when
    // probation is empty. -1 when every slot is reserved by a pending store
    int32_t evict_take_slot(evict_pool & ep) {
        if (!ep.free_slots.empty()) {
            const int32_t s = ep.free_slots.back();
            ep.free_slots.pop_back();
            return s;
        }
        std::list<int64_t> & l = !ep.prob.empty() ? ep.prob : ep.prot;
        if (l.empty()) {
            return -1; // every slot is reserved by a pending store
        }
        const int64_t key = l.back();
        const int32_t slot = ep.slot_of.at(key);
        l.pop_back();
        ep.slot_of.erase(key);
        ep.slot_key[(size_t) slot] = -1;
        n_l2_evictions++;
        ep.evictions++;
        ghost_add(ep, key); // a re-route of the victim admits it again
        return slot;
    }

    // a read-only presence test: unlike evict_touch a dropped expert must not
    // promote its entry in the SLRU
    bool evict_has(int pool, int64_t key) const {
        return evict_pools[(size_t) pool].slot_of.count(key) != 0;
    }

    // a hit: promote probation -> protected, or refresh protected
    int32_t evict_touch(int pool, int64_t key) {
        evict_pool & ep = evict_pools[(size_t) pool];
        const auto it = ep.slot_of.find(key);
        if (it == ep.slot_of.end()) {
            return -1;
        }
        const int32_t slot = it->second;
        if (ep.seg[(size_t) slot] == 0) {
            ep.prob.erase(ep.iter_of[(size_t) slot]);
            ep.seg[(size_t) slot] = 1;
            ep.prot.push_front(key);
            ep.iter_of[(size_t) slot] = ep.prot.begin();
            evict_trim(ep);
        } else {
            evict_mru(ep, slot);
        }
        return slot;
    }

    // a demotion (recently resident expert): insert into protected
    int32_t evict_alloc_prot(int pool, int64_t key) {
        evict_pool & ep = evict_pools[(size_t) pool];
        const int32_t slot = evict_take_slot(ep);
        if (slot < 0) {
            return -1;
        }
        evict_push(ep, key, slot, 1);
        evict_trim(ep);
        return slot;
    }

    // reserve a slot for an expert the worker will store into: the slot leaves
    // the free list but is not published, so no hit can read it and no eviction
    // can reuse it before the store lands. -1 when the pool has no slot to spare
    int32_t evict_reserve(int pool) {
        return evict_take_slot(evict_pools[(size_t) pool]);
    }

    // publish a reserved slot once its store has landed
    void evict_finalize(int pool, int64_t key, int32_t slot) {
        evict_pool & ep = evict_pools[(size_t) pool];
        evict_push(ep, key, slot, 0);
        evict_trim(ep);
    }

    int32_t evict_put(int pool, int64_t key) {
        const int32_t slot = evict_touch(pool, key);
        return slot >= 0 ? slot : evict_alloc_prot(pool, key);
    }

    void evict_remove(int pool, int64_t key) {
        evict_pool & ep = evict_pools[(size_t) pool];
        const auto it = ep.slot_of.find(key);
        if (it == ep.slot_of.end()) {
            return;
        }
        const int32_t slot = it->second;
        if (ep.seg[(size_t) slot] != 0) {
            ep.prot.erase(ep.iter_of[(size_t) slot]);
        } else {
            ep.prob.erase(ep.iter_of[(size_t) slot]);
        }
        ep.slot_of.erase(it);
        ep.slot_key[(size_t) slot] = -1;
        ep.free_slots.push_back(slot);
    }

    void evict_clear() {
        for (evict_pool & ep : evict_pools) {
            ep.prob.clear();
            ep.prot.clear();
            ep.slot_of.clear();
            ep.free_slots.resize((size_t) ep.cap);
            for (int32_t s = 0; s < ep.cap; ++s) {
                ep.free_slots[(size_t) s] = s;
            }
            std::fill(ep.slot_key.begin(), ep.slot_key.end(), (int64_t) -1);
            ep.ghost_q.clear();
            ep.ghost_stamp.clear();
            ep.ghost_next = 1;
        }
    }

    // double-buffered staging pipeline: one reader thread reads the next
    // stageable layer while the current layer computes. n_buf == 1 disables it
    // (a read-ahead would clobber the buffer in use)
    std::thread             reader;
    std::mutex              pipe_mu;
    std::condition_variable pipe_req_cv;
    std::condition_variable pipe_done_cv;
    int32_t  pipe_req    = -1;   // layer the main thread requested
    int32_t  pipe_issued = -1;   // layer the pipeline was last asked for
    int32_t  pipe_done   = -1;   // layer the reader finished
    bool     pipe_stop   = false;
    std::string pipe_error;      // non-empty once a read has failed
    int      n_buf = 2;
    std::vector<int32_t> next_stage; // layer -> next stageable layer, -1 if none
    std::vector<int8_t>  layer_buf;  // layer -> staging buffer index

    ~impl() {
        if (pool) {
            ggml_backend_buffer_free(pool);
        }
        if (cache_buf) {
            ggml_backend_buffer_free(cache_buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        if (cache_ctx) {
            ggml_free(cache_ctx);
        }
#if defined(_WIN32)
        if (iocp != nullptr) {
            CloseHandle(iocp);
        }
#endif
    }

    // Hold the decode cache's RAM for real. The cache replaces a disk read, so a
    // page that can be paged out is worse than no cache at all: evicting it costs
    // a pagefile write, and the next "hit" is a pagefile read off the same disk.
    // VirtualLock makes the range unpageable, and the touch makes the pages private
    // (a locked page that was never written can still be the shared zero page).
    bool lock_cache(void * base, size_t bytes) {
        if (!llama_mlock::SUPPORTED) {
            LLAMA_LOG_ERROR("%s: the decode cache cannot be held in RAM on this platform\n", __func__);
            return false;
        }

        // the minimum working set is the quota VirtualLock is measured against,
        // so raise it first; the extra MiB cover the lock bookkeeping itself
        if (!llama_mlock::reserve_working_set(bytes + 64 * 1024 * 1024)) {
            LLAMA_LOG_ERROR("%s: could not reserve a working set for the %.2f GiB decode cache; "
                            "lower --pin-hot-experts-budget-mib\n",
                            __func__, bytes / (1024.0 * 1024.0 * 1024.0));
            return false;
        }

        cache_lock = std::make_unique<llama_mlock>();
        cache_lock->init(base);
        cache_lock->grow_to(bytes);
        if (cache_lock->size() < bytes) {
            LLAMA_LOG_ERROR("%s: only %.2f of the %.2f GiB decode cache could be held in RAM; "
                            "lower --pin-hot-experts-budget-mib\n",
                            __func__, cache_lock->size() / (1024.0 * 1024.0 * 1024.0),
                            bytes / (1024.0 * 1024.0 * 1024.0));
            cache_lock.reset();
            return false;
        }

        const int64_t t0 = ggml_time_us();
        char * cp = (char *) base;
        for (size_t off = 0; off < bytes; off += disk_stage_align) {
            cp[off] = 0;
        }
        LLAMA_LOG_INFO("%s: decode cache held in RAM: %.2f GiB locked, touched in %.0f ms\n",
                       __func__, bytes / (1024.0 * 1024.0 * 1024.0), (ggml_time_us() - t0) / 1000.0);

        return true;
    }

    disk_stage_file * file_for(const std::string & path) {
        auto it = files.find(path);
        if (it != files.end()) {
            return it->second.get();
        }
        auto f = std::make_unique<disk_stage_file>();
        f->open(path);
#if defined(_WIN32)
        // associate here, once: a handle can only belong to one completion port
        if (CreateIoCompletionPort(f->h, iocp, 0, 0) == nullptr) {
            throw std::runtime_error("disk stage: failed to associate " + path + " with the completion port");
        }
        if (iocp_dec != nullptr) {
            f->open_dec(path);
            if (CreateIoCompletionPort(f->h_dec, iocp_dec, 0, 0) == nullptr) {
                throw std::runtime_error("disk stage: failed to associate " + path + " with the decode completion port");
            }
        }
#endif
        disk_stage_file * raw = f.get();
        files.emplace(path, std::move(f));
        return raw;
    }
};

static bool disk_stage_trace() {
    static const bool on = [] {
        const char * v = std::getenv("LLAMA_DISK_STAGE_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

bool llama_disk_stage::supported() {
#if defined(_WIN32)
    return true;
#else
    return false;
#endif
}

bool llama_disk_stage::is_active() const {
    return pimpl->active;
}

void llama_disk_stage::stats_snapshot(llama_expert_stats & out) const {
    impl & p = *pimpl;

    out.dio_active = p.active;
    out.decode_cache.locked_bytes = p.cache_lock ? p.cache_lock->size() : 0;

    std::lock_guard<std::mutex> lock(p.cache_mu);
    out.decode_cache.route_hits       = p.n_decode_cache_hits;
    out.decode_cache.route_misses     = p.n_decode_cache_misses;
    out.decode_cache.route_routed     = p.n_routed_routes;
    out.decode_cache.fills            = p.n_decode_cache_fills;
    out.decode_cache.resident_changes = p.n_decode_cache_resident_changes;
    out.decode_cache.dropped_routes   = p.n_dropped_routes;
    out.decode_cache.dropped_bytes    = p.n_dropped_bytes;
    out.decode_cache.substituted_routes = p.n_subst_routes;
    out.decode_cache.substituted_bytes  = p.n_subst_bytes;

    out.disk_l2.enabled = !p.evict_pools.empty();
    out.disk_l2.warm    = p.l2_warm;
    for (const auto & pool : p.evict_pools) {
        out.disk_l2.entries  += pool.slot_of.size();
        out.disk_l2.capacity += (uint64_t) pool.cap;
    }
    out.disk_l2.hits               = p.n_l2_all_hits;
    out.disk_l2.misses             = p.n_l2_all_misses;
    out.disk_l2.cold_lookups       = p.n_l2_cold;
    out.disk_l2.hit_bytes          = p.n_l2_hit_bytes;
    out.disk_l2.promotions         = p.n_l2_promotions;
    out.disk_l2.promotion_bytes    = p.n_l2_promo_bytes;
    out.disk_l2.evictions          = p.n_l2_evictions;
    out.disk_l2.demotions          = p.n_l2_demotions;
    out.disk_l2.decode_fill_calls  = p.n_dec_fill_calls;
    out.disk_l2.decode_fill_bytes  = p.n_dec_fill_bytes;
    out.disk_l2.decode_fill_microseconds = p.n_dec_fill_us;
}

bool llama_disk_stage::sparse_ubatch(int64_t n_tokens) const {
    return n_tokens > 1 && n_tokens < disk_stage_sparse_tokens;
}

bool llama_disk_stage::internal_decode_fill() const {
    return !pimpl->table_layer.empty();
}

void llama_disk_stage::node_prepare_callback(struct ggml_tensor * node, void * user_data) {
#if defined(_WIN32)
    auto * self = static_cast<llama_disk_stage *>(user_data);
    impl & p = *self->pimpl;
    if (!p.active || node == nullptr) {
        return;
    }

    // split-hot: the cold pass runs in its own split, after the hot one. Its
    // first node waits for the disk batch that the get_rows below handed to the
    // worker, so the hot compute overlapped the read. Every node of the cold
    // split carries the prefix, so the first node of the NEXT split is the
    // marker that the cold compute is over.
    if (p.split_hot_active) {
        static const char cold[] = "ffn_moe_cold_";
        if (std::strncmp(node->name, cold, sizeof(cold) - 1) == 0) {
            if (node->op == GGML_OP_MUL_MAT_ID) {
                const char * dash = std::strrchr(node->name, '-');
                if (dash != nullptr) {
                    self->fill_cache_wait(atoi(dash + 1));
                }
            }
            return;
        }
        self->split_cold_end();
    }

    // cache-aware dropping: the graph forces the weights get_rows onto the CPU
    // backend, so its probs (src[0]) and ids (src[1]) are host inputs of this
    // split. The fill is triggered here instead of at the table get_rows so the
    // scores are available, and it still lands before the split computes.
    // drop_active is cleared when the scores turn out unreadable: the table
    // get_rows then drives the plain fill again, so the cache still works
    if (p.drop_enabled()) {
        static const char wprefix[] = "ffn_moe_weights-";
        const bool is_weights = node->op == GGML_OP_GET_ROWS &&
                std::strncmp(node->name, wprefix, sizeof(wprefix) - 1) == 0;
        if (is_weights) {
            const char * dash = std::strrchr(node->name, '-');
            const int    il   = dash != nullptr ? atoi(dash + 1) : -1;
            const ggml_tensor * t_ids  = node->src[1];
            const ggml_tensor * t_prob = node->src[0];
            // multi-token weights: not a decode fill, leave dropping alone
            if (t_ids == nullptr || t_ids->ne[1] != 1) {
                return;
            }
            // drop was disabled (the scores turned out unreadable): the table
            // get_rows is no longer blocked, so it must own the fill. Filling here
            // as well would read the same layer twice
            if (!p.drop_active) {
                return;
            }
            if (il < 0 || il >= (int) p.cache.size() || p.cache[il].table == nullptr ||
                    t_prob == nullptr || t_ids->type != GGML_TYPE_I32 ||
                    t_prob->type != GGML_TYPE_F32 ||
                    t_ids->buffer == nullptr || t_prob->buffer == nullptr ||
                    !ggml_backend_buffer_is_host(t_ids->buffer) ||
                    !ggml_backend_buffer_is_host(t_prob->buffer) ||
                    t_ids->data == nullptr || t_prob->data == nullptr ||
                    t_ids->nb[0] != sizeof(int32_t)) {
                if (p.drop_active) {
                    if (!p.drop_disable_warned) {
                        p.drop_disable_warned = true;
                        LLAMA_LOG_WARN("%s: cache-aware dropping disabled: weights get_rows il=%d table=%d prob_null=%d ids_null=%d prob_host=%d ids_host=%d\n",
                                       __func__, il,
                                       (int) (il >= 0 && il < (int) p.cache.size() && p.cache[il].table != nullptr),
                                       (int) (t_prob == nullptr), (int) (t_ids == nullptr),
                                       (int) (t_prob != nullptr && t_prob->buffer != nullptr && ggml_backend_buffer_is_host(t_prob->buffer)),
                                       (int) (t_ids != nullptr && t_ids->buffer != nullptr && ggml_backend_buffer_is_host(t_ids->buffer)));
                    }
                    p.disable_drop();
                    // the table get_rows is skipped while dropping intercepts the
                    // weights, so fill here without the scores to keep the table
                    // current for this token
                    const bool ids_host = t_ids->type == GGML_TYPE_I32 &&
                            t_ids->buffer != nullptr && ggml_backend_buffer_is_host(t_ids->buffer) &&
                            t_ids->data != nullptr && t_ids->nb[0] == sizeof(int32_t) &&
                            t_ids->ne[1] == 1 && t_ids->ne[0] > 0;
                    if (ids_host && il >= 0 && il < (int) p.cache.size()) {
                        if (p.split_hot_active) {
                            self->fill_cache_begin(il, (const int32_t *) t_ids->data, t_ids->ne[0], nullptr);
                        } else {
                            self->fill_cache(il, (const int32_t *) t_ids->data, t_ids->ne[0], nullptr);
                        }
                    }
                }
                return;
            }
            const int64_t n_used = t_ids->ne[0];
            if (n_used <= 0) {
                return;
            }
            // probs is the reshaped [1, n_expert, n_tokens] gating output: for
            // decode ne[0] == ne[2] == 1, so expert e sits at e * nb[1]
            const char *    base = (const char *) t_prob->data;
            const int32_t * idp  = (const int32_t *) t_ids->data;
            p.fs_probs.resize((size_t) n_used);
            for (int64_t i = 0; i < n_used; ++i) {
                const int32_t id = idp[i];
                p.fs_probs[(size_t) i] = (id >= 0 && id < (int32_t) t_prob->ne[1])
                        ? *(const float *) (base + (int64_t) id * t_prob->nb[1]) : 1.0f;
            }
            // the substitution simulation and the substitution pass need the
            // whole distribution, not just the routed scores
            if (p.drop_probe || p.substitute_enabled()) {
                const int64_t n_all = t_prob->ne[1];
                p.fs_all.resize((size_t) n_all);
                for (int64_t e = 0; e < n_all; ++e) {
                    p.fs_all[(size_t) e] = *(const float *) (base + e * t_prob->nb[1]);
                }
                p.fs_n_all = n_all;
            } else {
                p.fs_n_all = 0;
            }
            if (p.split_hot_active) {
                self->fill_cache_begin(il, idp, n_used, p.fs_probs.data());
            } else {
                self->fill_cache(il, idp, n_used, p.fs_probs.data());
            }
            return;
        }
        // while dropping intercepts the weights, the table get_rows must not
        // trigger a second fill of the same layer
        if (p.drop_active) {
            return;
        }
    }

    if (node->op != GGML_OP_GET_ROWS || node->src[0] == nullptr || node->src[1] == nullptr) {
        return;
    }

    // the layer whose decode-cache id table this get_rows reads
    const auto it = p.table_layer.find(node->src[0]);
    if (it == p.table_layer.end()) {
        return;
    }

    // the original routed ids: a split input of this CPU get_rows, so already
    // host-side. Only single-token decode uses the table remap
    const ggml_tensor * t = node->src[1];
    if (t->type != GGML_TYPE_I32 || t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer) || t->ne[1] != 1) {
        return;
    }

    const int64_t n_used = t->ne[0];
    if (n_used <= 0) {
        return;
    }

    // ne[1] == 1, so the routed ids are one contiguous I32 row: read it in place
    if (t->data == nullptr || t->nb[0] != sizeof(int32_t)) {
        return;
    }

    if (p.split_hot_active) {
        self->fill_cache_begin(it->second, (const int32_t *) t->data, n_used);
    } else {
        self->fill_cache(it->second, (const int32_t *) t->data, n_used);
    }
#else
    GGML_UNUSED(node);
    GGML_UNUSED(user_data);
#endif
}

// a byte volume for the report: the disk counters reach the TiB range, while the
// sizes and budgets stay in MiB
static std::string report_volume(uint64_t n_bytes) {
    char buf[64];

    if (n_bytes >= 1024ull * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f GiB", n_bytes / (1024.0 * 1024.0 * 1024.0));
    } else {
        snprintf(buf, sizeof(buf), "%.2f MiB", n_bytes / (1024.0 * 1024.0));
    }
    return buf;
}

std::string llama_disk_stage::ram_layout() const {
    impl & p = *pimpl;

    if (!p.active) {
        return std::string();
    }

    std::string slots;
    for (size_t k = 0; k < p.pools.size(); ++k) {
        if (k > 0) {
            slots += " + ";
        }
        slots += std::to_string(p.pools[k].res_cap);
    }
    if (p.pools.size() > 1) {
        slots = "(" + slots + ")";
    }
    const size_t cache_bytes = p.cache_buf ? ggml_backend_buffer_get_size(p.cache_buf) : 0;

    char buf[512];
    snprintf(buf, sizeof(buf), "%zu pool(s) over %zu layer(s) | %s resident slots | %d transient per layer"
             " | decode cache %.2f MiB%s",
             p.pools.size(), p.cache.size(), slots.c_str(), p.n_trans,
             cache_bytes / (1024.0 * 1024.0), p.cache_lock ? " RAM-locked" : "");
    return buf;
}

void llama_disk_stage::print_stats(uint64_t decode_tokens, uint64_t routed, uint64_t base_l2,
                                   uint64_t base_sub, uint64_t base_drop, uint64_t base_disk) {
    impl & p = *pimpl;

    if (!p.active) {
        return;
    }

    // interval deltas against the previous report
    const uint64_t d_evict      = p.n_l2_evictions - p.prev_l2_evictions;
    const uint64_t d_demote     = p.n_l2_demotions - p.prev_l2_demotions;
    const uint64_t d_filtered   = p.n_l2_filtered - p.prev_l2_filtered;
    const uint64_t d_hit_bytes  = p.n_l2_hit_bytes - p.prev_l2_hit_bytes;
    const uint64_t d_promo      = p.n_l2_promo_bytes - p.prev_l2_promo_bytes;
    const uint64_t d_dec_us     = p.n_dec_fill_us - p.prev_dec_fill_us;
    const uint64_t d_dec_calls  = p.n_dec_fill_calls - p.prev_dec_fill_calls;
    const uint64_t d_dec_bytes  = p.n_dec_fill_bytes - p.prev_dec_fill_bytes;
    const uint64_t d_dropped    = p.n_dropped_routes - p.prev_dec_dropped;
    const uint64_t d_substituted = p.n_subst_routes - p.prev_dec_substituted;
    const double   d_routed_mass  = p.n_routed_mass - p.prev_dec_routed_mass;
    const double   d_dropped_mass = p.n_dropped_mass - p.prev_dec_dropped_mass;
    const uint64_t d_tokens     = decode_tokens > p.prev_dec_tokens ? decode_tokens - p.prev_dec_tokens : 0;
    const impl::split_acc split = p.split_tot - p.prev_split_tot;
    // funnel bases and the L2 hits over them, for the per-interval tier rates
    const uint64_t d_base_l2   = base_l2   > p.prev_base_l2   ? base_l2   - p.prev_base_l2   : 0;
    const uint64_t d_base_sub  = base_sub  > p.prev_base_sub  ? base_sub  - p.prev_base_sub  : 0;
    const uint64_t d_base_drop = base_drop > p.prev_base_drop ? base_drop - p.prev_base_drop : 0;
    const uint64_t d_l2_all    = p.n_l2_all_hits > p.prev_l2_all_hits ? p.n_l2_all_hits - p.prev_l2_all_hits : 0;

    // one themed line each. The report is split in four blocks (L2, substitution,
    // dropping, disk) and each block is emitted in one write
    std::string out;
    char buf[1024];
    auto line = [&out, &buf](const char * fmt, auto... args) {
        const int n_chars = snprintf(buf, sizeof(buf), fmt, args...);
        if (n_chars > 0) {
            out.append(buf, (size_t) std::min((size_t) n_chars, sizeof(buf) - 1));
        }
        out += '\n';
    };
    const auto pct = [](uint64_t part, uint64_t whole) {
        return whole ? 100.0 * (double) part / (double) whole : 0.0;
    };

    // every block leads with the selections the previous tier passed down
    const size_t s_bl2 = out.size();
    line("  base      : %" PRIu64 " RAM misses (%.1f%% of %" PRIu64 " routed)",
         base_l2, pct(base_l2, routed), routed);
    const std::string sec_base_l2 = out.substr(s_bl2);

    const size_t s_bsub = out.size();
    line("  base      : %" PRIu64 " L2 misses (%.1f%% of %" PRIu64 " routed)",
         base_sub, pct(base_sub, routed), routed);
    const std::string sec_base_sub = out.substr(s_bsub);

    const size_t s_bdrop = out.size();
    line("  base      : %" PRIu64 " after substitution (%.1f%% of %" PRIu64 " routed)",
         base_drop, pct(base_drop, routed), routed);
    const std::string sec_base_drop = out.substr(s_bdrop);

    const size_t s_bdisk = out.size();
    line("  base      : %" PRIu64 " reads (%.1f%% of %" PRIu64 " routed)",
         base_disk, pct(base_disk, routed), routed);
    const std::string sec_base_disk = out.substr(s_bdisk);

    // the sets read into the decode cache at load and the graph split state
    size_t n_base = 0;
    size_t n_warm = 0;
    size_t n_warm_loaded = 0;
    for (const auto & v : p.base_set) {
        n_base += v.size();
    }
    for (const auto & v : p.warm_set) {
        n_warm += v.size();
    }
    for (const auto & v : p.warm_loaded) {
        n_warm_loaded += v.size();
    }
    const size_t s_sets = out.size();
    line("  sets      : base %zu, warm %zu of %zu loaded | split hot/cold %s",
         n_base, n_warm_loaded, n_warm, p.split_hot_active ? "on" : "off");
    const std::string sec_sets = out.substr(s_sets);

    const uint64_t lookups = p.n_l2_hits + p.n_l2_misses;

    const size_t s_b = out.size();
    if (split.layers > 0 && d_tokens > 0) {
        const double tk = (double) d_tokens;
        line("  fill      : %.2f ms/call, %.2f ms/token span = plan %.2f + copy %.2f + hot %.2f + blocked %.2f",
             d_dec_calls ? (double) d_dec_us / 1000.0 / (double) d_dec_calls : 0.0,
             (double) d_dec_us / 1000.0 / tk,
             (double) split.plan_us / 1000.0 / tk,
             (double) split.copy_us / 1000.0 / tk,
             (double) split.hot_us / 1000.0 / tk,
             (double) split.wait_us / 1000.0 / tk);
    } else {
        line("  fill      : %.2f ms/call, %.2f ms/token span | %s read, %" PRIu64 " calls",
             d_dec_calls ? (double) d_dec_us / 1000.0 / (double) d_dec_calls : 0.0,
             d_tokens ? (double) d_dec_us / 1000.0 / (double) d_tokens : 0.0,
             report_volume(d_dec_bytes).c_str(), d_dec_calls);
    }
    line("  total     : %s read over %" PRIu64 " calls",
         report_volume(p.n_dec_fill_bytes).c_str(), p.n_dec_fill_calls);
    const std::string sec_b = out.substr(s_b);

    const size_t s_wait = out.size();
    if (p.drop_enabled() && !p.drop_warm) {
        line("  warming   : decode cache %.1f%% full of %d%% before dropping or probing",
             p.resident_fill(), p.drop_min_fill_percent);
    }
    const std::string sec_wait = out.substr(s_wait);

    const size_t s_sub = out.size();
    if (p.substitute_enabled() && p.drop_warm) {
        line("  config    : window %.2f-%.2f, pool %d",
             (double) p.substitute_rel, 1.0 / (double) p.substitute_rel, (int) p.substitute_pool);
        line("  substituted: %.1f%% cumulative (%" PRIu64 "/%" PRIu64 ") | %.1f%% this interval (%" PRIu64 "/%" PRIu64
             ") | %s skipped",
             pct(p.n_subst_routes, base_sub), p.n_subst_routes, base_sub,
             pct(d_substituted, d_base_sub), d_substituted, d_base_sub,
             report_volume(p.n_subst_bytes).c_str());
    }
    const std::string sec_sub = out.substr(s_sub);

    const size_t s_drop = out.size();
    if ((p.drop_fraction > 0.0f || p.drop_probe) && p.drop_warm) {
        char layer_cap[16];
        char tok_cap[16];
        if (p.drop_max_mass > 0.0f) {
            snprintf(layer_cap, sizeof(layer_cap), "%.1f%%", 100.0 * (double) p.drop_max_mass);
        } else {
            snprintf(layer_cap, sizeof(layer_cap), "off");
        }
        if (p.drop_max_mass_token > 0.0f) {
            snprintf(tok_cap, sizeof(tok_cap), "%.1f%%", 100.0 * (double) p.drop_max_mass_token);
        } else {
            snprintf(tok_cap, sizeof(tok_cap), "off");
        }
        line("  config    : fraction %.3f, below-rel %.2f, layer-cap %s, token-cap %s",
             (double) p.drop_fraction, (double) p.drop_below_rel, layer_cap, tok_cap);
        line("  dropped   : %.1f%% cumulative (%" PRIu64 "/%" PRIu64 ") | %.1f%% this interval (%" PRIu64 "/%" PRIu64
             ") | %s skipped | worst layer perturb %.1f%%, worst token perturb %.1f%%",
             pct(p.n_dropped_routes, base_drop), p.n_dropped_routes, base_drop,
             pct(d_dropped, d_base_drop), d_dropped, d_base_drop,
             report_volume(p.n_dropped_bytes).c_str(),
             100.0 * p.drop_worst_mass, 100.0 * p.drop_worst_token_mass);
    }
    const std::string sec_drop = out.substr(s_drop);

    const size_t s_l = out.size();
    // the L2 rate is over the RAM misses it works on, the same base the funnel
    // uses. The warm-phase rate excludes the lookups served before the pool
    // warmed, so it is printed apart
    const uint64_t l2_served = base_l2 > base_sub ? base_l2 - base_sub : 0;
    line("  hit       : %.1f%% cumulative (%" PRIu64 "/%" PRIu64 ") | %.1f%% this interval (%" PRIu64 "/%" PRIu64
         ") | warm-phase %.1f%% | %" PRIu64 " skipped cold",
         pct(l2_served, base_l2), l2_served, base_l2,
         pct(d_l2_all, d_base_l2), d_l2_all, d_base_l2,
         lookups ? 100.0 * p.n_l2_hits / lookups : 0.0, p.n_l2_cold);

    line("  l2 served : %s avoided this interval (%s promoted) | %s total",
         report_volume(d_hit_bytes + d_promo).c_str(), report_volume(d_promo).c_str(),
         report_volume(p.n_l2_hit_bytes + p.n_l2_promo_bytes).c_str());

    if (!p.evict_pools.empty()) {
        size_t n_entries  = 0;
        size_t n_capacity = 0;
        for (const auto & ep : p.evict_pools) {
            n_entries  += ep.slot_of.size();
            n_capacity += (size_t) ep.cap;
        }
        line("  l2 policy : %" PRIu64 " evictions, %" PRIu64 " demotions, %" PRIu64 " ghost rejects | %zu/%zu entries live%s",
             d_evict, d_demote, d_filtered, n_entries, n_capacity,
             p.l2_warm ? "" : " | not warm yet");
    }
    const std::string sec_l = out.substr(s_l);

    const size_t s_d = out.size();
    // split hot/cold: what the read costs on the critical path. hidden covers the
    // hot pass, outstanding is the part the cold split had to wait for, blocked is
    // the wall time actually spent in fill_cache_wait
    if (split.layers > 0 && d_tokens > 0) {
        const double tk = (double) d_tokens;
        const int64_t worker_us = split.wait_us > split.stall_us ? split.wait_us - split.stall_us : 0;
        line("  read      : %.2f ms/token drive, %.2f MiB/token over %d split(s) | %.0f%% hidden behind the %.2f ms/token hot pass",
             (double) split.read_us / 1000.0 / tk,
             (double) split.bytes / (1024.0 * 1024.0) / tk, split.layers,
             split.read_us > 0 ? 100.0 * (double) split.ovl_us / (double) split.read_us : 0.0,
             (double) split.hot_us / 1000.0 / tk);
        line("  blocked   : %.2f ms/token in fill_cache_wait (%.2f read outstanding on %d of %d split(s), %.2f worker copy/signal)",
             (double) split.wait_us / 1000.0 / tk,
             (double) split.stall_us / 1000.0 / tk, split.n_read_limited, split.layers,
             (double) worker_us / 1000.0 / tk);
    }
    const std::string sec_d = out.substr(s_d);

    const size_t s_o = out.size();
    if (!p.evict_pools.empty()) {
        std::string row = "  pools     :";
        for (size_t k = 0; k < p.evict_pools.size(); ++k) {
            const impl::evict_pool & ep = p.evict_pools[k];
            const uint64_t look = ep.hits + ep.misses;
            snprintf(buf, sizeof(buf), " pool %zu: %zu/%d live, hit %.1f%% (%" PRIu64 "/%" PRIu64
                     " warm), evictions %" PRIu64, k, ep.slot_of.size(), ep.cap,
                     look ? 100.0 * (double) ep.hits / (double) look : 0.0, ep.hits, look, ep.evictions);
            row += buf;
            // wrap after appending: checking first lets the last entry push the
            // row well past the width
            if (row.size() > 110 && k + 1 < p.evict_pools.size()) {
                out += row;
                out += '\n';
                row = "            ";
            }
        }
        out += row;
        out += '\n';
    }
    const std::string sec_o = out.substr(s_o);

    // four blocks, fastest tier first. The substitution and dropping blocks only
    // appear when their lever is on
    const std::string rno = " report #" + std::to_string(p.n_reports + 1) + "\n";

    LLAMA_LOG_INFO("%s", ("[l2-cache]" + rno + sec_base_l2 + sec_l + sec_o).c_str());

    if (p.substitute_enabled()) {
        LLAMA_LOG_INFO("%s", ("[substitution]" + rno + sec_base_sub + sec_wait + sec_sub).c_str());
    }
    if (p.drop_fraction > 0.0f || p.drop_probe) {
        std::string body = sec_base_drop + sec_drop;
        if (!p.substitute_enabled()) {
            body = sec_wait + body;
        }
        LLAMA_LOG_INFO("%s", ("[drop]" + rno + body).c_str());
    }

    LLAMA_LOG_INFO("%s", ("[disk]" + rno + sec_base_disk + sec_sets + sec_d + sec_b).c_str());

    p.prev_l2_hits           = p.n_l2_hits;
    p.prev_l2_all_hits       = p.n_l2_all_hits;
    p.prev_l2_misses         = p.n_l2_misses;
    p.prev_l2_cold           = p.n_l2_cold;
    p.prev_l2_evictions      = p.n_l2_evictions;
    p.prev_l2_demotions      = p.n_l2_demotions;
    p.prev_l2_hit_bytes      = p.n_l2_hit_bytes;
    p.prev_l2_promo_bytes    = p.n_l2_promo_bytes;
    p.prev_l2_filtered       = p.n_l2_filtered;
    p.prev_dec_fill_us       = p.n_dec_fill_us;
    p.prev_dec_fill_calls    = p.n_dec_fill_calls;
    p.prev_dec_fill_bytes    = p.n_dec_fill_bytes;
    p.prev_dec_cache_hits    = p.n_decode_cache_hits;
    p.prev_dec_cache_misses  = p.n_decode_cache_misses;
    p.prev_dec_routed_routes = p.n_routed_routes;
    p.prev_dec_cache_fills   = p.n_decode_cache_fills;
    p.prev_dec_cache_changes = p.n_decode_cache_resident_changes;
    p.prev_dec_dropped       = p.n_dropped_routes;
    p.prev_dec_substituted   = p.n_subst_routes;
    p.prev_dec_routed_mass   = p.n_routed_mass;
    p.prev_dec_dropped_mass  = p.n_dropped_mass;
    p.drop_worst_mass        = 0.0;
    p.drop_worst_token_mass  = 0.0;
    p.prev_dec_tokens        = decode_tokens;
    p.prev_base_l2           = base_l2;
    p.prev_base_sub          = base_sub;
    p.prev_base_drop         = base_drop;
    p.prev_split_tot         = p.split_tot;
    p.n_reports++;
}

// calibration dump: the cumulative probe and substitution simulation tables are
// for tuning, so they are printed once at shutdown instead of every interval
void llama_disk_stage::print_calibration() {
#if defined(_WIN32)
    impl & p = *pimpl;

    if (!p.active || !p.drop_probe || !p.drop_warm) {
        return;
    }

    std::string out;
    char buf[1024];
    auto line = [&out, &buf](const char * fmt, auto... args) {
        const int n_chars = snprintf(buf, sizeof(buf), fmt, args...);
        if (n_chars > 0) {
            out.append(buf, (size_t) std::min((size_t) n_chars, sizeof(buf) - 1));
        }
        out += '\n';
    };
    char cell[32];

    // cumulative probe: the distribution is what it is over the whole run, not
    // per interval. The quantiles are the inverse of the cold relative
    // cumulative, so a target drop share picks the floor directly
    const uint64_t calls = p.probe_calls;
    const auto hist_at = [&p, calls](double f) -> uint64_t {
        if (calls == 0) {
            return 0;
        }
        const uint64_t target = (uint64_t) (f * (double) calls);
        uint64_t acc = 0;
        for (uint64_t c = 0; c <= 32; ++c) {
            acc += p.probe_cold_hist[c];
            if (acc >= target) {
                return c;
            }
        }
        return 32;
    };
    const uint64_t cold         = p.probe_cold_n;
    const uint64_t rel_n        = p.probe_rel_n;
    const uint64_t probe_routed = p.probe_routed;
    line("  drop probe: %" PRIu64 " routed, %" PRIu64 " cold (%.1f%% of selections, %.1f%% of score mass)"
         " | cold/layer p50=%" PRIu64 " p90=%" PRIu64 " p99=%" PRIu64 " max=%" PRIu64,
         probe_routed, cold,
         probe_routed ? 100.0 * (double) cold / (double) probe_routed : 0.0,
         p.n_routed_mass > 0.0 ? 100.0 * p.n_cold_mass / p.n_routed_mass : 0.0,
         hist_at(0.50), hist_at(0.90), hist_at(0.99), p.probe_cold_max);
    line("  cold rel  : q10=%.2f q25=%.2f q50=%.2f q75=%.2f q90=%.2f q95=%.2f q99=%.2f (score / layer max)",
         (double) p.probe_rel_quantile(0.10f), (double) p.probe_rel_quantile(0.25f),
         (double) p.probe_rel_quantile(0.50f), (double) p.probe_rel_quantile(0.75f),
         (double) p.probe_rel_quantile(0.90f), (double) p.probe_rel_quantile(0.95f),
         (double) p.probe_rel_quantile(0.99f));
    static const float curve[] = { 0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f, 0.90f };
    std::string row_a = "  drop curve:  A     ";
    std::string row_n = "               cold% ";
    std::string row_m = "               mass% ";
    for (const float a : curve) {
        snprintf(cell, sizeof(cell), "%5.2f ", (double) a);
        row_a += cell;
        snprintf(cell, sizeof(cell), "%5.1f ",
                 rel_n ? 100.0 * (double) p.probe_rel_below(a) / (double) rel_n : 0.0);
        row_n += cell;
        snprintf(cell, sizeof(cell), "%5.1f ",
                 p.n_routed_mass > 0.0 ? 100.0 * p.probe_rel_mass_below(a) / p.n_routed_mass : 0.0);
        row_m += cell;
    }
    out += row_a + "\n";
    out += row_n + "\n";
    out += row_m + "\n";

    if (p.sim_cold > 0) {
        static const int   sim_p[impl::sim_n_p] = { 2, 4, 8 };
        static const float sim_s[impl::sim_n_s] = { 0.50f, 0.70f, 0.80f, 0.90f, 0.95f };
        std::string row_s = "  subst sim :   S    ";
        for (const float s : sim_s) {
            snprintf(cell, sizeof(cell), "%5.2f ", (double) s);
            row_s += cell;
        }
        out += row_s + "\n";
        for (int pi = 0; pi < impl::sim_n_p; ++pi) {
            std::string row_p;
            snprintf(cell, sizeof(cell), "                 P=%-2d ", sim_p[pi]);
            row_p += cell;
            for (int si = 0; si < impl::sim_n_s; ++si) {
                snprintf(cell, sizeof(cell), "%5.1f ",
                        100.0 * (double) p.sim_fired[pi][si] / (double) p.sim_cold);
                row_p += cell;
            }
            out += row_p + "\n";
        }
        out += "               (% of cold with a resident spare in [S, 1/S] of the cold score)\n";
        // median of |c - e| / layer_mass from the histogram
        uint64_t med_bin = 0;
        {
            const uint64_t target = (p.sim_ref_n + 1) / 2;
            uint64_t acc = 0;
            for (int i = 0; i < impl::sim_rel_bins; ++i) {
                acc += p.sim_ref_rel_hist[i];
                if (acc >= target) {
                    med_bin = (uint64_t) i;
                    break;
                }
                med_bin = (uint64_t) i;
            }
        }
        const double med_pct = 100.0 * ((double) med_bin + 0.5) / (double) impl::sim_rel_bins * impl::sim_rel_max;
        line("  subst cost: fires (S=0.90, P=4) %" PRIu64 " of %" PRIu64 " cold | spare weaker %.1f%%, stronger/equal %.1f%%"
             " | top-expert substitutions %" PRIu64 " | %s of reads avoidable",
             p.sim_ref_n, p.sim_cold,
             p.sim_ref_n ? 100.0 * (double) p.sim_ref_weaker / (double) p.sim_ref_n : 0.0,
             p.sim_ref_n ? 100.0 * (double) p.sim_ref_stronger / (double) p.sim_ref_n : 0.0,
             p.sim_top_fired, report_volume(p.sim_ref_bytes).c_str());
        line("  subst gap : (c-e) / layer mass: mean signed %+.2f%%, mean abs %.2f%%, median abs %.2f%%"
             " | per expert (c-e)/c: mean %+.1f%%",
             p.sim_ref_n ? 100.0 * p.sim_ref_rel_sum / (double) p.sim_ref_n : 0.0,
             p.sim_ref_n ? 100.0 * p.sim_ref_rel_abs / (double) p.sim_ref_n : 0.0,
             p.sim_ref_n ? med_pct : 0.0,
             p.sim_ref_n ? 100.0 * p.sim_ref_loss / (double) p.sim_ref_n : 0.0);
    }

    if (out.empty()) {
        return;
    }
    LLAMA_LOG_INFO("%s", ("[calibration] final\n" + out).c_str());
#endif
}

const llama_disk_stage_layer * llama_disk_stage::layer(int il) const {
    if (!pimpl->active || il < 0 || il >= (int) pimpl->layers.size()) {
        return nullptr;
    }
    const llama_disk_stage_layer & l = pimpl->layers[il];
    return l.gate != nullptr ? &l : nullptr;
}

const llama_disk_stage_cache_layer * llama_disk_stage::cache_layer(int il) const {
    if (il < 0 || il >= (int) pimpl->cache.size() || pimpl->cache[il].table == nullptr) {
        return nullptr;
    }
    return &pimpl->cache[il].pub;
}

int llama_disk_stage::n_pools() const {
    return (int) pimpl->pools.size();
}

int llama_disk_stage::pool_id(int il) const {
    if (il < 0 || il >= (int) pimpl->cache.size() || pimpl->cache[il].table == nullptr) {
        return -1;
    }
    return pimpl->cache[il].pool;
}

llama_disk_stage::llama_disk_stage(const llama_model & model, ggml_backend_dev_t dev,
                                   int32_t n_pin_experts, uint64_t cache_budget_bytes,
                                   int32_t pool_layers_max, const char * base_experts_path,
                                   const char * warm_experts_path, const char * base_template_path,
                                   bool split_hot, float drop_fraction, float drop_below_rel,
                                   float drop_max_mass, float drop_max_mass_token, float substitute_rel, int32_t substitute_pool,
                                   bool drop_probe) :
    pimpl(std::make_unique<impl>(model)) {
    impl & p = *pimpl;

    llama_mem_tag_scope mem_scope("disk");

#if !defined(_WIN32)
    GGML_UNUSED(dev);
    GGML_UNUSED(n_pin_experts);
    GGML_UNUSED(cache_budget_bytes);
    GGML_UNUSED(pool_layers_max);
    GGML_UNUSED(base_experts_path);
    GGML_UNUSED(warm_experts_path);
    GGML_UNUSED(base_template_path);
    GGML_UNUSED(split_hot);
    GGML_UNUSED(drop_fraction);
    GGML_UNUSED(drop_below_rel);
    GGML_UNUSED(drop_max_mass);
    GGML_UNUSED(drop_max_mass_token);
    GGML_UNUSED(substitute_rel);
    GGML_UNUSED(substitute_pool);
    GGML_UNUSED(drop_probe);
    return;
#else
    if (!model.has_disk_weights()) {
        return;
    }

    // cache-aware opportunistic dropping: the lowest fraction of the routed
    // experts of a token is dropped instead of read when it is cold and weak
    // relative to the layer's top. drop_below_rel is what keeps a flat layer
    // from losing a significant expert, and it always keeps the top expert.
    // SOFTMAX_WEIGHT scales the selected weights, so the multiplicative keep
    // mask the graph applies is wrong there: drop nothing for that gating
    if (model.hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        if (drop_fraction > 0.0f || drop_probe || substitute_pool > 0) {
            LLAMA_LOG_WARN("%s: dropping and substitution disabled: SOFTMAX_WEIGHT gating is not supported\n", __func__);
        }
        drop_fraction   = 0.0f;
        drop_probe      = false;
        substitute_pool = 0;
    }
    p.drop_fraction  = drop_fraction;
    p.drop_below_rel = std::min(std::max(drop_below_rel, 1e-3f), 1.0f);
    p.drop_max_mass       = std::min(std::max(drop_max_mass, 0.0f), 1.0f);
    p.drop_max_mass_token = std::min(std::max(drop_max_mass_token, 0.0f), 1.0f);
    p.substitute_rel  = std::min(std::max(substitute_rel, 0.0f), 0.999f);
    p.substitute_pool = std::min(std::max(substitute_pool, 0), 64);
    p.drop_probe      = drop_probe;
    p.drop_active    = p.drop_enabled();

    // split the host decode MoE into a hot and a cold pass so the cold disk read
    // overlaps the hot compute
    p.split_hot_active = split_hot;

    p.iocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (p.iocp == nullptr) {
        LLAMA_LOG_WARN("%s: failed to create the I/O completion port, disk staging disabled\n", __func__);
        return;
    }

    // the decode worker reads on its own completion port, so its reads never
    // queue behind the prefill reader on io_mu
    p.iocp_dec = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (p.iocp_dec == nullptr) {
        LLAMA_LOG_WARN("%s: failed to create the decode I/O completion port, the decode fill runs inline\n", __func__);
        p.split_hot_active = false;
    }

    // stages the separate gate/up/down layout only
    const int n_layer = (int) model.hparams.n_layer();
    p.layers.assign(n_layer, {});
    p.layer_regions.assign(n_layer, {});
    p.split_cold.assign(n_layer, 0);

    // collect the regions and their source files first so the pool size is known
    struct role_src {
        const char * suffix;
        int          slot; // 0 gate, 1 up, 2 down
    };
    const role_src roles[] = {
        { "ffn_gate_exps.weight", 0 },
        { "ffn_up_exps.weight",   1 },
        { "ffn_down_exps.weight", 2 },
    };

    struct layer_src {
        const ggml_tensor * t[3] = { nullptr, nullptr, nullptr };
        impl::region       r[3];
        bool               ok = false;
    };
    std::vector<layer_src> src(n_layer);

    for (int il = 0; il < n_layer; ++il) {
        const std::string base_name = "blk." + std::to_string(il) + ".";
        bool ok = true;
        for (const auto & role : roles) {
            const std::string name = base_name + role.suffix;
            const ggml_tensor * t = model.get_tensor(name.c_str());
            // data may be null in disk-stream mode: the region comes from the file offsets
            if (t == nullptr || t->ne[2] <= 0) {
                ok = false;
                break;
            }
            std::string path;
            size_t      t_off = 0;
            if (!model.tensor_file_region(t, path, t_off)) {
                ok = false;
                break;
            }
            const size_t size = (size_t) t->nb[2] * (size_t) t->ne[2];
            const size_t head = t_off & (disk_stage_align - 1);

            src[il].t[role.slot] = t;
            src[il].r[role.slot].head     = head;
            src[il].r[role.slot].stride   = (size_t) t->nb[2];
            src[il].r[role.slot].n_expert = (int32_t) t->ne[2];
            src[il].r[role.slot].read_len = align_up(head + size, disk_stage_align);
            src[il].r[role.slot].file_off = t_off - head;
            src[il].r[role.slot].file     = p.file_for(path);
        }
        src[il].ok = ok;
    }

    // one expert set file: parse it and validate it against the model's layer
    // layout, so a bad file fails before any pool is allocated
    const auto load_set = [&](const std::string & path, const char * what,
                              std::vector<std::vector<int32_t>> & out) {
        out = disk_stage_parse_expert_set(path, n_layer, what);
        int32_t n = 0;
        for (int il = 0; il < n_layer; ++il) {
            if (out[(size_t) il].empty()) {
                continue;
            }
            if (!src[il].ok) {
                throw std::runtime_error(std::string("disk stage: ") + what + " references layer " + std::to_string(il) +
                                         ", which is not stageable");
            }
            const int32_t n_expert = (int32_t) src[il].t[0]->ne[2];
            for (const int32_t id : out[(size_t) il]) {
                if (id >= n_expert) {
                    throw std::runtime_error(std::string("disk stage: ") + what + " references layer " + std::to_string(il) +
                                             " expert " + std::to_string(id) + ", out of range [0, " +
                                             std::to_string(n_expert) + ")");
                }
            }
            n += (int32_t) out[(size_t) il].size();
        }
        LLAMA_LOG_INFO("%s: %s '%s': %d expert(s) over %d layer(s)\n", __func__, what, path.c_str(), n, n_layer);
        return n;
    };

    // base-expert set: parse and validate now, so a bad file fails before any
    // pool is allocated
    if (base_template_path != nullptr && base_template_path[0] != '\0') {
        llama_expert_base_template_parse(base_template_path, p.base_template);
        p.have_base_template = true;
        const llama_expert_base_template & tmpl = p.base_template;
        if (base_experts_path != nullptr && base_experts_path[0] != '\0') {
            LLAMA_LOG_WARN("%s: --pin-experts-template replaces --pin-experts-from-profile, "
                           "ignoring '%s'\n", __func__, base_experts_path);
        }
        load_set(tmpl.base_file, "base-expert set", p.base_set);
        p.base_targets[""] = p.base_set;
        p.base_active = tmpl.default_set;

        // the mode sets, and with them the resident target of every mode: the
        // base set merged with that mode's set, so a switch is a lookup
        for (const auto & [name, file] : tmpl.set_files) {
            std::vector<std::vector<int32_t>> set;
            load_set(file, ("mode set '" + name + "'").c_str(), set);
            std::vector<std::vector<int32_t>> target((size_t) n_layer);
            for (int il = 0; il < n_layer; ++il) {
                const auto & b = p.base_set[(size_t) il];
                const auto & s = set[(size_t) il];
                target[(size_t) il].resize(b.size() + s.size());
                const auto end = std::set_union(b.begin(), b.end(), s.begin(), s.end(), target[(size_t) il].begin());
                target[(size_t) il].resize((size_t) (end - target[(size_t) il].begin()));
            }
            p.base_targets[name] = std::move(target);
        }
        LLAMA_LOG_INFO("%s: expert-set template '%s': %zu mode set(s), default '%s', base set is resident "
                       "in every mode\n", __func__, base_template_path, tmpl.set_files.size(),
                       p.base_active.empty() ? "<none>" : p.base_active.c_str());
    } else if (base_experts_path != nullptr && base_experts_path[0] != '\0') {
        load_set(base_experts_path, "base-expert set", p.base_set);
    }
    if (p.base_targets.find("") == p.base_targets.end()) {
        p.base_targets[""] = p.base_set;  // the base set alone is always a valid target
    }
    p.base_target = &p.base_targets.at(p.base_active);

    // warm-expert set: same format, candidates for the slots the base set leaves
    // free. Base experts, experts of non-stageable layers and out-of-range ids
    // are dropped, since the warm fill is best effort
    if (warm_experts_path != nullptr && warm_experts_path[0] != '\0') {
        // the slots the warm set competes for are the ones the base set and the
        // default mode set do not take
        static const std::vector<std::vector<int32_t>> empty_set;
        const auto base0_it = p.base_targets.find(p.base_active);
        const std::vector<std::vector<int32_t>> & base0 =
            base0_it == p.base_targets.end() ? empty_set : base0_it->second;
        p.warm_set = disk_stage_parse_expert_set(warm_experts_path, n_layer, "warm-expert set");
        int32_t n_kept    = 0;
        int32_t n_skipped = 0;
        for (int il = 0; il < n_layer; ++il) {
            std::vector<int32_t> & v = p.warm_set[(size_t) il];
            if (v.empty()) {
                continue;
            }
            if (!src[il].ok) {
                n_skipped += (int32_t) v.size();
                v.clear();
                continue;
            }
            const int32_t n_expert = (int32_t) src[il].t[0]->ne[2];
            const bool    have_base = il < (int) base0.size() && !base0[(size_t) il].empty();
            std::vector<int32_t> kept;
            kept.reserve(v.size());
            for (const int32_t id : v) {
                const bool in_base = have_base && std::binary_search(base0[(size_t) il].begin(),
                                                                     base0[(size_t) il].end(), id);
                if (id < n_expert && !in_base) {
                    kept.push_back(id);
                } else {
                    n_skipped++;
                }
            }
            v = std::move(kept);
            n_kept += (int32_t) v.size();
        }
        LLAMA_LOG_INFO("%s: warm-expert set '%s': %d candidate(s), %d skipped\n",
                       __func__, warm_experts_path, n_kept, n_skipped);
    }

    // every layer's staging tensors alias one region per role, so the pool holds
    // a single layer's worth of experts, not the whole model
    size_t region_len[3] = { 0, 0, 0 };
    for (int il = 0; il < n_layer; ++il) {
        if (!src[il].ok) {
            continue;
        }
        for (const auto & role : roles) {
            region_len[role.slot] = std::max(region_len[role.slot], src[il].r[role.slot].read_len);
        }
    }

    size_t region_off[3] = { 0, 0, 0 };
    size_t per_buffer = 0; // one layer's worth, all roles
    for (const auto & role : roles) {
        region_off[role.slot] = per_buffer;
        // one sector of slack: a run read is sector-aligned and can overrun the
        // region end into the next role's region
        per_buffer += region_len[role.slot] + disk_stage_align;
    }

    // two buffers so the read of layer i+1 overlaps the compute of layer i;
    // slack for aligning the pool base plus any per-tensor allocation rounding
    size_t total = (size_t) p.n_buf * per_buffer + 2 * disk_stage_align;

    // the pool holds one layer's expert region per role per buffer; anything far
    // larger means the layout math is wrong, so refuse rather than commit memory
    const size_t total_max = (size_t) 16 << 30;
    if (total > total_max) {
        LLAMA_LOG_WARN("%s: staging pool would be %.1f GiB, refusing, disk staging disabled\n",
                       __func__, total / (1024.0 * 1024.0 * 1024.0));
        return;
    }

    // one staging buffer, aliased by every layer's staging tensors. Prefer the
    // device's pinned host buffer so the host->VRAM offload copy reads from
    // page-locked memory; fall back to plain CPU memory
    ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
    if (buft == nullptr) {
        buft = ggml_backend_cpu_buffer_type();
    }

    p.pool = ggml_backend_buft_alloc_buffer(buft, total);
    if (p.pool == nullptr && p.n_buf > 1) {
        // no room for the second buffer: fall back to one, no read-ahead
        LLAMA_LOG_WARN("%s: could not allocate the %zu MiB double staging pool, "
                       "falling back to a single buffer (no prefill read-ahead)\n",
                       __func__, total / (1024 * 1024));
        p.n_buf = 1;
        total  = per_buffer + 2 * disk_stage_align;
        p.pool = ggml_backend_buft_alloc_buffer(buft, total);
    }
    if (p.pool == nullptr) {
        LLAMA_LOG_WARN("%s: failed to allocate the %zu MiB staging pool, disk staging disabled\n",
                       __func__, total / (1024 * 1024));
        return;
    }
    ggml_backend_buffer_set_usage(p.pool, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    p.base = (char *) align_up((uintptr_t) ggml_backend_buffer_get_base(p.pool), disk_stage_align);

    // alternate the staging buffer along the order of stageable layers, so a
    // layer and its read-ahead target never share one
    {
        std::vector<int32_t> order;
        order.reserve(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            if (src[il].ok) {
                order.push_back(il);
            }
        }
        p.layer_buf.assign(n_layer, 0);
        p.next_stage.assign(n_layer, -1);
        for (size_t k = 0; k < order.size(); ++k) {
            p.layer_buf[order[k]] = (int8_t) (k % (size_t) p.n_buf);
            if (k + 1 < order.size()) {
                p.next_stage[order[k]] = order[k + 1];
            }
        }
    }

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (n_layer * 3 + 16),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    p.ctx = ggml_init(ip);
    if (p.ctx == nullptr) {
        LLAMA_LOG_WARN("%s: failed to create the staging context, disk staging disabled\n", __func__);
        return;
    }

    int n_staged = 0;
    for (int il = 0; il < n_layer; ++il) {
        if (!src[il].ok) {
            continue;
        }

        ggml_tensor * tensors[3] = { nullptr, nullptr, nullptr };
        for (const auto & role : roles) {
            const ggml_tensor * s = src[il].t[role.slot];
            ggml_tensor * t = ggml_new_tensor_3d(p.ctx, s->type, s->ne[0], s->ne[1], s->ne[2]);
            ggml_format_name(t, "disk_stage_%s.%d", role.suffix, il);
            tensors[role.slot] = t;
        }
        for (const auto & role : roles) {
            impl::region & r = src[il].r[role.slot];
            r.pool_off = (size_t) p.layer_buf[il] * per_buffer + region_off[role.slot];

            void * addr = p.base + r.pool_off + r.head;
            if (ggml_backend_tensor_alloc(p.pool, tensors[role.slot], addr) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_WARN("%s: failed to bind a staging tensor, disk staging disabled\n", __func__);
                return;
            }
        }

        p.layers[il].gate = tensors[0];
        p.layers[il].up   = tensors[1];
        p.layers[il].down = tensors[2];

        p.layer_regions[il].resize(3);
        for (const auto & role : roles) {
            p.layer_regions[il][role.slot] = src[il].r[role.slot];
        }
        n_staged++;
    }

    if (n_staged == 0) {
        LLAMA_LOG_WARN("%s: no stageable MoE layer found, disk staging disabled\n", __func__);
        return;
    }

    // persistent decode cache: one shared slot array per pool of layers with
    // identical expert tensors. The graph remaps selected_experts through each
    // layer's table and reads the weights in place, so a resident expert is
    // served without a copy. Single-token decode needs the cache (the model
    // tensors are never mapped). A pool is shared only when every layer's tensor
    // data is sector-aligned in the file (head == 0), which the realign script
    // guarantees; otherwise each layer is its own pool, which reproduces the old
    // static per-layer layout
    const uint64_t cache_budget = cache_budget_bytes;
    {
        int ref = -1;
        for (int il = 0; il < n_layer; ++il) {
            if (src[il].ok) {
                ref = il;
                break;
            }
        }
        if (ref >= 0) {
            const int32_t n_expert = (int32_t) src[ref].t[0]->ne[2];
            const int32_t n_used   = std::max<int32_t>(1, (int32_t) model.hparams.n_expert_used());
            const int32_t n_trans  = std::min<int32_t>(n_used, n_expert);
            p.n_trans = n_trans;

            // size from the sum of every stageable layer's per-slot cost, not from
            // one reference layer: the bundle varies across layers (this quantization
            // mixes Q8_0 and Q5_1 down projections), and a reference layer's bundle
            // would leave the cheaper layers under-filled. The padded stride is what
            // the cache allocates per slot
            size_t cost_per_slot = 0;
            for (int il = 0; il < n_layer; ++il) {
                if (!src[il].ok) {
                    continue;
                }
                for (const auto & role : roles) {
                    cost_per_slot += (size_t) src[il].t[role.slot]->nb[2] + disk_stage_align;
                }
            }

            // a pool needs every layer's tensor data at the same sector
            // remainder, since one data pointer serves the whole pool and the
            // unbuffered read destination must be aligned; the realign script
            // makes that remainder zero
            bool all_aligned = true;
            for (int il = 0; il < n_layer && all_aligned; ++il) {
                if (!src[il].ok) {
                    continue;
                }
                for (const auto & role : roles) {
                    if (src[il].r[role.slot].head != 0) {
                        all_aligned = false;
                        break;
                    }
                }
            }

            // group the stageable layers: all-aligned layers with identical
            // expert tensors share one array per role, everything else gets a
            // private one. The cap trades cross-layer sharing for speed: the
            // CPU mul_mat_id scans every slot of its src tensor (n_as = ne02),
            // so a pool of N layers makes each layer pay for N layers' slots
            const int max_pool_layers = pool_layers_max > 0 ? pool_layers_max : (1 << 30);
            std::vector<std::vector<int>> groups;
            if (all_aligned) {
                for (int il = 0; il < n_layer; ++il) {
                    if (!src[il].ok) {
                        continue;
                    }
                    int g = -1;
                    for (size_t k = 0; k < groups.size(); ++k) {
                        if ((int) groups[k].size() >= max_pool_layers) {
                            continue;  // cap reached: start another sub-pool
                        }
                        const layer_src & a = src[groups[k][0]];
                        bool same = true;
                        for (const auto & role : roles) {
                            const ggml_tensor * ta = a.t[role.slot];
                            const ggml_tensor * tb = src[il].t[role.slot];
                            if (ta->type != tb->type || ta->ne[0] != tb->ne[0] || ta->ne[1] != tb->ne[1]) {
                                same = false;
                                break;
                            }
                        }
                        if (same) {
                            g = (int) k;
                            break;
                        }
                    }
                    if (g < 0) {
                        groups.push_back({});
                        g = (int) groups.size() - 1;
                    }
                    groups[g].push_back(il);
                }
            } else {
                for (int il = 0; il < n_layer; ++il) {
                    if (src[il].ok) {
                        groups.push_back({ il });
                    }
                }
            }

            // one shared transient window per pool, so the budget must hold one
            // expert of each pool's bundle on top of the resident slots
            size_t pool_cost = 0;
            for (const auto & grp : groups) {
                for (const auto & role : roles) {
                    pool_cost += (size_t) src[grp[0]].t[role.slot]->nb[2] + disk_stage_align;
                }
            }

            int32_t res_per_layer = 0;
            if (cache_budget > 0 && cost_per_slot > 0) {
                // every pool holds n_trans transient slots and one sentinel slot;
                // charge those in bytes instead of a per-layer share, which would
                // round a whole layer's bundle per term and leave slots unused
                const int64_t fixed = ((int64_t) n_trans + 1) * (int64_t) pool_cost
                                    + (int64_t) n_layer * 4 * (int64_t) disk_stage_align;
                if ((int64_t) cache_budget > fixed) {
                    const int64_t slots = ((int64_t) cache_budget - fixed) / (int64_t) cost_per_slot;
                    res_per_layer = (int32_t) std::min<int64_t>(slots, n_expert);
                }
            }
            if (n_pin_experts > 0) {
                // --pin-hot-experts N: N resident experts per layer on average.
                // With pools a hot layer may exceed N and a cold one fall short;
                // the budget bounds the total across the pool
                const int32_t want = std::min<int32_t>(n_pin_experts, n_expert);
                res_per_layer = res_per_layer > 0 ? std::min(res_per_layer, want) : want;
            }
            if (res_per_layer == 0) {
                res_per_layer = 1;
                LLAMA_LOG_WARN("%s: neither --pin-hot-experts nor --pin-hot-experts-budget-mib given, "
                               "using a minimal decode cache\n", __func__);
            }
            res_per_layer = std::min<int32_t>(res_per_layer, n_expert);

            // the base set is a requirement, not a hint: a pool must have room
            // for every one of its base experts, or the run refuses to start.
            // With a template the largest target is what has to fit: one mode is
            // resident at a time, and a switch demotes the previous mode before
            // it admits the new one
            for (const auto & [tname, target] : p.base_targets) {
                if (target.empty()) {
                    continue;
                }
                for (size_t g = 0; g < groups.size(); ++g) {
                    int32_t n_base = 0;
                    for (int il : groups[g]) {
                        n_base += (int32_t) target[(size_t) il].size();
                    }
                    const int32_t res_cap = (int32_t) groups[g].size() * res_per_layer;
                    if (n_base > res_cap) {
                        throw std::runtime_error("disk stage: base-expert set" +
                                                 std::string(tname.empty() ? "" : " '" + tname + "'") + " needs " +
                                                 std::to_string(n_base) +
                                                 " resident slot(s) in pool " + std::to_string(g) +
                                                 " but the decode cache has " + std::to_string(res_cap) +
                                                 "; raise --pin-hot-experts-budget-mib");
                    }
                }
            }

            if (res_per_layer > 0) {
                size_t cache_bytes = 64 * 1024;
                for (const auto & grp : groups) {
                    const int32_t n_pool_slots = (int32_t) grp.size() * res_per_layer + n_trans;
                    for (const auto & role : roles) {
                        const size_t stride = (size_t) src[grp[0]].t[role.slot]->nb[2];
                        const size_t head   = all_aligned ? 0 : src[grp[0]].r[role.slot].head;
                        // one extra aligned span per tensor: the per-expert read is
                        // rounded up to the sector size, which can overrun the last slot
                        cache_bytes += align_up(head + (size_t) (n_pool_slots + 1) * (stride + disk_stage_align) + disk_stage_align, disk_stage_align);
                    }
                    for (size_t j = 0; j < grp.size(); ++j) {
                        cache_bytes += align_up((size_t) n_expert * sizeof(int32_t), disk_stage_align);
                        cache_bytes += align_up((size_t) (n_pool_slots + 1) * sizeof(int32_t), disk_stage_align);
                        if (p.split_hot_active) {
                            // split-hot: the hot and cold slot tables
                            cache_bytes += 2 * align_up((size_t) (n_pool_slots + 1) * sizeof(int32_t), disk_stage_align);
                        }
                        if (p.drop_fraction > 0.0f || p.substitute_enabled()) {
                            // cache-aware dropping / substitution: the per-expert keep table
                            cache_bytes += align_up((size_t) n_expert * sizeof(float), disk_stage_align);
                        }
                    }
                }

                // the decode cache is read by the CPU mul_mat_id only: decode
                // never offloads (n_tokens == 1), so it is plain CPU memory.
                // A pinned (cudaMallocHost) buffer would page-lock the whole
                // cache, map it through the limited BAR1 aperture, and count
                // against the WDDM device budget - a 32 GiB cache then fails or
                // OOMs unrelated device allocations. The staging pool above stays
                // pinned because it IS the source of the host->VRAM offload copy.
                ggml_backend_buffer_type_t cbuft = ggml_backend_cpu_buffer_type();
                p.cache_buf = ggml_backend_buft_alloc_buffer(cbuft, cache_bytes);
                if (p.cache_buf == nullptr) {
                    LLAMA_LOG_WARN("%s: failed to allocate the %.2f GiB decode cache\n",
                                   __func__, cache_bytes / (1024.0 * 1024.0 * 1024.0));
                } else {
                    ggml_backend_buffer_set_usage(p.cache_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    char * cbase = (char *) align_up((uintptr_t) ggml_backend_buffer_get_base(p.cache_buf), disk_stage_align);

                    if (!p.lock_cache(cbase, cache_bytes)) {
                        throw std::runtime_error("failed to hold the MoE decode cache in RAM");
                    }

                    ggml_init_params cip = {
                        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (n_layer * 5 + groups.size() * 3 + 16),
                        /*.mem_buffer =*/ nullptr,
                        /*.no_alloc   =*/ true,
                    };
                    p.cache_ctx = ggml_init(cip);
                    if (p.cache_ctx != nullptr) {
                        p.cache.resize(n_layer);
                        p.pools.resize(groups.size());
                        size_t off = 0;
                        int    n_cache = 0;

                        for (size_t g = 0; g < groups.size() && p.cache_buf != nullptr; ++g) {
                            const std::vector<int> & grp = groups[g];
                            impl::cache_pool & pool = p.pools[g];
                            pool.n_layers = (int32_t) grp.size();
                            pool.res_base = n_trans;
                            pool.res_cap  = (int32_t) grp.size() * res_per_layer;
                            pool.sentinel = pool.res_base + pool.res_cap;

                            ggml_tensor * tensors[3] = { nullptr, nullptr, nullptr };
                            for (const auto & role : roles) {
                                const ggml_tensor * s = src[grp[0]].t[role.slot];
                                ggml_tensor * ct = ggml_new_tensor_3d(p.cache_ctx, s->type, s->ne[0], s->ne[1], pool.sentinel + 1);
                                ggml_format_name(ct, "disk_cache_%s.%d", role.suffix, (int) g);
                                // pad the slot stride: an expert read starts `head` bytes
                                // before its slot to stay sector-aligned, so with adjacent
                                // slots in flight it would clobber the tail of the
                                // previous slot's expert without the pad
                                ct->nb[2] = s->nb[2] + disk_stage_align;
                                const size_t head = all_aligned ? 0 : src[grp[0]].r[role.slot].head;
                                if (ggml_backend_tensor_alloc(p.cache_buf, ct, cbase + off + head) != GGML_STATUS_SUCCESS) {
                                    LLAMA_LOG_WARN("%s: failed to bind a decode cache tensor, cache disabled\n", __func__);
                                    ggml_backend_buffer_free(p.cache_buf);
                                    p.cache_buf = nullptr;
                                    break;
                                }
                                pool.data[role.slot]        = (char *) ct->data;
                                pool.slot_stride[role.slot] = (size_t) ct->nb[2];
                                off += align_up(head + (size_t) (pool.sentinel + 1) * ct->nb[2] + disk_stage_align, disk_stage_align);
                                tensors[role.slot] = ct;
                            }
                            if (p.cache_buf == nullptr) {
                                p.cache.clear();
                                p.pools.clear();
                                break;
                            }

                            pool.gate = tensors[0];
                            pool.up   = tensors[1];
                            pool.down = tensors[2];
                            pool.free_slots.resize((size_t) pool.res_cap);
                            for (int32_t s = 0; s < pool.res_cap; ++s) {
                                pool.free_slots[(size_t) s] = pool.res_base + s;
                            }
                            for (size_t j = 0; j < grp.size(); ++j) {
                                const int il = grp[j];
                                impl::cache_layer & c = p.cache[il];
                                c.pool           = (int) g;
                                c.resident_slot.assign((size_t) n_expert, -1);
                                c.resident_filled.assign((size_t) n_expert, 0);
                                c.vram.assign((size_t) n_expert, 0);
                                c.base.assign((size_t) n_expert, 0);

                                // 2d [1, n_expert] so ggml_get_rows can index the
                                // expert id along ne[1] during the decode remap
                                ggml_tensor * tab = ggml_new_tensor_2d(p.cache_ctx, GGML_TYPE_I32, 1, n_expert);
                                ggml_format_name(tab, "disk_cache_table.%d", il);
                                ggml_backend_tensor_alloc(p.cache_buf, tab, cbase + off);
                                std::memset(tab->data, 0, (size_t) n_expert * sizeof(int32_t));
                                off += align_up((size_t) n_expert * sizeof(int32_t), disk_stage_align);

                                // slot-indexed skip table for the host mul_mat_id: 1 at
                                // the sentinel, so a VRAM-served expert (whose table
                                // entry is the sentinel) is skipped instead of read
                                ggml_tensor * skip = ggml_new_tensor_2d(p.cache_ctx, GGML_TYPE_I32, 1, pool.sentinel + 1);
                                ggml_format_name(skip, "disk_cache_skip.%d", il);
                                ggml_backend_tensor_alloc(p.cache_buf, skip, cbase + off);
                                std::memset(skip->data, 0, (size_t) (pool.sentinel + 1) * sizeof(int32_t));
                                ((int32_t *) skip->data)[pool.sentinel] = 1;
                                off += align_up((size_t) (pool.sentinel + 1) * sizeof(int32_t), disk_stage_align);

                                // split-hot: static partition of the slots. skip_hot is 1
                                // on the transients and the sentinel, skip_cold is 1 on the
                                // residents and the sentinel, so the two host passes never
                                // compute the same expert
                                ggml_tensor * skip_hot  = nullptr;
                                ggml_tensor * skip_cold = nullptr;
                                if (p.split_hot_active) {
                                    skip_hot  = ggml_new_tensor_2d(p.cache_ctx, GGML_TYPE_I32, 1, pool.sentinel + 1);
                                    skip_cold = ggml_new_tensor_2d(p.cache_ctx, GGML_TYPE_I32, 1, pool.sentinel + 1);
                                    ggml_format_name(skip_hot,  "disk_cache_skip_hot.%d",  il);
                                    ggml_format_name(skip_cold, "disk_cache_skip_cold.%d", il);
                                    ggml_backend_tensor_alloc(p.cache_buf, skip_hot, cbase + off);
                                    off += align_up((size_t) (pool.sentinel + 1) * sizeof(int32_t), disk_stage_align);
                                    ggml_backend_tensor_alloc(p.cache_buf, skip_cold, cbase + off);
                                    off += align_up((size_t) (pool.sentinel + 1) * sizeof(int32_t), disk_stage_align);
                                    int32_t * hot = (int32_t *) skip_hot->data;
                                    int32_t * cld = (int32_t *) skip_cold->data;
                                    for (int32_t s = 0; s <= pool.sentinel; ++s) {
                                        hot[s] = (s < n_trans || s >= pool.sentinel) ? 1 : 0;
                                        cld[s] = (s >= n_trans) ? 1 : 0;
                                    }
                                }

                                // cache-aware dropping: F32 [1, n_expert], 1 = keep the
                                // expert this token, 0 = dropped (cold and below the
                                // threshold). The graph multiplies the weights by
                                // get_rows(keep, selected_experts) before norm_w
                                ggml_tensor * keep = nullptr;
                                if (p.drop_fraction > 0.0f || p.substitute_enabled()) {
                                    keep = ggml_new_tensor_2d(p.cache_ctx, GGML_TYPE_F32, 1, n_expert);
                                    ggml_format_name(keep, "disk_cache_keep.%d", il);
                                    ggml_backend_tensor_alloc(p.cache_buf, keep, cbase + off);
                                    for (size_t e = 0; e < (size_t) n_expert; ++e) {
                                        ((float *) keep->data)[e] = 1.0f;
                                    }
                                    off += align_up((size_t) n_expert * sizeof(float), disk_stage_align);
                                }

                                c.table          = tab;
                                c.slot_skip      = skip;
                                c.slot_skip_hot  = skip_hot;
                                c.slot_skip_cold = skip_cold;
                                c.keep           = keep;
                                p.table_layer.emplace(tab, il);
                                c.pub.gate           = pool.gate;
                                c.pub.up             = pool.up;
                                c.pub.down           = pool.down;
                                c.pub.table          = tab;
                                c.pub.slot_skip      = skip;
                                c.pub.slot_skip_hot  = skip_hot;
                                c.pub.slot_skip_cold = skip_cold;
                                c.pub.keep           = keep;
                                c.pub.force_weights_host = p.drop_enabled();
                                n_cache++;
                            }
                        }

                        if (p.cache_buf != nullptr) {
                            LLAMA_LOG_INFO("%s: decode cache active for %d layer(s), %.2f GiB, %zu pool(s)%s\n",
                                           __func__, n_cache, cache_bytes / (1024.0 * 1024.0 * 1024.0), p.pools.size(),
                                           all_aligned ? ", shared expert tensors" : ", per-layer tensors");
                            for (size_t g = 0; g < p.pools.size(); ++g) {
                                LLAMA_LOG_INFO("%s:   pool %zu: %d layer(s), %d slots/layer, %d resident slots\n",
                                               __func__, g, p.pools[g].n_layers, p.pools[g].sentinel / p.pools[g].n_layers,
                                               p.pools[g].res_cap);
                            }
                        }

                        // the prefill staging slabs are idle during decode: reuse
                        // their memory as the L2 pool. Requires aligned data: the
                        // pool reads an expert at its slot start, so head must be
                        // zero.
                        //
                        // The pool and the staging tensors are never live at once:
                        // every prefill clears the pool before it fills the slab,
                        // and decode never reads the staging tensors. The pool can
                        // therefore lay itself out compactly inside per_buffer
                        // instead of following the prefill regions, so a bundle
                        // with cheaper experts than the region budget fits more
                        // slots
                        if (all_aligned && p.n_buf >= 2) {
                            // group the stageable layers by expert-bundle type: a
                            // slot stride is fixed per tensor, so only identical
                            // tensors can share one pool. Start with one pool per
                            // type, then let any spare slab halve the largest type,
                            // so both staging slabs hold L2 slots even when the
                            // model has a single layout
                            std::vector<std::vector<int>> l2_groups;
                            for (int il = 0; il < n_layer; ++il) {
                                if (!src[il].ok) {
                                    continue;
                                }
                                int g = -1;
                                for (size_t k = 0; k < l2_groups.size(); ++k) {
                                    const layer_src & a = src[l2_groups[k][0]];
                                    bool same = true;
                                    for (const auto & role : roles) {
                                        const ggml_tensor * ta = a.t[role.slot];
                                        const ggml_tensor * tb = src[il].t[role.slot];
                                        if (ta->type != tb->type || ta->ne[0] != tb->ne[0] || ta->ne[1] != tb->ne[1]) {
                                            same = false;
                                            break;
                                        }
                                    }
                                    if (same) {
                                        g = (int) k;
                                        break;
                                    }
                                }
                                if (g < 0) {
                                    l2_groups.push_back({});
                                    g = (int) l2_groups.size() - 1;
                                }
                                l2_groups[g].push_back(il);
                            }

                            // more bundle types than staging slabs: pool the
                            // types that cover the most layers, the remaining
                            // layers keep streaming (a prefill clears the pools
                            // anyway, so a partial pool is safe)
                            if ((int) l2_groups.size() > p.n_buf) {
                                std::stable_sort(l2_groups.begin(), l2_groups.end(),
                                                 [](const std::vector<int> & a, const std::vector<int> & b) {
                                                     return a.size() > b.size();
                                                 });
                                l2_groups.resize((size_t) p.n_buf);
                            }

                            while ((int) l2_groups.size() < p.n_buf) {
                                size_t best = l2_groups.size();
                                for (size_t k = 0; k < l2_groups.size(); ++k) {
                                    if (l2_groups[k].size() < 2) {
                                        continue;
                                    }
                                    if (best == l2_groups.size() || l2_groups[k].size() > l2_groups[best].size()) {
                                        best = k;
                                    }
                                }
                                if (best == l2_groups.size()) {
                                    break;  // no type has a layer to spare
                                }
                                const size_t mid = l2_groups[best].size() / 2;
                                std::vector<int> half(l2_groups[best].begin() + mid, l2_groups[best].end());
                                l2_groups[best].resize(mid);
                                l2_groups.push_back(std::move(half));
                            }

                            p.evict_pool_id.assign(n_layer, -1);

                            for (size_t g = 0; g < l2_groups.size(); ++g) {
                                const int il  = l2_groups[g][0];
                                const int pid = (int) p.evict_pools.size();

                                size_t per_slot = 0;
                                int    n_role   = 0;
                                for (const auto & role : roles) {
                                    const size_t stride = src[il].r[role.slot].stride;
                                    if (stride > 0) {
                                        per_slot += align_up(stride, disk_stage_align);
                                        n_role++;
                                    }
                                }
                                if (per_slot == 0 || per_buffer <= (size_t) n_role * disk_stage_align) {
                                    continue;  // cannot size this type; leave the L2 off
                                }
                                const int32_t cap = (int32_t) ((per_buffer - (size_t) n_role * disk_stage_align) / per_slot);
                                if (cap <= 0) {
                                    continue;
                                }

                                impl::evict_pool ep;
                                ep.cap = cap;
                                // protected gets three quarters: one-shot misses
                                // enter probation and cannot displace a re-read
                                ep.prot_cap = cap - std::max(1, cap / 4);
                                ep.ghost_cap = (size_t) std::max(16, cap * 2);
                                size_t off = 0;
                                for (const auto & role : roles) {
                                    const size_t stride = src[il].r[role.slot].stride;
                                    if (stride == 0) {
                                        continue;
                                    }
                                    ep.data[role.slot]   = p.base + (size_t) pid * per_buffer + off;
                                    ep.stride[role.slot] = align_up(stride, disk_stage_align);
                                    off += (size_t) ep.cap * ep.stride[role.slot] + disk_stage_align;
                                }
                                ep.slot_key.assign((size_t) ep.cap, -1);
                                ep.iter_of.resize((size_t) ep.cap);
                                ep.seg.assign((size_t) ep.cap, 0);
                                ep.free_slots.resize((size_t) ep.cap);
                                for (int32_t s = 0; s < ep.cap; ++s) {
                                    ep.free_slots[(size_t) s] = s;
                                }
                                p.evict_pools.push_back(std::move(ep));

                                for (int jl : l2_groups[g]) {
                                    p.evict_pool_id[jl] = pid;
                                }
                            }

                            if (p.evict_pools.empty()) {
                                LLAMA_LOG_WARN("%s: no expert-bundle type fits a staging buffer, "
                                               "L2 eviction pool disabled\n", __func__);
                            } else {
                                int n_pooled = 0;
                                for (int il = 0; il < n_layer; ++il) {
                                    if (src[il].ok && p.evict_pool_id[il] >= 0) {
                                        n_pooled++;
                                    }
                                }
                                if (n_pooled < n_staged) {
                                    LLAMA_LOG_INFO("%s: L2 eviction pool covers %d of %d staged layers, "
                                                   "the other %d stream from disk\n",
                                                   __func__, n_pooled, n_staged, n_staged - n_pooled);
                                }
                                for (size_t k = 0; k < p.evict_pools.size(); ++k) {
                                    LLAMA_LOG_INFO("%s:   L2 pool %zu: %d slots, strides %zu/%zu/%zu bytes\n",
                                                   __func__, k, p.evict_pools[k].cap,
                                                   p.evict_pools[k].stride[0], p.evict_pools[k].stride[1],
                                                   p.evict_pools[k].stride[2]);
                                }
                                LLAMA_LOG_INFO("%s: L2 admission filter on (ghost %zu slots/pool)\n",
                                               __func__, p.evict_pools[0].ghost_cap);
                            }
                        }
                    }
                }
            } else {
                LLAMA_LOG_WARN("%s: decode cache budget too small for a decode cache\n", __func__);
            }

            if (!p.base_set.empty()) {
                if (p.cache_buf == nullptr || p.cache.empty()) {
                    throw std::runtime_error("disk stage: base-expert set requested but the decode cache is unavailable");
                }
                this->preload_base();
            }
            if (!p.warm_set.empty()) {
                if (p.cache_buf == nullptr || p.cache.empty()) {
                    throw std::runtime_error("disk stage: warm-expert set requested but the decode cache is unavailable");
                }
                this->preload_warm();
            }

            if (!p.base_set.empty() || !p.warm_set.empty()) {
                size_t n_cap  = 0;
                size_t n_free = 0;
                for (const auto & pool : p.pools) {
                    n_cap  += (size_t) pool.res_cap;
                    n_free += pool.free_slots.size();
                }
                size_t n_base = 0;
                if (p.base_target != nullptr) {
                    for (const auto & v : *p.base_target) {
                        n_base += v.size();
                    }
                }
                size_t n_warm = 0;
                for (const auto & v : p.warm_loaded) {
                    n_warm += v.size();
                }
                LLAMA_LOG_INFO("%s: decode cache prefilled: %zu/%zu resident slots at startup (%zu free, %zu base, %zu warm)\n",
                               __func__, n_cap - n_free, n_cap, n_free, n_base, n_warm);
            }
        }
    }

    if (p.n_buf > 1) {
        p.reader = std::thread([&p, this]() {
            for (;;) {
                int32_t il = -1;
                {
                    std::unique_lock<std::mutex> lk(p.pipe_mu);
                    p.pipe_req_cv.wait(lk, [&p] { return p.pipe_stop || p.pipe_req != -1; });
                    if (p.pipe_stop) {
                        return;
                    }
                    il = p.pipe_req;
                    p.pipe_req = -1;
                }

                std::string err;
                try {
                    this->fill_run(il);
                } catch (const std::exception & e) {
                    err = e.what();
                } catch (...) {
                    err = "disk stage: unbuffered read failed";
                }

                {
                    std::lock_guard<std::mutex> lk(p.pipe_mu);
                    p.pipe_done = il;
                    if (!err.empty()) {
                        p.pipe_error = err;
                    }
                }
                p.pipe_done_cv.notify_all();
            }
        });
    }

    if (p.iocp_dec != nullptr && !p.cache.empty()) {
        p.dec_io_active = true;
        p.dec_io = std::thread([&p]() {
            for (;;) {
                impl::dec_batch batch;
                {
                    std::unique_lock<std::mutex> lk(p.dec_io_mu);
                    p.dec_io_cv.wait(lk, [&p] { return p.dec_io_stop || p.dec_io_ready; });
                    if (p.dec_io_stop) {
                        return;
                    }
                    batch = std::move(p.dec_io_batch);
                    p.dec_io_ready      = false;
                    p.dec_io_running    = true;
                    p.dec_io_reads_done = false;
                }

                bool err = false;
                // the hit copies land in transient slots the hot pass skips, so
                // they run on a helper while the drive is busy
                std::thread copier;
                try {
                    size_t copy_bytes = 0;
                    for (const impl::pool_copy & cp : batch.copies) {
                        copy_bytes += cp.len;
                    }
                    if (copy_bytes >= disk_stage_copy_thread_min) {
                        try {
                            copier = std::thread([&batch]() {
                                for (const impl::pool_copy & cp : batch.copies) {
                                    std::memcpy(cp.dst, cp.src, cp.len);
                                }
                            });
                        } catch (...) {
                            // no thread: the copies run inline below
                        }
                    }
                    if (!batch.jobs.empty()) {
                        std::lock_guard<std::mutex> io(p.dec_io_disk_mu);
                        p.trace_split.rd0 = ggml_time_us();
                        disk_stage_run_jobs(batch.jobs, 32, p.iocp_dec);
                        p.trace_split.rd1 = ggml_time_us();
                    }
                } catch (...) {
                    err = true;
                }
                // join before the batch's descriptors go away, even on a read
                // failure, so a stray helper cannot write the next batch's slots
                if (copier.joinable()) {
                    copier.join();
                } else {
                    for (const impl::pool_copy & cp : batch.copies) {
                        std::memcpy(cp.dst, cp.src, cp.len);
                    }
                }

                // the read phase is over: the graph may compute the transient
                // slots while the stores keep filling the L2 pool. The next fill
                // drains them before it reuses the window
                {
                    std::lock_guard<std::mutex> lk(p.dec_io_mu);
                    p.dec_io_error      = err;
                    p.dec_io_reads_done = true;
                }
                p.dec_io_reads_cv.notify_all();

                if (!err) {
                    for (const impl::pool_copy & cp : batch.stores) {
                        p.store_l2(cp);
                    }
                    for (const impl::pool_copy & cp : batch.res_fills) {
                        std::memcpy(cp.dst, cp.src, cp.len);
                    }
                }

                {
                    std::lock_guard<std::mutex> lk(p.dec_io_mu);
                    p.dec_io_running = false;
                }
                p.dec_io_done_cv.notify_all();
            }
        });
        LLAMA_LOG_INFO("%s: decode I/O worker enabled, separate read queue, split-hot %s\n",
                       __func__, p.split_hot_active ? "on" : "off");
    }

    p.active = true;
    LLAMA_LOG_INFO("%s: disk staging active for %d layer(s), %.1f MiB pool, %d buffer(s), %zu-byte aligned unbuffered reads\n",
                   __func__, n_staged, total / (1024.0 * 1024.0), p.n_buf, disk_stage_align);

#endif
}

void llama_disk_stage::preload_base() {
    const size_t n = fill_base_experts();
    LLAMA_LOG_INFO("%s: base-expert set: %zu expert(s) resident in the decode cache\n", __func__, n);
}

// read the base experts whose bytes are not in the cache yet into their slots
size_t llama_disk_stage::fill_base_experts() {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (p.cache.empty()) {
        return 0;
    }
    const std::vector<std::vector<int32_t>> * target = p.base_target;
    if (target == nullptr) {
        return 0;
    }

    std::vector<disk_stage_job>          jobs;
    std::vector<std::pair<int, int32_t>> filled;
    size_t                               bytes      = 0;
    size_t                               n_resident = 0;

    // only called at load, before any graph runs; the cache lock still keeps a
    // concurrent VRAM upload from reusing a slot while the batch writes it
    std::lock_guard<std::mutex> cache_lock(p.cache_mu);
    for (int il = 0; il < (int) p.cache.size() && il < (int) target->size(); ++il) {
        impl::cache_layer & c = p.cache[(size_t) il];
        if (c.table == nullptr) {
            continue;
        }
        impl::cache_pool &                pool    = p.pools[(size_t) c.pool];
        const std::vector<impl::region> & regions = p.layer_regions[(size_t) il];

        for (const int32_t id : (*target)[(size_t) il]) {
            if (id < 0 || id >= (int32_t) c.resident_slot.size()) {
                continue;
            }
            if (c.vram[(size_t) id] != 0) {
                n_resident++;
                continue;  // served from VRAM, no decode-cache slot needed
            }
            if (c.resident_slot[(size_t) id] < 0) {
                // no slot yet: the fit check in the constructor makes this
                // unreachable at load, but a mode switch that ran out of
                // slots can leave the target partly unadmitted
                if (pool.free_slots.empty()) {
                    break;
                }
                const int32_t slot = pool.free_slots.back();
                pool.free_slots.pop_back();
                c.resident_slot[(size_t) id]   = slot;
                c.resident_filled[(size_t) id] = 0;
            } else if (c.resident_filled[(size_t) id] != 0) {
                n_resident++;
                continue;  // already cached
            }
            c.base[(size_t) id] = 1;

            for (int r = 0; r < 3; ++r) {
                const impl::region & sr = regions[(size_t) r];
                if (sr.file == nullptr || sr.stride == 0) {
                    continue;
                }
                const size_t read_len = align_up(sr.head + sr.stride, disk_stage_align);
                char *       dst      = pool.data[r] + (size_t) c.resident_slot[(size_t) id] * pool.slot_stride[r] - sr.head;
                jobs.push_back({ sr.file->h, dst, sr.file_off + (size_t) id * sr.stride, read_len });
                bytes += read_len;
            }
            filled.emplace_back(il, id);
            n_resident++;
        }
    }

    if (!jobs.empty()) {
        // io_mu only serializes the staging reads (the decode reads run on their
        // own completion port), and one deep batch is what the read queue wants
        std::lock_guard<std::mutex> io(p.io_mu);
        bool                        ok = false;
        try {
            disk_stage_run_jobs(jobs, 32, p.iocp);
            ok = true;
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("%s: base-expert read failed: %s\n", __func__, e.what());
        }
        if (ok) {
            for (const auto & [il, id] : filled) {
                p.cache[(size_t) il].resident_filled[(size_t) id] = 1;
            }
        }
        LLAMA_LOG_INFO("%s: loaded %zu base expert(s), %.1f MiB into the decode cache\n",
                       __func__, filled.size(), bytes / (1024.0 * 1024.0));
    }

    return n_resident;
#else
    return 0;
#endif
}

void llama_disk_stage::preload_warm() {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (p.warm_set.empty() || p.cache.empty()) {
        return;
    }

    p.warm_loaded.assign(p.warm_set.size(), {});

    std::vector<disk_stage_job>          jobs;
    std::vector<std::pair<int, int32_t>> filled;
    size_t                               bytes = 0;

    // one pool at a time: its free slots are handed out round-robin across the
    // layers that share it, so a layer with few candidates cannot starve the
    // others and every layer ends up as full as the candidate list allows
    for (size_t pid = 0; pid < p.pools.size(); ++pid) {
        impl::cache_pool & pool = p.pools[pid];

        std::vector<int>                  pl_layers;
        std::vector<std::vector<int32_t>> cand;
        for (int il = 0; il < (int) p.cache.size(); ++il) {
            if (p.cache[(size_t) il].table == nullptr || p.cache[(size_t) il].pool != (int) pid) {
                continue;
            }
            pl_layers.push_back(il);
            cand.emplace_back(p.warm_set[(size_t) il].begin(), p.warm_set[(size_t) il].end());
        }
        if (pl_layers.empty()) {
            continue;
        }

        std::vector<size_t> cursor(pl_layers.size(), 0);
        bool                progress = true;
        while (!pool.free_slots.empty() && progress) {
            progress = false;
            for (size_t i = 0; i < pl_layers.size() && !pool.free_slots.empty(); ++i) {
                if (cursor[i] >= cand[i].size()) {
                    continue;
                }
                const int     il = pl_layers[i];
                const int32_t id = cand[i][cursor[i]++];

                impl::cache_layer & c = p.cache[(size_t) il];
                if (c.resident_slot[(size_t) id] >= 0) {
                    continue;  // already resident (base); the parse dropped those
                }
                const int32_t slot = pool.free_slots.back();
                pool.free_slots.pop_back();
                c.resident_slot[(size_t) id]   = slot;
                c.resident_filled[(size_t) id] = 0;

                const std::vector<impl::region> & regions = p.layer_regions[(size_t) il];
                for (int r = 0; r < 3; ++r) {
                    const impl::region & sr = regions[(size_t) r];
                    if (sr.file == nullptr || sr.stride == 0) {
                        continue;
                    }
                    const size_t read_len = align_up(sr.head + sr.stride, disk_stage_align);
                    char *       dst      = pool.data[r] + (size_t) slot * pool.slot_stride[r] - sr.head;
                    jobs.push_back({ sr.file->h, dst, sr.file_off + (size_t) id * sr.stride, read_len });
                    bytes += read_len;
                }
                filled.emplace_back(il, id);
                p.warm_loaded[(size_t) il].push_back(id);
                progress = true;
            }
        }
    }

    if (!jobs.empty()) {
        std::lock_guard<std::mutex> io(p.io_mu);
        disk_stage_run_jobs(jobs, 32, p.iocp);
        for (const auto & [il, id] : filled) {
            p.cache[(size_t) il].resident_filled[(size_t) id] = 1;
        }
    }

    LLAMA_LOG_INFO("%s: warm-expert set: %zu expert(s), %.1f MiB preloaded (count 0, evictable)\n",
                   __func__, filled.size(), bytes / (1024.0 * 1024.0));
#else
#endif
}

llama_disk_stage::~llama_disk_stage() {
#if defined(_WIN32)
    if (pimpl) {
        print_calibration();
        {
            std::lock_guard<std::mutex> lk(pimpl->pipe_mu);
            pimpl->pipe_stop = true;
        }
        pimpl->pipe_req_cv.notify_all();
        if (pimpl->reader.joinable()) {
            pimpl->reader.join();
        }
        {
            std::lock_guard<std::mutex> lk(pimpl->dec_io_mu);
            pimpl->dec_io_stop = true;
        }
        pimpl->dec_io_cv.notify_all();
        pimpl->dec_io_done_cv.notify_all();
        pimpl->dec_io_reads_cv.notify_all();
        if (pimpl->dec_io.joinable()) {
            pimpl->dec_io.join();
        }
        if (pimpl->iocp_dec != nullptr) {
            CloseHandle(pimpl->iocp_dec);
            pimpl->iocp_dec = nullptr;
        }
        if (pimpl->iocp != nullptr) {
            CloseHandle(pimpl->iocp);
            pimpl->iocp = nullptr;
        }
    }
#endif
}

bool llama_disk_stage::split_hot() const {
    return pimpl->split_hot_active;
}

void llama_disk_stage::set_split_cold(int il, bool cold) const {
#if defined(_WIN32)
    if (il < 0 || il >= (int) pimpl->split_cold.size()) {
        return;
    }
    pimpl->split_cold[(size_t) il] = cold ? 1 : 0;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(cold);
#endif
}

bool llama_disk_stage::split_cold(int il) const {
#if defined(_WIN32)
    return il >= 0 && il < (int) pimpl->split_cold.size() && pimpl->split_cold[(size_t) il] != 0;
#else
    GGML_UNUSED(il);
    return false;
#endif
}

bool llama_disk_stage::split_trace_once() {
#if defined(_WIN32)
    static const bool trace = disk_stage_trace();
    static bool done = false;
    if (!trace || done) {
        return false;
    }
    done = true;
    return true;
#else
    return false;
#endif
}

void llama_disk_stage::fill(int il) {
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.layer_regions.size() || p.layer_regions[il].empty()) {
        return;
    }

#if defined(_WIN32)
    if (p.evict_populated) {
        // the staging slabs hold the L2 pool, so wait out any decode store still
        // writing them before the prefill clears the pool
        dec_io_drain();
        std::lock_guard<std::mutex> lock(p.cache_mu);
        p.evict_clear();
        p.evict_populated = false;
    }

    if (p.n_buf < 2) {
        // single staging buffer: no read-ahead possible, fill synchronously
        fill_run(il);
        return;
    }

    // wait for this layer's read, which the pipeline started at the previous
    // layer, then start the read for the next stageable layer so it overlaps
    // with this layer's compute
    {
        std::unique_lock<std::mutex> lk(p.pipe_mu);
        if (p.pipe_issued != il) {
            p.pipe_req    = il;
            p.pipe_issued = il;
            p.pipe_req_cv.notify_one();
        }
        p.pipe_done_cv.wait(lk, [&p, il] { return p.pipe_done == il || !p.pipe_error.empty(); });
        const bool        failed = !p.pipe_error.empty();
        const std::string err    = p.pipe_error;
        p.pipe_done = -1;
        if (failed) {
            throw std::runtime_error(err);
        }
    }

    const int32_t nxt = p.next_stage[il];
    if (nxt >= 0) {
        std::lock_guard<std::mutex> lk(p.pipe_mu);
        p.pipe_req    = nxt;
        p.pipe_issued = nxt;
        p.pipe_req_cv.notify_one();
    }
#else
    GGML_UNUSED(il);
#endif
}

#if defined(_WIN32)
// cpu time of the CALLING thread, to tell a slow read from a descheduled thread
static int64_t disk_stage_thread_cpu_us() {
    FILETIME c, e, k, u;
    if (!GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u)) {
        return 0;
    }
    ULARGE_INTEGER kk, uu;
    kk.LowPart = k.dwLowDateTime; kk.HighPart = k.dwHighDateTime;
    uu.LowPart = u.dwLowDateTime; uu.HighPart = u.dwHighDateTime;
    return (int64_t) ((kk.QuadPart + uu.QuadPart) / 10);
}
#else
static int64_t disk_stage_thread_cpu_us() { return 0; }
#endif

void llama_disk_stage::fill_run(int il, const uint8_t * used) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (il < 0 || il >= (int) p.layer_regions.size() || p.layer_regions[il].empty()) {
        return;
    }

    // the queue depth that saturated the drive in the diskspd measurement
    const int queue_depth = 32;

    struct cache_copy {
        const char * src;
        char *       dst;
        size_t       len;
    };

    std::vector<disk_stage_job> jobs;
    std::vector<cache_copy>     copies;  // whole residents, served from the cache
    std::vector<cache_copy>     patches; // the bytes an aligned read overruns into a neighbour

    const int64_t cpu0 = disk_stage_thread_cpu_us();

    {
        // the resident set only moves on a single-token ubatch, so it is stable
        // for a whole prefill; the lock also keeps a slot from being reused
        // while the reads below are in flight
        std::lock_guard<std::mutex> lock(p.cache_mu);

        const impl::cache_layer * c    = nullptr;
        const impl::cache_pool *  pool = nullptr;
        if (il < (int) p.cache.size() && p.cache[il].table != nullptr) {
            c    = &p.cache[il];
            pool = &p.pools[c->pool];
        }

        for (int role = 0; role < (int) p.layer_regions[il].size(); ++role) {
            const impl::region & r = p.layer_regions[il][role];

            const int32_t n_expert = c != nullptr ? (int32_t) c->resident_slot.size() : r.n_expert;
            const size_t  stride   = (c != nullptr || used != nullptr) ? r.stride : 0;

            if (stride == 0 || n_expert <= 0) {
                // no decode cache for this layer: read the whole region
                char * dst = p.base + r.pool_off;
                for (size_t done = 0; done < r.read_len; done += disk_stage_chunk) {
                    const size_t len = std::min(disk_stage_chunk, r.read_len - done);
                    jobs.push_back({ r.file->h, dst + done, r.file_off + done, len });
                }
                continue;
            }

            // a resident expert already sits in the cache's CPU memory, so copy
            // it instead of reading it again. An expert served by the VRAM cache
            // has no RAM slot (vram_commit freed it), so it is read like any
            // non-resident. A promoted-but-unread resident must be read too
            auto resident = [&](int32_t id) {
                return c != nullptr && c->resident_slot[id] >= 0 && c->resident_filled[id] != 0 && c->vram[id] == 0;
            };
            // sparse fill reads only the routed experts, the whole slab reads all
            auto selected = [&](int32_t id) {
                return used == nullptr || used[id] != 0;
            };

            char * slab = p.base + r.pool_off + r.head;

            int32_t id = 0;
            while (id < n_expert) {
                if (!selected(id)) {
                    ++id;
                    continue;
                }
                if (resident(id)) {
                    const size_t slot = (size_t) c->resident_slot[id];
                    copies.push_back({ pool->data[role] + slot * pool->slot_stride[role],
                                       slab + (size_t) id * stride, stride });
                    ++id;
                    continue;
                }

                int32_t last = id;
                while (last < n_expert && selected(last) && !resident(last)) {
                    ++last;
                }

                // one aligned read covers the whole run. It starts where the
                // previous resident ended, so the destination carries the same
                // alignment prefix the whole-region read had
                const size_t start  = r.file_off + r.head + (size_t) id * stride;
                const size_t aoff   = start & ~(disk_stage_align - 1);
                const size_t prefix = start - aoff;
                const size_t rlen   = align_up(prefix + (size_t) (last - id) * stride, disk_stage_align);
                char *       dst    = slab + (size_t) id * stride - prefix;

                for (size_t done = 0; done < rlen; done += disk_stage_chunk) {
                    const size_t len = std::min(disk_stage_chunk, rlen - done);
                    jobs.push_back({ r.file->h, dst + done, aoff + done, len });
                }

                // The aligned read starts `prefix` bytes before the run and ends up
                // to one sector past it, so it overruns the tail of the preceding
                // resident and the head of the following one. Restore both from the
                // cache once the reads are done: that is what frees the resident
                // copies to run while the drive is busy instead of after it.
                if (prefix > 0 && id > 0 && resident(id - 1)) {
                    const size_t slot = (size_t) c->resident_slot[id - 1];
                    patches.push_back({ pool->data[role] + slot * pool->slot_stride[role] + (stride - prefix),
                                        slab + (size_t) id * stride - prefix, prefix });
                }
                if (last < n_expert && resident(last)) {
                    const size_t over = (size_t) ((dst + rlen) - (slab + (size_t) last * stride));
                    if (over > 0 && over <= stride) {
                        const size_t slot = (size_t) c->resident_slot[last];
                        patches.push_back({ pool->data[role] + slot * pool->slot_stride[role],
                                            slab + (size_t) last * stride, over });
                    }
                }
                id = last;
            }
        }

        size_t disk_bytes = 0;
        for (const disk_stage_job & j : jobs) {
            disk_bytes += j.len;
        }

        size_t copy_bytes = 0;
        for (const cache_copy & cp : copies) {
            copy_bytes += cp.len;
        }
        size_t patch_bytes = 0;
        for (const cache_copy & cp : patches) {
            patch_bytes += cp.len;
        }

        // every read now writes bytes the copies do not own (the overruns into
        // the neighbours are patched below), so the copies can run on a second
        // thread while the drive is busy
        std::thread copier;
        if (!copies.empty()) {
            try {
                copier = std::thread([&copies]() {
                    for (const cache_copy & cp : copies) {
                        std::memcpy(cp.dst, cp.src, cp.len);
                    }
                });
            } catch (...) {
                // no thread: the copies are done inline below
            }
        }

        const int64_t t1 = ggml_time_us();
        try {
            // the pin worker reads through the same completion port, so only one
            // batch may be reaped at a time
            std::lock_guard<std::mutex> io(p.io_mu);
            disk_stage_run_jobs(jobs, queue_depth, p.iocp);
        } catch (...) {
            if (copier.joinable()) {
                copier.join();
            }
            throw;
        }
        const int64_t t2 = ggml_time_us();

        if (copier.joinable()) {
            copier.join();
        } else {
            for (const cache_copy & cp : copies) {
                std::memcpy(cp.dst, cp.src, cp.len);
            }
        }

        // last, so it is final: restore the bytes the aligned reads overran
        for (const cache_copy & cp : patches) {
            std::memcpy(cp.dst, cp.src, cp.len);
        }

        if (disk_stage_trace()) {
            const double s_read = (t2 - t1) / 1e6;
            const double s_all  = (ggml_time_us() - t1) / 1e6;
            LLAMA_LOG_INFO("%s: layer %d fill: %.1f MiB read in %.1f ms (cpu %.1f ms) = %.2f GB/s + %.1f MiB copied + %.1f MiB patched, total %.1f ms\n",
                           __func__, il, disk_bytes / (1024.0 * 1024.0), s_read * 1000.0,
                           (disk_stage_thread_cpu_us() - cpu0) / 1000.0,
                           s_read > 0 ? disk_bytes / (s_read * 1e9) : 0.0,
                           copy_bytes / (1024.0 * 1024.0), patch_bytes / (1024.0 * 1024.0), s_all * 1000.0);
        }
    }
#else
    GGML_UNUSED(il);
#endif
}

void llama_disk_stage::fill_selected(int il, const int32_t * ids, int64_t n_ids) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.layer_regions.size() || p.layer_regions[il].empty() || n_ids <= 0) {
        return;
    }

    const int32_t n_expert = p.layer_regions[il][0].n_expert;
    if (n_expert <= 0) {
        return;
    }

    // the routed experts; the rest of the slab keeps stale data and is never
    // read, the CPU mul_mat_id and the VRAM copy both touch only these ids
    std::vector<uint8_t> used((size_t) n_expert, 0);
    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t id = ids[i];
        if (id >= 0 && id < n_expert) {
            used[(size_t) id] = 1;
        }
    }

    // a prefill clobbers the L2 pool decode reuses the staging slabs for
    if (p.evict_populated) {
        dec_io_drain();
        std::lock_guard<std::mutex> lock(p.cache_mu);
        p.evict_clear();
        p.evict_populated = false;
    }

    fill_run(il, used.data());
#else
    GGML_UNUSED(il);
    GGML_UNUSED(ids);
    GGML_UNUSED(n_ids);
#endif
}

bool llama_disk_stage::fill_cache_plan(int il, const int32_t * ids, int64_t n_ids, const float * probs) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (il < 0 || il >= (int) p.cache.size()) {
        return false;
    }
    impl::cache_layer & c = p.cache[il];
    if (c.table == nullptr || n_ids <= 0) {
        return false;
    }
    impl::cache_pool & pool = p.pools[c.pool];

    int32_t * table = (int32_t *) c.table->data;

    std::vector<disk_stage_job> & jobs            = p.fs_jobs;
    std::vector<int32_t>        & newly_filled    = p.fs_newly_filled;
    std::vector<impl::pool_copy>& copies          = p.fs_copies;
    std::vector<impl::pool_copy>& copies_pre      = p.fs_copies_pre;
    std::vector<impl::pool_copy>& stores          = p.fs_stores;
    std::vector<impl::pool_copy>& res_fills       = p.fs_res_fills;
    std::vector<impl::l2_pending>  & l2_pending   = p.fs_l2_pending;
    std::vector<impl::res_pending> & res_pending  = p.fs_res_pending;
    std::vector<std::pair<int, int64_t>> & l2_consume = p.fs_l2_consume;
    jobs.clear();
    newly_filled.clear();
    copies.clear();
    copies_pre.clear();
    stores.clear();
    res_fills.clear();
    l2_pending.clear();
    res_pending.clear();
    l2_consume.clear();

    const std::vector<impl::region> & regions = p.layer_regions[il];
    const int epid = p.evict_pool_for(il);

    // the decode worker owns its own read queue, so it reads through the decode
    // handle when one exists
    const auto job_handle = [&p](const impl::region & r) {
        if (r.file == nullptr) {
            return INVALID_HANDLE_VALUE;
        }
        return r.file->h_dec != INVALID_HANDLE_VALUE ? r.file->h_dec : r.file->h;
    };

    // per-expert aligned read length per role (region.read_len is the whole-slab
    // length, one expert is shorter)
    size_t expert_read_len[3] = { 0, 0, 0 };
    for (int r = 0; r < 3; ++r) {
        if (regions[r].file != nullptr && regions[r].stride != 0) {
            expert_read_len[r] = align_up(regions[r].head + regions[r].stride, disk_stage_align);
        }
    }

    // read the rows of expert id into transient slot use: source starts `head`
    // bytes before the expert data so the aligned read lands the row where the
    // tensor expects it, and the length is rounded up to the sector size the
    // unbuffered read requires (the extra bytes fall in the per-tensor slack)
    const auto read_to_transient = [&](int32_t id, int32_t use) {
        for (int r = 0; r < 3; ++r) {
            const impl::region & sr = regions[r];
            if (sr.file == nullptr || sr.stride == 0) {
                continue;
            }
            char * dst = pool.data[r] + (size_t) use * pool.slot_stride[r] - sr.head;
            jobs.push_back({ job_handle(sr), dst, sr.file_off + (size_t) id * sr.stride, expert_read_len[r] });
        }
    };

    // the pool keeps one transient window: only the layer being computed holds
    // routed-but-not-resident experts
    const int32_t trans_begin = 0;
    const int32_t trans_end   = p.n_trans;
    int32_t transient = trans_begin;

    // routed ids served from RAM / read from disk / skipped (VRAM), for the trace:
    // a layer with cold ids and no residents has no compute to hide the read behind
    p.trace_n_res  = 0;
    p.trace_n_cold = 0;
    p.trace_n_vram = 0;

    // slot assignment under the lock; the unbuffered reads below run without it
    {
        std::lock_guard<std::mutex> lock(p.cache_mu);

        // cache-aware opportunistic dropping: mark the cold, weak experts to
        // skip. The fraction cap and the relative floor need all scores at once,
        // so the decision is a pre-pass over the (small) routed set; the main
        // loop only honors it. The probe records the distribution
        std::vector<uint8_t> & drop        = p.fs_drop;
        std::vector<uint8_t> & subst       = p.fs_subst;
        std::vector<int32_t> & subst_slot  = p.fs_subst_slot;
        std::vector<float>   & subst_scale = p.fs_subst_scale;
        std::vector<int32_t> & subst_used  = p.fs_subst_used;
        bool drop_planned = false;
        if (p.drop_enabled() && probs != nullptr && p.resident_warm()) {
            drop.assign((size_t) n_ids, 0);
            drop_planned = true;
            // the layer's highest routed score: the reference the relative floor
            // is taken against, over warm and cold experts alike
            float max_score = 0.0f;
            for (int64_t i = 0; i < n_ids; ++i) {
                if (probs[i] > max_score) {
                    max_score = probs[i];
                }
            }
            const float floor_score = p.drop_below_rel * max_score;

            // candidates: cold experts below the relative floor. The highest
            // score is never below A * max, so the top expert is always kept
            std::vector<int64_t> & cold = p.fs_cold;
            uint32_t n_cold_all = 0;
            double   layer_mass = 0.0;
            cold.clear();
            for (int64_t i = 0; i < n_ids; ++i) {
                const int32_t id = ids[i];
                if (id < 0 || id >= (int32_t) c.resident_slot.size()) {
                    continue;
                }
                // served without a disk read: VRAM, filled RAM resident, L2 pool,
                // or a permanent base resident. A base expert is never dropped
                const bool available = c.vram[id] || c.base[id] ||
                        (c.resident_slot[id] >= 0 && c.resident_filled[id] != 0) ||
                        (epid >= 0 && p.evict_has(epid, ((int64_t) il << 32) | (uint32_t) id));
                p.n_routed_mass += (double) probs[i];
                layer_mass      += (double) probs[i];
                if (p.drop_probe) {
                    p.probe_routed++;
                    if (!available) {
                        p.probe_cold_n++;
                        if (max_score > 0.0f) {
                            const size_t rb = (size_t) p.probe_rel_bin(probs[i] / max_score);
                            p.probe_rel[rb]++;
                            p.probe_rel_mass[rb] += (double) probs[i];
                            p.probe_rel_n++;
                        }
                    }
                }
                if (!available) {
                    if (p.drop_probe) {
                        p.n_cold_mass += (double) probs[i];
                    }
                    n_cold_all++;
                    if (probs[i] < floor_score) {
                        cold.push_back(i);
                    }
                }
            }
            if (p.drop_probe) {
                p.probe_cold_hist[std::min<size_t>(n_cold_all, 32)]++;
                p.probe_cold_max = std::max<uint64_t>(p.probe_cold_max, n_cold_all);
                p.probe_calls++;
            }
            // a pool of the highest scoring non-routed experts: the substitution
            // source, and what the probe simulation measures. The router already
            // scores every expert, so the pool costs no extra compute
            std::vector<std::pair<float,int32_t>> & sppool = p.fs_pool;
            if ((p.drop_probe || p.substitute_enabled()) &&
                    p.fs_n_all > 0 && p.fs_n_all <= (int64_t) p.fs_all.size()) {
                sppool.clear();
                for (int64_t e = 0; e < p.fs_n_all; ++e) {
                    bool selected = false;
                    for (int64_t i = 0; i < n_ids; ++i) {
                        if ((int32_t) e == ids[i]) {
                            selected = true;
                            break;
                        }
                    }
                    if (!selected) {
                        sppool.emplace_back(p.fs_all[(size_t) e], (int32_t) e);
                    }
                }
                const auto by_score = [](const std::pair<float,int32_t> & a, const std::pair<float,int32_t> & b) {
                    return a.first > b.first;
                };
                const size_t p_max = (size_t) std::max(8, (int) p.substitute_pool);
                if (sppool.size() > p_max) {
                    std::partial_sort(sppool.begin(), sppool.begin() + p_max, sppool.end(), by_score);
                    sppool.resize(p_max);
                } else {
                    std::sort(sppool.begin(), sppool.end(), by_score);
                }
            } else {
                sppool.clear();
            }
            // substitution simulation: over the layer's cold selections, does a
            // resident spare from just outside the top-k come close enough in
            // score to stand in? Accounting only, the routing is untouched
            if (p.drop_probe) {
                static const int   sim_p[impl::sim_n_p] = { 2, 4, 8 };
                static const float sim_s[impl::sim_n_s] = { 0.50f, 0.70f, 0.80f, 0.90f, 0.95f };
                const auto spare_available = [&](int32_t sid) {
                    // the table+keep substitution can only point at a host
                    // resident slot, so vram and l2-only spares do not count
                    return sid >= 0 && sid < (int32_t) c.resident_slot.size() &&
                            c.resident_slot[sid] >= 0 && c.resident_filled[sid] != 0;
                };
                for (int64_t i = 0; i < n_ids; ++i) {
                    const int32_t id = ids[i];
                    if (id < 0 || id >= (int32_t) c.resident_slot.size()) {
                        continue;
                    }
                    const bool is_cold = !(c.vram[id] || c.base[id] ||
                            (c.resident_slot[id] >= 0 && c.resident_filled[id] != 0) ||
                            (epid >= 0 && p.evict_has(epid, ((int64_t) il << 32) | (uint32_t) id)));
                    if (!is_cold || !(probs[i] > 0.0f)) {
                        continue;
                    }
                    p.sim_cold++;
                    const float cs = probs[i];
                    // two-sided window: the spare must sit inside the ratio
                    // [S, 1/S] of the cold score. A spare that is much stronger
                    // is not a stand-in, so the highest resident spare inside
                    // the window wins, not the highest overall
                    float e_best[impl::sim_n_p][impl::sim_n_s];
                    for (auto & e_row : e_best) {
                        for (float & e_val : e_row) {
                            e_val = -1.0f;
                        }
                    }
                    for (int pi = 0; pi < impl::sim_n_p; ++pi) {
                        for (int k = 0; k < sim_p[pi] && k < (int) sppool.size(); ++k) {
                            if (!spare_available(sppool[(size_t) k].second)) {
                                continue;
                            }
                            const float e = sppool[(size_t) k].first;
                            for (int si = 0; si < impl::sim_n_s; ++si) {
                                if (e_best[pi][si] >= 0.0f) {
                                    continue;
                                }
                                const float s = sim_s[si];
                                if (e > cs / s || e < s * cs) {
                                    continue;
                                }
                                e_best[pi][si] = e;
                            }
                        }
                    }
                    for (int pi = 0; pi < impl::sim_n_p; ++pi) {
                        for (int si = 0; si < impl::sim_n_s; ++si) {
                            if (e_best[pi][si] >= 0.0f) {
                                p.sim_fired[pi][si]++;
                            }
                        }
                    }
                    const float ref = e_best[impl::sim_ref_p][impl::sim_ref_s];
                    if (ref >= 0.0f) {
                        p.sim_ref_n++;
                        p.sim_ref_loss += (double) (cs - ref) / (double) cs;
                        const double gap = (double) cs - (double) ref;
                        if (gap > 0.0) {
                            p.sim_ref_weaker++;
                        } else {
                            p.sim_ref_stronger++;
                        }
                        if (layer_mass > 0.0) {
                            const double rel = gap / layer_mass;
                            p.sim_ref_rel_sum += rel;
                            p.sim_ref_rel_abs += std::fabs(rel);
                            int bi = (int) (std::fabs(rel) / impl::sim_rel_max * impl::sim_rel_bins);
                            if (bi < 0) {
                                bi = 0;
                            }
                            if (bi >= impl::sim_rel_bins) {
                                bi = impl::sim_rel_bins - 1;
                            }
                            p.sim_ref_rel_hist[bi]++;
                        }
                        for (int r = 0; r < 3; ++r) {
                            if (regions[r].stride != 0) {
                                p.sim_ref_bytes += regions[r].stride;
                            }
                        }
                        if (cs == max_score) {
                            p.sim_top_fired++;
                        }
                    }
                }
            }
            // substitution: replace a cold routed expert with a nearby resident
            // one, so neither the read nor the renormalization is paid. The top
            // expert is never substituted, the spare must sit inside the score
            // window [S, 1/S] and be host-resident, and the table points the
            // cold id at the spare slot while keep scales its weight to the
            // spare score
            subst.assign((size_t) n_ids, 0);
            subst_slot.assign((size_t) n_ids, -1);
            subst_scale.assign((size_t) n_ids, 1.0f);
            subst_used.clear();
            double layer_subst_mass = 0.0;
            if (p.substitute_enabled() && !sppool.empty()) {
                const bool   layer_cap    = p.drop_max_mass > 0.0f;
                const double layer_budget = (double) p.drop_max_mass * layer_mass;
                const bool   token_cap    = p.drop_max_mass_token > 0.0f;
                const double token_budget = (double) p.drop_max_mass_token * (p.tok_routed_mass + layer_mass);
                int64_t top_pos = 0;
                for (int64_t i = 1; i < n_ids; ++i) {
                    if (probs[i] > probs[top_pos]) {
                        top_pos = i;
                    }
                }
                for (int64_t i = 0; i < n_ids; ++i) {
                    const int32_t id = ids[i];
                    if (i == top_pos || id < 0 || id >= (int32_t) c.resident_slot.size() || !(probs[i] > 0.0f)) {
                        continue;
                    }
                    const bool available = c.vram[id] || c.base[id] ||
                            (c.resident_slot[id] >= 0 && c.resident_filled[id] != 0) ||
                            (epid >= 0 && p.evict_has(epid, ((int64_t) il << 32) | (uint32_t) id));
                    if (available) {
                        continue;
                    }
                    const float cs = probs[i];
                    int32_t best = -1;
                    for (int k = 0; k < (int) sppool.size() && k < (int) p.substitute_pool; ++k) {
                        const float   e   = sppool[(size_t) k].first;
                        const int32_t bid = sppool[(size_t) k].second;
                        if (e > cs / p.substitute_rel) {
                            continue; // stronger than the window, not a stand-in
                        }
                        if (e < p.substitute_rel * cs) {
                            break; // sorted, everything after is weaker
                        }
                        if (bid < 0 || bid >= (int32_t) c.resident_slot.size()) {
                            continue;
                        }
                        if (c.resident_slot[bid] < 0 || c.resident_filled[bid] == 0) {
                            continue; // only host residents can be pointed at
                        }
                        bool used = false;
                        for (const int32_t u : subst_used) {
                            if (u == bid) {
                                used = true;
                                break;
                            }
                        }
                        if (used) {
                            continue;
                        }
                        best = k;
                        break;
                    }
                    if (best < 0) {
                        continue;
                    }
                    const float  e       = sppool[(size_t) best].first;
                    const double perturb = std::fabs((double) cs - (double) e);
                    if (layer_cap && layer_subst_mass + perturb > layer_budget) {
                        continue;
                    }
                    if (token_cap && p.tok_dropped_mass + layer_subst_mass + perturb > token_budget) {
                        continue;
                    }
                    subst[(size_t) i]       = 1;
                    subst_slot[(size_t) i]  = c.resident_slot[sppool[(size_t) best].second];
                    subst_scale[(size_t) i] = e / cs;
                    subst_used.push_back(sppool[(size_t) best].second);
                    layer_subst_mass += perturb;
                }
            }
            // a substituted expert is already handled, so it is not a drop
            // candidate
            {
                size_t w = 0;
                for (size_t r = 0; r < cold.size(); ++r) {
                    if (subst[(size_t) cold[r]] == 0) {
                        cold[w++] = cold[r];
                    }
                }
                cold.resize(w);
            }
            // drop the lowest-score candidates, up to round(Q * n_expert_used):
            // the floor decides which experts are weak and Q caps how many of
            // them are dropped. The token ceiling is a running budget over the
            // token's layer fills, so the final perturbation is bounded
            double dropped_mass = 0.0;
            if (p.drop_fraction > 0.0f) {
                int32_t budget = (int32_t) std::llround((double) p.drop_fraction * (double) n_ids);
                budget = std::min<int32_t>(budget, (int32_t) cold.size());
                std::sort(cold.begin(), cold.end(), [probs](int64_t a, int64_t b) {
                    return probs[a] < probs[b];
                });
                const bool   layer_cap    = p.drop_max_mass > 0.0f;
                const double layer_budget = (double) p.drop_max_mass * layer_mass;
                const bool   token_cap    = p.drop_max_mass_token > 0.0f;
                const double token_budget = (double) p.drop_max_mass_token * (p.tok_routed_mass + layer_mass);
                for (int32_t k = 0; k < budget; ++k) {
                    const double w = (double) probs[cold[(size_t) k]];
                    if (layer_cap && layer_subst_mass + dropped_mass + w > layer_budget) {
                        break;
                    }
                    if (token_cap && p.tok_dropped_mass + layer_subst_mass + dropped_mass + w > token_budget) {
                        break;
                    }
                    drop[(size_t) cold[(size_t) k]] = 1;
                    dropped_mass += w;
                }
            }
            // token ceiling accounting: the running budget the next layer sees.
            // The per-layer worst tracks the same combined perturbation as the
            // cap, so a substitution-only run is not reported as 0%
            if (p.drop_fraction > 0.0f || p.substitute_enabled()) {
                const double layer_perturb = layer_subst_mass + dropped_mass;
                if (layer_mass > 0.0 && layer_perturb / layer_mass > p.drop_worst_mass) {
                    p.drop_worst_mass = layer_perturb / layer_mass;
                }
                p.tok_routed_mass  += layer_mass;
                p.tok_dropped_mass += layer_perturb;
            }
        }

        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = ids[i];
            if (id < 0 || id >= (int32_t) c.resident_slot.size()) {
                continue;
            }
            p.n_routed_routes++;

            // cache-aware dropping: a dropped expert points at the sentinel, so
            // the host mul_mat_id skips it and nothing is read. keep[id] feeds
            // the graph's weight renormalization
            const bool dropped     = drop_planned && i < (int64_t) drop.size()  && drop[(size_t) i]  != 0;
            const bool substituted = drop_planned && i < (int64_t) subst.size() && subst[(size_t) i] != 0;
            if (c.keep != nullptr) {
                ((float *) c.keep->data)[id] = dropped ? 0.0f : (substituted ? subst_scale[(size_t) i] : 1.0f);
            }
            if (substituted) {
                // the position reads the spare's resident slot instead of a
                // transient read for this id; keep scaled the gathered weight
                // from the cold score to the spare score
                table[id] = subst_slot[(size_t) i];
                p.n_subst_routes++;
                if (probs != nullptr) {
                    const double cw = (double) probs[i];
                    p.n_subst_mass += std::fabs(cw - cw * (double) subst_scale[(size_t) i]);
                }
                for (int r = 0; r < 3; ++r) {
                    if (regions[r].stride != 0) {
                        p.n_subst_bytes += regions[r].stride;
                    }
                }
                continue;
            }
            if (dropped) {
                table[id] = pool.sentinel;
                p.n_dropped_routes++;
                if (probs != nullptr) {
                    p.n_dropped_mass += (double) probs[i];
                }
                for (int r = 0; r < 3; ++r) {
                    if (regions[r].stride != 0) {
                        p.n_dropped_bytes += regions[r].stride;
                    }
                }
                continue;
            }

            // served by the VRAM cache: point the table at the sentinel so the
            // host mul_mat_id skips it, and read nothing
            if (c.vram[id]) {
                table[id] = pool.sentinel;
                p.trace_n_vram++;
                continue;
            }

            // a resident slot is only valid once this expert's bytes have been
            // read into it
            const int32_t slot = c.resident_slot[id];
            if (slot >= 0 && c.resident_filled[id] != 0) {
                table[id] = slot;
                p.trace_n_res++;
                p.n_decode_cache_hits++;
                continue;
            }

            if (slot >= 0) {
                // promoted but not read yet. The L2 pool may already hold the
                // bytes: copy them into the resident slot the hot pass reads, so
                // that copy runs before the batch is handed over.
                const int64_t key = ((int64_t) il << 32) | (uint32_t) id;
                const int32_t eslot = epid >= 0 ? p.evict_touch(epid, key) : -1;
                if (eslot >= 0) {
                    table[id] = slot;
                    p.trace_n_res++;
                    p.n_decode_cache_misses++;
                    p.n_decode_cache_fills++;
                    p.n_l2_all_hits++;
                    impl::evict_pool & ep = p.evict_pools[(size_t) epid];
                    for (int r = 0; r < 3; ++r) {
                        const size_t len = regions[r].stride;
                        if (len == 0) {
                            continue;
                        }
                        copies_pre.push_back({ ep.data[r] + (size_t) eslot * ep.stride[r],
                                               pool.data[r] + (size_t) slot * pool.slot_stride[r], len, true });
                        p.n_l2_promo_bytes += len;
                    }
                    p.n_l2_promotions++;
                    p.evict_populated = true;
                    l2_consume.emplace_back(epid, key);
                    newly_filled.push_back(id);
                    continue;
                }

                // no L2 entry: read straight into a transient slot, then fill
                // the resident slot from it on the worker. The hot pass skips
                // this id (its table entry is the transient), so the resident is
                // not read until the next route, after the store has landed
                if (transient >= trans_end) {
                    transient = trans_begin;
                }
                const int32_t use = transient++;
                p.trace_n_cold++;
                p.n_decode_cache_misses++;
                read_to_transient(id, use);
                for (int r = 0; r < 3; ++r) {
                    const size_t len = regions[r].stride;
                    if (len == 0) {
                        continue;
                    }
                    res_fills.push_back({ pool.data[r] + (size_t) use * pool.slot_stride[r],
                                          pool.data[r] + (size_t) slot * pool.slot_stride[r], len, true });
                }
                res_pending.push_back({ il, id });
                table[id] = use;
                continue;
            }

            // not resident: serve it from this layer's transient window. The L2
            // pool hit copies the rows in; a miss reads the disk straight into
            // the transient slot the graph computes from and queues the store
            // into the pool for the worker to run while the layer computes.
            if (transient >= trans_end) {
                transient = trans_begin;
            }
            const int32_t use = transient++;
            p.trace_n_cold++;
            p.n_decode_cache_misses++;
            table[id] = use;

            if (epid >= 0) {
                impl::evict_pool & ep = p.evict_pools[(size_t) epid];
                const int64_t      key = ((int64_t) il << 32) | (uint32_t) id;
                const int32_t      hit = p.evict_touch(epid, key);
                if (hit >= 0) {
                    p.n_l2_all_hits++;
                    if (p.l2_warm) {
                        p.n_l2_hits++;
                        ep.hits++;
                    } else {
                        p.n_l2_cold++;
                    }
                    for (int r = 0; r < 3; ++r) {
                        const size_t len = regions[r].stride;
                        if (len == 0) {
                            continue;
                        }
                        copies.push_back({ ep.data[r] + (size_t) hit * ep.stride[r],
                                           pool.data[r] + (size_t) use * pool.slot_stride[r], len, false });
                        p.n_l2_hit_bytes += len;
                    }
                    continue;
                }

                // miss: admit the expert only when the ghost saw it before, so a
                // one-shot miss neither evicts a re-read nor pays a store. The
                // disk read goes into the transient slot either way.
                p.n_l2_all_misses++;
                if (p.l2_warm) {
                    p.n_l2_misses++;
                    ep.misses++;
                } else {
                    p.n_l2_cold++;
                }
                p.evict_populated = true;
                int32_t eslot = -1;
                if (!ep.free_slots.empty()) {
                    // the pool still has room: admit without waiting for a second
                    // sight, so it fills to capacity and the ghost only governs
                    // admission once it is full
                    impl::ghost_remove(ep, key);
                    eslot = p.evict_reserve(epid);
                } else if (impl::ghost_has(ep, key)) {
                    impl::ghost_remove(ep, key);
                    eslot = p.evict_reserve(epid);
                    if (eslot < 0) {
                        impl::ghost_add(ep, key); // no slot: keep the candidate
                    }
                } else {
                    impl::ghost_add(ep, key);
                    p.n_l2_filtered++;
                }
                read_to_transient(id, use);
                if (eslot >= 0) {
                    for (int r = 0; r < 3; ++r) {
                        const size_t len = regions[r].stride;
                        if (len == 0) {
                            continue;
                        }
                        stores.push_back({ pool.data[r] + (size_t) use * pool.slot_stride[r],
                                           ep.data[r] + (size_t) eslot * ep.stride[r], len, false });
                    }
                    l2_pending.push_back({ epid, key, eslot });
                }
                continue;
            }

            // no L2 pool: read straight into the transient slot
            read_to_transient(id, use);
        }
    }

    // fill_cache()/fill_cache_begin() run the copies and hand the disk batch to
    // the worker; dec_io_drain() publishes the stores once they land
    return true;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(ids);
    GGML_UNUSED(n_ids);
    GGML_UNUSED(probs);
    return false;
#endif
}

void llama_disk_stage::fill_cache(int il, const int32_t * ids, int64_t n_ids, const float * probs) {
#if defined(_WIN32)
    impl & p = *pimpl;
    const int64_t t0 = ggml_time_us();

    // join the previous layer's L2 stores and publish their slots before the
    // new plan reads or writes the pool
    dec_io_drain();

    if (!fill_cache_plan(il, ids, n_ids, probs)) {
        return;
    }
    std::vector<disk_stage_job> & jobs = p.fs_jobs;
    std::vector<impl::pool_copy>& copies_pre = p.fs_copies_pre;
    std::vector<impl::pool_copy>& stores = p.fs_stores;
    std::vector<impl::pool_copy>& res_fills = p.fs_res_fills;

    size_t       disk_bytes = 0;
    const size_t n_jobs     = jobs.size();
    for (const disk_stage_job & j : jobs) {
        disk_bytes += j.len;
    }

    // a copy into a resident slot must land before the hot pass reads it
    for (const impl::pool_copy & cp : copies_pre) {
        std::memcpy(cp.dst, cp.src, cp.len);
    }

    // resident fills that consumed an L2 entry: drop the now-redundant copy. The
    // entry was touched to the MRU above, so it is still present
    if (!p.fs_l2_consume.empty() || !p.fs_newly_filled.empty()) {
        std::lock_guard<std::mutex> lock(p.cache_mu);
        for (const auto & [e, key] : p.fs_l2_consume) {
            p.evict_remove(e, key);
        }
        for (const int32_t id : p.fs_newly_filled) {
            p.cache[il].resident_filled[id] = 1;
        }
    }

    impl::dec_batch & batch = p.dec_io_batch;
    batch.copies    = p.fs_copies;
    batch.jobs      = std::move(jobs);
    batch.stores    = stores;
    batch.res_fills = res_fills;

    if (!p.dec_io_active) {
        // no decode completion port: run the batch inline, stores included, so
        // the fill still works without the worker
        for (const impl::pool_copy & cp : batch.copies) {
            std::memcpy(cp.dst, cp.src, cp.len);
        }
        if (!batch.jobs.empty()) {
            std::lock_guard<std::mutex> io(p.io_mu);
            disk_stage_run_jobs(batch.jobs, 32, p.iocp);
        }
        for (const impl::pool_copy & cp : batch.stores) {
            p.store_l2(cp);
        }
        for (const impl::pool_copy & cp : batch.res_fills) {
            std::memcpy(cp.dst, cp.src, cp.len);
        }
        std::lock_guard<std::mutex> lock(p.cache_mu);
        for (const impl::l2_pending & m : p.fs_l2_pending) {
            p.evict_finalize(m.pid, m.key, m.slot);
        }
        for (const impl::res_pending & m : p.fs_res_pending) {
            p.cache[(size_t) m.il].resident_filled[(size_t) m.id] = 1;
        }
        p.fs_l2_pending.clear();
        p.fs_res_pending.clear();
    } else {
        bool handoff = false;
        {
            std::lock_guard<std::mutex> lk(p.dec_io_mu);
            p.dec_io_error      = false;
            p.dec_io_reads_done = false;
            handoff = !batch.jobs.empty() || !batch.copies.empty() ||
                      !batch.stores.empty() || !batch.res_fills.empty();
            p.dec_io_ready = handoff;
        }
        if (handoff) {
            p.dec_io_cv.notify_one();
            // wait for the read phase only: the stores that fill the L2 pool keep
            // running while the layer computes, and the next fill drains them
            std::unique_lock<std::mutex> lk(p.dec_io_mu);
            p.dec_io_reads_cv.wait(lk, [&p] { return p.dec_io_reads_done; });
            if (p.dec_io_error) {
                throw std::runtime_error("disk stage: unbuffered read failed");
            }
        }
    }

    // decode fill timing: how much of a decode step is the blocking read, and
    // (trace) the gap to the previous layer's fill, i.e. the compute in between
    const int64_t t1 = ggml_time_us();
    {
        std::lock_guard<std::mutex> lock(p.cache_mu);
        p.n_dec_fill_calls++;
        p.n_dec_fill_us    += (uint64_t) (t1 - t0);
        p.n_dec_fill_bytes += disk_bytes;
    }

    if (disk_stage_trace()) {
        const double fill_ms = (t1 - t0) / 1000.0;
        if (p.trace_prev_il >= 0 && il <= p.trace_prev_il) {
            // the layer index wrapped back: a new decode token, report the last
            const double tok_us = (double) (t0 - p.trace_tok_start_us);
            LLAMA_LOG_INFO("[disk-stage] decode token: %" PRIu64 " fills, fill=%.2f ms gap=%.2f ms wall=%.2f ms (read %.0f%%)\n",
                           p.trace_tok_calls, p.trace_tok_fill_us / 1000.0, p.trace_tok_gap_us / 1000.0,
                           tok_us / 1000.0, tok_us > 0.0 ? 100.0 * p.trace_tok_fill_us / tok_us : 0.0);
            p.trace_tok_start_us = t0;
            p.trace_tok_fill_us  = 0;
            p.trace_tok_gap_us   = 0;
            p.trace_tok_calls    = 0;
        } else if (p.trace_prev_il < 0) {
            p.trace_tok_start_us = t0;
        }
        p.trace_tok_calls++;
        p.trace_tok_fill_us += (uint64_t) (t1 - t0);

        // gap since the previous fill: the compute that ran between the two
        // fills when the scheduler splits the graph per layer, ~0 when it
        // prepares several layers in one split
        const int64_t gap_us = p.trace_prev_us > 0 ? t0 - p.trace_prev_us : 0;
        if (gap_us > 0) {
            p.trace_tok_gap_us += (uint64_t) gap_us;
        }
        LLAMA_LOG_INFO("[disk-stage] decode L%d: fill=%.2f ms jobs=%zu (%.2f MiB) res=%" PRId64 " cold=%" PRId64
                       " vram=%" PRId64 " gap=%.2f ms\n",
                       il, fill_ms, n_jobs, disk_bytes / (1024.0 * 1024.0),
                       p.trace_n_res, p.trace_n_cold, p.trace_n_vram, gap_us / 1000.0);
        p.trace_prev_il = il;
        p.trace_prev_us = t1;
    }
#else
    GGML_UNUSED(il);
    GGML_UNUSED(ids);
    GGML_UNUSED(n_ids);
    GGML_UNUSED(probs);
#endif
}

// wait out the decode I/O worker and publish the L2 slots its stores filled, so
// the next plan can hit them. Also called before a prefill clobbers the staging
// slabs the L2 pool lives on
void llama_disk_stage::dec_io_drain() {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.dec_io_active) {
        return;
    }
    {
        std::unique_lock<std::mutex> lk(p.dec_io_mu);
        p.dec_io_done_cv.wait(lk, [&p] { return !p.dec_io_running && !p.dec_io_ready; });
    }
    p.dec_wait_il = -1;
    p.trace_split = {};

    std::lock_guard<std::mutex> lock(p.cache_mu);
    if (!p.dec_io_error) {
        for (const impl::l2_pending & m : p.fs_l2_pending) {
            p.evict_finalize(m.pid, m.key, m.slot);
        }
        for (const impl::res_pending & m : p.fs_res_pending) {
            impl::cache_layer & c = p.cache[(size_t) m.il];
            if (c.table != nullptr && m.id >= 0 && m.id < (int32_t) c.resident_filled.size()) {
                c.resident_filled[(size_t) m.id] = 1;
            }
        }
    }
    p.fs_l2_pending.clear();
    p.fs_res_pending.clear();
#endif
}

void llama_disk_stage::fill_cache_begin(int il, const int32_t * ids, int64_t n_ids, const float * probs) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.split_hot_active) {
        p.dec_wait_il = -1;
        fill_cache(il, ids, n_ids, probs);
        return;
    }
    // a batch left in flight by an aborted graph would be overwritten below
    dec_io_drain();
    if (il < 0 || il >= (int) p.cache.size() || p.cache[il].table == nullptr || n_ids <= 0) {
        p.dec_wait_il = -1;
        return;
    }
    // no cold pass was emitted for this layer, so nobody would ever wait on the
    // batch: fill it synchronously instead of stranding it on the worker
    if (!split_cold(il)) {
        fill_cache(il, ids, n_ids, probs);
        return;
    }

    const int64_t t_begin = ggml_time_us();
    p.trace_split          = {};
    p.trace_split.il       = il;
    p.trace_split.t_begin  = t_begin;
    if (!fill_cache_plan(il, ids, n_ids, probs)) {
        p.dec_wait_il = -1;
        return;
    }
    const int64_t t_plan0 = ggml_time_us();

    std::vector<disk_stage_job> & jobs = p.fs_jobs;
    std::vector<impl::pool_copy>& copies_pre = p.fs_copies_pre;
    std::vector<impl::pool_copy>& stores = p.fs_stores;
    std::vector<impl::pool_copy>& res_fills = p.fs_res_fills;
    impl::dec_batch & batch = p.dec_io_batch;

    p.dec_bytes = 0;
    for (const disk_stage_job & j : jobs) {
        p.dec_bytes += j.len;
    }

    // a copy into a resident slot must land before the hot pass reads it, so it
    // runs here; the copies into transient slots are skipped by the hot pass, so
    // they go to the worker and overlap both the read and the hot compute
    for (const impl::pool_copy & cp : copies_pre) {
        std::memcpy(cp.dst, cp.src, cp.len);
    }
    const int64_t t_copy = ggml_time_us();

    // resident fills that consumed an L2 entry: drop the now-redundant copy
    if (!p.fs_l2_consume.empty() || !p.fs_newly_filled.empty()) {
        std::lock_guard<std::mutex> lock(p.cache_mu);
        for (const auto & [e, key] : p.fs_l2_consume) {
            p.evict_remove(e, key);
        }
        for (const int32_t id : p.fs_newly_filled) {
            p.cache[il].resident_filled[id] = 1;
        }
    }

    // hand the batch over, so the drive works while the hot pass computes. The
    // marks must be in place first: the worker writes its read window into them
    p.trace_split.t_plan0 = t_plan0;
    p.trace_split.t_copy  = t_copy;
    p.trace_split.t_hot   = ggml_time_us();

    batch.copies    = p.fs_copies;
    batch.jobs      = std::move(jobs);
    batch.stores    = stores;
    batch.res_fills = res_fills;
    bool handoff = false;
    {
        std::lock_guard<std::mutex> lk(p.dec_io_mu);
        p.dec_io_error      = false;
        p.dec_io_reads_done = false;
        handoff = !batch.jobs.empty() || !batch.copies.empty() ||
                  !batch.stores.empty() || !batch.res_fills.empty();
        p.dec_io_ready = handoff;
    }
    if (handoff) {
        p.dec_io_cv.notify_one();
    }

    p.dec_wait_il = handoff ? il : -1;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(ids);
    GGML_UNUSED(n_ids);
    GGML_UNUSED(probs);
#endif
}

void llama_disk_stage::fill_cache_wait(int il) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (p.dec_wait_il != il) {
        return; // no batch in flight for this layer
    }
    p.dec_wait_il = -1;

    const int64_t tw = ggml_time_us();
    // wait for the read phase only: the L2 stores keep running while the cold
    // pass computes, and the next fill drains them
    {
        std::unique_lock<std::mutex> lk(p.dec_io_mu);
        p.dec_io_reads_cv.wait(lk, [&p] { return p.dec_io_reads_done; });
    }

    bool failed = false;
    {
        std::lock_guard<std::mutex> lk(p.dec_io_mu);
        failed = p.dec_io_error;
    }

    const int64_t t_join = ggml_time_us();
    p.trace_split.t_cold  = tw;
    p.trace_split.t_join  = t_join;
    p.trace_split.started = true;

    {
        std::lock_guard<std::mutex> lock(p.cache_mu);
        p.n_dec_fill_calls++;
        p.n_dec_fill_us    += (uint64_t) (t_join - p.trace_split.t_begin);
        p.n_dec_fill_bytes += p.dec_bytes;
    }

    if (failed) {
        throw std::runtime_error("disk stage: unbuffered read failed");
    }
#else
    GGML_UNUSED(il);
#endif
}

// one layer of the split, as deltas between the two calls: how long the drive
// took, how much of it the hot pass covered, and what the cold pass then cost
void llama_disk_stage::split_report() {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (p.trace_split.il < 0 || !p.trace_split.started) {
        return;
    }
    const impl::split_phase & s = p.trace_split;
    const int64_t t_end = ggml_time_us();

    const int64_t read_us = s.rd1 > s.rd0 ? s.rd1 - s.rd0 : 0;
    // the read only overlaps the hot pass while the compute thread is running
    // it, i.e. from the handoff to the cold split
    int64_t ovl_us   = 0;
    int64_t stall_us = 0;
    if (read_us > 0) {
        const int64_t lo = std::max(s.rd0, s.t_hot);
        const int64_t hi = std::min(s.rd1, s.t_cold);
        ovl_us           = hi > lo ? hi - lo : 0;
        stall_us         = s.rd1 > s.t_cold ? s.rd1 - s.t_cold : 0;
    }

    if (disk_stage_trace()) {
        LLAMA_LOG_INFO("[disk-stage] split L%d: bytes=%.2f MiB span=%.3f ms plan=%.3f copy=%.3f"
                       " read=%.3f hot=%.3f wait=%.3f cold=%.3f hidden=%.3f stall=%.3f lim=%d"
                       " res=%" PRId64 " cold_ids=%" PRId64 " vram=%" PRId64 "\n",
                       s.il, p.dec_bytes / (1024.0 * 1024.0), (t_end - s.t_begin) / 1000.0,
                       (s.t_plan0 - s.t_begin) / 1000.0,
                       (s.t_copy - s.t_plan0) / 1000.0,
                       read_us / 1000.0,
                       (s.t_cold - s.t_hot) / 1000.0,
                       (s.t_join - s.t_cold) / 1000.0,
                       (t_end - s.t_join) / 1000.0, ovl_us / 1000.0, stall_us / 1000.0,
                       s.rd1 > s.t_cold ? 1 : 0,
                       p.trace_n_res, p.trace_n_cold, p.trace_n_vram);
    }

    const int64_t wait_us = s.t_join > s.t_cold ? s.t_join - s.t_cold : 0;

    impl::split_acc & a = p.trace_split_tok;
    a.layers++;
    a.plan_us    += s.t_plan0   > 0 ? s.t_plan0   - s.t_begin  : 0;
    a.copy_us    += s.t_copy    > 0 ? s.t_copy    - s.t_plan0  : 0;
    a.hot_us     += s.t_cold - s.t_hot;
    a.read_us    += read_us;
    a.cold_us    += (t_end - s.t_join);
    a.ovl_us     += ovl_us;
    a.stall_us   += stall_us;
    a.wait_us    += wait_us;
    a.span_us    += t_end - s.t_begin;
    a.bytes      += (int64_t) p.dec_bytes;
    if (read_us > 0 && s.rd1 > s.t_cold) {
        a.n_read_limited++;
    }

    p.trace_split = {};
#else
#endif
}

// the cold pass of the pending layer has run: this is the first node of the
// next split, so its compute is over
void llama_disk_stage::split_cold_end() {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (p.trace_split.il < 0 || !p.trace_split.started) {
        return;
    }
    split_report();
#endif
}

// end of a decode step: close the layer no later split reported, print the totals
void llama_disk_stage::split_token_end() {
#if defined(_WIN32)
    impl & p = *pimpl;
    // the graph is over: give the weights intercept another chance, in case the
    // scores were unreadable on this token and the table fill drove it instead
    p.drop_active = p.drop_enabled();
    split_cold_end();
    // close the token ceiling budget: keep the worst token share of the interval
    // for the report, then reset for the next token
    if (p.tok_routed_mass > 0.0) {
        const double frac = p.tok_dropped_mass / p.tok_routed_mass;
        if (frac > p.drop_worst_token_mass) {
            p.drop_worst_token_mass = frac;
        }
    }
    p.tok_routed_mass  = 0.0;
    p.tok_dropped_mass = 0.0;
    const impl::split_acc & a = p.trace_split_tok;
    if (a.layers == 0) {
        return;
    }
    if (disk_stage_trace()) {
        LLAMA_LOG_INFO("[disk-stage] split token: layers=%d bytes=%.2f MiB span=%.3f ms plan=%.3f copy=%.3f"
                       " read=%.3f hot=%.3f wait=%.3f cold=%.3f hidden=%.3f stall=%.3f lim=%d/%d\n",
                       a.layers, a.bytes / (1024.0 * 1024.0), a.span_us / 1000.0,
                       a.plan_us / 1000.0, a.copy_us / 1000.0,
                       a.read_us / 1000.0, a.hot_us / 1000.0,
                       a.wait_us / 1000.0, a.cold_us / 1000.0,
                       a.ovl_us / 1000.0, a.stall_us / 1000.0, a.n_read_limited, a.layers);
    }
    // the token is over: fold it into the interval totals the report prints.
    // This is the only place the token accumulator is whole, split_report() runs
    // once per layer and would add the same partial token again and again
    p.split_tot += a;
    p.trace_split_tok = {};
#endif
}

int32_t llama_disk_stage::resident_capacity(int il) const {
    const int pid = pool_id(il);
    return pid >= 0 ? pimpl->pools[(size_t) pid].res_cap : 0;
}

int32_t llama_disk_stage::resident_capacity() const {
    int32_t cap = 0;
    for (const auto & pool : pimpl->pools) {
        cap = std::max(cap, pool.res_cap);
    }
    return cap;
}

const std::vector<std::vector<int32_t>> & llama_disk_stage::base_experts() const {
    return pimpl->base_target != nullptr ? *pimpl->base_target : pimpl->base_set;
}

const std::vector<std::vector<int32_t>> * llama_disk_stage::base_set_by_name(const std::string & name) const {
    const auto it = pimpl->base_targets.find(name);
    return it == pimpl->base_targets.end() ? nullptr : &it->second;
}

std::string llama_disk_stage::select_base_set(const std::vector<std::string> & tools) const {
    return pimpl->have_base_template ? pimpl->base_template.select(tools) : std::string();
}

std::string llama_disk_stage::fence_base_set() const {
    return pimpl->have_base_template ? pimpl->base_template.fence_set : std::string();
}

bool llama_disk_stage::set_base_target(const std::string & name) {
    impl & p = *pimpl;
    const auto it = p.base_targets.find(name);
    if (it == p.base_targets.end()) {
        return false;
    }
    p.base_target = &it->second;
    p.base_active = name;
    return true;
}

const std::string & llama_disk_stage::active_base_set() const {
    return pimpl->base_active;
}

bool llama_disk_stage::base_add(int il, int32_t id) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.cache.size()) {
        return false;
    }
    impl::cache_layer & c = p.cache[(size_t) il];
    if (c.table == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(p.cache_mu);
    if (c.base[(size_t) id] != 0 || c.resident_slot[(size_t) id] >= 0 || c.vram[(size_t) id] != 0) {
        c.base[(size_t) id] = 1;
        return true;
    }
    impl::cache_pool & pool = p.pools[(size_t) c.pool];
    if (pool.free_slots.empty()) {
        return false;
    }
    c.resident_slot[(size_t) id]   = pool.free_slots.back();
    pool.free_slots.pop_back();
    c.resident_filled[(size_t) id] = 0;
    c.base[(size_t) id]            = 1;
    p.n_decode_cache_resident_changes++;
    return true;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
    return false;
#endif
}

bool llama_disk_stage::resident_slot_held(int il, int32_t id) const {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (il < 0 || il >= (int) p.cache.size()) {
        return false;
    }
    const impl::cache_layer & c = p.cache[(size_t) il];
    if (id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(p.cache_mu);
    return c.resident_slot[(size_t) id] >= 0 && c.vram[(size_t) id] == 0;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
    return false;
#endif
}

void llama_disk_stage::base_remove(int il, int32_t id) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (il < 0 || il >= (int) p.cache.size()) {
        return;
    }
    impl::cache_layer & c = p.cache[(size_t) il];
    if (id < 0 || id >= (int32_t) c.base.size()) {
        return;
    }
    std::lock_guard<std::mutex> lock(p.cache_mu);
    c.base[(size_t) id] = 0;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
#endif
}


const std::vector<std::vector<int32_t>> & llama_disk_stage::warm_experts() const {
    return pimpl->warm_loaded;
}

bool llama_disk_stage::resident_add(int il, int32_t id) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.cache.size()) {
        return false;
    }
    impl::cache_layer & c = p.cache[il];
    if (c.table == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(p.cache_mu);
    if (c.vram[id]) {
        return false;  // served by the VRAM cache: never a RAM resident
    }
    if (c.resident_slot[id] >= 0) {
        return true;  // already resident
    }
    impl::cache_pool & pool = p.pools[c.pool];
    if (pool.free_slots.empty()) {
        return false;  // the pool's resident slots are all taken
    }

    // reserve only; fill_cache() reads the bytes in when the expert is routed,
    // from the L2 pool when its first route left a copy there, else from disk
    const int32_t slot = pool.free_slots.back();
    pool.free_slots.pop_back();
    c.resident_slot[id]   = slot;
    c.resident_filled[id] = 0;
    p.n_decode_cache_resident_changes++;

    // the RAM tier is full once no pool has a free slot: from then on the L2
    // stats describe steady state (see fill_cache)
    if (!p.l2_warm) {
        bool full = true;
        for (const impl::cache_pool & cp : p.pools) {
            if (!cp.free_slots.empty()) {
                full = false;
                break;
            }
        }
        p.l2_warm = full;
    }
    return true;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
    return false;
#endif
}

void llama_disk_stage::resident_remove(int il, int32_t id, bool keep_l2) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.cache.size()) {
        return;
    }
    impl::cache_layer & c = p.cache[il];
    if (c.table == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return;
    }

    std::lock_guard<std::mutex> lock(p.cache_mu);
    if (c.base[id] != 0) {
        return;  // base expert: a permanent resident, never evicted
    }
    if (c.vram[id]) {
        return;  // already belongs to the VRAM cache
    }
    const int32_t slot = c.resident_slot[id];
    if (slot < 0) {
        return;
    }
    impl::cache_pool & pool = p.pools[c.pool];

    // demote the bytes to the L2 before the slot can be reused: an evicted
    // resident is a recent-hot expert, so it is the best pool candidate. A slot
    // that was never filled holds no valid data, so skip it
    const int epid = p.evict_pool_for(il);
    if (keep_l2 && epid >= 0 && c.resident_filled[id] != 0) {
        impl::evict_pool & ep = p.evict_pools[(size_t) epid];
        const int64_t      key = ((int64_t) il << 32) | (uint32_t) id;
        const int32_t      eslot = p.evict_put(epid, key);
        if (eslot >= 0) {
            for (int r = 0; r < 3; ++r) {
                const size_t len = p.layer_regions[il][r].stride;
                if (len == 0) {
                    continue;
                }
                std::memcpy(ep.data[r] + (size_t) eslot * ep.stride[r],
                            pool.data[r] + (size_t) slot * pool.slot_stride[r], len);
            }
            p.n_l2_demotions++;
            p.evict_populated = true;
        }
    }

    c.resident_slot[id]   = -1;
    c.resident_filled[id] = 0;
    pool.free_slots.push_back(slot);
    p.n_decode_cache_resident_changes++;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
    GGML_UNUSED(keep_l2);
#endif
}

bool llama_disk_stage::resident_filled(int il, int32_t id) const {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.cache.size()) {
        return false;
    }
    const impl::cache_layer & c = p.cache[il];
    if (c.table == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(p.cache_mu);
    return !c.vram[id] && c.resident_slot[id] >= 0 && c.resident_filled[id] != 0;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
    return false;
#endif
}

bool llama_disk_stage::resident_copy(int il, int32_t id, const size_t sz[3], void * dst, size_t dst_cap) const {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.cache.size() || dst == nullptr) {
        return false;
    }
    const impl::cache_layer & c = p.cache[il];
    if (c.table == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(p.cache_mu);
    const int32_t slot = c.resident_slot[id];
    if (slot < 0 || c.vram[id] || c.resident_filled[id] == 0) {
        return false;
    }
    const impl::cache_pool & pool = p.pools[c.pool];

    size_t off = 0;
    char * out = (char *) dst;
    for (int role = 0; role < 3; ++role) {
        const size_t len = p.layer_regions[il][role].stride;
        if (len == 0 || pool.slot_stride[role] == 0) {
            continue;
        }
        if (len > sz[role] || off + len > dst_cap) {
            return false;  // caller buffer too small for this role
        }
        std::memcpy(out + off, pool.data[role] + (size_t) slot * pool.slot_stride[role], len);
        off += len;
    }
    return true;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
    GGML_UNUSED(sz);
    GGML_UNUSED(dst);
    GGML_UNUSED(dst_cap);
    return false;
#endif
}

void llama_disk_stage::vram_commit(int il, int32_t id) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.cache.size()) {
        return;
    }
    impl::cache_layer & c = p.cache[il];
    if (c.table == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return;
    }

    std::lock_guard<std::mutex> lock(p.cache_mu);
    if (c.vram[id]) {
        return;
    }
    c.vram[id] = 1;

    // served from VRAM now: drop a stale L2 copy
    const int epid = p.evict_pool_for(il);
    if (epid >= 0) {
        p.evict_remove(epid, ((int64_t) il << 32) | (uint32_t) id);
    }

    impl::cache_pool & pool = p.pools[c.pool];
    ((int32_t *) c.table->data)[id] = pool.sentinel;

    // the RAM copy is redundant now: free the slot for the next promotion (it
    // may already be gone: the hot tier can have evicted it during the upload)
    const int32_t slot = c.resident_slot[id];
    if (slot >= 0) {
        c.resident_slot[id]   = -1;
        c.resident_filled[id] = 0;
        pool.free_slots.push_back(slot);
        p.n_decode_cache_resident_changes++;
    }
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
#endif
}

void llama_disk_stage::vram_release(int il, int32_t id) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.cache.size()) {
        return;
    }
    impl::cache_layer & c = p.cache[il];
    if (c.table == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return;
    }

    std::lock_guard<std::mutex> lock(p.cache_mu);
    c.vram[id] = 0;
    // the table entry stays at the sentinel; the next fill_cache() reassigns it
    // (the expert is not a RAM resident any more, so it takes a transient slot)
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
#endif
}

bool llama_disk_stage::is_vram(int il, int32_t id) const {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.cache.size()) {
        return false;
    }
    const impl::cache_layer & c = p.cache[il];
    if (c.table == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(p.cache_mu);
    return c.vram[id] != 0;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
    return false;
#endif
}
