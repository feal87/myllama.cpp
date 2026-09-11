#include "llama-disk-stage.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

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

static size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

#if defined(_WIN32)
struct disk_stage_file {
    HANDLE h = INVALID_HANDLE_VALUE;

    ~disk_stage_file() {
        if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
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
        disk_stage_file *   file     = nullptr;
        uint64_t            file_off = 0; // aligned-down source offset
    };

    explicit impl(const llama_model & m) : model(m) {}

    const llama_model & model;
    bool active = false;

#if defined(_WIN32)
    HANDLE iocp = nullptr; // one completion port, every staging file associated with it
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
        int32_t       res_base = 0; // first resident slot
        int32_t       res_cap  = 0; // resident slots
        int32_t       sentinel = 0; // == res_base + res_cap: first spare slot
        std::vector<int32_t> free_slots;   // resident slots with no expert (LIFO)
    };
    struct cache_layer {
        int           pool           = -1;
        int           pool_layer_idx = 0;  // index within its pool, selects the transient region
        ggml_tensor * table    = nullptr;  // I32 [n_expert], expert id -> slot
        ggml_tensor * slot_skip = nullptr; // I32 [n_slots + 1], 1 at the sentinel
        std::vector<int32_t> resident_slot;   // expert id -> slot, -1 when not resident
        std::vector<uint8_t> resident_filled; // expert id -> its slot holds this expert's data
        std::vector<uint8_t> vram;            // expert id -> served by the VRAM cache (host chain skips it)
        llama_disk_stage_cache_layer pub;     // public view returned by cache_layer()
    };
    ggml_backend_buffer_t cache_buf  = nullptr;
    ggml_context *        cache_ctx  = nullptr;
    std::unique_ptr<llama_mlock> cache_lock; // decode cache held in RAM for the process lifetime
    std::vector<cache_pool>  pools;
    std::vector<cache_layer> cache;
    std::mutex   cache_mu;       // guards resident_slot / free_slots and table writes
    std::mutex   io_mu;          // one reaper at a time: the IOCP is shared by all reads
    int32_t      n_trans = 0;    // transient slots per layer, 0 when no cache

    // second-level (L2) expert pool. During decode the two prefill staging
    // slabs are idle, so they hold the experts the resident cache missed. A miss
    // is read into a pool slot and copied to the transient slot the graph
    // executes from; a later hit copies from the pool instead of reading the
    // disk. One pool per expert-bundle type (one slab each): a slot stride is
    // fixed per tensor, and this model mixes Q8_0 and Q5_1 down projections.
    // Evicted residents enter at the MRU end and outlive mere miss entries.
    struct evict_pool {
        char *  data[3]   = { nullptr, nullptr, nullptr }; // role -> slab region
        size_t  stride[3] = { 0, 0, 0 };
        int32_t cap       = 0;
        // LRU: lru.front() is the most recent; key = ((int64_t) il << 32) | id
        std::list<int64_t>                        lru;
        std::unordered_map<int64_t, int32_t>      slot_of;   // key -> slot
        std::vector<std::list<int64_t>::iterator> iter_of;   // slot -> position in lru
        std::vector<int64_t>                      slot_key;  // slot -> key, -1 when empty
        std::vector<int32_t>                      free_slots;
    };
    std::vector<evict_pool> evict_pools;
    std::vector<int>        evict_pool_id;    // layer -> pool, -1 when not pooled
    bool                    evict_populated = false; // pools hold entries a prefill would clobber

    uint64_t n_l2_hits        = 0;
    uint64_t n_l2_misses      = 0;
    uint64_t n_l2_cold        = 0; // lookups while the RAM tier was still filling, excluded from the hit rate
    uint64_t n_l2_evictions   = 0;
    uint64_t n_l2_demotions   = 0;
    uint64_t n_l2_hit_bytes   = 0;
    uint64_t n_l2_promos      = 0; // resident fills served from the L2 instead of the disk
    uint64_t n_l2_promo_bytes = 0;
    bool     l2_warm          = false; // latched when every resident pool is full

    int evict_pool_for(int il) const {
        return (il >= 0 && il < (int) evict_pool_id.size()) ? evict_pool_id[(size_t) il] : -1;
    }

    int32_t evict_touch(int pool, int64_t key) {
        evict_pool & ep = evict_pools[(size_t) pool];
        const auto it = ep.slot_of.find(key);
        if (it == ep.slot_of.end()) {
            return -1;
        }
        const int32_t slot = it->second;
        if (ep.iter_of[(size_t) slot] != ep.lru.begin()) {
            ep.lru.splice(ep.lru.begin(), ep.lru, ep.iter_of[(size_t) slot]);
            ep.iter_of[(size_t) slot] = ep.lru.begin();
        }
        return slot;
    }

    int32_t evict_alloc(int pool, int64_t key) {
        evict_pool & ep = evict_pools[(size_t) pool];
        int32_t slot;
        if (!ep.free_slots.empty()) {
            slot = ep.free_slots.back();
            ep.free_slots.pop_back();
        } else {
            const int64_t victim = ep.lru.back();
            ep.lru.pop_back();
            slot = ep.slot_of.at(victim);
            ep.slot_of.erase(victim);
            ep.slot_key[(size_t) slot] = -1;
            n_l2_evictions++;
        }
        ep.slot_key[(size_t) slot] = key;
        ep.slot_of[key] = slot;
        ep.lru.push_front(key);
        ep.iter_of[(size_t) slot] = ep.lru.begin();
        return slot;
    }

    int32_t evict_put(int pool, int64_t key) {
        const int32_t slot = evict_touch(pool, key);
        return slot >= 0 ? slot : evict_alloc(pool, key);
    }

    void evict_remove(int pool, int64_t key) {
        evict_pool & ep = evict_pools[(size_t) pool];
        const auto it = ep.slot_of.find(key);
        if (it == ep.slot_of.end()) {
            return;
        }
        const int32_t slot = it->second;
        ep.lru.erase(ep.iter_of[(size_t) slot]);
        ep.slot_of.erase(it);
        ep.slot_key[(size_t) slot] = -1;
        ep.free_slots.push_back(slot);
    }

    void evict_clear() {
        for (evict_pool & ep : evict_pools) {
            ep.lru.clear();
            ep.slot_of.clear();
            ep.free_slots.resize((size_t) ep.cap);
            for (int32_t s = 0; s < ep.cap; ++s) {
                ep.free_slots[(size_t) s] = s;
            }
            std::fill(ep.slot_key.begin(), ep.slot_key.end(), (int64_t) -1);
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
#endif
        disk_stage_file * raw = f.get();
        files.emplace(path, std::move(f));
        return raw;
    }
};

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

void llama_disk_stage::print_stats() const {
    const impl & p = *pimpl;
    if (p.evict_pools.empty()) {
        return;
    }
    const uint64_t lookups = p.n_l2_hits + p.n_l2_misses;
    LLAMA_LOG_INFO("[disk-stage] L2 pool: hit=%.1f%% (%" PRIu64 "/%" PRIu64 " warm routed, %" PRIu64 " cold fill skipped)"
                   " | disk avoided=%.2f MiB (promo %.2f MiB) | evictions=%" PRIu64 " | demotions=%" PRIu64 "\n",
                   lookups ? 100.0 * p.n_l2_hits / lookups : 0.0, p.n_l2_hits, lookups, p.n_l2_cold,
                   p.n_l2_hit_bytes / (1024.0 * 1024.0), p.n_l2_promo_bytes / (1024.0 * 1024.0),
                   p.n_l2_evictions, p.n_l2_demotions);
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
                                   int32_t pool_layers_max) :
    pimpl(std::make_unique<impl>(model)) {
    impl & p = *pimpl;

#if !defined(_WIN32)
    GGML_UNUSED(dev);
    GGML_UNUSED(n_pin_experts);
    GGML_UNUSED(cache_budget_bytes);
    GGML_UNUSED(pool_layers_max);
    return;
#else
    if (!model.has_disk_weights()) {
        return;
    }

    p.iocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (p.iocp == nullptr) {
        LLAMA_LOG_WARN("%s: failed to create the I/O completion port, disk staging disabled\n", __func__);
        return;
    }

    // stages the separate gate/up/down layout only
    const int n_layer = (int) model.hparams.n_layer();
    p.layers.assign(n_layer, {});
    p.layer_regions.assign(n_layer, {});

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
            src[il].r[role.slot].read_len = align_up(head + size, disk_stage_align);
            src[il].r[role.slot].file_off = t_off - head;
            src[il].r[role.slot].file     = p.file_for(path);
        }
        src[il].ok = ok;
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
            // the cache allocates per slot, and one sentinel slot per layer shares
            // the same tensor, so the usable slot count is one below the quotient
            size_t cost_per_slot = 0;
            for (int il = 0; il < n_layer; ++il) {
                if (!src[il].ok) {
                    continue;
                }
                for (const auto & role : roles) {
                    cost_per_slot += (size_t) src[il].t[role.slot]->nb[2] + disk_stage_align;
                }
            }

            int32_t slots_per_layer = 0;
            if (cache_budget > 0 && cost_per_slot > 0) {
                const int64_t slots = (int64_t) (cache_budget / cost_per_slot) - 1;
                slots_per_layer = slots > 0 ? (int32_t) std::min<int64_t>(slots, n_expert) : 0;
            }
            if (n_pin_experts > 0) {
                // --pin-hot-experts N: N resident experts per layer on average,
                // plus the transient slots the routed-but-not-resident experts
                // need. With pools a hot layer may exceed N and a cold one fall
                // short; the budget bounds the total across the pool
                const int32_t want = std::min<int32_t>(n_pin_experts + n_trans, n_expert);
                slots_per_layer = slots_per_layer > 0 ? std::min(slots_per_layer, want) : want;
            }
            if (slots_per_layer == 0) {
                slots_per_layer = std::min<int32_t>(n_trans + 1, n_expert);
                LLAMA_LOG_WARN("%s: neither --pin-hot-experts nor --pin-hot-experts-budget-mib given, "
                               "using a minimal %d-slot decode cache per layer\n",
                               __func__, (int) slots_per_layer);
            }
            slots_per_layer = std::min<int32_t>(slots_per_layer, n_expert);

            if (slots_per_layer > n_trans) {
                const int32_t res_per_layer = slots_per_layer - n_trans;

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

                size_t cache_bytes = 64 * 1024;
                for (const auto & grp : groups) {
                    const int32_t n_pool_slots = (int32_t) grp.size() * (n_trans + res_per_layer);
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
                            pool.res_base = pool.n_layers * n_trans;
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
                                c.pool_layer_idx = (int) j;
                                c.resident_slot.assign((size_t) n_expert, -1);
                                c.resident_filled.assign((size_t) n_expert, 0);
                                c.vram.assign((size_t) n_expert, 0);

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

                                c.table     = tab;
                                c.slot_skip = skip;
                                c.pub.gate      = pool.gate;
                                c.pub.up        = pool.up;
                                c.pub.down      = pool.down;
                                c.pub.table     = tab;
                                c.pub.slot_skip = skip;
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
                        // them as the L2 pool, one slab per expert-bundle type.
                        // Requires aligned data: the pool reads an expert at its
                        // slot start, so head must be zero
                        const char * l2_env = std::getenv("LLAMA_DISK_STAGE_L2");
                        const bool   l2_on  = l2_env == nullptr || (l2_env[0] != '\0' && l2_env[0] != '0');
                        if (!l2_on) {
                            LLAMA_LOG_INFO("%s: L2 eviction pool disabled by LLAMA_DISK_STAGE_L2\n", __func__);
                        } else if (all_aligned && p.n_buf >= 2) {
                            p.evict_pool_id.assign(n_layer, -1);

                            for (int il = 0; il < n_layer && (int) p.evict_pools.size() < p.n_buf; ++il) {
                                if (!src[il].ok || p.evict_pool_id[il] >= 0) {
                                    continue;
                                }
                                const int pid = (int) p.evict_pools.size();

                                int32_t cap = INT32_MAX;
                                for (const auto & role : roles) {
                                    const size_t stride = src[il].r[role.slot].stride;
                                    if (stride > 0) {
                                        cap = std::min<int32_t>(cap, (int32_t) (region_len[role.slot] / stride));
                                    }
                                }
                                if (cap <= 0 || cap == INT32_MAX) {
                                    continue;  // cannot size this type; leave the L2 off
                                }

                                impl::evict_pool ep;
                                ep.cap = cap;
                                for (const auto & role : roles) {
                                    ep.data[role.slot]   = p.base + (size_t) pid * per_buffer + region_off[role.slot] + src[il].r[role.slot].head;
                                    ep.stride[role.slot] = src[il].r[role.slot].stride;
                                }
                                ep.slot_key.assign((size_t) ep.cap, -1);
                                ep.iter_of.resize((size_t) ep.cap);
                                ep.free_slots.resize((size_t) ep.cap);
                                for (int32_t s = 0; s < ep.cap; ++s) {
                                    ep.free_slots[(size_t) s] = s;
                                }
                                p.evict_pools.push_back(std::move(ep));

                                for (int jl = il; jl < n_layer; ++jl) {
                                    if (!src[jl].ok) {
                                        continue;
                                    }
                                    bool same = true;
                                    for (const auto & role : roles) {
                                        const ggml_tensor * a = src[il].t[role.slot];
                                        const ggml_tensor * b = src[jl].t[role.slot];
                                        if (a->type != b->type || a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1]) {
                                            same = false;
                                            break;
                                        }
                                    }
                                    if (same) {
                                        p.evict_pool_id[jl] = pid;
                                    }
                                }
                            }

                            bool all_pooled = true;
                            for (int il = 0; il < n_layer; ++il) {
                                if (src[il].ok && p.evict_pool_id[il] < 0) {
                                    all_pooled = false;
                                    break;
                                }
                            }
                            if (!all_pooled) {
                                // more bundle types than staging slabs: the pool
                                // cannot cover them all, so leave it off
                                p.evict_pools.clear();
                                p.evict_pool_id.assign(n_layer, -1);
                                LLAMA_LOG_WARN("%s: more expert-bundle types than staging buffers, "
                                               "L2 eviction pool disabled\n", __func__);
                            } else {
                                for (size_t k = 0; k < p.evict_pools.size(); ++k) {
                                    LLAMA_LOG_INFO("%s:   L2 pool %zu: %d slots, strides %zu/%zu/%zu bytes\n",
                                                   __func__, k, p.evict_pools[k].cap,
                                                   p.evict_pools[k].stride[0], p.evict_pools[k].stride[1],
                                                   p.evict_pools[k].stride[2]);
                                }
                            }
                        }
                    }
                }
            } else {
                LLAMA_LOG_WARN("%s: decode cache budget too small for %d transient slots/layer\n", __func__, n_trans);
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

    p.active = true;
    LLAMA_LOG_INFO("%s: disk staging active for %d layer(s), %.1f MiB pool, %d buffer(s), %zu-byte aligned unbuffered reads\n",
                   __func__, n_staged, total / (1024.0 * 1024.0), p.n_buf, disk_stage_align);

#endif
}

llama_disk_stage::~llama_disk_stage() {
#if defined(_WIN32)
    if (pimpl) {
        {
            std::lock_guard<std::mutex> lk(pimpl->pipe_mu);
            pimpl->pipe_stop = true;
        }
        pimpl->pipe_req_cv.notify_all();
        if (pimpl->reader.joinable()) {
            pimpl->reader.join();
        }
    }
#endif
}

void llama_disk_stage::fill(int il) {
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.layer_regions.size() || p.layer_regions[il].empty()) {
        return;
    }

#if defined(_WIN32)
    if (p.evict_populated) {
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

static bool disk_stage_trace() {
    static const bool on = [] {
        const char * v = std::getenv("LLAMA_DISK_STAGE_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
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

void llama_disk_stage::fill_run(int il) {
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

            const size_t  stride   = c != nullptr ? r.stride : 0;
            const int32_t n_expert = c != nullptr ? (int32_t) c->resident_slot.size() : 0;

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
                return c->resident_slot[id] >= 0 && c->resident_filled[id] != 0 && c->vram[id] == 0;
            };

            char * slab = p.base + r.pool_off + r.head;

            int32_t id = 0;
            while (id < n_expert) {
                if (resident(id)) {
                    const size_t slot = (size_t) c->resident_slot[id];
                    copies.push_back({ pool->data[role] + slot * pool->slot_stride[role],
                                       slab + (size_t) id * stride, stride });
                    ++id;
                    continue;
                }

                int32_t last = id;
                while (last < n_expert && !resident(last)) {
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

void llama_disk_stage::fill_cache(int il, const int32_t * ids, int64_t n_ids) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (il < 0 || il >= (int) p.cache.size()) {
        return;
    }
    impl::cache_layer & c = p.cache[il];
    if (c.table == nullptr || n_ids <= 0) {
        return;
    }
    impl::cache_pool & pool = p.pools[c.pool];

    struct pool_copy {
        const char * src;
        char *       dst;
        size_t       len;
    };

    int32_t * table = (int32_t *) c.table->data;
    std::vector<disk_stage_job> jobs;
    std::vector<int32_t>    newly_filled; // residents whose slot was just read
    std::vector<pool_copy>  copies;       // L2 pool rows to copy into a transient or resident slot
    std::vector<std::pair<int, int64_t>> l2_consume; // L2 entries consumed by a resident fill

    // this layer's own transient window inside the pool: [0, res_base) is split
    // into one n_trans-wide window per layer
    const int32_t trans_begin = c.pool_layer_idx * p.n_trans;
    const int32_t trans_end   = trans_begin + p.n_trans;
    int32_t transient = trans_begin;

    // slot assignment under the lock; the unbuffered reads below run without it
    {
        std::lock_guard<std::mutex> lock(p.cache_mu);

        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = ids[i];
            if (id < 0 || id >= (int32_t) c.resident_slot.size()) {
                continue;
            }

            // served by the VRAM cache: point the table at the sentinel so the
            // host mul_mat_id skips it, and read nothing
            if (c.vram[id]) {
                table[id] = pool.sentinel;
                continue;
            }

            // a resident slot is only valid once this expert's bytes have been
            // read into it; that first read happens here, in the same batch as
            // the transients
            const int32_t slot = c.resident_slot[id];
            if (slot >= 0) {
                table[id] = slot;
                if (c.resident_filled[id] == 0) {
                    // a promoted expert may still sit in the L2 pool from its
                    // first route: copy it in instead of reading the disk again
                    const int epid = p.evict_pool_for(il);
                    const int64_t key = ((int64_t) il << 32) | (uint32_t) id;
                    const int32_t eslot = epid >= 0 ? p.evict_touch(epid, key) : -1;
                    if (eslot >= 0) {
                        impl::evict_pool & ep = p.evict_pools[(size_t) epid];
                        for (int r = 0; r < 3; ++r) {
                            const size_t len = p.layer_regions[il][r].stride;
                            if (len == 0) {
                                continue;
                            }
                            copies.push_back({ ep.data[r] + (size_t) eslot * ep.stride[r],
                                               pool.data[r] + (size_t) slot * pool.slot_stride[r], len });
                            p.n_l2_promo_bytes += len;
                        }
                        p.n_l2_promos++;
                        p.evict_populated = true;
                        l2_consume.emplace_back(epid, key);
                    } else {
                        for (int r = 0; r < 3; ++r) {
                            const impl::region & sr = p.layer_regions[il][r];
                            if (sr.file == nullptr || sr.stride == 0) {
                                continue;
                            }
                            char * dst = pool.data[r] + (size_t) slot * pool.slot_stride[r] - sr.head;
                            jobs.push_back({ sr.file->h, dst, sr.file_off + (size_t) id * sr.stride, align_up(sr.head + sr.stride, disk_stage_align) });
                        }
                    }
                    newly_filled.push_back(id);
                }
                continue;
            }

            // not resident: serve it from this layer's transient window. When
            // the L2 pool holds it, copy the rows in instead of reading the
            // disk; otherwise read the disk into a pool slot, which fills the
            // pool, and copy from there
            if (transient >= trans_end) {
                transient = trans_begin;
            }
            const int32_t use = transient++;

            const int epid = p.evict_pool_for(il);
            if (epid >= 0) {
                impl::evict_pool & ep = p.evict_pools[(size_t) epid];
                const int64_t      key = ((int64_t) il << 32) | (uint32_t) id;
                const int32_t      hit = p.evict_touch(epid, key);
                const int32_t      eslot = hit >= 0 ? hit : p.evict_alloc(epid, key);

                p.evict_populated = true;
                // the cold prefix is the RAM tier fill: a promoted expert is
                // served from residence next time, so those lookups cannot hit
                const bool l2_warm = p.l2_warm;
                if (hit < 0) {
                    if (l2_warm) {
                        p.n_l2_misses++;
                    } else {
                        p.n_l2_cold++;
                    }
                    for (int r = 0; r < 3; ++r) {
                        const impl::region & sr = p.layer_regions[il][r];
                        if (sr.file == nullptr || sr.stride == 0) {
                            continue;
                        }
                        char * dst = ep.data[r] + (size_t) eslot * ep.stride[r] - sr.head;
                        jobs.push_back({ sr.file->h, dst, sr.file_off + (size_t) id * sr.stride,
                                         align_up(sr.head + sr.stride, disk_stage_align) });
                    }
                } else {
                    if (l2_warm) {
                        p.n_l2_hits++;
                    } else {
                        p.n_l2_cold++;
                    }
                    for (int r = 0; r < 3; ++r) {
                        p.n_l2_hit_bytes += p.layer_regions[il][r].stride;
                    }
                }

                // the graph reads the transient slot: copy the pool rows there
                for (int r = 0; r < 3; ++r) {
                    const size_t len = p.layer_regions[il][r].stride;
                    if (len == 0) {
                        continue;
                    }
                    copies.push_back({ ep.data[r] + (size_t) eslot * ep.stride[r],
                                       pool.data[r] + (size_t) use * pool.slot_stride[r], len });
                }
                table[id] = use;
                continue;
            }

            // no L2 pool: read the expert into its slot: source starts `head`
            // bytes before the expert data so the aligned read lands the row
            // where the tensor expects it, and the length is rounded up to the
            // sector size the unbuffered read requires (the extra bytes fall in
            // the per-tensor slack)
            for (int r = 0; r < 3; ++r) {
                const impl::region & sr = p.layer_regions[il][r];
                if (sr.file == nullptr || sr.stride == 0) {
                    continue;
                }
                char * dst = pool.data[r] + (size_t) use * pool.slot_stride[r] - sr.head;
                jobs.push_back({ sr.file->h, dst, sr.file_off + (size_t) id * sr.stride, align_up(sr.head + sr.stride, disk_stage_align) });
            }
            table[id] = use;
        }
    }

    if (!jobs.empty()) {
        std::lock_guard<std::mutex> io(p.io_mu);
        disk_stage_run_jobs(jobs, 32, p.iocp);
    }

    // pool rows are copied after the reads above, so a miss always copies the
    // bytes the read just landed in its pool slot
    for (const pool_copy & cp : copies) {
        std::memcpy(cp.dst, cp.src, cp.len);
    }

    // resident fills that consumed an L2 entry: drop the now-redundant copy so
    // it does not hold a slot. The entry was touched to the MRU above, so it is
    // still present (evict_remove is a no-op otherwise)
    if (!l2_consume.empty()) {
        std::lock_guard<std::mutex> lock(p.cache_mu);
        for (const auto & [epid, key] : l2_consume) {
            p.evict_remove(epid, key);
        }
    }

    if (!newly_filled.empty()) {
        std::lock_guard<std::mutex> lock(p.cache_mu);
        for (const int32_t id : newly_filled) {
            c.resident_filled[id] = 1;
        }
    }
#else
    GGML_UNUSED(il);
    GGML_UNUSED(ids);
    GGML_UNUSED(n_ids);
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

void llama_disk_stage::resident_remove(int il, int32_t id) {
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
    if (epid >= 0 && c.resident_filled[id] != 0) {
        impl::evict_pool & ep = p.evict_pools[(size_t) epid];
        const int64_t      key = ((int64_t) il << 32) | (uint32_t) id;
        const int32_t      eslot = p.evict_put(epid, key);
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

    c.resident_slot[id]   = -1;
    c.resident_filled[id] = 0;
    pool.free_slots.push_back(slot);
#else
    GGML_UNUSED(il);
    GGML_UNUSED(id);
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
