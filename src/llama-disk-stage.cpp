#include "llama-disk-stage.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

    // persistent decode cache: one compact slot array per layer, aliased by the
    // graph through a per-layer expert-id -> slot table
    struct cache_slot_region {
        disk_stage_file * file     = nullptr;
        size_t            file_off = 0; // aligned-down file offset of expert 0
        size_t            head     = 0; // bytes from file_off to expert 0 data
        size_t            stride   = 0; // bytes per expert
        size_t            slot_stride = 0; // padded bytes per slot in the cache tensor
    };
    struct cache_layer {
        ggml_tensor * gate  = nullptr;
        ggml_tensor * up    = nullptr;
        ggml_tensor * down  = nullptr;
        ggml_tensor * table = nullptr; // I32 [n_expert]
        ggml_tensor * slot_skip = nullptr; // I32 [n_slots + 1], 1 at the sentinel
        char *        data[3] = { nullptr, nullptr, nullptr };
        cache_slot_region r[3];
        int32_t n_slots    = 0;
        int32_t n_resident = 0;
        int32_t sentinel   = 0; // spare slot for VRAM-served experts
        std::vector<int32_t> resident_slot; // expert id -> slot, -1 when not resident
        std::vector<uint8_t> resident_filled; // expert id -> its slot holds this expert's data
        std::vector<uint8_t> vram;           // expert id -> served by the VRAM cache (host chain skips it)
        std::vector<int32_t> free_slots;    // resident slots with no expert (LIFO)
        llama_disk_stage_cache_layer pub;    // public view returned by cache_layer()
    };
    ggml_backend_buffer_t cache_pool = nullptr;
    ggml_context *        cache_ctx  = nullptr;
    std::unique_ptr<llama_mlock> cache_lock; // decode cache held in RAM for the process lifetime
    std::vector<cache_layer> cache;
    std::mutex   cache_mu;       // guards resident_slot / free_slots and table writes
    std::mutex   io_mu;          // one reaper at a time: the IOCP is shared by all reads
    int32_t      n_resident = 0; // resident slots per layer (uniform), 0 when no cache

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
        if (cache_pool) {
            ggml_backend_buffer_free(cache_pool);
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

const llama_disk_stage_layer * llama_disk_stage::layer(int il) const {
    if (!pimpl->active || il < 0 || il >= (int) pimpl->layers.size()) {
        return nullptr;
    }
    const llama_disk_stage_layer & l = pimpl->layers[il];
    return l.gate != nullptr ? &l : nullptr;
}

const llama_disk_stage_cache_layer * llama_disk_stage::cache_layer(int il) const {
    if (il < 0 || il >= (int) pimpl->cache.size() || pimpl->cache[il].gate == nullptr) {
        return nullptr;
    }
    return &pimpl->cache[il].pub;
}

llama_disk_stage::llama_disk_stage(const llama_model & model, ggml_backend_dev_t dev,
                                   int32_t n_pin_experts, uint64_t cache_budget_bytes) :
    pimpl(std::make_unique<impl>(model)) {
    impl & p = *pimpl;

#if !defined(_WIN32)
    GGML_UNUSED(dev);
    GGML_UNUSED(n_pin_experts);
    GGML_UNUSED(cache_budget_bytes);
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

    // one pool, aliased by every layer's staging tensors. Prefer the device's
    // pinned host buffer so the offload copy into VRAM is not staged through
    // prefer the device's pinned host buffer so the host->VRAM offload copy reads
    // from page-locked memory; fall back to plain CPU memory
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

    // persistent decode cache: one compact slot array per layer, filled by the
    // same unbuffered reader. The graph remaps selected_experts through the
    // layer's table and reads the weights in place, so a resident expert is
    // served without a copy. Single-token decode needs it (the model tensors are
    // never mapped), so it is always built: --pin-hot-experts sets the resident
    // experts per layer and --pin-hot-experts-budget-mib caps the total across
    // all layers
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

            size_t per_expert = 0;
            for (const auto & role : roles) {
                per_expert += src[ref].t[role.slot]->nb[2];
            }

            int32_t slots_per_layer = cache_budget > 0
                    ? (int32_t) (cache_budget / ((size_t) n_layer * std::max<size_t>(per_expert, 1)))
                    : 0;
            if (n_pin_experts > 0) {
                // --pin-hot-experts N: N resident experts per layer, plus the
                // transient slots the routed-but-not-resident experts need
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
                const int32_t n_resident = slots_per_layer - n_trans;

                size_t cache_bytes = 64 * 1024;
                for (int il = 0; il < n_layer; ++il) {
                    if (!src[il].ok) {
                        continue;
                    }
                    // one extra aligned span per tensor: the per-expert read is
                    // rounded up to the sector size, which can overrun the last slot
                    for (const auto & role : roles) {
                        const size_t stride = (size_t) src[il].t[role.slot]->nb[2];
                        cache_bytes += align_up(src[il].r[role.slot].head + (size_t) (slots_per_layer + 1) * (stride + disk_stage_align) + disk_stage_align, disk_stage_align);
                    }
                    cache_bytes += align_up((size_t) n_expert * sizeof(int32_t), disk_stage_align);
                    cache_bytes += align_up((size_t) (slots_per_layer + 1) * sizeof(int32_t), disk_stage_align);
                }

                // the decode cache is read by the CPU mul_mat_id only: decode
                // never offloads (n_tokens == 1), so it is plain CPU memory.
                // A pinned (cudaMallocHost) buffer would page-lock the whole
                // cache, map it through the limited BAR1 aperture, and count
                // against the WDDM device budget - a 32 GiB cache then fails or
                // OOMs unrelated device allocations. The staging pool above stays
                // pinned because it IS the source of the host->VRAM offload copy.
                ggml_backend_buffer_type_t cbuft = ggml_backend_cpu_buffer_type();
                p.cache_pool = ggml_backend_buft_alloc_buffer(cbuft, cache_bytes);
                if (p.cache_pool == nullptr) {
                    LLAMA_LOG_WARN("%s: failed to allocate the %.2f GiB decode cache\n",
                                   __func__, cache_bytes / (1024.0 * 1024.0 * 1024.0));
                } else {
                    ggml_backend_buffer_set_usage(p.cache_pool, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    char * cbase = (char *) align_up((uintptr_t) ggml_backend_buffer_get_base(p.cache_pool), disk_stage_align);

                    if (!p.lock_cache(cbase, cache_bytes)) {
                        throw std::runtime_error("failed to hold the MoE decode cache in RAM");
                    }

                    ggml_init_params cip = {
                        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (n_layer * 5 + 16),
                        /*.mem_buffer =*/ nullptr,
                        /*.no_alloc   =*/ true,
                    };
                    p.cache_ctx = ggml_init(cip);
                    if (p.cache_ctx != nullptr) {
                        p.cache.resize(n_layer);
                        size_t off = 0;
                        int    n_cache = 0;
                        for (int il = 0; il < n_layer && p.cache_pool != nullptr; ++il) {
                            if (!src[il].ok) {
                                continue;
                            }
                            impl::cache_layer & c = p.cache[il];
                            c.n_slots    = slots_per_layer;
                            c.n_resident = n_resident;
                            c.sentinel   = slots_per_layer;
                            c.resident_slot.assign((size_t) n_expert, -1);
                            c.resident_filled.assign((size_t) n_expert, 0);
                            c.vram.assign((size_t) n_expert, 0);
                            c.free_slots.resize((size_t) n_resident);
                            for (int32_t s = 0; s < n_resident; ++s) {
                                c.free_slots[(size_t) s] = s;
                            }

                            ggml_tensor * tensors[3] = { nullptr, nullptr, nullptr };
                            for (const auto & role : roles) {
                                const ggml_tensor * s = src[il].t[role.slot];
                                ggml_tensor * ct = ggml_new_tensor_3d(p.cache_ctx, s->type, s->ne[0], s->ne[1], slots_per_layer + 1);
                                ggml_format_name(ct, "disk_cache_%s.%d", role.suffix, il);
                                // pad the slot stride: an expert read starts `head` bytes
                                // before its slot to stay sector-aligned, so with adjacent
                                // slots in flight it would clobber the tail of the
                                // previous slot's expert without the pad
                                ct->nb[2] = s->nb[2] + disk_stage_align;
                                c.r[role.slot] = { src[il].r[role.slot].file, src[il].r[role.slot].file_off,
                                                   src[il].r[role.slot].head, (size_t) s->nb[2], (size_t) ct->nb[2] };
                                if (ggml_backend_tensor_alloc(p.cache_pool, ct, cbase + off + src[il].r[role.slot].head) != GGML_STATUS_SUCCESS) {
                                    LLAMA_LOG_WARN("%s: failed to bind a decode cache tensor, cache disabled\n", __func__);
                                    ggml_backend_buffer_free(p.cache_pool);
                                    p.cache_pool = nullptr;
                                    break;
                                }
                                c.data[role.slot] = (char *) ct->data;
                                off += align_up(src[il].r[role.slot].head + (size_t) (slots_per_layer + 1) * ct->nb[2] + disk_stage_align, disk_stage_align);
                                tensors[role.slot] = ct;
                            }
                            if (p.cache_pool == nullptr) {
                                p.cache.clear();
                                break;
                            }

                            // 2d [1, n_expert] so ggml_get_rows can index the
                            // expert id along ne[1] during the decode remap
                            ggml_tensor * tab = ggml_new_tensor_2d(p.cache_ctx, GGML_TYPE_I32, 1, n_expert);
                            ggml_format_name(tab, "disk_cache_table.%d", il);
                            ggml_backend_tensor_alloc(p.cache_pool, tab, cbase + off);
                            std::memset(tab->data, 0, (size_t) n_expert * sizeof(int32_t));
                            off += align_up((size_t) n_expert * sizeof(int32_t), disk_stage_align);

                            // slot-indexed skip table for the host mul_mat_id: 1 at
                            // the sentinel, so a VRAM-served expert (whose table
                            // entry is the sentinel) is skipped instead of read
                            ggml_tensor * skip = ggml_new_tensor_2d(p.cache_ctx, GGML_TYPE_I32, 1, slots_per_layer + 1);
                            ggml_format_name(skip, "disk_cache_skip.%d", il);
                            ggml_backend_tensor_alloc(p.cache_pool, skip, cbase + off);
                            std::memset(skip->data, 0, (size_t) (slots_per_layer + 1) * sizeof(int32_t));
                            ((int32_t *) skip->data)[slots_per_layer] = 1;
                            off += align_up((size_t) (slots_per_layer + 1) * sizeof(int32_t), disk_stage_align);

                            c.gate  = tensors[0];
                            c.up    = tensors[1];
                            c.down  = tensors[2];
                            c.table = tab;
                            c.slot_skip = skip;
                            c.pub.gate    = c.gate;
                            c.pub.up      = c.up;
                            c.pub.down    = c.down;
                            c.pub.table   = c.table;
                            c.pub.slot_skip = c.slot_skip;
                            c.pub.n_slots = c.n_slots;
                            c.pub.sentinel = c.sentinel;
                            n_cache++;
                        }
                        if (p.cache_pool != nullptr) {
                            p.n_resident = n_resident;
                            LLAMA_LOG_INFO("%s: decode cache active for %d layer(s), %.2f GiB, %d slots/layer (%d resident)\n",
                                           __func__, n_cache, cache_bytes / (1024.0 * 1024.0 * 1024.0), slots_per_layer, n_resident);
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

        const impl::cache_layer * c = nullptr;
        if (il < (int) p.cache.size() && p.cache[il].gate != nullptr) {
            c = &p.cache[il];
        }

        for (int role = 0; role < (int) p.layer_regions[il].size(); ++role) {
            const impl::region & r = p.layer_regions[il][role];

            const size_t  stride   = c != nullptr ? c->r[role].stride : 0;
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
                    copies.push_back({ c->data[role] + slot * c->r[role].slot_stride,
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
                    patches.push_back({ c->data[role] + slot * c->r[role].slot_stride + (stride - prefix),
                                        slab + (size_t) id * stride - prefix, prefix });
                }
                if (last < n_expert && resident(last)) {
                    const size_t over = (size_t) ((dst + rlen) - (slab + (size_t) last * stride));
                    if (over > 0 && over <= stride) {
                        const size_t slot = (size_t) c->resident_slot[last];
                        patches.push_back({ c->data[role] + slot * c->r[role].slot_stride,
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
    if (c.gate == nullptr || c.table == nullptr || n_ids <= 0) {
        return;
    }

    int32_t * table = (int32_t *) c.table->data;
    std::vector<disk_stage_job> jobs;
    std::vector<int32_t>    newly_filled; // residents whose slot was just read
    int32_t transient = c.n_resident;

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
                table[id] = c.sentinel;
                continue;
            }

            // a resident slot is only valid once this expert's bytes have been
            // read into it; that first read happens here, in the same batch as
            // the transients
            const int32_t slot = c.resident_slot[id];
            if (slot >= 0) {
                table[id] = slot;
                if (c.resident_filled[id] == 0) {
                    for (int r = 0; r < 3; ++r) {
                        const impl::cache_slot_region & sr = c.r[r];
                        if (sr.file == nullptr || sr.stride == 0) {
                            continue;
                        }
                        char * dst = c.data[r] + (size_t) slot * sr.slot_stride - sr.head;
                        jobs.push_back({ sr.file->h, dst, sr.file_off + (size_t) id * sr.stride, align_up(sr.head + sr.stride, disk_stage_align) });
                    }
                    newly_filled.push_back(id);
                }
                continue;
            }

            // not resident: serve it from a transient slot for this ubatch
            if (transient >= c.n_slots) {
                transient = c.n_resident;
            }
            const int32_t use = transient++;

            // read the expert into its slot: source starts `head` bytes before the
            // expert data so the aligned read lands the row where the tensor expects
            // it, and the length is rounded up to the sector size the unbuffered read
            // requires (the extra bytes fall in the per-tensor slack)
            for (int r = 0; r < 3; ++r) {
                const impl::cache_slot_region & sr = c.r[r];
                if (sr.file == nullptr || sr.stride == 0) {
                    continue;
                }
                char * dst = c.data[r] + (size_t) use * sr.slot_stride - sr.head;
                jobs.push_back({ sr.file->h, dst, sr.file_off + (size_t) id * sr.stride, align_up(sr.head + sr.stride, disk_stage_align) });
            }
            table[id] = use;
        }
    }

    if (!jobs.empty()) {
        std::lock_guard<std::mutex> io(p.io_mu);
        disk_stage_run_jobs(jobs, 32, p.iocp);
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

int32_t llama_disk_stage::resident_capacity() const {
    return pimpl->n_resident;
}

bool llama_disk_stage::resident_add(int il, int32_t id) {
#if defined(_WIN32)
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.cache.size()) {
        return false;
    }
    impl::cache_layer & c = p.cache[il];
    if (c.gate == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(p.cache_mu);
    if (c.vram[id]) {
        return false;  // served by the VRAM cache: never a RAM resident
    }
    if (c.resident_slot[id] >= 0) {
        return true;  // already resident
    }
    if (c.free_slots.empty()) {
        return false;  // all resident slots taken
    }

    // reserve only; fill_cache() reads the bytes in when the expert is routed
    const int32_t slot = c.free_slots.back();
    c.free_slots.pop_back();
    c.resident_slot[id]   = slot;
    c.resident_filled[id] = 0;
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
    if (c.gate == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
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
    c.resident_slot[id]   = -1;
    c.resident_filled[id] = 0;
    c.free_slots.push_back(slot);
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
    if (c.gate == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
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
    if (c.gate == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(p.cache_mu);
    const int32_t slot = c.resident_slot[id];
    if (slot < 0 || c.vram[id] || c.resident_filled[id] == 0) {
        return false;
    }

    size_t off = 0;
    char * out = (char *) dst;
    for (int role = 0; role < 3; ++role) {
        const size_t len = c.r[role].stride;
        if (len == 0 || c.r[role].slot_stride == 0) {
            continue;
        }
        if (len > sz[role] || off + len > dst_cap) {
            return false;  // caller buffer too small for this role
        }
        std::memcpy(out + off, c.data[role] + (size_t) slot * c.r[role].slot_stride, len);
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
    if (c.gate == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
        return;
    }

    std::lock_guard<std::mutex> lock(p.cache_mu);
    if (c.vram[id]) {
        return;
    }
    c.vram[id] = 1;
    ((int32_t *) c.table->data)[id] = c.sentinel;

    // the RAM copy is redundant now: free the slot for the next promotion (it
    // may already be gone: the hot tier can have evicted it during the upload)
    const int32_t slot = c.resident_slot[id];
    if (slot >= 0) {
        c.resident_slot[id]   = -1;
        c.resident_filled[id] = 0;
        c.free_slots.push_back(slot);
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
    if (c.gate == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
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
    if (c.gate == nullptr || id < 0 || id >= (int32_t) c.resident_slot.size()) {
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
