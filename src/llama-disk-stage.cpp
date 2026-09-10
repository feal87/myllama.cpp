#include "llama-disk-stage.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
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

static bool env_flag_on(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "off") != 0 && std::strcmp(v, "no") != 0;
}

// default when the variable is unset
static bool env_flag_default_on(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr) {
        return true;
    }
    return v[0] != '\0' && std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "off") != 0 && std::strcmp(v, "no") != 0;
}

// integer env value, 0 when unset or unparsable
static uint64_t env_u64(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr) {
        return 0;
    }
    return strtoull(v, nullptr, 10);
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
        char *        data[3] = { nullptr, nullptr, nullptr };
        cache_slot_region r[3];
        int32_t n_slots         = 0;
        int32_t n_resident      = 0;
        int32_t n_resident_used = 0;
        std::vector<int32_t> resident_slot; // expert id -> slot, -1 when not resident
        llama_disk_stage_cache_layer pub;    // public view returned by cache_layer()
    };
    ggml_backend_buffer_t cache_pool = nullptr;
    ggml_context *        cache_ctx  = nullptr;
    std::vector<cache_layer> cache;

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

bool llama_disk_stage::decode_full() {
    return env_flag_on("LLAMA_DISK_STAGE_DECODE_FULL");
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

llama_disk_stage::llama_disk_stage(const llama_model & model, ggml_backend_dev_t dev) :
    pimpl(std::make_unique<impl>(model)) {
    impl & p = *pimpl;

#if !defined(_WIN32)
    GGML_UNUSED(dev);
    return;
#else
    if (!env_flag_on("LLAMA_DISK_STAGE")) {
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
    size_t pool_off_total = 0;
    for (const auto & role : roles) {
        region_off[role.slot] = pool_off_total;
        pool_off_total += region_len[role.slot];
    }
    // slack for aligning the pool base plus any per-tensor allocation rounding
    const size_t total = pool_off_total + 2 * disk_stage_align;

    // the pool holds one layer's expert region per role; anything far larger
    // means the layout math is wrong, so refuse rather than commit huge memory
    const size_t total_max = (size_t) 16 << 30;
    if (total > total_max) {
        LLAMA_LOG_WARN("%s: staging pool would be %.1f GiB, refusing, disk staging disabled\n",
                       __func__, total / (1024.0 * 1024.0 * 1024.0));
        return;
    }

    // one pool, aliased by every layer's staging tensors. Prefer the device's
    // pinned host buffer so the offload copy into VRAM is not staged through
    // pageable memory; LLAMA_DISK_STAGE_PIN=0 forces the plain CPU buffer
    ggml_backend_buffer_type_t buft = nullptr;
    if (env_flag_default_on("LLAMA_DISK_STAGE_PIN")) {
        buft = ggml_backend_dev_host_buffer_type(dev);
    }
    if (buft == nullptr) {
        buft = ggml_backend_cpu_buffer_type();
    }

    p.pool = ggml_backend_buft_alloc_buffer(buft, total);
    if (p.pool == nullptr) {
        LLAMA_LOG_WARN("%s: failed to allocate the %zu MiB staging pool, disk staging disabled\n",
                       __func__, total / (1024 * 1024));
        return;
    }
    ggml_backend_buffer_set_usage(p.pool, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    p.base = (char *) align_up((uintptr_t) ggml_backend_buffer_get_base(p.pool), disk_stage_align);

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
            r.pool_off = region_off[role.slot];

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

    // persistent decode cache (LLAMA_DISK_STAGE_CACHE_MIB): one compact slot
    // array per layer, filled by the same unbuffered reader. The graph remaps
    // selected_experts through the layer's table and reads the weights in place,
    // so a resident expert is served without a copy.
    const uint64_t cache_mib = env_u64("LLAMA_DISK_STAGE_CACHE_MIB");
    if (cache_mib > 0) {
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

            int32_t slots_per_layer = (int32_t) ((cache_mib << 20) /
                    ((size_t) n_layer * std::max<size_t>(per_expert, 1)));
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
                        cache_bytes += align_up(src[il].r[role.slot].head + (size_t) slots_per_layer * (stride + disk_stage_align) + disk_stage_align, disk_stage_align);
                    }
                    cache_bytes += align_up((size_t) n_expert * sizeof(int32_t), disk_stage_align);
                }

                ggml_backend_buffer_type_t cbuft = env_flag_default_on("LLAMA_DISK_STAGE_PIN")
                    ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
                if (cbuft == nullptr) {
                    cbuft = ggml_backend_cpu_buffer_type();
                }
                p.cache_pool = ggml_backend_buft_alloc_buffer(cbuft, cache_bytes);
                if (p.cache_pool == nullptr) {
                    LLAMA_LOG_WARN("%s: failed to allocate the %.2f GiB decode cache\n",
                                   __func__, cache_bytes / (1024.0 * 1024.0 * 1024.0));
                } else {
                    ggml_backend_buffer_set_usage(p.cache_pool, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    char * cbase = (char *) align_up((uintptr_t) ggml_backend_buffer_get_base(p.cache_pool), disk_stage_align);

                    ggml_init_params cip = {
                        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (n_layer * 4 + 16),
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
                            c.resident_slot.assign((size_t) n_expert, -1);

                            ggml_tensor * tensors[3] = { nullptr, nullptr, nullptr };
                            for (const auto & role : roles) {
                                const ggml_tensor * s = src[il].t[role.slot];
                                ggml_tensor * ct = ggml_new_tensor_3d(p.cache_ctx, s->type, s->ne[0], s->ne[1], slots_per_layer);
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
                                off += align_up(src[il].r[role.slot].head + (size_t) slots_per_layer * ct->nb[2] + disk_stage_align, disk_stage_align);
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

                            c.gate  = tensors[0];
                            c.up    = tensors[1];
                            c.down  = tensors[2];
                            c.table = tab;
                            c.pub.gate    = c.gate;
                            c.pub.up      = c.up;
                            c.pub.down    = c.down;
                            c.pub.table   = c.table;
                            c.pub.n_slots = c.n_slots;
                            n_cache++;
                        }
                        if (p.cache_pool != nullptr) {
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

    p.active = true;
    LLAMA_LOG_INFO("%s: disk staging active for %d layer(s), %.1f MiB pool, %zu-byte aligned unbuffered reads\n",
                   __func__, n_staged, total / (1024.0 * 1024.0), disk_stage_align);

    // control: run the same reader over one long contiguous span of a model
    // file, the shape diskspd measured. If this hits ~7 GB/s while the per-layer
    // fills stay near 3, the reader is fine and the access pattern (gaps between
    // expert tensors, three shards) is what costs the bandwidth
    if (env_flag_on("LLAMA_DISK_STAGE_SELFTEST") && !p.files.empty()) {
        disk_stage_file * f = p.files.begin()->second.get();
        const std::string & fname = p.files.begin()->first;
        const size_t len = pool_off_total;

        std::vector<disk_stage_job> jobs;
        jobs.reserve(len / disk_stage_chunk + 1);
        for (size_t off = 0; off < len; off += disk_stage_chunk) {
            jobs.push_back({ f->h, p.base + off, off, std::min(disk_stage_chunk, len - off) });
        }
        for (int rep = 0; rep < 2; ++rep) {
            const int64_t t0 = ggml_time_us();
            disk_stage_run_jobs(jobs, 32, p.iocp);
            const int64_t t1 = ggml_time_us();
            const double secs = (t1 - t0) / 1e6;
            LLAMA_LOG_INFO("%s: selftest contiguous %.0f MiB from %s in %.1f ms = %.2f GB/s (qd=32)\n",
                           __func__, len / (1024.0 * 1024.0), fname.c_str(), secs * 1000.0,
                           secs > 0 ? len / (secs * 1e9) : 0.0);
        }
    }
#endif
}

llama_disk_stage::~llama_disk_stage() = default;

void llama_disk_stage::fill(int il) {
    impl & p = *pimpl;
    if (!p.active || il < 0 || il >= (int) p.layer_regions.size() || p.layer_regions[il].empty()) {
        return;
    }


#if defined(_WIN32)
    std::vector<disk_stage_job> jobs;
    for (const impl::region & r : p.layer_regions[il]) {
        char * dst = p.base + r.pool_off;
        for (size_t done = 0; done < r.read_len; done += disk_stage_chunk) {
            const size_t len = std::min(disk_stage_chunk, r.read_len - done);
            jobs.push_back({ r.file->h, dst + done, r.file_off + done, len });
        }
    }

    // queue depth 32 matches the diskspd measurement that saturated the drive
    const int queue_depth = 32;

    size_t bytes = 0;
    for (const disk_stage_job & j : jobs) {
        bytes += j.len;
    }

    const int64_t t0 = ggml_time_us();
    disk_stage_run_jobs(jobs, queue_depth, p.iocp);
    const int64_t t1 = ggml_time_us();

    // one line per layer: the raw fill rate, separate from the prefill average
    // (the disk is idle while the offload copies the staged layer into VRAM)
    const double secs = (t1 - t0) / 1e6;
    LLAMA_LOG_INFO("%s: layer %d fill: %.1f MiB in %.1f ms = %.2f GB/s (qd=%d)\n",
                   __func__, il, bytes / (1024.0 * 1024.0), secs * 1000.0,
                   secs > 0 ? bytes / (secs * 1e9) : 0.0, queue_depth);
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
    int32_t transient = c.n_resident;

    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || id >= (int32_t) c.resident_slot.size()) {
            continue;
        }

        const int32_t slot = c.resident_slot[id];
        if (slot >= 0) {
            // resident: filled once, stays in place
            table[id] = slot;
            continue;
        }

        int32_t use = -1;
        if (c.n_resident_used < c.n_resident) {
            // first-come resident set; promotion/eviction is a later step
            use = c.n_resident_used++;
            c.resident_slot[id] = use;
        } else {
            if (transient >= c.n_slots) {
                transient = c.n_resident;
            }
            use = transient++;
        }

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

    if (!jobs.empty()) {
        disk_stage_run_jobs(jobs, 32, p.iocp);
    }
#else
    GGML_UNUSED(il);
    GGML_UNUSED(ids);
    GGML_UNUSED(n_ids);
#endif
}
