#include "server-ckpt-store.h"

#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <system_error>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

constexpr const char * FILE_PREFIX = "llama-slot-";
constexpr const char * FILE_SUFFIX = ".bin";

const uint8_t zeros[server_ckpt_store::align_bytes] = {};

} // namespace

server_ckpt_store::server_ckpt_store(std::string dir, std::string key, bool use_dio, size_t max_bytes)
    : dir(std::move(dir)), key(std::move(key)), use_dio(use_dio), max_bytes(max_bytes) {}

server_ckpt_store::~server_ckpt_store() {
    for (auto & kv : slots) {
        if (kv.second.fp != nullptr) {
            std::fclose(kv.second.fp);
            kv.second.fp = nullptr;
        }
    }
}

server_ckpt_store::slot_state & server_ckpt_store::state_for(int slot_id) {
    return slots[slot_id];
}

bool server_ckpt_store::start_session(int slot_id, slot_state & s) {
    if (s.fp != nullptr) {
        return true;
    }

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    // id includes the model key hash so a different model never reuses the file
    char name[128];
    std::snprintf(name, sizeof(name), "%s%d-%08x-%" PRId64 "%s",
            FILE_PREFIX, slot_id, (unsigned) (std::hash<std::string>{}(key) & 0xffffffff),
            ggml_time_us(), FILE_SUFFIX);

    s.path = (std::filesystem::path(dir) / name).string();
    s.fp   = std::fopen(s.path.c_str(), "wb");
    s.end  = 0;
    s.live = 0;
    s.dead = 0;
    s.recs.clear();

    if (s.fp == nullptr) {
        return false;
    }

    enforce_cap(0);
    return true;
}

uint64_t server_ckpt_store::append(int slot_id, const uint8_t * data, size_t size) {
    slot_state & s = state_for(slot_id);

    if (!start_session(slot_id, s)) {
        return UINT64_MAX;
    }

    const uint64_t off = s.end;
    const uint64_t pad = round_up(size) - size;

    if (std::fwrite(data, 1, size, s.fp) != size) {
        return UINT64_MAX;
    }
    if (pad > 0 && std::fwrite(zeros, 1, pad, s.fp) != pad) {
        return UINT64_MAX;
    }

    s.end  += round_up(size);
    s.live += round_up(size);
    s.recs.push_back({ off, size, true });

    st.appends++;
    st.write_bytes += size;
    LOG_TRC("[ckpt-store] append slot %d off=%" PRIu64 " size=%zu\n", slot_id, off, size);
    return off;
}

bool server_ckpt_store::read(int slot_id, uint64_t off, size_t size, uint8_t * dst) {
    auto it = slots.find(slot_id);
    if (it == slots.end()) {
        return false;
    }

    // the record may still be in the write buffer of the append handle
    if (it->second.fp != nullptr) {
        std::fflush(it->second.fp);
    }

    const bool ok = read_file(it->second.path, off, size, dst);
    if (ok) {
        st.reads++;
        st.read_bytes += size;
        LOG_TRC("[ckpt-store] read slot %d off=%" PRIu64 " size=%zu\n", slot_id, off, size);
    }
    return ok;
}

void server_ckpt_store::release(int slot_id, uint64_t off) {
    auto it = slots.find(slot_id);
    if (it == slots.end()) {
        return;
    }

    for (auto & r : it->second.recs) {
        if (r.off == off && r.live) {
            r.live = false;
            const uint64_t n = round_up(r.size);
            it->second.live -= std::min(it->second.live, n);
            it->second.dead += n;
            return;
        }
    }
}

void server_ckpt_store::maybe_compact(int slot_id, std::vector<std::pair<uint64_t, uint64_t>> & remap) {
    auto it = slots.find(slot_id);
    if (it == slots.end()) {
        return;
    }
    slot_state & s = it->second;

    if (s.fp == nullptr || s.dead <= s.live) {
        return;
    }

    // close the write handle so the rewrite can read the same path on Windows
    std::fflush(s.fp);
    std::fclose(s.fp);
    s.fp = nullptr;

    const std::string tmp = s.path + ".tmp";
    FILE * out = std::fopen(tmp.c_str(), "wb");
    if (out == nullptr) {
        s.fp = std::fopen(s.path.c_str(), "ab");
        return;
    }

    std::vector<uint8_t> buf;
    std::vector<rec>     new_recs;
    uint64_t             new_end = 0;

    for (auto & r : s.recs) {
        if (!r.live) {
            continue;
        }

        buf.resize(r.size);
        if (!read_file(s.path, r.off, r.size, buf.data())) {
            std::fclose(out);
            std::remove(tmp.c_str());
            s.fp = std::fopen(s.path.c_str(), "ab");
            return;
        }

        const uint64_t pad = round_up(r.size) - r.size;
        if (std::fwrite(buf.data(), 1, r.size, out) != r.size ||
                (pad > 0 && std::fwrite(zeros, 1, pad, out) != pad)) {
            std::fclose(out);
            std::remove(tmp.c_str());
            s.fp = std::fopen(s.path.c_str(), "ab");
            return;
        }

        remap.emplace_back(r.off, new_end);
        new_recs.push_back({ new_end, r.size, true });
        new_end += round_up(r.size);
    }

    std::fclose(out);
    std::remove(s.path.c_str());
    std::rename(tmp.c_str(), s.path.c_str());

    const uint64_t dead_before = s.dead;
    const uint64_t live_before = s.live;

    s.fp   = std::fopen(s.path.c_str(), "ab");
    s.end  = new_end;
    s.live = new_end;
    s.dead = 0;
    s.recs = std::move(new_recs);

    st.compactions++;
    st.compact_read += new_end;

    LOG_INF("[ckpt-store] compacted slot %d: %.1f MiB written (was %.1f MiB live + %.1f MiB dead)\n",
            slot_id, new_end / (1024.0 * 1024.0), live_before / (1024.0 * 1024.0), dead_before / (1024.0 * 1024.0));
}

void server_ckpt_store::finalize(int slot_id) {
    auto it = slots.find(slot_id);
    if (it == slots.end()) {
        return;
    }

    if (it->second.fp != nullptr) {
        std::fflush(it->second.fp);
        std::fclose(it->second.fp);
        it->second.fp = nullptr;
    }

    it->second.recs.clear();
    it->second.path.clear();
    it->second.end  = 0;
    it->second.live = 0;
    it->second.dead = 0;
    it->second.session++;
}

size_t server_ckpt_store::total_bytes() const {
    std::error_code ec;
    size_t res = 0;
    for (const auto & e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) {
            continue;
        }
        const std::string name = e.path().filename().string();
        if (name.rfind(FILE_PREFIX, 0) != 0) {
            continue;
        }
        res += (size_t) e.file_size();
    }
    return res;
}

size_t server_ckpt_store::n_files() const {
    std::error_code ec;
    size_t res = 0;
    for (const auto & e : std::filesystem::directory_iterator(dir, ec)) {
        if (e.is_regular_file() && e.path().filename().string().rfind(FILE_PREFIX, 0) == 0) {
            res++;
        }
    }
    return res;
}

void server_ckpt_store::enforce_cap(size_t incoming) {
    if (max_bytes == 0) {
        return;
    }

    struct entry {
        std::string path;
        size_t size;
        std::filesystem::file_time_type mtime;
    };

    std::error_code ec;
    std::vector<entry> entries;
    size_t total = 0;

    for (const auto & e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) {
            continue;
        }
        const std::string name = e.path().filename().string();
        if (name.rfind(FILE_PREFIX, 0) != 0) {
            continue;
        }
        entries.push_back({ e.path().string(), (size_t) e.file_size(), e.last_write_time(ec) });
        total += (size_t) e.file_size();
    }

    if (total + incoming <= max_bytes) {
        return;
    }

    std::sort(entries.begin(), entries.end(), [](const entry & a, const entry & b) {
        return a.mtime < b.mtime;
    });

    for (const auto & e : entries) {
        if (total + incoming <= max_bytes) {
            break;
        }
        std::error_code ec2;
        if (std::filesystem::remove(e.path, ec2)) {
            total -= std::min(total, e.size);
            st.evicted++;
            st.evicted_bytes += e.size;
            LOG_INF("[ckpt-store] evicted %s (%.1f MiB)\n",
                    std::filesystem::path(e.path).filename().string().c_str(), e.size / (1024.0 * 1024.0));
        }
    }
}

bool server_ckpt_store::read_file(const std::string & path, uint64_t off, size_t size, uint8_t * dst) const {
    if (size == 0) {
        return true;
    }

    if (use_dio) {
#ifdef _WIN32
        HANDLE h = CreateFileA(path.c_str(), GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                FILE_FLAG_NO_BUFFERING, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            const size_t rsize = (size_t) round_up(size);
            void * buf = _aligned_malloc(rsize, align_bytes);
            if (buf != nullptr) {
                OVERLAPPED ov = {};
                ov.Offset     = (DWORD) (off & 0xffffffff);
                ov.OffsetHigh = (DWORD) (off >> 32);
                DWORD got = 0;
                const BOOL ok = ReadFile(h, buf, (DWORD) rsize, &got, &ov);
                const bool res = ok && got >= size;
                if (res) {
                    std::memcpy(dst, buf, size);
                }
                _aligned_free(buf);
                CloseHandle(h);
                if (res) {
                    return true;
                }
            } else {
                CloseHandle(h);
            }
        }
#else
        const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd >= 0) {
            const size_t rsize = (size_t) round_up(size);
            void * buf = nullptr;
            if (posix_memalign(&buf, align_bytes, rsize) == 0) {
                const ssize_t got = pread(fd, buf, rsize, (off_t) off);
                const bool res = got >= (ssize_t) size;
                if (res) {
                    std::memcpy(dst, buf, size);
                }
                free(buf);
                close(fd);
                if (res) {
                    return true;
                }
            } else {
                close(fd);
            }
        }
#endif
    }

    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = std::fseek(f, (long) off, SEEK_SET) == 0 &&
                    std::fread(dst, 1, size, f) == size;
    std::fclose(f);
    return ok;
}

void server_ckpt_store::print_stats() const {
    LOG_INF("[ckpt-store] files=%zu bytes=%.1f MiB | appends=%" PRIu64
            " writes=%.1f MiB | reads=%" PRIu64 " (%.1f MiB)"
            " | compactions=%" PRIu64 " (%.1f MiB rewritten)"
            " | evicted=%" PRIu64 " (%.1f MiB)\n",
            n_files(), total_bytes() / (1024.0 * 1024.0),
            st.appends, st.write_bytes / (1024.0 * 1024.0),
            st.reads, st.read_bytes / (1024.0 * 1024.0),
            st.compactions, st.compact_read / (1024.0 * 1024.0),
            st.evicted, st.evicted_bytes / (1024.0 * 1024.0));
}
