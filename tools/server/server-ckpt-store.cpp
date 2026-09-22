#include "server-ckpt-store.h"

#include "server-disk-meta.h"

#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
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

constexpr const char * DIR_PREFIX    = "llama-session-";
constexpr const char * LEGACY_PREFIX = "llama-slot-";
constexpr const char * META_NAME     = "meta.bin";
constexpr const char * ATTN_NAME     = "attn.bin";
constexpr const char * REC_PREFIX    = "rec-";
constexpr const char * REC_SUFFIX    = ".bin";
constexpr const char * TMP_SUFFIX    = ".tmp";

constexpr char META_MAGIC[8] = {'L', 'L', 'S', 'L', 'O', 'T', 'C', 'K'};
constexpr uint32_t META_VERSION = 5;
constexpr uint32_t META_FLAG_MTMD     = 1u << 0;
constexpr uint32_t META_FLAG_HAS_FULL = 1u << 1;
constexpr uint32_t META_FLAG_PARTIAL  = 1u << 2;
constexpr uint64_t META_HEADER_SIZE = 80;
constexpr uint64_t META_CKPT_SIZE   = 104;

const uint8_t zeros[server_ckpt_store::align_bytes] = {};

std::string record_name(uint64_t file_id) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%s%" PRIu64 "%s", REC_PREFIX, file_id, REC_SUFFIX);
    return buf;
}

bool parse_record_id(const std::string & name, uint64_t & file_id) {
    const size_t prefix_len = std::strlen(REC_PREFIX);
    const size_t suffix_len = std::strlen(REC_SUFFIX);
    if (name.rfind(REC_PREFIX, 0) != 0 || name.size() <= prefix_len + suffix_len ||
            !string_ends_with(name, REC_SUFFIX)) {
        return false;
    }

    uint64_t value = 0;
    for (size_t i = prefix_len; i < name.size() - suffix_len; ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return false;
        }
        value = value * 10 + (uint64_t) (name[i] - '0');
    }
    file_id = value;
    return true;
}

uint64_t dir_size(const std::filesystem::path & dir) {
    std::error_code ec;
    uint64_t total = 0;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        total += (uint64_t) entry.file_size(ec);
        if (ec) {
            ec.clear();
        }
    }
    return total;
}

} // namespace

server_ckpt_store::server_ckpt_store(config cfg) : cfg(std::move(cfg)) {
    load_index();
    worker = std::thread([this]() { worker_loop(); });
}

server_ckpt_store::~server_ckpt_store() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stopping = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

void server_ckpt_store::enqueue(std::function<void()> fn) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push_back(std::move(fn));
        pending++;
    }
    cv.notify_one();
}

void server_ckpt_store::wait_idle() {
    std::unique_lock<std::mutex> lock(mtx);
    cv_done.wait(lock, [this]() { return pending == 0; });
}

void server_ckpt_store::worker_loop() {
    while (true) {
        std::function<void()> fn;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]() { return stopping || !queue.empty(); });
            if (queue.empty()) {
                if (stopping) {
                    return;
                }
                continue;
            }
            fn = std::move(queue.front());
            queue.pop_front();
        }

        try {
            fn();
        } catch (const std::exception & e) {
            LOG_WRN("[ckpt-store] write job failed: %s\n", e.what());
        } catch (...) {
            LOG_WRN("%s", "[ckpt-store] write job failed\n");
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            pending--;
        }
        cv_done.notify_all();
    }
}

server_ckpt_store::slot_state & server_ckpt_store::state_for(int slot_id) {
    return slots[slot_id];
}

std::string server_ckpt_store::record_path(const std::string & dir, uint64_t file_id) const {
    return (std::filesystem::path(dir) / record_name(file_id)).string();
}

std::string server_ckpt_store::attn_path(const std::string & dir) const {
    return (std::filesystem::path(dir) / ATTN_NAME).string();
}

void server_ckpt_store::remove_dir(const std::string & dir) const {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void server_ckpt_store::load_index() {
    std::error_code ec;
    std::filesystem::create_directories(cfg.dir, ec);

    std::vector<std::filesystem::path> metas;
    for (const auto & entry : std::filesystem::directory_iterator(cfg.dir, ec)) {
        if (ec) {
            break;
        }

        const std::string name = entry.path().filename().string();
        if (!entry.is_directory(ec) || ec) {
            ec.clear();
            // remove files from an older format, they are not compatible
            if (entry.is_regular_file(ec) && name.rfind(LEGACY_PREFIX, 0) == 0) {
                std::error_code ec2;
                std::filesystem::remove(entry.path(), ec2);
            }
            continue;
        }

        if (name.rfind(DIR_PREFIX, 0) != 0) {
            continue;
        }

        const std::filesystem::path meta_path = entry.path() / META_NAME;
        if (std::filesystem::exists(meta_path, ec) && !ec) {
            metas.push_back(meta_path);
        } else {
            ec.clear();
            LOG_WRN("[ckpt-store] removing incomplete session %s\n", entry.path().string().c_str());
            remove_dir(entry.path().string());
        }
    }
    if (ec) {
        LOG_WRN("[ckpt-store] failed to scan %s: %s\n", cfg.dir.c_str(), ec.message().c_str());
        return;
    }

    std::sort(metas.begin(), metas.end(), [](const auto & a, const auto & b) {
        std::error_code ea;
        std::error_code eb;
        const auto ta = std::filesystem::last_write_time(a, ea);
        const auto tb = std::filesystem::last_write_time(b, eb);
        if (ea || eb) {
            return a.string() < b.string();
        }
        return ta < tb;
    });

    for (const auto & meta_path : metas) {
        session s;
        bool ok = false;
        try {
            ok = read_meta(meta_path, s);
        } catch (const std::exception & e) {
            LOG_WRN("[ckpt-store] failed to read sidecar %s: %s\n", meta_path.string().c_str(), e.what());
        }
        if (!ok) {
            LOG_WRN("[ckpt-store] removing incompatible or invalid session %s\n", meta_path.parent_path().string().c_str());
            remove_dir(meta_path.parent_path().string());
            st.discarded++;
            continue;
        }

        // drop record files written after the last sidecar commit
        {
            std::vector<uint64_t> referenced;
            if (s.full.file_id != 0) {
                referenced.push_back(s.full.file_id);
            }
            for (const auto & c : s.checkpoints) {
                referenced.push_back(c.disk_id);
            }

            std::error_code ec2;
            for (const auto & entry : std::filesystem::directory_iterator(s.dir, ec2)) {
                if (ec2 || !entry.is_regular_file(ec2) || ec2) {
                    ec2.clear();
                    continue;
                }
                const std::string name = entry.path().filename().string();
                if (string_ends_with(name, TMP_SUFFIX)) {
                    std::error_code ec3;
                    std::filesystem::remove(entry.path(), ec3);
                    continue;
                }

                uint64_t file_id = 0;
                if (!parse_record_id(name, file_id)) {
                    continue;
                }
                if (std::find(referenced.begin(), referenced.end(), file_id) == referenced.end()) {
                    LOG_TRC("[ckpt-store] removing orphan record %s\n", entry.path().string().c_str());
                    std::error_code ec3;
                    std::filesystem::remove(entry.path(), ec3);
                }
            }
        }

        index.push_back(std::move(s));
        st.restored++;
    }

    if (!index.empty()) {
        LOG_INF("[ckpt-store] restored %zu slot checkpoint sessions from %s\n", index.size(), cfg.dir.c_str());
    }
}

bool server_ckpt_store::read_meta(const std::filesystem::path & meta_path, session & out) const {
    std::error_code ec;
    const uintmax_t metadata_file_size = std::filesystem::file_size(meta_path, ec);
    if (ec || metadata_file_size < META_HEADER_SIZE) {
        return false;
    }

    const std::filesystem::path dir = meta_path.parent_path();

    std::ifstream input(meta_path, std::ios::binary);
    if (!input) {
        return false;
    }

    char magic[sizeof(META_MAGIC)];
    input.read(magic, sizeof(magic));

    uint32_t version = 0;
    uint32_t flags = 0;
    uint64_t key_size = 0;
    uint64_t tokens_size = 0;
    uint64_t checkpoint_count = 0;
    uint64_t full_file_id = 0;
    uint64_t off_full_tgt = 0;
    uint64_t size_full_tgt = 0;
    uint64_t off_full_dft = 0;
    uint64_t size_full_dft = 0;

    bool ok = input.good() &&
        server_disk_meta_read(input, version) &&
        server_disk_meta_read(input, flags) &&
        server_disk_meta_read(input, key_size) &&
        server_disk_meta_read(input, tokens_size) &&
        server_disk_meta_read(input, checkpoint_count) &&
        server_disk_meta_read(input, full_file_id) &&
        server_disk_meta_read(input, off_full_tgt) &&
        server_disk_meta_read(input, size_full_tgt) &&
        server_disk_meta_read(input, off_full_dft) &&
        server_disk_meta_read(input, size_full_dft);

    const bool has_full = (flags & META_FLAG_HAS_FULL) != 0;

    uint64_t checkpoint_bytes = 0;
    uint64_t expected_metadata_size = META_HEADER_SIZE;
    ok = ok &&
        std::memcmp(magic, META_MAGIC, sizeof(magic)) == 0 &&
        version == META_VERSION &&
        (flags & ~(META_FLAG_MTMD | META_FLAG_HAS_FULL | META_FLAG_PARTIAL)) == 0 &&
        ((flags & META_FLAG_MTMD) != 0) == cfg.has_mtmd &&
        ((flags & META_FLAG_PARTIAL) != 0) == cfg.partial_ckpt &&
        key_size == cfg.key.size() &&
        tokens_size > 0 &&
        tokens_size <= 1024ull * 1024ull * 1024ull &&
        tokens_size % sizeof(llama_token) == 0 &&
        checkpoint_count <= 1024 * 1024 &&
        server_disk_u64_mul(checkpoint_count, META_CKPT_SIZE, checkpoint_bytes) &&
        server_disk_u64_add(expected_metadata_size, key_size) &&
        server_disk_u64_add(expected_metadata_size, tokens_size) &&
        server_disk_u64_add(expected_metadata_size, checkpoint_bytes) &&
        expected_metadata_size == (uint64_t) metadata_file_size &&
        (checkpoint_count > 0 || has_full) &&
        (has_full
            ? size_full_tgt > 0 && off_full_tgt == 0 && off_full_dft % align_bytes == 0
            : size_full_tgt == 0 && size_full_dft == 0 && off_full_tgt == 0 && off_full_dft == 0);
    if (!ok) {
        LOG_WRN("[ckpt-store] reject %s: header (ver=%u flags=%u key_size=%" PRIu64 " expected=%zu tokens=%" PRIu64 " ckpt=%" PRIu64 " meta=%" PRIu64 " got=%" PRIu64 ")\n",
                meta_path.string().c_str(), version, flags, key_size, cfg.key.size(), tokens_size, checkpoint_count,
                expected_metadata_size, (uint64_t) metadata_file_size);
        return false;
    }

    auto record_size = [&](uint64_t file_id, uint64_t & size) -> bool {
        if (file_id == 0) {
            return false;
        }
        std::error_code ec2;
        const uintmax_t s = std::filesystem::file_size(dir / record_name(file_id), ec2);
        if (ec2) {
            return false;
        }
        size = (uint64_t) s;
        return true;
    };

    if (has_full) {
        uint64_t size_file = 0;
        if (!record_size(full_file_id, size_file) ||
                !server_disk_range_valid(off_full_tgt, size_full_tgt, size_file) ||
                (size_full_dft > 0 && (!server_disk_range_valid(off_full_dft, size_full_dft, size_file) || off_full_dft == 0))) {
            LOG_WRN("[ckpt-store] reject %s: invalid full state record\n", meta_path.string().c_str());
            return false;
        }
    }

    std::string key(key_size, '\0');
    input.read(key.data(), key_size);
    if (!input.good() || key != cfg.key) {
        LOG_WRN("[ckpt-store] reject %s: cache key mismatch\n", meta_path.string().c_str());
        return false;
    }

    std::vector<char> serialized_tokens(tokens_size);
    input.read(serialized_tokens.data(), tokens_size);
    if (!input.good()) {
        LOG_WRN("[ckpt-store] reject %s: truncated token list\n", meta_path.string().c_str());
        return false;
    }

    llama_tokens packed_tokens(tokens_size / sizeof(llama_token));
    std::memcpy(packed_tokens.data(), serialized_tokens.data(), tokens_size);
    server_tokens tokens = server_tokens::deserialize(packed_tokens, cfg.has_mtmd);
    if (tokens.empty()) {
        LOG_WRN("[ckpt-store] reject %s: empty token list\n", meta_path.string().c_str());
        return false;
    }

    std::list<common_prompt_checkpoint> checkpoints;
    for (uint64_t i = 0; i < checkpoint_count; ++i) {
        int64_t n_tokens = 0;
        int32_t id_task = -1;
        uint32_t reserved = 0;
        int64_t pos_min = 0;
        int64_t pos_max = 0;
        uint64_t disk_id = 0;
        uint64_t off_tgt = 0;
        uint64_t size_tgt = 0;
        uint64_t off_dft = 0;
        uint64_t size_dft = 0;
        uint64_t off_spec = 0;
        uint64_t size_spec = 0;
        int64_t len_ctx = -1;
        uint64_t fingerprint = 0;

        ok = server_disk_meta_read(input, n_tokens) &&
            server_disk_meta_read(input, id_task) &&
            server_disk_meta_read(input, reserved) &&
            server_disk_meta_read(input, pos_min) &&
            server_disk_meta_read(input, pos_max) &&
            server_disk_meta_read(input, disk_id) &&
            server_disk_meta_read(input, off_tgt) &&
            server_disk_meta_read(input, size_tgt) &&
            server_disk_meta_read(input, off_dft) &&
            server_disk_meta_read(input, size_dft) &&
            server_disk_meta_read(input, off_spec) &&
            server_disk_meta_read(input, size_spec) &&
            server_disk_meta_read(input, len_ctx) &&
            server_disk_meta_read(input, fingerprint);
        if (!ok) {
            LOG_WRN("[ckpt-store] reject %s: truncated checkpoint %" PRIu64 "\n", meta_path.string().c_str(), i);
            return false;
        }

        // a checkpoint that does not fit the token list is stale, not corrupt:
        // drop it and keep the session usable
        uint64_t size_file = 0;
        const bool valid =
            reserved == 0 &&
            n_tokens > 0 && (uint64_t) n_tokens <= tokens.size() &&
            pos_min >= std::numeric_limits<llama_pos>::min() && pos_min <= std::numeric_limits<llama_pos>::max() &&
            pos_max >= std::numeric_limits<llama_pos>::min() && pos_max <= std::numeric_limits<llama_pos>::max() &&
            record_size(disk_id, size_file) &&
            off_tgt == 0 && size_tgt > 0 &&
            server_disk_range_valid(off_tgt, size_tgt, size_file) &&
            (size_dft > 0 ? (off_dft != 0 && off_dft % align_bytes == 0 && server_disk_range_valid(off_dft, size_dft, size_file)) : off_dft == 0) &&
            (size_spec > 0 ? (off_spec != 0 && off_spec % align_bytes == 0 && server_disk_range_valid(off_spec, size_spec, size_file)) : off_spec == 0);
        if (!valid) {
            LOG_WRN("[ckpt-store] dropping invalid checkpoint %" PRIu64 " of %s (n_tokens=%" PRId64 ")\n",
                    i, meta_path.string().c_str(), n_tokens);
            continue;
        }

        common_prompt_checkpoint checkpoint;
        checkpoint.n_tokens  = n_tokens;
        checkpoint.id_task   = -1; // task ids from a previous run are meaningless
        checkpoint.pos_min   = (llama_pos) pos_min;
        checkpoint.pos_max   = (llama_pos) pos_max;
        checkpoint.len_ctx     = len_ctx;
        checkpoint.fingerprint = fingerprint;
        checkpoint.on_disk   = true;
        checkpoint.disk_id   = disk_id;
        checkpoint.off_tgt   = off_tgt;
        checkpoint.size_tgt  = size_tgt;
        checkpoint.off_dft   = off_dft;
        checkpoint.size_dft  = size_dft;
        checkpoint.off_spec  = off_spec;
        checkpoint.size_spec = size_spec;
        checkpoints.push_back(std::move(checkpoint));
    }

    if (checkpoints.empty() && !has_full) {
        LOG_WRN("[ckpt-store] reject %s: no usable checkpoint left\n", meta_path.string().c_str());
        return false;
    }

    out.dir = dir.string();
    out.tokens = std::move(tokens);
    if (has_full) {
        out.full.file_id  = full_file_id;
        out.full.off_tgt  = off_full_tgt;
        out.full.size_tgt = size_full_tgt;
        out.full.off_dft  = off_full_dft;
        out.full.size_dft = size_full_dft;
    }
    out.checkpoints = std::move(checkpoints);
    std::error_code ec_mtime;
    out.mtime = std::filesystem::last_write_time(meta_path, ec_mtime);
    return true;
}

bool server_ckpt_store::start_session(int slot_id, slot_state & s) {
    if (!s.dir.empty()) {
        return true;
    }

    std::error_code ec;
    std::filesystem::create_directories(cfg.dir, ec);

    // the key hash keeps a different model from reusing the directory
    char name[128];
    std::snprintf(name, sizeof(name), "%s%08x-%" PRId64,
            DIR_PREFIX, (unsigned) (std::hash<std::string>{}(cfg.key) & 0xffffffff), ggml_time_us());

    s.dir = (std::filesystem::path(cfg.dir) / name).string();
    if (!std::filesystem::create_directories(s.dir, ec) && ec) {
        s.dir.clear();
        return false;
    }

    s.next_file_id    = 1;
    s.full_committed  = 0;
    s.attn_covered    = 0;
    s.attn_tokens.clear();
    s.pending_remove.clear();
    s.has_meta        = false;
    s.full            = {};
    s.meta_tokens.clear();
    s.meta_ckpts.clear();

    enforce_cap(0);
    return true;
}

namespace {

// write tgt/dft/spec back to back, each padded to the alignment so direct IO
// reads never cross the end of the file
bool write_record_file(const std::string & path,
        const std::vector<uint8_t> & tgt,
        const std::vector<uint8_t> & dft,
        const std::vector<uint8_t> & spec) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }

    bool ok = true;
    auto put = [&](const std::vector<uint8_t> & data) {
        if (!ok || data.empty()) {
            return;
        }
        if (std::fwrite(data.data(), 1, data.size(), f) != data.size()) {
            ok = false;
            return;
        }
        const size_t pad = (size_t) (server_ckpt_store::round_up(data.size()) - data.size());
        if (pad > 0 && std::fwrite(zeros, 1, pad, f) != pad) {
            ok = false;
        }
    };

    put(tgt);
    put(dft);
    put(spec);

    if (std::fflush(f) != 0) {
        ok = false;
    }
    if (std::fclose(f) != 0) {
        ok = false;
    }
    return ok;
}

} // namespace

bool server_ckpt_store::write_checkpoint(int slot_id,
        std::vector<uint8_t> tgt,
        std::vector<uint8_t> dft,
        std::vector<uint8_t> spec,
        common_prompt_checkpoint & out) {
    if (tgt.empty()) {
        return false;
    }

    slot_state & s = state_for(slot_id);
    if (!start_session(slot_id, s)) {
        return false;
    }

    const uint64_t file_id  = s.next_file_id;
    const uint64_t size_tgt = tgt.size();
    const uint64_t size_dft = dft.size();
    const uint64_t size_spec = spec.size();
    s.next_file_id = file_id + 1;

    out.on_disk   = true;
    out.disk_id   = file_id;
    out.off_tgt   = 0;
    out.size_tgt  = size_tgt;
    out.off_dft   = size_dft  > 0 ? round_up(size_tgt) : 0;
    out.size_dft  = size_dft;
    out.off_spec  = size_spec > 0 ? round_up(size_tgt) + round_up(size_dft) : 0;
    out.size_spec = size_spec;

    st.appends++;
    st.write_bytes += size_tgt + size_dft + size_spec;
    LOG_TRC("[ckpt-store] write checkpoint slot %d file=%" PRIu64 " size=%" PRIu64 "\n", slot_id, file_id, size_tgt);

    const std::string path = record_path(s.dir, file_id);
    enqueue([path, tgt = std::move(tgt), dft = std::move(dft), spec = std::move(spec)]() {
        if (!write_record_file(path, tgt, dft, spec)) {
            LOG_WRN("[ckpt-store] failed to write record %s\n", path.c_str());
        }
    });
    return true;
}

bool server_ckpt_store::write_full(int slot_id, std::vector<uint8_t> tgt, std::vector<uint8_t> dft) {
    if (tgt.empty()) {
        return false;
    }

    slot_state & s = state_for(slot_id);
    if (!start_session(slot_id, s)) {
        return false;
    }

    // drop a full state that was written but never committed; the queue is FIFO,
    // so the delete lands after the pending write of that generation
    if (s.full.file_id != 0 && s.full.file_id != s.full_committed) {
        const std::string stale = record_path(s.dir, s.full.file_id);
        enqueue([stale]() {
            std::error_code ec;
            std::filesystem::remove(stale, ec);
        });
    }

    const uint64_t file_id  = s.next_file_id;
    const uint64_t size_tgt = tgt.size();
    const uint64_t size_dft = dft.size();
    s.next_file_id = file_id + 1;

    full_state full;
    full.file_id  = file_id;
    full.off_tgt  = 0;
    full.size_tgt = size_tgt;
    full.off_dft  = size_dft > 0 ? round_up(size_tgt) : 0;
    full.size_dft = size_dft;
    s.full = full;

    st.appends++;
    st.write_bytes += size_tgt + size_dft;
    LOG_TRC("[ckpt-store] write full slot %d file=%" PRIu64 " size=%" PRIu64 "\n", slot_id, file_id, size_tgt);

    const std::string path = record_path(s.dir, file_id);
    enqueue([path, tgt = std::move(tgt), dft = std::move(dft)]() {
        if (!write_record_file(path, tgt, dft, {})) {
            LOG_WRN("[ckpt-store] failed to write record %s\n", path.c_str());
        }
    });
    return true;
}

bool server_ckpt_store::append_attention(int slot_id, int64_t pos_end, const server_tokens & tokens,
        const std::function<std::vector<uint8_t>(int64_t, int64_t)> & gen) {
    auto it = slots.find(slot_id);
    if (it == slots.end() || it->second.dir.empty() || pos_end <= 0) {
        return false;
    }
    slot_state & s = it->second;

    // the log is valid up to the common prefix with the history it was built
    // from. a log that was just adopted, or one that runs past either of those,
    // has to be rebuilt from zero.
    const size_t lcp = s.attn_tokens.empty() ? 0 : s.attn_tokens.get_common_prefix(tokens);
    const bool reset = s.attn_covered <= 0 || s.attn_covered > pos_end || (int64_t) lcp < s.attn_covered;
    const int64_t begin = reset ? 0 : s.attn_covered;
    if (pos_end <= begin) {
        return true;
    }

    std::vector<uint8_t> data = gen(begin, pos_end);
    if (data.empty()) {
        return false;
    }
    const size_t n_bytes = data.size();

    const std::string path = attn_path(s.dir);
    enqueue([path, data = std::move(data), reset]() {
        FILE * f = std::fopen(path.c_str(), reset ? "wb" : "ab");
        if (f == nullptr) {
            LOG_WRN("[ckpt-store] failed to open attention %s\n", path.c_str());
            return;
        }
        bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
        if (std::fflush(f) != 0) {
            ok = false;
        }
        if (std::fclose(f) != 0) {
            ok = false;
        }
        if (!ok) {
            LOG_WRN("[ckpt-store] failed to write attention %s\n", path.c_str());
        }
    });

    s.attn_covered = pos_end;
    s.attn_tokens  = tokens.clone();
    st.write_bytes += n_bytes;
    LOG_TRC("[ckpt-store] append attention slot %d [%" PRId64 ", %" PRId64 ") size=%zu reset=%d\n",
            slot_id, begin, pos_end, n_bytes, (int) reset);
    return true;
}

bool server_ckpt_store::session_diverged(int slot_id, const server_tokens & tokens) const {
    auto it = slots.find(slot_id);
    if (it == slots.end() || it->second.attn_covered <= 0 || it->second.attn_tokens.empty()) {
        return false;
    }

    const slot_state & s = it->second;
    return (int64_t) s.attn_tokens.get_common_prefix(tokens) < s.attn_covered;
}

bool server_ckpt_store::load_attention(int slot_id, llama_context * ctx, llama_seq_id seq_id) {
    auto it = slots.find(slot_id);
    if (it == slots.end() || it->second.dir.empty()) {
        return false;
    }

    // a rebuild from the log is the only consumer, so drain first
    wait_idle();

    std::ifstream input(attn_path(it->second.dir), std::ios::binary);
    if (!input) {
        return true; // no log yet
    }

    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (buf.empty()) {
        return true;
    }

    size_t off = 0;
    while (off < buf.size()) {
        const size_t n = llama_state_seq_set_data_range_ext(ctx, buf.data() + off, buf.size() - off, seq_id, -1, -1, 0);
        if (n == 0) {
            LOG_WRN("[ckpt-store] failed to rebuild attention at offset %zu of %zu\n", off, buf.size());
            return false;
        }
        off += n;
    }

    st.reads++;
    st.read_bytes += buf.size();

    // the log carries attention only, so the recurrent cache would still be
    // empty. put it back at the newest checkpoint, which the log covers, so
    // the caches agree on a position again.
    const common_prompt_checkpoint * best = nullptr;
    for (const auto & c : it->second.meta_ckpts) {
        if (c.on_disk && c.size_tgt > 0 && (best == nullptr || c.n_tokens > best->n_tokens)) {
            best = &c;
        }
    }
    if (best == nullptr) {
        LOG_WRN("[ckpt-store] attention log for slot %d has no checkpoint to pair with\n", slot_id);
        return false;
    }

    std::vector<uint8_t> recr(best->size_tgt);
    if (!read_file(record_path(it->second.dir, best->disk_id), best->off_tgt, best->size_tgt, recr.data()) ||
            llama_state_seq_set_data_ext(ctx, recr.data(), recr.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != best->size_tgt) {
        LOG_WRN("[ckpt-store] failed to restore the recurrent state for slot %d\n", slot_id);
        return false;
    }

    LOG_INF("[ckpt-store] rebuilt attention for slot %d from %.1f MiB, recurrent at %" PRId64 "\n",
            slot_id, buf.size() / 1024.0 / 1024.0, best->n_tokens);
    return true;
}

void server_ckpt_store::write_meta(int slot_id, const server_tokens & tokens, const std::list<common_prompt_checkpoint> & checkpoints) {
    auto it = slots.find(slot_id);
    if (it == slots.end() || it->second.dir.empty()) {
        return;
    }
    slot_state & s = it->second;

    std::list<common_prompt_checkpoint> disk;
    uint32_t flags = tokens.has_mtmd ? META_FLAG_MTMD : 0;
    if (cfg.partial_ckpt) {
        flags |= META_FLAG_PARTIAL;
    }
    if (s.full.valid()) {
        flags |= META_FLAG_HAS_FULL;
    }
    for (const auto & c : checkpoints) {
        if (!c.on_disk) {
            continue;
        }
        // a checkpoint can outlive the prompt it was made from when the session
        // switches to a shorter history; never let it into the sidecar
        if (c.n_tokens <= 0 || c.n_tokens > (int64_t) tokens.size()) {
            LOG_WRN("[ckpt-store] dropping checkpoint n_tokens=%" PRId64 " for a %zu token session\n",
                    c.n_tokens, tokens.size());
            continue;
        }
        disk.push_back(c);
    }

    const std::vector<char> serialized_tokens = tokens.serialize();

    const uint64_t key_size = cfg.key.size();
    const uint64_t tokens_size = serialized_tokens.size();
    const uint64_t checkpoint_count = disk.size();
    const uint64_t full_file_id  = s.full.file_id;
    const uint64_t off_full_tgt  = s.full.off_tgt;
    const uint64_t size_full_tgt = s.full.size_tgt;
    const uint64_t off_full_dft  = s.full.off_dft;
    const uint64_t size_full_dft = s.full.size_dft;

    std::ostringstream body;
    body.write(META_MAGIC, sizeof(META_MAGIC));
    bool ok = body.good() &&
        server_disk_meta_write(body, META_VERSION) &&
        server_disk_meta_write(body, flags) &&
        server_disk_meta_write(body, key_size) &&
        server_disk_meta_write(body, tokens_size) &&
        server_disk_meta_write(body, checkpoint_count) &&
        server_disk_meta_write(body, full_file_id) &&
        server_disk_meta_write(body, off_full_tgt) &&
        server_disk_meta_write(body, size_full_tgt) &&
        server_disk_meta_write(body, off_full_dft) &&
        server_disk_meta_write(body, size_full_dft);

    if (ok && key_size > 0) {
        body.write(cfg.key.data(), key_size);
        ok = body.good();
    }
    if (ok && tokens_size > 0) {
        body.write(serialized_tokens.data(), tokens_size);
        ok = body.good();
    }

    for (const auto & c : disk) {
        const int64_t  n_tokens  = c.n_tokens;
        const int32_t  id_task   = c.id_task;
        const uint32_t reserved  = 0;
        const int64_t  pos_min   = c.pos_min;
        const int64_t  pos_max   = c.pos_max;
        const uint64_t disk_id   = c.disk_id;
        const uint64_t off_tgt   = c.off_tgt;
        const uint64_t size_tgt  = c.size_tgt;
        const uint64_t off_dft   = c.off_dft;
        const uint64_t size_dft  = c.size_dft;
        const uint64_t off_spec  = c.off_spec;
        const uint64_t size_spec = c.size_spec;
        const int64_t  len_ctx     = c.len_ctx;
        const uint64_t fingerprint = c.fingerprint;

        ok = ok &&
            server_disk_meta_write(body, n_tokens) &&
            server_disk_meta_write(body, id_task) &&
            server_disk_meta_write(body, reserved) &&
            server_disk_meta_write(body, pos_min) &&
            server_disk_meta_write(body, pos_max) &&
            server_disk_meta_write(body, disk_id) &&
            server_disk_meta_write(body, off_tgt) &&
            server_disk_meta_write(body, size_tgt) &&
            server_disk_meta_write(body, off_dft) &&
            server_disk_meta_write(body, size_dft) &&
            server_disk_meta_write(body, off_spec) &&
            server_disk_meta_write(body, size_spec) &&
            server_disk_meta_write(body, len_ctx) &&
            server_disk_meta_write(body, fingerprint);
    }

    if (!ok) {
        LOG_WRN("[ckpt-store] failed to serialize sidecar for slot %d\n", slot_id);
        return;
    }

    // paths to delete once the new sidecar is in place
    std::vector<std::string> to_delete;
    if (s.full_committed != 0 && s.full_committed != s.full.file_id) {
        to_delete.push_back(record_path(s.dir, s.full_committed));
    }
    for (const uint64_t file_id : s.pending_remove) {
        to_delete.push_back(record_path(s.dir, file_id));
    }

    const std::string meta_path = (std::filesystem::path(s.dir) / META_NAME).string();
    std::string body_str = body.str();

    enqueue([meta_path, body_str = std::move(body_str), to_delete = std::move(to_delete)]() {
        const bool committed = server_disk_meta_commit(meta_path, [&](std::ostream & output) {
            output.write(body_str.data(), body_str.size());
            return output.good();
        });
        if (!committed) {
            LOG_WRN("[ckpt-store] failed to write sidecar %s\n", meta_path.c_str());
            return;
        }
        for (const auto & path : to_delete) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    });

    s.full_committed = s.full.file_id;
    s.pending_remove.clear();
    s.has_meta = true;
    s.meta_tokens = tokens.clone();
    s.meta_ckpts = std::move(disk);
    st.meta_writes++;
}

bool server_ckpt_store::find_best(const server_tokens & tokens, size_t & out_index, float & out_f_keep, float & out_f_sim) const {
    if (tokens.empty()) {
        return false;
    }

    bool found = false;
    size_t best = 0;
    size_t best_lcp = 0;

    for (size_t i = 0; i < index.size(); ++i) {
        const session & s = index[i];
        if (s.tokens.empty() || (s.checkpoints.empty() && !s.full.valid())) {
            continue;
        }

        const size_t lcp = s.tokens.get_common_prefix(tokens);
        // the stored session must cover a prefix of the slot prompt, so its
        // checkpoint positions are valid for the prompt being processed. when
        // there is no full state, only checkpoints are reused, so a session
        // that runs past the request is fine too.
        if (lcp != s.tokens.size() && !(lcp == tokens.size() && !s.full.valid())) {
            continue;
        }

        if (!found || lcp > best_lcp) {
            found = true;
            best = i;
            best_lcp = lcp;
        }
    }

    if (!found) {
        return false;
    }

    out_index  = best;
    out_f_keep = 1.0f;
    out_f_sim  = float(best_lcp) / tokens.size();
    return true;
}

bool server_ckpt_store::adopt(int slot_id, size_t index_pos, server_tokens & tokens,
        full_state & full, std::list<common_prompt_checkpoint> & checkpoints) {
    if (index_pos >= index.size()) {
        return false;
    }

    session s = std::move(index[index_pos]);
    index.erase(index.begin() + index_pos);
    if (s.checkpoints.empty() && !s.full.valid()) {
        return false;
    }

    // make room: the slot may still hold an active session
    auto it = slots.find(slot_id);
    if (it != slots.end() && !it->second.dir.empty()) {
        finalize(slot_id);
    }

    uint64_t next_file_id = 1;
    if (s.full.file_id >= next_file_id) {
        next_file_id = s.full.file_id + 1;
    }
    for (const auto & c : s.checkpoints) {
        if (c.disk_id >= next_file_id) {
            next_file_id = c.disk_id + 1;
        }
    }

    slot_state & st_slot = slots[slot_id];
    st_slot.dir            = std::move(s.dir);
    st_slot.next_file_id   = next_file_id;
    st_slot.full           = s.full;
    st_slot.full_committed = s.full.file_id;
    // the committed extent of the attention log is not tracked across restarts,
    // so the next checkpoint rebuilds it from zero
    st_slot.attn_covered   = 0;
    st_slot.attn_tokens.clear();
    st_slot.pending_remove.clear();
    st_slot.has_meta       = true;
    st_slot.meta_tokens    = std::move(s.tokens);
    st_slot.meta_ckpts     = std::move(s.checkpoints);

    tokens      = st_slot.meta_tokens.clone();
    full        = st_slot.full;
    checkpoints = st_slot.meta_ckpts;
    st.adopted++;

    LOG_INF("[ckpt-store] adopted session %s for slot %d (%zu checkpoints)\n",
            st_slot.dir.c_str(), slot_id, checkpoints.size());
    return true;
}

bool server_ckpt_store::read(int slot_id, uint64_t file_id, uint64_t off, size_t size, uint8_t * dst) {
    auto it = slots.find(slot_id);
    if (it == slots.end() || it->second.dir.empty()) {
        return false;
    }

    // the record may still be queued for the write worker
    wait_idle();

    const bool ok = read_file(record_path(it->second.dir, file_id), off, size, dst);
    if (ok) {
        st.reads++;
        st.read_bytes += size;
        LOG_TRC("[ckpt-store] read slot %d file=%" PRIu64 " off=%" PRIu64 " size=%zu\n", slot_id, file_id, off, size);
    }
    return ok;
}

void server_ckpt_store::release_file(int slot_id, uint64_t file_id) {
    auto it = slots.find(slot_id);
    if (it == slots.end() || it->second.dir.empty() || file_id == 0) {
        return;
    }

    it->second.pending_remove.push_back(file_id);
}

void server_ckpt_store::finalize(int slot_id) {
    auto it = slots.find(slot_id);
    if (it == slots.end()) {
        return;
    }
    slot_state & s = it->second;

    // pending records that the committed sidecar does not reference are orphans
    for (const uint64_t file_id : s.pending_remove) {
        bool referenced = file_id == s.full.file_id;
        for (const auto & c : s.meta_ckpts) {
            if (c.disk_id == file_id) {
                referenced = true;
                break;
            }
        }
        if (!referenced) {
            std::error_code ec;
            std::filesystem::remove(record_path(s.dir, file_id), ec);
        }
    }

    // the sidecar is already current; keep the finished session matchable
    if (s.has_meta && (!s.meta_ckpts.empty() || s.full.valid()) && !s.dir.empty()) {
        session ses;
        ses.dir         = s.dir;
        ses.tokens      = std::move(s.meta_tokens);
        ses.full        = s.full;
        ses.checkpoints = std::move(s.meta_ckpts);
        std::error_code ec;
        ses.mtime = std::filesystem::last_write_time(s.dir, ec);
        index.push_back(std::move(ses));
    }

    s.dir.clear();
    s.next_file_id   = 1;
    s.full_committed = 0;
    s.attn_covered   = 0;
    s.attn_tokens.clear();
    s.pending_remove.clear();
    s.has_meta       = false;
    s.full           = {};
    s.meta_tokens.clear();
    s.meta_ckpts.clear();
    s.session++;
}

size_t server_ckpt_store::total_bytes() const {
    std::error_code ec;
    size_t res = 0;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(cfg.dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        res += (size_t) entry.file_size(ec);
        if (ec) {
            ec.clear();
        }
    }
    return res;
}

size_t server_ckpt_store::n_files() const {
    std::error_code ec;
    size_t res = 0;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(cfg.dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        res++;
    }
    return res;
}

void server_ckpt_store::enforce_cap(size_t incoming) {
    if (cfg.max_bytes == 0) {
        return;
    }

    // do not evict a directory while a queued write still targets it
    wait_idle();

    struct entry {
        std::string dir;
        size_t size;
        std::filesystem::file_time_type mtime;
    };

    std::error_code ec;
    std::vector<entry> entries;
    size_t total = 0;

    for (const auto & e : std::filesystem::directory_iterator(cfg.dir, ec)) {
        if (ec) {
            break;
        }
        if (!e.is_directory(ec) || ec) {
            ec.clear();
            continue;
        }

        const std::string name = e.path().filename().string();
        if (name.rfind(DIR_PREFIX, 0) != 0) {
            continue;
        }

        const std::string dir = e.path().string();

        // never evict the session that a slot is currently writing to
        bool active = false;
        for (const auto & kv : slots) {
            if (kv.second.dir == dir) {
                active = true;
                break;
            }
        }
        if (active) {
            continue;
        }

        const size_t size = (size_t) dir_size(e.path());
        entries.push_back({ dir, size, e.last_write_time(ec) });
        total += size;
    }

    if (total + incoming <= cfg.max_bytes) {
        return;
    }

    std::sort(entries.begin(), entries.end(), [](const entry & a, const entry & b) {
        return a.mtime < b.mtime;
    });

    for (const auto & e : entries) {
        if (total + incoming <= cfg.max_bytes) {
            break;
        }
        std::error_code ec2;
        std::filesystem::remove_all(e.dir, ec2);
        if (!ec2) {
            total -= std::min(total, e.size);
            st.evicted++;
            st.evicted_bytes += e.size;
            index.erase(std::remove_if(index.begin(), index.end(), [&](const session & s) {
                return s.dir == e.dir;
            }), index.end());
            LOG_INF("[ckpt-store] evicted %s (%.1f MiB)\n",
                    std::filesystem::path(e.dir).filename().string().c_str(), e.size / (1024.0 * 1024.0));
        }
    }
}

bool server_ckpt_store::read_file(const std::string & path, uint64_t off, size_t size, uint8_t * dst) const {
    if (size == 0) {
        return true;
    }

    if (cfg.use_dio) {
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
            " | evicted=%" PRIu64 " (%.1f MiB)"
            " | meta=%" PRIu64 " restored=%" PRIu64 " discarded=%" PRIu64 " adopted=%" PRIu64 "\n",
            n_files(), total_bytes() / (1024.0 * 1024.0),
            st.appends, st.write_bytes / (1024.0 * 1024.0),
            st.reads, st.read_bytes / (1024.0 * 1024.0),
            st.evicted, st.evicted_bytes / (1024.0 * 1024.0),
            st.meta_writes, st.restored, st.discarded, st.adopted);
}
