#include "server-ckpt-store.h"

#include "server-disk-meta.h"

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

constexpr char META_MAGIC[8] = {'L', 'L', 'S', 'L', 'O', 'T', 'C', 'K'};
constexpr uint32_t META_VERSION = 2;
constexpr uint32_t META_FLAG_MTMD     = 1u << 0;
constexpr uint32_t META_FLAG_HAS_FULL = 1u << 1;
constexpr uint64_t META_HEADER_SIZE = 80;
constexpr uint64_t META_CKPT_SIZE   = 80;

const uint8_t zeros[server_ckpt_store::align_bytes] = {};

} // namespace

server_ckpt_store::server_ckpt_store(config cfg) : cfg(std::move(cfg)) {
    load_index();
}

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

void server_ckpt_store::remove_files(const std::string & path) const {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::remove(path + ".meta", ec);
    ec.clear();
    std::filesystem::remove(path + ".meta.tmp", ec);
}

void server_ckpt_store::load_index() {
    std::error_code ec;
    std::filesystem::create_directories(cfg.dir, ec);

    std::vector<std::filesystem::path> metas;
    for (const auto & entry : std::filesystem::directory_iterator(cfg.dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }

        const std::string name = entry.path().filename().string();
        if (name.rfind(FILE_PREFIX, 0) != 0) {
            continue;
        }

        if (string_ends_with(name, ".bin.meta")) {
            metas.push_back(entry.path());
        } else if (string_ends_with(name, ".bin.meta.tmp")) {
            std::filesystem::remove(entry.path(), ec);
            ec.clear();
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
            std::filesystem::path data_path = meta_path;
            data_path.replace_extension();
            LOG_WRN("[ckpt-store] removing incompatible or invalid session %s\n", data_path.string().c_str());
            remove_files(data_path.string());
            st.discarded++;
            continue;
        }

        index.push_back(std::move(s));
        st.restored++;
    }

    // raw files without a committed sidecar are interrupted writes
    for (const auto & entry : std::filesystem::directory_iterator(cfg.dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }

        const std::string name = entry.path().filename().string();
        if (name.rfind(FILE_PREFIX, 0) != 0 || !string_ends_with(name, ".bin")) {
            continue;
        }

        const std::filesystem::path meta_path = entry.path().string() + ".meta";
        if (!std::filesystem::exists(meta_path, ec) || ec) {
            ec.clear();
            LOG_WRN("[ckpt-store] removing incomplete session %s\n", entry.path().string().c_str());
            remove_files(entry.path().string());
        }
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
    uint64_t file_size = 0;
    uint64_t checkpoint_count = 0;
    uint64_t off_full_tgt = 0;
    uint64_t size_full_tgt = 0;
    uint64_t off_full_dft = 0;
    uint64_t size_full_dft = 0;

    bool ok = input.good() &&
        server_disk_meta_read(input, version) &&
        server_disk_meta_read(input, flags) &&
        server_disk_meta_read(input, key_size) &&
        server_disk_meta_read(input, tokens_size) &&
        server_disk_meta_read(input, file_size) &&
        server_disk_meta_read(input, checkpoint_count) &&
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
        (flags & ~(META_FLAG_MTMD | META_FLAG_HAS_FULL)) == 0 &&
        ((flags & META_FLAG_MTMD) != 0) == cfg.has_mtmd &&
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
            ? size_full_tgt > 0 &&
              off_full_tgt % align_bytes == 0 &&
              off_full_dft % align_bytes == 0 &&
              server_disk_range_valid(off_full_tgt, size_full_tgt, file_size) &&
              server_disk_range_valid(off_full_dft, size_full_dft, file_size)
            : size_full_tgt == 0 && size_full_dft == 0 && off_full_tgt == 0 && off_full_dft == 0);
    if (!ok) {
        LOG_WRN("[ckpt-store] reject %s: header (ver=%u flags=%u key_size=%" PRIu64 " expected=%zu tokens=%" PRIu64 " ckpt=%" PRIu64 " meta=%" PRIu64 " got=%" PRIu64 ")\n",
                meta_path.string().c_str(), version, flags, key_size, cfg.key.size(), tokens_size, checkpoint_count,
                expected_metadata_size, (uint64_t) metadata_file_size);
        return false;
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
        uint64_t off_tgt = 0;
        uint64_t size_tgt = 0;
        uint64_t off_dft = 0;
        uint64_t size_dft = 0;
        uint64_t off_spec = 0;
        uint64_t size_spec = 0;

        ok = server_disk_meta_read(input, n_tokens) &&
            server_disk_meta_read(input, id_task) &&
            server_disk_meta_read(input, reserved) &&
            server_disk_meta_read(input, pos_min) &&
            server_disk_meta_read(input, pos_max) &&
            server_disk_meta_read(input, off_tgt) &&
            server_disk_meta_read(input, size_tgt) &&
            server_disk_meta_read(input, off_dft) &&
            server_disk_meta_read(input, size_dft) &&
            server_disk_meta_read(input, off_spec) &&
            server_disk_meta_read(input, size_spec);

        ok = ok &&
            reserved == 0 &&
            n_tokens >= 0 && (uint64_t) n_tokens <= tokens.size() &&
            pos_min >= std::numeric_limits<llama_pos>::min() && pos_min <= std::numeric_limits<llama_pos>::max() &&
            pos_max >= std::numeric_limits<llama_pos>::min() && pos_max <= std::numeric_limits<llama_pos>::max() &&
            off_tgt % align_bytes == 0 &&
            size_tgt > 0 &&
            server_disk_range_valid(off_tgt, size_tgt, file_size) &&
            off_dft % align_bytes == 0 &&
            server_disk_range_valid(off_dft, size_dft, file_size) &&
            off_spec % align_bytes == 0 &&
            server_disk_range_valid(off_spec, size_spec, file_size) &&
            (size_dft > 0 || off_dft == 0) &&
            (size_spec > 0 || off_spec == 0);
        if (!ok) {
            LOG_WRN("[ckpt-store] reject %s: bad checkpoint %" PRIu64 "\n", meta_path.string().c_str(), i);
            return false;
        }

        common_prompt_checkpoint checkpoint;
        checkpoint.n_tokens  = n_tokens;
        checkpoint.id_task   = -1; // task ids from a previous run are meaningless
        checkpoint.pos_min   = (llama_pos) pos_min;
        checkpoint.pos_max   = (llama_pos) pos_max;
        checkpoint.on_disk   = true;
        checkpoint.off_tgt   = off_tgt;
        checkpoint.size_tgt  = size_tgt;
        checkpoint.off_dft   = off_dft;
        checkpoint.size_dft  = size_dft;
        checkpoint.off_spec  = off_spec;
        checkpoint.size_spec = size_spec;
        checkpoints.push_back(std::move(checkpoint));
    }

    std::filesystem::path data_path = meta_path;
    data_path.replace_extension();
    ec.clear();
    const uintmax_t data_file_size = std::filesystem::file_size(data_path, ec);
    if (ec || (uint64_t) data_file_size < file_size) {
        LOG_WRN("[ckpt-store] reject %s: data file missing or short (want %" PRIu64 ")\n", meta_path.string().c_str(), file_size);
        return false;
    }

    out.path = data_path.string();
    out.file_size = file_size;
    out.tokens = std::move(tokens);
    out.full.off_tgt  = off_full_tgt;
    out.full.size_tgt = size_full_tgt;
    out.full.off_dft  = off_full_dft;
    out.full.size_dft = size_full_dft;
    out.checkpoints = std::move(checkpoints);
    std::error_code ec_mtime;
    out.mtime = std::filesystem::last_write_time(meta_path, ec_mtime);
    return true;
}

bool server_ckpt_store::start_session(int slot_id, slot_state & s) {
    if (s.fp != nullptr) {
        return true;
    }

    std::error_code ec;
    std::filesystem::create_directories(cfg.dir, ec);

    // id includes the model key hash so a different model never reuses the file
    char name[128];
    std::snprintf(name, sizeof(name), "%s%d-%08x-%" PRId64 "%s",
            FILE_PREFIX, slot_id, (unsigned) (std::hash<std::string>{}(cfg.key) & 0xffffffff),
            ggml_time_us(), FILE_SUFFIX);

    s.path = (std::filesystem::path(cfg.dir) / name).string();
    s.fp   = std::fopen(s.path.c_str(), "wb");
    s.end  = 0;
    s.live = 0;
    s.dead = 0;
    s.recs.clear();
    s.has_meta = false;
    s.full = {};
    s.meta_tokens.clear();
    s.meta_ckpts.clear();

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

bool server_ckpt_store::write_full(int slot_id, const uint8_t * tgt, size_t size_tgt, const uint8_t * dft, size_t size_dft) {
    if (size_tgt == 0) {
        return false;
    }

    // a superseded snapshot is dead weight, drop it so compaction can reclaim it
    auto prev = slots.find(slot_id);
    if (prev != slots.end() && prev->second.full.off_tgt != 0) {
        release(slot_id, prev->second.full.off_tgt);
        if (prev->second.full.off_dft != 0) {
            release(slot_id, prev->second.full.off_dft);
        }
        prev->second.full = {};
    }

    const uint64_t off_tgt = append(slot_id, tgt, size_tgt);
    if (off_tgt == UINT64_MAX) {
        return false;
    }

    uint64_t off_dft = 0;
    if (size_dft > 0) {
        off_dft = append(slot_id, dft, size_dft);
        if (off_dft == UINT64_MAX) {
            return false;
        }
    }

    slot_state & s = state_for(slot_id);
    s.full.off_tgt  = off_tgt;
    s.full.size_tgt = size_tgt;
    s.full.off_dft  = off_dft;
    s.full.size_dft = size_dft;
    return true;
}

void server_ckpt_store::write_meta(int slot_id, const server_tokens & tokens, const std::list<common_prompt_checkpoint> & checkpoints) {
    auto it = slots.find(slot_id);
    if (it == slots.end() || it->second.fp == nullptr || it->second.path.empty()) {
        return;
    }
    slot_state & s = it->second;

    std::list<common_prompt_checkpoint> disk;
    uint32_t flags = tokens.has_mtmd ? META_FLAG_MTMD : 0;
    if (s.full.valid()) {
        flags |= META_FLAG_HAS_FULL;
    }
    for (const auto & c : checkpoints) {
        if (!c.on_disk) {
            continue;
        }
        disk.push_back(c);
    }

    const std::vector<char> serialized_tokens = tokens.serialize();

    const uint64_t key_size = cfg.key.size();
    const uint64_t tokens_size = serialized_tokens.size();
    const uint64_t file_size = s.end;
    const uint64_t checkpoint_count = disk.size();
    const uint64_t off_full_tgt  = s.full.off_tgt;
    const uint64_t size_full_tgt = s.full.size_tgt;
    const uint64_t off_full_dft  = s.full.off_dft;
    const uint64_t size_full_dft = s.full.size_dft;

    const std::string meta_path = s.path + ".meta";
    std::fflush(s.fp);

    const bool ok = server_disk_meta_commit(meta_path, [&](std::ostream & output) {
        output.write(META_MAGIC, sizeof(META_MAGIC));
        bool ok = output.good() &&
            server_disk_meta_write(output, META_VERSION) &&
            server_disk_meta_write(output, flags) &&
            server_disk_meta_write(output, key_size) &&
            server_disk_meta_write(output, tokens_size) &&
            server_disk_meta_write(output, file_size) &&
            server_disk_meta_write(output, checkpoint_count) &&
            server_disk_meta_write(output, off_full_tgt) &&
            server_disk_meta_write(output, size_full_tgt) &&
            server_disk_meta_write(output, off_full_dft) &&
            server_disk_meta_write(output, size_full_dft);

        if (ok && key_size > 0) {
            output.write(cfg.key.data(), key_size);
            ok = output.good();
        }
        if (ok && tokens_size > 0) {
            output.write(serialized_tokens.data(), tokens_size);
            ok = output.good();
        }

        for (const auto & c : disk) {
            const int64_t  n_tokens  = c.n_tokens;
            const int32_t  id_task   = c.id_task;
            const uint32_t reserved  = 0;
            const int64_t  pos_min   = c.pos_min;
            const int64_t  pos_max   = c.pos_max;
            const uint64_t off_tgt   = c.off_tgt;
            const uint64_t size_tgt  = c.size_tgt;
            const uint64_t off_dft   = c.off_dft;
            const uint64_t size_dft  = c.size_dft;
            const uint64_t off_spec  = c.off_spec;
            const uint64_t size_spec = c.size_spec;

            ok = ok &&
                server_disk_meta_write(output, n_tokens) &&
                server_disk_meta_write(output, id_task) &&
                server_disk_meta_write(output, reserved) &&
                server_disk_meta_write(output, pos_min) &&
                server_disk_meta_write(output, pos_max) &&
                server_disk_meta_write(output, off_tgt) &&
                server_disk_meta_write(output, size_tgt) &&
                server_disk_meta_write(output, off_dft) &&
                server_disk_meta_write(output, size_dft) &&
                server_disk_meta_write(output, off_spec) &&
                server_disk_meta_write(output, size_spec);
        }

        return ok;
    });

    if (!ok) {
        LOG_WRN("[ckpt-store] failed to write sidecar %s\n", meta_path.c_str());
        return;
    }

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
        // checkpoint positions are valid for the prompt being processed
        if (lcp != s.tokens.size()) {
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

    // make room: the slot may still hold an active (unfinished) session
    auto it = slots.find(slot_id);
    if (it != slots.end() && it->second.fp != nullptr) {
        finalize(slot_id);
    }

    // a crash between an append and the sidecar commit leaves bytes past the
    // committed size; drop them so the record offsets stay aligned
    std::error_code ec;
    std::filesystem::resize_file(s.path, s.file_size, ec);

    FILE * fp = std::fopen(s.path.c_str(), "ab");
    if (fp == nullptr) {
        LOG_WRN("[ckpt-store] failed to open session %s for append\n", s.path.c_str());
        return false;
    }

    slot_state & st_slot = slots[slot_id];
    st_slot.path = s.path;
    st_slot.fp   = fp;
    st_slot.end  = s.file_size;
    st_slot.live = 0;
    st_slot.dead = 0;
    st_slot.recs.clear();
    st_slot.full = s.full;
    for (const auto & c : s.checkpoints) {
        st_slot.recs.push_back({ c.off_tgt, c.size_tgt, true });
        st_slot.live += round_up(c.size_tgt);
    }
    st_slot.has_meta    = true;
    st_slot.meta_tokens = std::move(s.tokens);
    st_slot.meta_ckpts  = std::move(s.checkpoints);

    tokens      = st_slot.meta_tokens.clone();
    full        = st_slot.full;
    checkpoints = st_slot.meta_ckpts;
    st.adopted++;

    LOG_INF("[ckpt-store] adopted session %s for slot %d (%zu checkpoints, %.1f MiB)\n",
            st_slot.path.c_str(), slot_id, checkpoints.size(), st_slot.end / (1024.0 * 1024.0));
    return true;
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

    // the full state record moved too, keep the sidecar reference valid
    auto remap_off = [&](uint64_t & off) {
        if (off == 0) {
            return;
        }
        for (const auto & m : remap) {
            if (m.first == off) {
                off = m.second;
                return;
            }
        }
    };
    remap_off(s.full.off_tgt);
    remap_off(s.full.off_dft);

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
    slot_state & s = it->second;

    // the sidecar is already current; keep the finished session matchable
    if (s.has_meta && (!s.meta_ckpts.empty() || s.full.valid()) && !s.path.empty()) {
        session ses;
        ses.path      = s.path;
        ses.file_size = s.end;
        ses.tokens    = std::move(s.meta_tokens);
        ses.full      = s.full;
        ses.checkpoints = std::move(s.meta_ckpts);
        std::error_code ec;
        ses.mtime = std::filesystem::last_write_time(s.path, ec);
        index.push_back(std::move(ses));
    }

    if (s.fp != nullptr) {
        std::fflush(s.fp);
        std::fclose(s.fp);
        s.fp = nullptr;
    }

    s.recs.clear();
    s.path.clear();
    s.end  = 0;
    s.live = 0;
    s.dead = 0;
    s.has_meta = false;
    s.full = {};
    s.meta_tokens.clear();
    s.meta_ckpts.clear();
    s.session++;
}

size_t server_ckpt_store::total_bytes() const {
    std::error_code ec;
    size_t res = 0;
    for (const auto & e : std::filesystem::directory_iterator(cfg.dir, ec)) {
        if (!e.is_regular_file()) {
            continue;
        }
        const std::string name = e.path().filename().string();
        if (name.rfind(FILE_PREFIX, 0) != 0 || !string_ends_with(name, FILE_SUFFIX)) {
            continue;
        }
        res += (size_t) e.file_size();
    }
    return res;
}

size_t server_ckpt_store::n_files() const {
    std::error_code ec;
    size_t res = 0;
    for (const auto & e : std::filesystem::directory_iterator(cfg.dir, ec)) {
        if (e.is_regular_file()) {
            const std::string name = e.path().filename().string();
            if (name.rfind(FILE_PREFIX, 0) == 0 && string_ends_with(name, FILE_SUFFIX)) {
                res++;
            }
        }
    }
    return res;
}

void server_ckpt_store::enforce_cap(size_t incoming) {
    if (cfg.max_bytes == 0) {
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

    for (const auto & e : std::filesystem::directory_iterator(cfg.dir, ec)) {
        if (!e.is_regular_file()) {
            continue;
        }
        const std::string name = e.path().filename().string();
        if (name.rfind(FILE_PREFIX, 0) != 0 || !string_ends_with(name, ".bin")) {
            continue;
        }

        const std::string path = e.path().string();

        // never evict a file that is currently being appended to
        bool active = false;
        for (const auto & kv : slots) {
            if (kv.second.path == path) {
                active = true;
                break;
            }
        }
        if (active) {
            continue;
        }

        entries.push_back({ path, (size_t) e.file_size(), e.last_write_time(ec) });
        total += (size_t) e.file_size();
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
        if (std::filesystem::remove(e.path, ec2)) {
            total -= std::min(total, e.size);
            st.evicted++;
            st.evicted_bytes += e.size;
            remove_files(e.path);
            index.erase(std::remove_if(index.begin(), index.end(), [&](const session & s) {
                return s.path == e.path;
            }), index.end());
            LOG_INF("[ckpt-store] evicted %s (%.1f MiB)\n",
                    std::filesystem::path(e.path).filename().string().c_str(), e.size / (1024.0 * 1024.0));
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
            " | compactions=%" PRIu64 " (%.1f MiB rewritten)"
            " | evicted=%" PRIu64 " (%.1f MiB)"
            " | meta=%" PRIu64 " restored=%" PRIu64 " discarded=%" PRIu64 " adopted=%" PRIu64 "\n",
            n_files(), total_bytes() / (1024.0 * 1024.0),
            st.appends, st.write_bytes / (1024.0 * 1024.0),
            st.reads, st.read_bytes / (1024.0 * 1024.0),
            st.compactions, st.compact_read / (1024.0 * 1024.0),
            st.evicted, st.evicted_bytes / (1024.0 * 1024.0),
            st.meta_writes, st.restored, st.discarded, st.adopted);
}
