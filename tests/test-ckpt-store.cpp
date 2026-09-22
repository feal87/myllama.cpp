// Tests the server checkpoint store sidecar invariants.
//
// A checkpoint that is longer than the session token list used to be written
// as-is, which made read_meta reject the whole session on restart. Cover both
// the write-side filter and the tolerant read-side drop.

#include "server-ckpt-store.h"

#include "common.h"
#include "ggml.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " #cond " (line " << __LINE__ << ")\n"; \
        ++n_fail; \
    } \
} while (0)

static std::string test_dir(const char * name) {
    auto p = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.string();
}

static server_tokens make_tokens(int n) {
    llama_tokens t;
    for (int i = 0; i < n; ++i) {
        t.push_back(1000 + i);
    }
    return server_tokens(t, false);
}

static std::string find_meta(const std::string & dir) {
    for (const auto & e : std::filesystem::directory_iterator(dir)) {
        if (!e.is_directory()) {
            continue;
        }
        const auto meta = e.path() / "meta.bin";
        if (std::filesystem::exists(meta)) {
            return meta.string();
        }
    }
    return {};
}

// overwrite the n_tokens field of checkpoint record `idx`: header, key and token
// list are fixed size, each record is 104 bytes
static bool patch_checkpoint_n_tokens(const std::string & meta_path, std::streamoff record_idx, int64_t value) {
    std::fstream f(meta_path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f) {
        return false;
    }

    char magic[8] = {};
    uint32_t version = 0;
    uint32_t flags = 0;
    uint64_t key_size = 0;
    uint64_t tokens_size = 0;

    f.read(magic, sizeof(magic));
    f.read(reinterpret_cast<char *>(&version), sizeof(version));
    f.read(reinterpret_cast<char *>(&flags), sizeof(flags));
    f.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
    f.read(reinterpret_cast<char *>(&tokens_size), sizeof(tokens_size));
    if (!f) {
        return false;
    }

    const std::streamoff off = 80 + (std::streamoff) key_size + (std::streamoff) tokens_size + record_idx * 104;
    f.seekp(off);
    f.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return f.good();
}

// write one valid checkpoint and one that is longer than the token list
static void write_mixed_checkpoints(const std::string & dir, const server_tokens & tokens, const char * key) {
    server_ckpt_store::config cfg;
    cfg.dir = dir;
    cfg.key = key;

    server_ckpt_store store(cfg);

    std::vector<uint8_t> tgt(256, 0xab);

    common_prompt_checkpoint valid;
    CHECK(store.write_checkpoint(0, tgt, {}, {}, valid));
    valid.n_tokens = 32;
    valid.pos_min  = 31;
    valid.pos_max  = 31;

    common_prompt_checkpoint stale = valid;
    stale.n_tokens = (int64_t) tokens.size() + 1;
    stale.pos_min  = (llama_pos) tokens.size();
    stale.pos_max  = (llama_pos) tokens.size();

    std::list<common_prompt_checkpoint> ckpts = { valid, stale };
    store.write_meta(0, tokens, ckpts);
    // the destructor drains the write worker
}

static void test_write_filter(const std::string & dir, const server_tokens & tokens, const char * key) {
    std::cout << "test_write_filter\n";

    write_mixed_checkpoints(dir, tokens, key);

    server_ckpt_store::config cfg;
    cfg.dir = dir;
    cfg.key = key;
    server_ckpt_store store(cfg);

    CHECK(store.stats().restored == 1);
    CHECK(store.stats().discarded == 0);

    size_t index = 0;
    float f_keep = 0.0f;
    float f_sim = 0.0f;
    CHECK(store.find_best(tokens, index, f_keep, f_sim));

    server_tokens out_tokens;
    server_ckpt_store::full_state full;
    std::list<common_prompt_checkpoint> checkpoints;
    CHECK(store.adopt(0, index, out_tokens, full, checkpoints));
    CHECK(checkpoints.size() == 1);
    if (!checkpoints.empty()) {
        CHECK(checkpoints.front().n_tokens == 32);
    }
}

static void test_read_tolerance(const std::string & dir, const server_tokens & tokens, const char * key) {
    std::cout << "test_read_tolerance\n";

    {
        server_ckpt_store::config cfg;
        cfg.dir = dir;
        cfg.key = key;
        server_ckpt_store store(cfg);

        std::vector<uint8_t> tgt(256, 0xcd);

        common_prompt_checkpoint c0;
        CHECK(store.write_checkpoint(0, tgt, {}, {}, c0));
        c0.n_tokens = 32;
        c0.pos_min  = 31;
        c0.pos_max  = 31;

        common_prompt_checkpoint c1;
        CHECK(store.write_checkpoint(0, tgt, {}, {}, c1));
        c1.n_tokens = 48;
        c1.pos_min  = 47;
        c1.pos_max  = 47;

        std::list<common_prompt_checkpoint> ckpts = { c0, c1 };
        store.write_meta(0, tokens, ckpts);
    }

    const std::string meta = find_meta(dir);
    CHECK(!meta.empty());
    CHECK(patch_checkpoint_n_tokens(meta, 1, (int64_t) tokens.size() + 1));

    server_ckpt_store::config cfg;
    cfg.dir = dir;
    cfg.key = key;
    server_ckpt_store store(cfg);

    CHECK(store.stats().restored == 1);
    CHECK(store.stats().discarded == 0);

    size_t index = 0;
    float f_keep = 0.0f;
    float f_sim = 0.0f;
    CHECK(store.find_best(tokens, index, f_keep, f_sim));

    server_tokens out_tokens;
    server_ckpt_store::full_state full;
    std::list<common_prompt_checkpoint> checkpoints;
    CHECK(store.adopt(0, index, out_tokens, full, checkpoints));
    CHECK(checkpoints.size() == 1);
    if (!checkpoints.empty()) {
        CHECK(checkpoints.front().n_tokens == 32);
    }
}

int main() {
    ggml_time_init();

    const server_tokens tokens = make_tokens(64);
    const char * key = "test-key";

    test_write_filter(test_dir("test-ckpt-store-write"), tokens, key);
    test_read_tolerance(test_dir("test-ckpt-store-read"), tokens, key);

    if (n_fail != 0) {
        std::cerr << n_fail << " check(s) failed\n";
        return 1;
    }

    std::cout << "all checks passed\n";
    return 0;
}
