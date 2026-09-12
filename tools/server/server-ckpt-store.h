#pragma once

// Disk-backed store for slot context checkpoints.
//
// One append-only file per slot session, records padded to a fixed alignment so
// reads can go through direct IO without a bounce buffer per record. Only the
// newest blob stays in RAM, everything else lives here. Metadata (position,
// size, offset) stays in common_prompt_checkpoint.
//
// Each session also keeps a <file>.bin.meta sidecar with the token list and the
// checkpoint index. The sidecar carries the prompt cache settings key, so a
// config change invalidates it. At startup the sidecars rebuild an in-RAM index
// of sessions that can be adopted by a slot after a restart.

#include "common.h"
#include "server-common.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct server_ckpt_store {
    // 4096 keeps both the O_DIRECT block size and the Windows unbuffered sector
    // size happy; records start aligned and are padded to this.
    static constexpr uint64_t align_bytes = 4096;

    static uint64_t round_up(uint64_t n) {
        return (n + align_bytes - 1) & ~(align_bytes - 1);
    }

    struct config {
        std::string dir;
        std::string key;
        bool        use_dio  = false;
        bool        has_mtmd = false;
        size_t      max_bytes = 0; // 0 = no limit
    };

    // a full state snapshot (LLAMA_STATE_SEQ_FLAGS_NONE), self-sufficient after
    // a restart: it carries the attention cache and the recurrent state
    struct full_state {
        uint64_t off_tgt  = 0;
        uint64_t size_tgt = 0;
        uint64_t off_dft  = 0;
        uint64_t size_dft = 0;

        bool valid() const { return size_tgt > 0; }
    };

    explicit server_ckpt_store(config cfg);
    ~server_ckpt_store();

    server_ckpt_store(const server_ckpt_store &) = delete;
    server_ckpt_store & operator=(const server_ckpt_store &) = delete;

    // append a blob to the slot's active file, returns its aligned offset.
    // starts a new file when the slot has no active one.
    uint64_t append(int slot_id, const uint8_t * data, size_t size);

    // append the full target/draft state and remember it for the next write_meta
    bool write_full(int slot_id, const uint8_t * tgt, size_t size_tgt, const uint8_t * dft, size_t size_dft);

    // persist the session sidecar: token list, full state and on-disk checkpoints
    void write_meta(int slot_id, const server_tokens & tokens, const std::list<common_prompt_checkpoint> & checkpoints);

    // pick the persisted session whose token list is the longest prefix of `tokens`
    bool find_best(const server_tokens & tokens, size_t & index, float & f_keep, float & f_sim) const;

    // make a persisted session the active one for the slot, returns its tokens,
    // full state and checkpoints
    bool adopt(int slot_id, size_t index, server_tokens & tokens,
            full_state & full, std::list<common_prompt_checkpoint> & checkpoints);

    // read a blob back, dst must hold size bytes
    bool read(int slot_id, uint64_t off, size_t size, uint8_t * dst);

    // mark a record dead; compaction may follow
    void release(int slot_id, uint64_t off);

    // compact when dead bytes exceed live bytes, fills the old -> new offset map
    void maybe_compact(int slot_id, std::vector<std::pair<uint64_t, uint64_t>> & remap);

    // stop appending to the slot's file; the session becomes matchable
    void finalize(int slot_id);

    // delete the oldest slot files until the dir fits within the cap
    void enforce_cap(size_t incoming);

    size_t total_bytes() const;
    size_t n_files() const;

    struct stats_t {
        uint64_t appends      = 0;
        uint64_t reads        = 0;
        uint64_t read_bytes   = 0;
        uint64_t write_bytes  = 0;
        uint64_t compactions  = 0;
        uint64_t compact_read = 0;
        uint64_t evicted      = 0;
        uint64_t evicted_bytes = 0;
        uint64_t meta_writes  = 0;
        uint64_t restored     = 0;
        uint64_t discarded    = 0;
        uint64_t adopted      = 0;
    };

    const stats_t & stats() const { return st; }
    void print_stats() const;

private:
    struct rec {
        uint64_t off;
        uint64_t size;
        bool     live;
    };

    struct session {
        std::string path;
        uint64_t    file_size = 0;
        server_tokens tokens;
        full_state  full;
        std::list<common_prompt_checkpoint> checkpoints;
        std::filesystem::file_time_type mtime;
    };

    struct slot_state {
        int         session = 0;
        std::string path;
        FILE *      fp      = nullptr;
        uint64_t    end     = 0;
        uint64_t    live    = 0;
        uint64_t    dead    = 0;
        std::vector<rec> recs;

        // last metadata written for this session, moved to the index on finalize
        bool        has_meta = false;
        server_tokens meta_tokens;
        full_state  full;
        std::list<common_prompt_checkpoint> meta_ckpts;
    };

    void load_index();
    bool read_meta(const std::filesystem::path & meta_path, session & out) const;
    void remove_files(const std::string & path) const;

    slot_state & state_for(int slot_id);
    bool start_session(int slot_id, slot_state & s);

    // read raw bytes, direct IO when enabled and possible
    bool read_file(const std::string & path, uint64_t off, size_t size, uint8_t * dst) const;

    config cfg;

    std::map<int, slot_state> slots;
    std::vector<session>      index;

    stats_t st;
};
