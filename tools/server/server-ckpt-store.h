#pragma once

// Disk-backed store for slot context checkpoints.
//
// One append-only file per slot session, records padded to a fixed alignment so
// reads can go through direct IO without a bounce buffer per record. Only the
// newest blob stays in RAM, everything else lives here. Metadata (position,
// size, offset) stays in common_prompt_checkpoint.

#include <cstdint>
#include <cstdio>
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

    server_ckpt_store(std::string dir, std::string key, bool use_dio, size_t max_bytes);
    ~server_ckpt_store();

    server_ckpt_store(const server_ckpt_store &) = delete;
    server_ckpt_store & operator=(const server_ckpt_store &) = delete;

    // append a blob to the slot's active file, returns its aligned offset.
    // starts a new file when the slot has no active one.
    uint64_t append(int slot_id, const uint8_t * data, size_t size);

    // read a blob back, dst must hold size bytes
    bool read(int slot_id, uint64_t off, size_t size, uint8_t * dst);

    // mark a record dead; compaction may follow
    void release(int slot_id, uint64_t off);

    // compact when dead bytes exceed live bytes, fills the old -> new offset map
    void maybe_compact(int slot_id, std::vector<std::pair<uint64_t, uint64_t>> & remap);

    // stop appending to the slot's file; the next append starts a new session
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
    };

    const stats_t & stats() const { return st; }
    void print_stats() const;

private:
    struct rec {
        uint64_t off;
        uint64_t size;
        bool     live;
    };

    struct slot_state {
        int         session = 0;
        std::string path;
        FILE *      fp      = nullptr;
        uint64_t    end     = 0;
        uint64_t    live    = 0;
        uint64_t    dead    = 0;
        std::vector<rec> recs;
    };

    slot_state & state_for(int slot_id);
    bool start_session(int slot_id, slot_state & s);

    // read raw bytes, direct IO when enabled and possible
    bool read_file(const std::string & path, uint64_t off, size_t size, uint8_t * dst) const;

    std::string dir;
    std::string key;
    bool        use_dio  = false;
    size_t      max_bytes = 0;

    std::map<int, slot_state> slots;

    stats_t st;
};
