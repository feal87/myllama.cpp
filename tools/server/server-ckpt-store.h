#pragma once

// Disk-backed store for slot context checkpoints.
//
// One directory per session holds one immutable file per checkpoint and a
// generation-tagged file for the full state. Nothing is reused in place, so
// releasing a record is an unlink and no compaction pass is needed.
//
// Writes run on a single worker thread: the caller does the bookkeeping and
// hands the buffers over, so decoding is not blocked by the record I/O. The
// queue is FIFO, which keeps the write, sidecar and delete ordering.
//
// Each session also keeps a meta.bin sidecar with the token list and the record
// index. The sidecar carries the prompt cache settings key, so a config change
// invalidates it. At startup the sidecars rebuild an in-RAM index of sessions
// that can be adopted by a slot after a restart.

#include "common.h"
#include "server-common.h"

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <thread>
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
        bool        partial_ckpt = false; // hybrid/recurrent model: checkpoints carry the recurrent state only
        bool        attn_log     = false; // keep one append-only attention file per session instead of per-turn full states
        size_t      max_bytes = 0; // 0 = no limit
    };

    // a full state snapshot (LLAMA_STATE_SEQ_FLAGS_NONE), self-sufficient after
    // a restart: it carries the attention cache and the recurrent state
    struct full_state {
        uint64_t file_id  = 0;
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

    // write one checkpoint as its own record file. the buffers are moved to the
    // write worker. on failure the partial file is removed and `out` stays off
    // disk, so the caller can fall back to RAM.
    bool write_checkpoint(int slot_id,
            std::vector<uint8_t> tgt,
            std::vector<uint8_t> dft,
            std::vector<uint8_t> spec,
            common_prompt_checkpoint & out);

    // write the full target/draft state under a new generation
    bool write_full(int slot_id, std::vector<uint8_t> tgt, std::vector<uint8_t> dft);

    // append attention to the session's attention log so that it covers
    // [0, pos_end). `gen` produces the serialized range state for a range.
    // the log is restarted when the checked-out history forked, which is when
    // `tokens` no longer shares a prefix with what the log was built from.
    bool append_attention(int slot_id, int64_t pos_end, const server_tokens & tokens,
            const std::function<std::vector<uint8_t>(int64_t, int64_t)> & gen);

    // rebuild the live attention of a slot from its attention log. returns true
    // when the whole log was applied.
    bool load_attention(int slot_id, llama_context * ctx, llama_seq_id seq_id);

    // persist the session sidecar: token list, full state and on-disk checkpoints
    void write_meta(int slot_id, const server_tokens & tokens, const std::list<common_prompt_checkpoint> & checkpoints);

    // pick the persisted session whose token list is the longest prefix of `tokens`
    bool find_best(const server_tokens & tokens, size_t & index, float & f_keep, float & f_sim) const;

    // make a persisted session the active one for the slot, returns its tokens,
    // full state and checkpoints
    bool adopt(int slot_id, size_t index, server_tokens & tokens,
            full_state & full, std::list<common_prompt_checkpoint> & checkpoints);

    // read a record part back, dst must hold size bytes
    bool read(int slot_id, uint64_t file_id, uint64_t off, size_t size, uint8_t * dst);

    // drop a record file; the unlink is deferred until the next sidecar commit
    void release_file(int slot_id, uint64_t file_id);

    // stop appending to the slot's session; it becomes matchable
    void finalize(int slot_id);

    // delete the oldest session directories until the dir fits within the cap
    void enforce_cap(size_t incoming);

    size_t total_bytes() const;
    size_t n_files() const;

    struct stats_t {
        uint64_t appends      = 0;
        uint64_t reads        = 0;
        uint64_t read_bytes   = 0;
        uint64_t write_bytes  = 0;
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
    struct session {
        std::string dir;
        server_tokens tokens;
        full_state  full;
        std::list<common_prompt_checkpoint> checkpoints;
        std::filesystem::file_time_type mtime;
    };

    struct slot_state {
        int         session = 0;
        std::string dir;
        uint64_t    next_file_id = 1;

        // extent of the attention log: highest position it covers (pos_max + 1)
        // and the token history it was built from
        int64_t     attn_covered = 0;
        server_tokens attn_tokens;

        // record file in the committed sidecar; the previous generation is
        // deleted once a newer one is committed
        uint64_t    full_committed = 0;

        // released records, unlinked only after the sidecar stops referencing them
        std::vector<uint64_t> pending_remove;

        // last metadata written for this session, moved to the index on finalize
        bool        has_meta = false;
        server_tokens meta_tokens;
        full_state  full;
        std::list<common_prompt_checkpoint> meta_ckpts;
    };

    void load_index();
    bool read_meta(const std::filesystem::path & meta_path, session & out) const;
    void remove_dir(const std::string & dir) const;

    slot_state & state_for(int slot_id);
    bool start_session(int slot_id, slot_state & s);

    std::string record_path(const std::string & dir, uint64_t file_id) const;
    std::string attn_path  (const std::string & dir) const;

    // read raw bytes, direct IO when enabled and possible
    bool read_file(const std::string & path, uint64_t off, size_t size, uint8_t * dst) const;

    // queue a job for the write worker
    void enqueue(std::function<void()> fn);
    // block until every queued write and sidecar commit has finished
    void wait_idle();
    void worker_loop();

    config cfg;

    std::map<int, slot_state> slots;
    std::vector<session>      index;

    std::thread             worker;
    std::mutex              mtx;
    std::condition_variable cv;
    std::condition_variable cv_done;
    std::deque<std::function<void()>> queue;
    bool                    stopping = false;
    size_t                  pending  = 0;

    stats_t st;
};
