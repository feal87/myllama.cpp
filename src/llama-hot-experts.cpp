#include "llama-hot-experts.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

llama_hot_expert_cache::llama_hot_expert_cache(const llama_model & model,
                                               int32_t             n_pin_experts,
                                               uint64_t            budget_bytes,
                                               uint64_t            stats_interval,
                                               uint64_t            decay_interval) :
    model(model),
    n_pin(n_pin_experts),
    budget_bytes(budget_bytes),
    stats_interval(stats_interval),
    decay_interval(decay_interval) {
    // Count MoE layers by checking which layers have ffn_down_exps.weight
    int32_t n_moe_layers = 0;
    for (int32_t il = 0; il < (int32_t) model.hparams.n_layer(); il++) {
        const std::string name = "blk." + std::to_string(il) + ".ffn_down_exps.weight";
        if (model.get_tensor(name.c_str()) != nullptr) {
            n_moe_layers++;
        }
    }
    n_pin_total = n_pin * n_moe_layers;

    if (n_moe_layers == 0) {
        LLAMA_LOG_WARN("%s: no MoE layers detected in model, --pin-hot-experts has no effect\n", __func__);
    } else if (budget_bytes > 0) {
        LLAMA_LOG_INFO(
            "%s: pinning (mlock) up to %d hottest MoE experts per layer (%d MoE layers, "
            "%d total global slots) in place, budget %.2f MiB total, pin/evict on the fly\n",
            __func__, n_pin, n_moe_layers, n_pin_total, budget_bytes / (1024.0 * 1024.0));
    } else {
        LLAMA_LOG_WARN(
            "%s: --pin-hot-experts has NO memory budget cap (--pin-hot-experts-budget-mib "
            "was not set); with enough layers/experts this WILL try to lock more memory "
            "than physically fits and can be killed by the OOM killer. Setting an explicit "
            "budget that leaves headroom for the KV cache and compute buffers is strongly "
            "recommended.\n",
            __func__);
    }
    if (budget_bytes > 0 && llama_mlock::SUPPORTED && !llama_mlock::reserve_working_set(budget_bytes)) {
        LLAMA_LOG_WARN(
            "%s: could not raise the Windows working-set minimum to the pin budget; "
            "VirtualLock will retry and report failures as needed\n",
            __func__);
    }
    if (decay_interval > 0) {
        LLAMA_LOG_INFO("%s: halving all usage counts every %" PRIu64 " tokens of content\n", __func__, decay_interval);
    }
    if (stats_interval > 0) {
        LLAMA_LOG_INFO("%s: printing pinning stats to stderr every %" PRIu64 " router observations\n", __func__,
                       stats_interval);
    }
    if (n_pin > 0 && llama_mlock::SUPPORTED) {
        pin_worker = std::thread(&llama_hot_expert_cache::pin_worker_main, this);
    }

    if (!llama_mlock::SUPPORTED) {
        LLAMA_LOG_WARN(
            "%s: mlock is not supported on this platform, --pin-hot-experts will only "
            "track usage statistics and will not actually lock any memory (the per-layer "
            "prefetch still keeps recently-used expert rows in the page cache)\n",
            __func__);
    }

}

llama_hot_expert_cache::~llama_hot_expert_cache() {
    print_stats();

    // stop the pin worker and drain whatever is queued (pending jobs are simply
    // abandoned: the lock guards they would have created are irrelevant now)
    {
        std::lock_guard<std::mutex> lock(mu);
        pin_stop = true;
    }
    pin_cv.notify_all();
    if (pin_worker.joinable()) {
        pin_worker.join();
    }
}

void llama_hot_expert_cache::print_stats() const {
    std::lock_guard<std::mutex> lock(mu);

    size_t   total_distinct_seen  = counts.size();
    size_t   total_pinned         = pinned.size();
    uint64_t global_coldest_count = UINT64_MAX;
    uint64_t global_hottest_count = 0;

    // Per-layer breakdown of pinned experts
    std::unordered_map<int, size_t> pinned_per_layer;
    for (const auto & [key, pe] : pinned) {
        pinned_per_layer[key.layer]++;
    }

    if (!pinned_rank.empty()) {
        global_coldest_count = std::get<0>(*pinned_rank.begin());
        global_hottest_count = std::get<0>(*pinned_rank.rbegin());
    }

    LLAMA_LOG_INFO("[pin-hot-experts] ub=%" PRIu64
                   " obs=%" PRIu64
                   " | experts_locked=%.2f MiB (lock calls=%" PRIu64 " slow=%" PRIu64 " fails=%" PRIu64 ")"
                   " | moe_layers=%zu | "
                   "pinned=%zu/%d (global, N=%d x layers=%zu) | distinct (layer,expert) seen=%zu"
                   " | prefetch calls=%" PRIu64 " bytes=%.2f MiB failures=%" PRIu64 " | decays=%" PRIu64,
                   n_ubatches, n_eval_calls,
                   n_bytes_locked / (1024.0 * 1024.0), n_lock_calls, n_lock_slow, n_pin_failures,
                   layers.size(), total_pinned, n_pin_total, n_pin,
                   layers.size(), total_distinct_seen, n_prefetch_calls,
                   n_prefetch_bytes / (1024.0 * 1024.0), n_prefetch_failures, n_decays);

    if (!pinned_rank.empty()) {
        LLAMA_LOG_CONT(" | pinned count range=[%" PRIu64 ", %" PRIu64 "]", global_coldest_count, global_hottest_count);
    }

    if (!pinned_per_layer.empty()) {
        LLAMA_LOG_CONT(" | per-layer: {");
        // Sort by layer index for readable output
        std::vector<std::pair<int, size_t>> sorted_layers(pinned_per_layer.begin(), pinned_per_layer.end());
        std::sort(sorted_layers.begin(), sorted_layers.end());
        for (size_t i = 0; i < sorted_layers.size(); i++) {
            const auto & [il, cnt] = sorted_layers[i];
            LLAMA_LOG_CONT("L%d=%zu%s", il, cnt, (i + 1 < sorted_layers.size()) ? ", " : "");
        }
        LLAMA_LOG_CONT("}");
    }
    LLAMA_LOG_CONT("\n");
}

bool llama_hot_expert_cache::eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * self = static_cast<llama_hot_expert_cache *>(user_data);

    // we only need the *values* of the top-k expert-selection tensor, named
    // "ffn_moe_topk-<il>" by llm_graph_context::cb() -> llama_context::graph_get_cb()
    static const char prefix[] = "ffn_moe_topk-";
    if (strncmp(t->name, prefix, sizeof(prefix) - 1) != 0) {
        return ask ? false : true;  // not interested, let the scheduler carry on either way
    }

    if (ask) {
        // true = break the graph here and make the tensor readable on the host
        // false = keep going; this layer is skipped entirely this ubatch
        return self->wants_observe(atoi(t->name + sizeof(prefix) - 1));
    }

    self->observe(atoi(t->name + sizeof(prefix) - 1), t);

    return true;
}

// ask phase. Layers that cannot be pinned (experts offloaded to a device, or
// pinning disabled) and layers that this ubatch's sampling pattern skips return
// false, so the scheduler does not chunk the graph at their topk tensor.
bool llama_hot_expert_cache::wants_observe(int il) {
    std::lock_guard<std::mutex> lock(mu);

    if (n_pin <= 0) {
        return false;
    }

    layer_state & ls = layers[il];
    if (!ls.resolved_tensors) {
        resolve_tensors(il, ls);
    }
    if (!ls.tensors_are_host) {
        return false;
    }

    return true;
}

void llama_hot_expert_cache::observe(int il, const struct ggml_tensor * t) {
    if (t->type != GGML_TYPE_I32) {
        return;  // unexpected, be defensive rather than misinterpret bytes
    }

    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];
    const int64_t n_ids         = n_expert_used * n_tokens;

    if (n_ids <= 0) {
        return;
    }

    std::vector<int32_t> ids(n_ids);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(ids.data(), t->data, n_ids * sizeof(int32_t));
    } else {
        ggml_backend_tensor_get(t, ids.data(), 0, n_ids * sizeof(int32_t));
    }

    uint64_t eval_calls_now = 0;
    std::vector<std::pair<const void *, size_t>> ranges;  // rows to prefetch (filled under the lock)
    {
        std::lock_guard<std::mutex> lock(mu);

        layer_state & ls = layers[il];
        if (!ls.resolved_tensors) {
            resolve_tensors(il, ls);
        }
        if (!ls.tensors_are_host) {
            return;  // nothing to pin or prefetch for this layer
        }

        n_eval_calls++;
        eval_calls_now = n_eval_calls;

        for (int32_t id : ids) {
            if (id >= 0) {
                observe_expert(il, ls, id);
            }
        }

        // The expert FFNs run on the CPU, and ggml-cpu processes the routed
        // experts of each token in the order they appear in the ids tensor.
        // Reorder each token's ids so the resident (pinned) experts come first:
        // the cold (unpinned) experts are then faulted in only after the pinned
        // ones have computed, which gives the prefetch (issued below) time to
        // land before the CPU reaches them. Positional pairing with the per-
        // expert gating weights is preserved because every consumer reads the
        // same permuted tensor. Only possible when the ids tensor is in host
        // memory - i.e. it is the exact buffer the CPU mul_mat_id reads.
        if (ggml_backend_buffer_is_host(t->buffer)) {
            for (int64_t j = 0; j < n_tokens; ++j) {
                int32_t * row = ids.data() + j * n_expert_used;
                std::stable_partition(row, row + n_expert_used, [&](int32_t id) {
                    return id >= 0 && pinned.count(expert_key{ il, id }) != 0;
                });
            }
            std::memcpy(t->data, ids.data(), n_ids * sizeof(int32_t));
        }

        // prefetch the routed-but-unpinned rows NOW, while the rest of this
        // ubatch still computes: the reads run in the background and the same
        // experts are routed again on the next token with high probability, so
        // their rows are resident when the next FFN wants them. This is what
        // keeps the disk busy while the GPU works (see the class comment).
        // Dedupe first: prompt-processing batches route hundreds of experts per
        // layer and a duplicate would append the same rows once per token.
        std::vector<int32_t> uniq = ids;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        for (int32_t id : uniq) {
            if (id < 0) {
                continue;
            }
            expert_key key{ il, id };
            if (pinned.count(key) == 0) {
                add_expert_ranges(ls, id, ranges);
            }
        }
    }  // lock released here

    if (!ranges.empty()) {
        n_prefetch_calls++;
        for (const auto & range : ranges) {
            n_prefetch_bytes += range.second;
        }
        if (!llama_mmap::prefetch(ranges)) {
            n_prefetch_failures++;
        }
    }

    // print_stats() takes the same mutex itself, so this must run outside the
    // scope above (the mutex is not recursive)
    if (stats_interval > 0 && eval_calls_now % stats_interval == 0) {
        print_stats();
    }
}

void llama_hot_expert_cache::resolve_tensors(int il, layer_state & ls) {
    const std::string base = "blk." + std::to_string(il) + ".";

    ls.t_gate    = model.get_tensor((base + "ffn_gate_exps.weight").c_str());
    ls.t_up      = model.get_tensor((base + "ffn_up_exps.weight").c_str());
    ls.t_down    = model.get_tensor((base + "ffn_down_exps.weight").c_str());
    ls.t_gate_up = model.get_tensor((base + "ffn_gate_up_exps.weight").c_str());

    ls.resolved_tensors = true;

    if (!ls.t_down || (!ls.t_gate && !ls.t_gate_up)) {
        LLAMA_LOG_WARN(
            "%s: layer %d does not look like a (supported) MoE FFN layer, "
            "hot-expert pinning disabled for this layer\n",
            __func__, il);
        return;
    }

    // pinning only makes sense (and is only safe to do in place) when the expert
    // tensors actually live in host memory, e.g. all experts kept on the CPU
    // while only the dense/router parts are offloaded to VRAM
    const ggml_tensor * repr = ls.t_down;
    ls.tensors_are_host      = ggml_backend_buffer_is_host(repr->buffer);

    if (!ls.tensors_are_host) {
        LLAMA_LOG_WARN(
            "%s: layer %d's MoE experts are not in host memory (offloaded to a "
            "device buffer), --pin-hot-experts has no effect for this layer\n",
            __func__, il);
    }
}

void llama_hot_expert_cache::observe_expert(int il, layer_state & ls, int32_t expert_id) {
    expert_key key{ il, expert_id };

    uint64_t &     count     = counts[key];  // default-constructs to 0
    const uint64_t old_count = count;
    count++;
    const uint64_t new_count = count;

    if (!ls.tensors_are_host || n_pin <= 0 || !llama_mlock::SUPPORTED) {
        return;  // stats-only mode, nothing to pin
    }

    if (pinned.count(key)) {
        // already pinned: keep its ordered-set position up to date
        pinned_rank.erase({ old_count, il, expert_id });
        pinned_rank.insert({ new_count, il, expert_id });
        return;
    }

    if (pin_inflight.count(key)) {
        return;  // a pin for this expert is already queued
    }

    // build the job up front so the budget/slot checks below are exact and the
    // takeover never needs a rollback (nothing is evicted before the job is sure
    // to be accepted)
    pin_job job;
    job.il        = il;
    job.expert_id = expert_id;
    job.t_gate    = ls.t_gate;
    job.t_gate_up = ls.t_gate_up;
    job.t_up      = ls.t_up;
    job.t_down    = ls.t_down;
    job.expected_bytes = expert_row_bytes(ls, expert_id);
    if (job.expected_bytes == 0) {
        return;
    }

    const size_t committed = pinned.size() + pin_inflight.size();
    if (committed < (size_t) n_pin_total) {
        // a slot is free: pin without evicting anyone (budget is checked inside
        // enqueue_pin; if there is no headroom the job is dropped and the expert
        // stays unpinned until the budget frees up again)
        enqueue_pin(job);
        return;
    }

    // all slots are committed: only take over if we just overtook the coldest
    // pinned expert. The munlock of the victim is cheap (no page-in) and stays
    // inline; only the mlock of the newcomer is deferred to the worker.
    const auto coldest = pinned_rank.begin();
    if (coldest == pinned_rank.end() || new_count <= std::get<0>(*coldest)) {
        return;
    }

    const int      evict_layer = std::get<1>(*coldest);
    const int32_t  evict_id    = std::get<2>(*coldest);

    // if the worker queue is already full the replacement cannot be queued, so
    // evicting the victim would only waste a resident expert: skip the takeover
    if (pin_queue.size() >= pin_queue_max) {
        return;
    }

    // the takeover may exceed the budget even though evicting the victim frees
    // its bytes first: check the post-eviction total up front so we never evict
    // a resident expert for a pin that cannot fit
    const auto vit = pinned.find(expert_key{ evict_layer, evict_id });
    const uint64_t victim_bytes = vit != pinned.end() ? vit->second.nbytes_locked : 0;
    if (budget_bytes > 0 &&
        n_bytes_locked + n_bytes_reserved + job.expected_bytes > budget_bytes + victim_bytes) {
        return;
    }

    pinned_rank.erase(coldest);

    auto & evict_ls = layers[evict_layer];
    if (!evict_ls.resolved_tensors) {
        resolve_tensors(evict_layer, evict_ls);
    }
    unpin_expert(evict_layer, evict_ls, evict_id);

    if (!enqueue_pin(job)) {
        // only possible if the queue filled or the budget was exhausted between
        // the checks above and the reservation; the slot stays empty and the
        // next promotion refills it
        LLAMA_LOG_DEBUG("%s: pin of layer %d expert %d dropped after evicting %d/%d\n", __func__, il, expert_id,
                        evict_layer, evict_id);
    }
}

size_t llama_hot_expert_cache::expert_row_bytes(const layer_state & ls, int32_t expert_id) const {
    size_t nbytes = 0;

    const ggml_tensor * ws[] = { ls.t_gate_up ? ls.t_gate_up : ls.t_gate, ls.t_up, ls.t_down };
    for (const ggml_tensor * w : ws) {
        if (w == nullptr || !ggml_backend_buffer_is_host(w->buffer) || w->data == nullptr) {
            continue;
        }
        if (expert_id < 0 || expert_id >= w->ne[2]) {
            continue;
        }
        nbytes += ggml_nbytes(w) / (size_t) w->ne[2];
    }
    return nbytes;
}

bool llama_hot_expert_cache::enqueue_pin(const pin_job & job) {
    // caller holds mu
    expert_key key{ job.il, job.expert_id };
    if (pin_inflight.count(key) != 0) {
        return false;
    }
    if (pin_queue.size() >= pin_queue_max) {
        return false;
    }
    if (budget_bytes > 0 && n_bytes_locked + n_bytes_reserved + job.expected_bytes > budget_bytes) {
        return false;
    }

    pin_queue.push_back(job);
    pin_inflight.insert(key);
    n_bytes_reserved += job.expected_bytes;
    pin_cv.notify_one();
    return true;
}

void llama_hot_expert_cache::pin_worker_main() {
    std::unique_lock<std::mutex> lock(mu);

    for (;;) {
        pin_cv.wait(lock, [&]() { return pin_stop || !pin_queue.empty(); });

        if (pin_queue.empty()) {
            break;  // pin_stop and nothing left to do
        }

        pin_job job = std::move(pin_queue.front());
        pin_queue.pop_front();

        // run the syscalls OUTSIDE the mutex: grow_to() faults the expert's rows
        // in and can take tens of ms, which must not block any decode-thread work
        lock.unlock();

        pinned_expert pe;
        uint64_t      lock_calls = 0;
        uint64_t      lock_slow  = 0;
        bool          syscall_error = false;

        auto lock_row = [&](const ggml_tensor *                           w,
                            std::unique_ptr<llama_mlock, mlock_deleter> & out) -> size_t {
            if (!w || job.expert_id < 0 || job.expert_id >= w->ne[2] || !llama_mlock::SUPPORTED) {
                return 0;
            }
            if (!ggml_backend_buffer_is_host(w->buffer) || w->data == nullptr) {
                return 0;
            }

            const size_t nbytes = ggml_nbytes(w) / (size_t) w->ne[2];
            const size_t offset = (size_t) job.expert_id * w->nb[2];

            // lock the expert's rows IN PLACE inside the model's own tensor -- the
            // exact memory ggml_mul_mat_id() reads during build_moe_ffn()
            std::unique_ptr<llama_mlock, mlock_deleter> ml(new llama_mlock());
            ml->init((uint8_t *) w->data + offset);
            const int64_t t0 = ggml_time_us();
            ml->grow_to(nbytes);
            const int64_t dt = ggml_time_us() - t0;

            lock_calls++;
            if (dt > lock_slow_us) {
                lock_slow++;
                LLAMA_LOG_DEBUG("%s: async mlock of expert %d (tensor %s) took %" PRId64 " us\n", __func__,
                                job.expert_id, ggml_get_name(w), dt);
            }

            // only keep what was ACTUALLY locked -- grow_to() silently stops (and
            // logs a warning) on failure rather than throwing
            const size_t locked = ml->size();
            if (locked == 0) {
                ml.reset();
                return 0;
            }
            if (locked < nbytes) {
                LLAMA_LOG_WARN(
                    "%s: only locked %zu/%zu bytes for expert %d (system out of lockable "
                    "memory?) -- consider lowering --pin-hot-experts N or its budget\n",
                    __func__, locked, nbytes, job.expert_id);
            }
            out = std::move(ml);
            return locked;
        };

        try {
            if (job.t_gate_up) {
                pe.nbytes_locked += lock_row(job.t_gate_up, pe.gate_up_lock);
            } else {
                pe.nbytes_locked += lock_row(job.t_gate, pe.gate_lock);
            }
            pe.nbytes_locked += lock_row(job.t_up, pe.up_lock);
            pe.nbytes_locked += lock_row(job.t_down, pe.down_lock);
        } catch (...) {
            // never let an exception escape the worker (it would terminate the
            // process); any rows already locked in pe are released on unwind and
            // the job is counted as a failure below
            syscall_error = true;
        }

        lock.lock();

        // bookkeeping: release the budget reservation and account for the result
        n_bytes_reserved -= job.expected_bytes;
        n_lock_calls += lock_calls;
        n_lock_slow += lock_slow;

        expert_key key{ job.il, job.expert_id };
        pin_inflight.erase(key);

        if (pe.nbytes_locked == 0 || syscall_error) {
            n_pin_failures++;
            LLAMA_LOG_DEBUG("%s: async pin of layer %d expert %d locked nothing\n", __func__, job.il, job.expert_id);
            continue;
        }

        n_bytes_locked += pe.nbytes_locked;

        // the expert's count may have kept climbing (or been decayed) since the
        // job was queued: insert its rank entry with the current value
        uint64_t c = 0;
        auto     it = counts.find(key);
        if (it != counts.end()) {
            c = it->second;
        }
        pinned.emplace(key, std::move(pe));
        pinned_rank.insert({ c, job.il, job.expert_id });
    }
}

void llama_hot_expert_cache::unpin_expert(int il, layer_state & /*ls*/, int32_t expert_id) {
    expert_key key{ il, expert_id };
    auto       it = pinned.find(key);
    if (it == pinned.end()) {
        return;
    }

    n_bytes_locked -= it->second.nbytes_locked;
    pinned.erase(it);  // pinned_expert's destructor releases the mlock guards
}

size_t llama_hot_expert_cache::add_expert_ranges(const layer_state &                    ls,
                                                 int32_t                                 expert_id,
                                                 std::vector<std::pair<const void *, size_t>> & out) {
    const ggml_tensor * ws[] = { ls.t_gate, ls.t_up, ls.t_down, ls.t_gate_up };
    size_t              nbytes = 0;
    for (const ggml_tensor * w : ws) {
        if (w == nullptr || !ggml_backend_buffer_is_host(w->buffer) || w->data == nullptr || w->ne[2] <= 0) {
            continue;
        }
        if (expert_id < 0 || expert_id >= w->ne[2]) {
            continue;
        }
        const size_t esize = w->nb[2];
        out.emplace_back((const char *) w->data + expert_id * esize, esize);
        nbytes += esize;
    }
    return nbytes;
}

void llama_hot_expert_cache::on_ubatch_begin(int64_t n_tokens) {
    n_ubatches++;

    // periodic decay of the usage counts: keeps the pin set tracking the recent
    // routing mix instead of lifetime leaders. The clock is content tokens, not
    // ubatches (a prefill ubatch covers hundreds of tokens, a decode ubatch one
    // or a few), accumulated here and checked between graph computes.
    if (decay_interval > 0 && n_tokens > 0) {
        n_tokens_seen += n_tokens;
        while (n_tokens_seen >= decay_interval) {
            n_tokens_seen -= decay_interval;
            decay_counts();
        }
    }
}

void llama_hot_expert_cache::decay_counts() {
    std::lock_guard<std::mutex> lock(mu);

    for (auto it = counts.begin(); it != counts.end(); ++it) {
        const expert_key & key         = it->first;
        const uint64_t     old_count   = it->second;
        const uint64_t     new_count   = (old_count + 1) / 2;  // floor at 1, halves everything else
        if (new_count == old_count) {
            continue;  // count == 1: nothing left to decay
        }
        it->second = new_count;
        if (pinned.count(key) != 0) {
            // keep the ordered-set key in sync with the count
            pinned_rank.erase({ old_count, key.layer, key.expert_id });
            pinned_rank.insert({ new_count, key.layer, key.expert_id });
        }
    }

    n_decays++;
}

bool llama_hot_expert_cache::is_pinned(int il, int32_t expert_id) const {
    std::lock_guard<std::mutex> lock(mu);

    expert_key key{ il, expert_id };
    return pinned.count(key) != 0;
}
