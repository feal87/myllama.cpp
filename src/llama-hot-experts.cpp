#include "llama-hot-experts.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-disk-stage.h"
#include "llama-expert-predict.h"
#include "llama-model.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// --expert-ngram-record: one raw binary stream, appended across runs. The header
// is written once and re-verified on every append, so a recording cannot be
// merged across models. A record is one byte of type followed by its payload;
// the decode_step payload is the token a graph consumed plus, per observed MoE
// layer, the experts it routed. scripts/expert-ngram parses this.
static constexpr char     NGRAM_REC_MAGIC[8]     = { 'L', 'E', 'N', 'G', 'R', 'E', 'C', '1' };
static constexpr uint32_t NGRAM_REC_VERSION      = 1;
static constexpr uint8_t  NGRAM_REC_PROMPT_BEGIN = 1;
static constexpr uint8_t  NGRAM_REC_DECODE_STEP  = 2;

static uint64_t ngram_fnv1a(uint64_t h, const void * data, size_t n) {
    const unsigned char * p = (const unsigned char *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// true when `f` starts with a header matching this model and tokenizer
static bool ngram_header_matches(std::FILE * f, uint64_t model_fp, uint64_t tok_fp) {
    char     magic[8];
    uint32_t version = 0;
    uint32_t n_layer = 0, n_expert = 0, n_used = 0;
    uint64_t mfp = 0, tfp = 0;
    if (std::fseek(f, 0, SEEK_SET) != 0) {
        return false;
    }
    if (std::fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        std::fread(&version, sizeof(version), 1, f) != 1 ||
        std::fread(&n_layer, sizeof(n_layer), 1, f) != 1 ||
        std::fread(&n_expert, sizeof(n_expert), 1, f) != 1 ||
        std::fread(&n_used, sizeof(n_used), 1, f) != 1 ||
        std::fread(&mfp, sizeof(mfp), 1, f) != 1 ||
        std::fread(&tfp, sizeof(tfp), 1, f) != 1) {
        return false;
    }
    return std::memcmp(magic, NGRAM_REC_MAGIC, sizeof(NGRAM_REC_MAGIC)) == 0 &&
           version == NGRAM_REC_VERSION && mfp == model_fp && tfp == tok_fp;
}

llama_hot_expert_cache::llama_hot_expert_cache(const llama_model & model,
                                               int32_t             n_pin_experts,
                                               uint64_t            budget_bytes,
                                               uint64_t            decay_interval,
                                               uint64_t            recent_decay_interval,
                                               uint64_t            prompt_decay,
                                               uint64_t            min_pin_count,
                                               bool                prefetch_enabled,
                                               bool                track_rank,
                                               bool                disk_mode,
                                               const char *        profile_path,
                                               const char *        ngram_record_path,
                                               const char *        ngram_profile_path,
                                               int32_t             ngram_prefetch_max) :
    model(model),
    n_pin(n_pin_experts),
    budget_bytes(budget_bytes),
    decay_interval(decay_interval),
    recent_decay_interval(recent_decay_interval),
    prompt_decay(prompt_decay),
    min_pin_count(min_pin_count),
    prefetch_enabled(prefetch_enabled),
    track_rank(track_rank),
    ngram_prefetch_max(ngram_prefetch_max) {
    // Count MoE layers by checking which layers have ffn_down_exps.weight
    int32_t n_moe_layers = 0;
    for (int32_t il = 0; il < (int32_t) model.hparams.n_layer(); il++) {
        const std::string name = "blk." + std::to_string(il) + ".ffn_down_exps.weight";
        if (model.get_tensor(name.c_str()) != nullptr) {
            n_moe_layers++;
        }
    }
    n_pin_total = n_pin * n_moe_layers;
    n_pinned_layer.assign((size_t) model.hparams.n_layer(), 0);
    layers.resize((size_t) model.hparams.n_layer());
    pinned_rank_pool.resize((size_t) n_pools);

    if (profile_path != nullptr && profile_path[0] != '\0') {
        profile_file = std::fopen(profile_path, "ab");
        if (profile_file == nullptr) {
            LLAMA_LOG_WARN("%s: failed to open expert profile '%s', profiling disabled\n", __func__, profile_path);
        } else {
            LLAMA_LOG_INFO("%s: writing decode-only expert profiles to %s\n", __func__, profile_path);
        }
    }

    if (ngram_record_path != nullptr && ngram_record_path[0] != '\0') {
        // fingerprints guard the append path: a recording only merges with one
        // made by the same model weights and tokenizer
        const std::string arch = model.arch_name();
        const uint64_t    dims[4] = {
            (uint64_t) model.hparams.n_layer(),
            (uint64_t) model.hparams.n_expert,
            (uint64_t) model.hparams.n_expert_used(),
            (uint64_t) model.hparams.n_embd,
        };
        ngram_model_fp     = ngram_fnv1a(1469598103934665603ULL, arch.data(), arch.size());
        ngram_model_fp     = ngram_fnv1a(ngram_model_fp, dims, sizeof(dims));
        const uint64_t n_vocab = model.vocab.n_tokens();
        ngram_tokenizer_fp = ngram_fnv1a(1469598103934665603ULL, &n_vocab, sizeof(n_vocab));

        ngram_file = std::fopen(ngram_record_path, "ab+");
        if (ngram_file == nullptr) {
            LLAMA_LOG_WARN("%s: failed to open expert n-gram recording '%s', recording disabled\n", __func__, ngram_record_path);
        } else {
            std::fseek(ngram_file, 0, SEEK_END);
            const long size = std::ftell(ngram_file);
            if (size < 0) {
                LLAMA_LOG_WARN("%s: expert n-gram recording '%s' is not seekable, recording disabled\n", __func__, ngram_record_path);
                std::fclose(ngram_file);
                ngram_file = nullptr;
            } else if (size == 0) {
                write_ngram_header();
                std::fflush(ngram_file);
                LLAMA_LOG_INFO("%s: recording decode tokens and routed experts to %s\n", __func__, ngram_record_path);
            } else if (!ngram_header_matches(ngram_file, ngram_model_fp, ngram_tokenizer_fp)) {
                LLAMA_LOG_WARN("%s: expert n-gram recording '%s' was made by a different model or tokenizer, recording disabled\n",
                               __func__, ngram_record_path);
                std::fclose(ngram_file);
                ngram_file = nullptr;
            } else {
                std::fseek(ngram_file, 0, SEEK_END);
                LLAMA_LOG_INFO("%s: appending decode token + expert recording to %s\n", __func__, ngram_record_path);
            }
            if (ngram_file != nullptr) {
                // a run boundary is a prompt boundary: the offline study must not
                // build an n-gram across two independent sessions
                std::fwrite(&NGRAM_REC_PROMPT_BEGIN, sizeof(NGRAM_REC_PROMPT_BEGIN), 1, ngram_file);
                std::fflush(ngram_file);
            }
        }
    }

    if (ngram_profile_path != nullptr && ngram_profile_path[0] != '\0') {
        try {
            ngram_predict = std::make_unique<llama_expert_predict>(ngram_profile_path,
                    (int32_t) model.hparams.n_layer(), (int32_t) model.hparams.n_expert);
            const size_t n_layer = (size_t) model.hparams.n_layer();
            ngram_predicted.resize(n_layer);
            ngram_pred_layer.assign(n_layer, 0);
            ngram_hit_layer.assign(n_layer, 0);
            LLAMA_LOG_INFO("%s: expert prediction profile %s: max order %d, top %d, %d/%d layers, "
                           "prefetching up to %d expert(s) per step\n",
                           __func__, ngram_profile_path, ngram_predict->max_order(),
                           ngram_predict->top_m(), ngram_predict->n_enabled(),
                           ngram_predict->n_layer(), ngram_prefetch_max);
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("%s: %s, expert prefetch disabled\n", __func__, e.what());
            ngram_predict.reset();
        }
    }

    if (n_moe_layers == 0) {
        if (n_pin > 0) {
            LLAMA_LOG_WARN("%s: no MoE layers detected in model, --pin-hot-experts has no effect\n", __func__);
        }
    } else if (n_pin > 0) {
        if (disk_mode) {
            LLAMA_LOG_INFO(
                "%s: keeping the %d hottest MoE experts per layer (%d MoE layers) resident in the "
                "disk decode cache, nothing mlock'd\n",
                __func__, n_pin, n_moe_layers);
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
    }
    if (n_pin > 0 && !disk_mode) {
        if (budget_bytes > 0 && llama_mlock::SUPPORTED && !llama_mlock::reserve_working_set(budget_bytes)) {
            LLAMA_LOG_WARN(
                "%s: could not raise the Windows working-set minimum to the pin budget; "
                "VirtualLock will retry and report failures as needed\n",
                __func__);
        }
        if (llama_mlock::SUPPORTED) {
            pin_worker = std::thread(&llama_hot_expert_cache::pin_worker_main, this);
        } else {
            LLAMA_LOG_WARN(
                "%s: mlock is not supported on this platform, --pin-hot-experts will only "
                "track usage statistics and will not actually lock any memory (--hot-experts-prefetch "
                "still keeps recently-used expert rows in the page cache)\n",
                __func__);
        }
    } else if (disk_mode) {
        // the disk decode cache is the RAM tier; logged above, nothing is mlock'd
    } else if (track_rank) {
        // ranking-only mode: the router observation feeds the VRAM MoE tier
        // (llama_moe_cache) and/or the prefetch, but nothing is mlock'd
        LLAMA_LOG_INFO("%s: tracking MoE router usage, nothing pinned%s\n", __func__,
                       prefetch_enabled ? " (prefetching routed expert rows)" : "");
    } else {
        // prefetch-only mode: no ranking consumer exists, so only multi-token
        // (batch/prefill) ubatches are observed, purely to read the routed rows
        // ahead; nothing is counted or pinned
        LLAMA_LOG_INFO("%s: prefetching routed expert rows on batch/prefill ubatches, "
                       "no usage tracking\n", __func__);
    }
    if (n_pin > 0 && min_pin_count > 0) {
        LLAMA_LOG_INFO("%s: %s only experts with usage count >= %" PRIu64 " (--pin-hot-experts-min-count)\n",
                       __func__, disk_mode ? "admitting" : "pinning", min_pin_count);
    }
    if (decay_interval > 0) {
        LLAMA_LOG_INFO("%s: halving all usage counts every %" PRIu64 " decode tokens\n", __func__, decay_interval);
    }
    if (prompt_decay > 1) {
        LLAMA_LOG_INFO("%s: dividing all usage counts by %" PRIu64 " at every prompt start\n", __func__, prompt_decay);
    }
    if (recent_decay_interval > 0) {
        LLAMA_LOG_INFO("%s: VRAM tier ranks by recent routing, halved every %" PRIu64 " decode tokens\n", __func__, recent_decay_interval);
    }

}

llama_hot_expert_cache::~llama_hot_expert_cache() {
    if (n_pin > 0 || prefetch_enabled || ngram_predict != nullptr) {
        print_stats(/* final_report = */ true);
    }
    if (profile_file != nullptr) {
        std::lock_guard<std::mutex> lock(mu);
        write_profile();
        std::fclose(profile_file);
        profile_file = nullptr;
    }
    if (ngram_file != nullptr) {
        std::lock_guard<std::mutex> lock(mu);
        std::fflush(ngram_file);
        std::fclose(ngram_file);
        ngram_file = nullptr;
    }
    if (ngram_records > 0 || ngram_tokens_missed > 0) {
        LLAMA_LOG_INFO("%s: expert n-gram recording: %" PRIu64 " decode steps, %" PRIu64 " steps skipped (no token)\n",
                       __func__, ngram_records, ngram_tokens_missed);
    }

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

    if (obs_stage_buf != nullptr) {
        ggml_backend_buffer_free(obs_stage_buf);
        obs_stage_buf = nullptr;
        obs_stage     = nullptr;
    }
}

void llama_hot_expert_cache::print_stats(bool final_report, uint64_t ram_hit, uint64_t ram_total) {
    // gather everything under mu, then log after unlocking: console writes are
    // synchronous on Windows and must not stall the pin worker or a VRAM takeover
    size_t   total_distinct_seen = 0;
    size_t   total_pinned        = 0;
    size_t   n_new               = 0;
    size_t   n_base_ram          = 0;
    uint64_t routed_hit          = 0;
    uint64_t routed_total        = 0;
    uint64_t min_count           = 0;
    uint64_t max_count           = 0;
    bool     any_rank            = false;
    uint64_t bytes_locked        = 0;
    uint64_t lock_calls          = 0;
    uint64_t lock_slow           = 0;
    uint64_t pin_failures        = 0;
    uint64_t eval_calls          = 0;
    uint64_t ubatches            = 0;
    uint64_t prefetch_calls      = 0;
    uint64_t prefetch_bytes      = 0;
    uint64_t prefetch_failures   = 0;
    uint64_t decays              = 0;
    uint64_t hysteresis_holds    = 0;
    uint64_t min_count_holds     = 0;
    uint64_t route_base          = 0;
    uint64_t route_total         = 0;
    uint64_t prev_base           = 0;
    uint64_t prev_total          = 0;
    size_t   n_base_used         = 0;
    int32_t  base_count          = 0;
    uint64_t base_switches       = 0;
    uint64_t base_admitted       = 0;
    uint64_t base_released       = 0;
    uint64_t base_refused        = 0;
    std::string base_name;
    uint64_t prev_hit            = 0;
    uint64_t prev_routed         = 0;
    uint64_t n_reserved          = 0;
    uint64_t n_prompt_decays     = 0;
    uint64_t recent_decays       = 0;
    size_t   n_inflight          = 0;
    size_t   n_queued            = 0;
    size_t   n_held_unfilled     = 0;
    size_t   n_moe_layers        = 0;
    uint32_t n_reports           = 0;
    bool     prefetch_active     = false;
    uint64_t predict_steps       = 0;
    uint64_t predict_matched     = 0;
    uint64_t predict_total       = 0;
    uint64_t predict_hits        = 0;
    std::vector<uint64_t> predict_layer;
    std::vector<uint64_t> predict_hit;

    // one slot per layer, filled under mu and read after
    std::vector<size_t> per_layer(layers.size(), 0);

    {
        std::lock_guard<std::mutex> lock(mu);

        total_distinct_seen = (size_t) n_distinct;
        total_pinned        = pinned.size();
        // the disk stage decides RAM residency at the read, before the graph
        // finishes, while the pin state is only current after it. Prefer the
        // disk stage's realized count when the caller passes it
        if (ram_total > 0) {
            routed_hit   = ram_hit;
            routed_total = ram_total;
        } else {
            routed_hit   = n_route_hit;
            routed_total = n_route_hit + n_route_miss;
        }
        bytes_locked        = n_bytes_locked;
        lock_calls          = n_lock_calls;
        lock_slow           = n_lock_slow;
        pin_failures        = n_pin_failures;
        eval_calls          = n_eval_calls;
        ubatches            = n_ubatches;
        prefetch_calls      = n_prefetch_calls;
        prefetch_bytes      = n_prefetch_bytes;
        prefetch_failures   = n_prefetch_failures;
        decays              = n_decays;
        hysteresis_holds    = n_hysteresis_holds;
        min_count_holds     = n_min_count_holds;
        route_base          = n_route_base;
        route_total         = n_route_total;
        prev_base           = prev_route_base;
        prev_total          = prev_route_total;
        prev_hit            = ram_total > 0 ? prev_ext_hit   : prev_route_hit;
        prev_routed         = ram_total > 0 ? prev_ext_total : prev_routed_total;
        n_reserved          = n_bytes_reserved;
        n_prompt_decays     = n_decays_prompt;
        recent_decays       = n_recent_decays;
        base_count          = n_base;
        base_switches       = n_base_switches;
        base_admitted       = n_base_admitted;
        base_released       = n_base_released;
        base_refused        = n_base_refused;
        base_name           = active_base_set;
        n_inflight          = pin_inflight.size();
        n_queued            = pin_queue.size();
        n_reports           = n_reports_total;
        // the row prefetch needs the mmap'd model pages, which the disk decode
        // cache bypasses: it is skipped entirely there, so it is not reported
        prefetch_active     = prefetch_enabled && disk_stage == nullptr;
        predict_steps       = ngram_predict ? ngram_predict->n_steps() : 0;
        predict_matched     = n_ngram_matched;
        predict_total       = n_ngram_pred_routes;
        predict_hits        = n_ngram_pred_hits;
        predict_layer       = ngram_pred_layer;
        predict_hit         = ngram_hit_layer;

        // RAM occupancy: pinned holds every resident, base included. Base entries
        // are permanent (never churn), so churn and the count range cover the
        // dynamic ones only. The range is read from the counts directly, so the
        // report no longer rebuilds the eviction rank just to make it exact
        const uint32_t last_stamp = stats_stamp;
        const uint32_t next_stamp = last_stamp + 1;
        min_count = UINT64_MAX;
        for (auto & [key, pe] : pinned) {
            const layer_state * ls   = layer_of(key.layer);
            const bool          base = ls != nullptr && key.expert_id >= 0 && (uint32_t) key.expert_id < ls->n_experts &&
                                       (ls->pin_state[(size_t) key.expert_id] & PIN_BASE) != 0;
            if (key.layer >= 0 && key.layer < (int) per_layer.size()) {
                per_layer[(size_t) key.layer]++;
            }
            if (base) {
                n_base_ram++;
                continue;
            }
            if (pe.seen_stamp != last_stamp) {
                n_new++;  // arrived since the previous report
            }
            pe.seen_stamp = next_stamp;
            const uint64_t c = count_of(key.layer, key.expert_id);
            any_rank  = true;
            min_count = std::min(min_count, c);
            max_count = std::max(max_count, c);
        }

        // base utilization: base experts routed at least once since startup (the
        // count floors at 1 once set, so counts[id] == 0 means never routed)
        for (size_t il = 0; il < layers.size(); ++il) {
            const layer_state & ls = layers[il];
            if (!ls.resolved_tensors) {
                continue;
            }
            n_moe_layers++;
            for (uint32_t id = 0; id < ls.n_experts; ++id) {
                if ((ls.pin_state[(size_t) id] & PIN_BASE) != 0 && ls.counts[(size_t) id] > 0) {
                    n_base_used++;
                }
            }
        }

        prev_route_base  = n_route_base;
        prev_route_total = n_route_total;
        prev_route_hit   = n_route_hit;
        prev_routed_total = n_route_hit + n_route_miss;
        if (ram_total > 0) {
            prev_ext_hit   = ram_hit;
            prev_ext_total = ram_total;
        }
        n_reports_total++;
        stats_stamp = next_stamp;
    }

    const size_t n_dynamic   = total_pinned - n_base_ram;
    const int    n_base_vram = (int) (base_count - (int32_t) n_base_ram);
    const size_t n_free      = (size_t) std::max((int32_t) n_pin_total - (int32_t) total_pinned, 0);
    const double mib         = 1024.0 * 1024.0;

    // resident slots without bytes in them: a promotion whose expert has not
    // been routed again yet, so its fill has not run
    if (disk_stage != nullptr) {
        n_held_unfilled = disk_stage->resident_held_unfilled();
    }

    // per-layer breakdown, already in layer order
    size_t n_layers_used = 0;
    for (const size_t c : per_layer) {
        n_layers_used += c != 0 ? 1 : 0;
    }

    // one themed line each, emitted as a single write: a console write per line
    // was the only measurable cost of the report (Windows console writes are
    // synchronous)
    std::string out = "[pin-hot-experts] RAM tier report #";
    out += std::to_string(n_reports + 1);
    out += final_report ? " (final)\n" : "\n";

    char buf[1024];
    auto line = [&out, &buf](const char * fmt, auto... args) {
        const int n_chars = snprintf(buf, sizeof(buf), fmt, args...);
        if (n_chars > 0) {
            out.append(buf, (size_t) std::min((size_t) n_chars, sizeof(buf) - 1));
        }
        out += '\n';
    };

    line("  slots    : %zu/%d residents (dynamic %zu, base %zu ram + %d vram) | free %zu | unfilled %zu"
         " | pools %d | layers %zu/%zu | min-count %" PRIu64 " | in flight %zu, queued %zu",
         total_pinned, n_pin_total, n_dynamic, n_base_ram, n_base_vram, n_free, n_held_unfilled, n_pools,
         n_layers_used, n_moe_layers ? n_moe_layers : per_layer.size(), min_pin_count, n_inflight,
         n_queued);

    line("  hit      : %.1f%% cumulative (%" PRIu64 "/%" PRIu64 " routed) | %.1f%% this interval (%" PRIu64 "/%" PRIu64 ")",
         routed_total ? 100.0 * routed_hit / routed_total : 0.0, routed_hit, routed_total,
         routed_total > prev_routed ? 100.0 * (routed_hit - prev_hit) / (routed_total - prev_routed) : 0.0,
         routed_hit - prev_hit, routed_total - prev_routed);

    line("  churn    : %.1f%% of the dynamic set changed (%" PRIu64 "/%zu) | takeovers held %" PRIu64
         " | min-count holds %" PRIu64,
         n_dynamic ? 100.0 * n_new / n_dynamic : 0.0, (uint64_t) n_new, n_dynamic,
         hysteresis_holds, min_count_holds);

    line("  memory   : %.2f MiB resident of a %.2f MiB budget (%.1f%%) | %.2f MiB reserved by in-flight pins",
         bytes_locked / mib, budget_bytes / mib,
         budget_bytes ? 100.0 * bytes_locked / (double) budget_bytes : 0.0, n_reserved / mib);

    if (disk_stage != nullptr) {
        const std::string lay = disk_stage->ram_layout();
        if (!lay.empty()) {
            line("  layout   : %s", lay.c_str());
        }
    }

    if (n_pin > 0 && disk_stage == nullptr) {
        line("  locks    : %" PRIu64 " calls, %" PRIu64 " slow (over %" PRId64 " us), %" PRIu64 " failed",
             lock_calls, lock_slow, lock_slow_us, pin_failures);
    }

    char range[64] = "n/a (no ranked resident)";
    if (any_rank) {
        snprintf(range, sizeof(range), "%" PRIu64 "..%" PRIu64, min_count, max_count);
    }
    line("  ranking  : %zu distinct (layer, expert) pairs seen | %" PRIu64 " observations in %" PRIu64 " ubatches"
         " | pinned counts %s",
         total_distinct_seen, eval_calls, ubatches, range);

    const std::string window     = decay_interval > 0 ? "every " + std::to_string(decay_interval) + " tokens" : "off";
    const std::string divisor    = prompt_decay > 1 ? std::to_string(prompt_decay) : "off (1)";
    line("  decays   : %" PRIu64 " total, %" PRIu64 " periodic (%s), %" PRIu64 " at prompt start (divisor %s), %" PRIu64 " recent",
         decays, decays - n_prompt_decays, window.c_str(), n_prompt_decays, divisor.c_str(), recent_decays);

    // base set feedback: how much of the decode routing the base experts serve and
    // how many of them were ever routed (a large gap to n_base means dead weight)
    if (base_count > 0) {
        const uint64_t d_base  = route_base  - prev_base;
        const uint64_t d_total = route_total - prev_total;
        line("  base set : %s, %d experts, %" PRIu64 " switch(es), %.1f%% of routes (%" PRIu64 "/%" PRIu64 ")"
             " | %.1f%% this interval (%" PRIu64 "/%" PRIu64 ") | %zu/%d ever routed"
             " | admitted %" PRIu64 ", released %" PRIu64 ", refused %" PRIu64,
             base_name.empty() ? "<base>" : base_name.c_str(), base_count, base_switches,
             route_total ? 100.0 * route_base / route_total : 0.0, route_base, route_total,
             d_total ? 100.0 * d_base / d_total : 0.0, d_base, d_total, n_base_used, base_count,
             base_admitted, base_released, base_refused);
    }

    // the row prefetch works on the mmap'd model pages, which the disk decode
    // cache bypasses: it is skipped there, so it is not reported either
    if (prefetch_active) {
        line("  prefetch : %" PRIu64 " calls | %.2f MiB read ahead | %" PRIu64 " failed",
             prefetch_calls, prefetch_bytes / mib, prefetch_failures);
    }

    if (predict_total > 0) {
        line("  predict  : %" PRIu64 "/%" PRIu64 " steps matched | %" PRIu64 " predicted, "
             "%" PRIu64 " routed (%.1f%% precision)",
             predict_matched, predict_steps, predict_total, predict_hits,
             predict_total ? 100.0 * (double) predict_hits / (double) predict_total : 0.0);
        std::string row = "  predict/L:";
        bool wrapped = false;
        for (size_t il = 0; il < predict_layer.size(); ++il) {
            if (predict_layer[il] == 0) {
                continue;
            }
            char cell[48];
            snprintf(cell, sizeof(cell), " L%zu=%" PRIu64 "/%" PRIu64,
                     il, predict_hit[il], predict_layer[il]);
            row += cell;
            if (row.size() > 110 && !wrapped) {
                out += row;
                out += '\n';
                row = "            ";
                wrapped = true;
            }
        }
        out += row;
        out += '\n';
    }

    if (n_layers_used > 0) {
        std::string row = "  layers   :";
        bool wrapped = false;
        for (size_t il = 0; il < per_layer.size(); ++il) {
            if (per_layer[il] == 0) {
                continue;
            }
            row += " L" + std::to_string(il) + "=" + std::to_string(per_layer[il]);
            // wrap after appending: checking first lets the last entry push the
            // row well past the width
            if (row.size() > 110 && !wrapped) {
                out += row;
                out += '\n';
                row = "            ";
                wrapped = true;
            }
        }
        out += row;
        out += '\n';
    }

    LLAMA_LOG_INFO("%s", out.c_str());
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

// ask phase. Without disk staging llama_context installs the eval callback only
// when the ubatch is multi-token AND --hot-experts-prefetch is on: the read-ahead
// is the only reason to break the graph mid-ubatch, since decode ubatches feed the
// ranking post-compute via observe_decode(). Layers with experts offloaded to a
// device cannot be prefetched and return false, so the scheduler does not chunk
// the graph at their topk.
bool llama_hot_expert_cache::wants_observe(int il) {
    // disk staging fills through this callback on every ubatch, so the graph has
    // to break at every layer's topk: the whole layer slab on a multi-token one,
    // the decode cache on a single-token one
    if (disk_stage != nullptr) {
        return n_tokens_cur > 1 ? disk_stage->layer(il) != nullptr
                                : disk_stage->cache_layer(il) != nullptr;
    }

    if (!prefetch_enabled || n_tokens_cur <= 1) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mu);

    layer_state & ls = layers[il];
    if (!ls.resolved_tensors) {
        resolve_tensors(il, ls);
    }
    if (!ls.tensors_are_host) {
        return false;
    }

    return true;
}

// the top-k tensor is a view into the argsort workspace, so its row stride can
// be much larger than its n_expert_used values: read it row by row
static void read_topk_ids(const struct ggml_tensor * t, int32_t * ids) {
    const int64_t n_used = t->ne[0];
    const int64_t n_tok  = t->ne[1];
    if (ggml_backend_buffer_is_host(t->buffer)) {
        const char * src = (const char *) t->data;
        for (int64_t i = 0; i < n_tok; ++i) {
            std::memcpy(ids + i * n_used, src + i * t->nb[1], n_used * sizeof(int32_t));
        }
    } else {
        for (int64_t i = 0; i < n_tok; ++i) {
            ggml_backend_tensor_get(t, ids + i * n_used, i * t->nb[1], n_used * sizeof(int32_t));
        }
    }
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

    // disk staging fills through this callback: the ubatch decides how much. A
    // multi-token ubatch fills the whole layer slab (any routed expert may be
    // read); a single-token ubatch fills only the routed experts of the layer's
    // decode cache. Gate on the ubatch size, not on t->ne[1]: one layer can be
    // fed a 1-token slice inside a multi-token ubatch, and its staging must
    // still be filled
    if (disk_stage != nullptr) {
        obs_scratch.resize(n_ids);
        int32_t * ids = obs_scratch.data();
        read_topk_ids(t, ids);
        if (n_tokens_cur > 1) {
            if (disk_stage->sparse_ubatch(n_tokens_cur)) {
                disk_stage->fill_selected(il, ids, n_ids);
            } else {
                disk_stage->fill(il);
            }
        } else {
            disk_stage->fill_cache(il, ids, n_ids);
        }
        return;
    }

    // the rest of this function is the multi-token prefetch read-ahead
    if (!prefetch_enabled || n_tokens_cur <= 1) {
        return;
    }

    // reuse one scratch for the routed ids (and one for the VRAM residency
    // snapshot below): observation runs once per layer per ubatch on the compute
    // thread, so per-call allocation is pure churn
    obs_scratch.resize(n_ids);
    int32_t * ids = obs_scratch.data();
    read_topk_ids(t, ids);

    prefetch_ranges.clear();
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

        // (A per-token reorder to put pinned experts first was tried here and
        // removed: the CPU mul_mat_id batches each expert across the whole ubatch
        // and walks experts in expert-id order, so the position of an expert
        // inside a token's top-k ids does not influence when its rows are read.
        // Prefetch below is what overlaps the page-ins with the ongoing compute.)

        // prefetch the routed-but-unpinned rows NOW, while the rest of this
        // ubatch still computes: a multi-token ubatch routes a wide expert set
        // that the per-layer FFNs read from host memory right after this point,
        // and overlapping the page-ins with the ongoing compute is what keeps
        // prefill fast (demand-paging every expert row one fault at a time
        // stalls it). Gated on --hot-experts-prefetch AND on multi-token
        // (batch/prefill) ubatches only: single-token decode re-reads the same
        // few experts every token (pinned and/or VRAM-resident after warm-up),
        // where per-token prefetch is pure syscall/page-cache churn. VRAM-served
        // experts are prefetched like any other: multi-token graphs never use
        // the VRAM tier (its chain is single-token only), so prefill reads their
        // host rows too. Dedupe first: prompt-processing batches route hundreds
        // of experts per layer and a duplicate would append the same rows once
        // per token.
        if (prefetch_enabled && n_tokens > 1 && disk_stage == nullptr) {
            // dedupe in place: prompt-processing batches route hundreds of experts
            // per layer and a duplicate would append the same rows once per token
            int32_t * b = obs_scratch.data();
            int32_t * e = b + obs_scratch.size();
            std::sort(b, e);
            e = std::unique(b, e);
            for (int32_t * p = b; p != e; ++p) {
                const int32_t id = *p;
                if (id < 0 || id >= (int32_t) ls.n_experts) {
                    continue;
                }
                if ((ls.pin_state[(size_t) id] & PIN_RESIDENT) != 0) {
                    continue;  // already resident: nothing to read ahead
                }
                add_expert_ranges(ls, id, prefetch_ranges);
            }
        }
    }  // lock released here

    if (!prefetch_ranges.empty()) {
        n_prefetch_calls++;
        for (const auto & range : prefetch_ranges) {
            n_prefetch_bytes += range.second;
        }
        if (!llama_mmap::prefetch(prefetch_ranges)) {
            n_prefetch_failures++;
        }
    }
}

void llama_hot_expert_cache::observe_decode_begin(const std::vector<ggml_tensor *> & topk, ggml_backend_sched_t sched) {
    // plan the staged readback: which layers to observe and where each layer's
    // ids go in the shared scratch. One pass over the layers of the decode graph
    // that just finished computing; the gates mirror what the old single-pass
    // observe_decode() applied (valid tensor, single token, experts on host).
    obs_off.assign(topk.size(), -1);
    obs_cnt.assign(topk.size(), 0);
    obs_ids = nullptr;

    size_t total = 0;
    for (int il = 0; il < (int) topk.size(); ++il) {
        const ggml_tensor * t = topk[il];
        if (t == nullptr || t->type != GGML_TYPE_I32) {
            continue;
        }
        const int64_t n_ids = t->ne[0] * t->ne[1];
        if (n_ids <= 0 || t->ne[1] != 1) {
            continue;  // decode graphs route a single token per ubatch
        }

        // lock only when a layer still needs resolving: after warm-up this pass
        // takes no lock at all instead of one acquisition per layer
        layer_state & ls = layers[il];
        if (!ls.resolved_tensors) {
            std::lock_guard<std::mutex> lock(mu);
            if (!ls.resolved_tensors) {
                resolve_tensors(il, ls);
            }
        }
        if (!ls.tensors_are_host && profile_file == nullptr && ngram_file == nullptr) {
            continue;  // experts offloaded to a device: nothing to count or pin
        }

        obs_off[il] = (int) total;
        obs_cnt[il] = (int) n_ids;
        total += (size_t) n_ids;
    }

    if (total == 0) {
        return;
    }

    // stage the routed ids: one async D2H copy per device top-k tensor, queued
    // on its backend stream right behind the decode graph, plus direct copies
    // for host tensors (their compute already finished). The scratch is pinned
    // host memory when the device exposes a host buft, so the copies are truly
    // async (async D2H into pageable memory blocks until each copy completes,
    // which serialized the whole readback behind the decode graph); llama_context
    // syncs once after this, so the whole ubatch costs a single device sync.
    const size_t bytes_total = total * sizeof(int32_t);

    // size pinned staging from the first device-resident tensor's backend
    ggml_backend_t stage_backend = nullptr;
    if (sched) {
        for (int il = 0; il < (int) topk.size() && stage_backend == nullptr; ++il) {
            if (obs_off[il] < 0) {
                continue;
            }
            const ggml_tensor * t = topk[il];
            if (ggml_backend_buffer_is_host(t->buffer)) {
                continue;
            }
            stage_backend = ggml_backend_sched_get_tensor_backend(sched, const_cast<ggml_tensor *>(t));
        }
    }

    int32_t * dst = nullptr;
    if (stage_backend && ensure_obs_stage(stage_backend, bytes_total)) {
        dst = (int32_t *) obs_stage;
    } else {
        obs_scratch.resize(total);
        dst = obs_scratch.data();
    }
    obs_ids = dst;

    for (int il = 0; il < (int) topk.size(); ++il) {
        if (obs_off[il] < 0) {
            continue;
        }
        const ggml_tensor * t = topk[il];
        int32_t * dst_il = dst + obs_off[il];
        const size_t nbytes = (size_t) obs_cnt[il] * sizeof(int32_t);
        if (ggml_backend_buffer_is_host(t->buffer)) {
            std::memcpy(dst_il, t->data, nbytes);
            continue;
        }
        ggml_backend_t backend = sched ? ggml_backend_sched_get_tensor_backend(sched, const_cast<ggml_tensor *>(t)) : nullptr;
        if (backend) {
            ggml_backend_tensor_get_async(backend, t, dst_il, 0, nbytes);
        } else {
            ggml_backend_tensor_get(t, dst_il, 0, nbytes);
        }
    }
}

bool llama_hot_expert_cache::ensure_obs_stage(ggml_backend_t backend, size_t bytes) {
    if (obs_stage_buf != nullptr && obs_stage_cap >= bytes) {
        return true;
    }

    ggml_backend_dev_t           dev       = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t   host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    if (host_buft == nullptr) {
        return false;
    }

    if (obs_stage_buf != nullptr) {
        ggml_backend_buffer_free(obs_stage_buf);
        obs_stage_buf = nullptr;
        obs_stage     = nullptr;
    }
    {
        llama_mem_tag_scope mem_scope("hot");
        obs_stage_buf = ggml_backend_buft_alloc_buffer(host_buft, bytes);
    }
    if (obs_stage_buf == nullptr) {
        obs_stage_cap = 0;
        return false;
    }
    obs_stage     = ggml_backend_buffer_get_base(obs_stage_buf);
    obs_stage_cap = bytes;
    return true;
}

void llama_hot_expert_cache::observe_decode_finish() {
    if (obs_ids == nullptr) {
        // no layer produced an observable top-k this step, but the token still
        // belongs in the recording: a missing step would shift every following
        // n-gram window by one token
        if (ngram_file != nullptr) {
            std::lock_guard<std::mutex> lock(mu);
            write_ngram_step();
        }
        if (ngram_predict) {
            std::lock_guard<std::mutex> lock(mu);
            for (std::vector<int32_t> & v : ngram_predicted) {
                v.clear();
            }
        }
        return;  // nothing was staged (no observed layer)
    }

    const int32_t * ids_all = obs_ids;

    // one lock for the whole ubatch (the old per-layer lock/unlock was 48
    // uncontended acquisitions per decode token on the same mutex)
    std::lock_guard<std::mutex> lock(mu);

    // with the disk stage active, RAM residency is decided by the decode-cache
    // slot actually holding the bytes, not by the pin state
    const bool ds_active = disk_stage != nullptr && disk_stage->is_active();

    for (int il = 0; il < (int) obs_off.size(); ++il) {
        if (obs_off[il] < 0) {
            continue;
        }
        layer_state & ls = layers[il];  // resolved in observe_decode_begin()
        const int32_t * ids = ids_all + obs_off[il];
        const int n_ids = obs_cnt[il];

        n_eval_calls++;

        // score this step's prediction against what the layer just routed: of
        // the predicted experts, how many the layer actually selected (precision)
        if (ngram_predict) {
            const std::vector<int32_t> & pred = ngram_predicted[(size_t) il];
            for (const int32_t pid : pred) {
                n_ngram_pred_routes++;
                ngram_pred_layer[(size_t) il]++;
                bool routed = false;
                for (int k = 0; k < n_ids; ++k) {
                    if (ids[k] == pid) {
                        routed = true;
                        break;
                    }
                }
                if (routed) {
                    n_ngram_pred_hits++;
                    ngram_hit_layer[(size_t) il]++;
                }
            }
        }

        // snapshot this layer's VRAM residency: the VRAM tier only publishes and
        // evicts at the ubatch boundary (tick), never during a graph, so the flags
        // read here describe exactly what the decode graph just used. The table
        // is borrowed in place (null when the layer has no device cache)
        const uint8_t * vram_flags = vram_query ? vram_query(vram_ud, il) : nullptr;
        const bool vram_tier_layer = vram_flags != nullptr;

        // flat per-expert tables, indexed by routed expert id
        const int   n_experts = (int) ls.n_experts;
        uint64_t *  counts    = ls.counts.data();
        uint64_t *  recent_counts = ls.recent_counts.empty() ? nullptr : ls.recent_counts.data();
        uint64_t *  profile_counts = profile_file != nullptr ? ls.profile_counts.data() : nullptr;
        uint8_t *   pin_state = ls.pin_state.data();

        for (int k = 0; k < n_ids; ++k) {
            const int32_t id = ids[k];
            if (id < 0 || id >= n_experts) {
                continue;
            }
            // base set telemetry: the base flag survives a VRAM takeover, so a
            // route to a base expert is counted wherever it is served from
            n_route_total++;
            if ((pin_state[id] & PIN_BASE) != 0) {
                n_route_base++;
            }
            const bool served = vram_flags != nullptr && vram_flags[(size_t) id] != 0;
            if (served) {
                // decode-time VRAM-tier hit: the VRAM copy served the expert
                // and this host read was skipped
                ls.n_vram_hit++;
            } else {
                // realized RAM-tier hit rate. PIN_RESIDENT is set when the slot is
                // reserved, so with the disk stage active the bytes must also be in
                // the slot, else the read still happens and this is not a hit
                const bool hit = ds_active
                        ? disk_stage->resident_filled(il, id)
                        : (pin_state[id] & (PIN_RESIDENT | PIN_BASE)) != 0;
                if (hit) {
                    n_route_hit++;
                } else {
                    n_route_miss++;
                }
                if (vram_tier_layer) {
                    ls.n_vram_miss++;
                }
            }
            uint64_t & c = counts[id];
            if (c == 0) {
                n_distinct++;  // first route of this (layer, expert)
            }
            c++;
            if (recent_counts != nullptr) {
                recent_counts[id]++;
            }
            if (profile_counts != nullptr) {
                profile_counts[id]++;
                profile_routes++;
            }
            if (!served) {
                // try_promote() returns immediately for VRAM-resident experts, so
                // skipping the call here is behaviour-identical and keeps the
                // hot path on the flat tables
                try_promote(il, ls, id, c, false, true);
            }
        }
    }

    write_ngram_step();

    if (ngram_predict) {
        for (std::vector<int32_t> & v : ngram_predicted) {
            v.clear();
        }
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

    // size the flat per-expert tables (usage counts + pin state mirrors). Routed
    // expert ids are validated against this range on the hot path
    ls.n_experts = (uint32_t) repr->ne[2];
    ls.counts.assign(ls.n_experts, 0);
    if (recent_decay_interval > 0) {
        ls.recent_counts.assign(ls.n_experts, 0);
    }
    if (profile_file != nullptr) {
        ls.profile_counts.assign(ls.n_experts, 0);
    }
    ls.pin_state.assign(ls.n_experts, 0);

    if (!ls.tensors_are_host) {
        LLAMA_LOG_WARN(
            "%s: layer %d's MoE experts are not in host memory (offloaded to a "
            "device buffer), --pin-hot-experts has no effect for this layer\n",
            __func__, il);
    }
}

uint64_t llama_hot_expert_cache::count_of(int il, int32_t expert_id) const {
    // caller holds mu. Cold path: any layer may be queried, resolved or not
    const layer_state * ls = layer_of(il);
    if (ls == nullptr || !ls->resolved_tensors) {
        return 0;
    }
    if (expert_id < 0 || (uint32_t) expert_id >= ls->n_experts) {
        return 0;
    }
    return ls->counts[(size_t) expert_id];
}

void llama_hot_expert_cache::set_disk_stage(llama_disk_stage * ds) {
    disk_stage = ds;
    if (ds == nullptr) {
        return;
    }
    // the disk cache's pools become the RAM tier's pools; the not-aligned
    // fallback (one pool per layer) makes this the old per-layer behaviour
    n_pools = ds->n_pools() > 0 ? ds->n_pools() : 1;
    pinned_rank_pool.assign((size_t) n_pools, std::set<std::tuple<uint64_t, int, int32_t>>());
    layer_pool.assign((size_t) model.hparams.n_layer(), -1);
    for (int il = 0; il < (int) layer_pool.size(); ++il) {
        layer_pool[(size_t) il] = ds->pool_id(il);
    }
    // real total capacity: one representative layer per pool, since the pool
    // capacity is shared across its layers
    n_pin_total = 0;
    for (int pid = 0; pid < n_pools; ++pid) {
        for (int il = 0; il < (int) layer_pool.size(); ++il) {
            if (layer_pool[(size_t) il] == pid) {
                n_pin_total += ds->resident_capacity(il);
                break;
            }
        }
    }

    // base experts: permanent RAM residents. PIN_BASE keeps them out of the
    // promotion/victim policy while still letting them join the VRAM tier
    // through the shared ranking
    const std::vector<std::vector<int32_t>> & base = ds->base_experts();
    active_base_set = ds->active_base_set();
    for (int il = 0; il < (int) base.size(); ++il) {
        if (base[(size_t) il].empty() || il >= (int) layers.size()) {
            continue;
        }
        layer_state & ls = layers[(size_t) il];
        if (!ls.resolved_tensors) {
            resolve_tensors(il, ls);
        }
        for (const int32_t id : base[(size_t) il]) {
            if (id < 0 || (uint32_t) id >= ls.n_experts) {
                continue;
            }
            ls.pin_state[(size_t) id] |= PIN_BASE;
            n_base++;
            add_base_pin(il, ls, id);
        }
    }

    // warm experts: count-0 RAM residents that fill the cache at startup. They
    // are normal pins (PIN_RESIDENT, in the victim rank), so the first expert of
    // the current topic that heats up evicts them
    const std::vector<std::vector<int32_t>> & warm = ds->warm_experts();
    for (int il = 0; il < (int) warm.size(); ++il) {
        if (warm[(size_t) il].empty() || il >= (int) layers.size()) {
            continue;
        }
        layer_state & ls = layers[(size_t) il];
        if (!ls.resolved_tensors) {
            resolve_tensors(il, ls);
        }
        for (const int32_t id : warm[(size_t) il]) {
            if (id < 0 || (uint32_t) id >= ls.n_experts ||
                    (ls.pin_state[(size_t) id] & (PIN_BASE | PIN_RESIDENT)) != 0) {
                continue;
            }
            complete_disk_pin(il, ls, id, expert_row_bytes(ls, id));
        }
    }

    // the base+warm startup fill is not churn: stamp the current residents so the
    // first report starts clean
    for (auto & [key, pe] : pinned) {
        pe.seen_stamp = stats_stamp;
    }
}

int llama_hot_expert_cache::pool_of_layer(int il) const {
    if (disk_stage == nullptr || layer_pool.empty() || il < 0 || il >= (int) layer_pool.size()) {
        return 0;
    }
    const int p = layer_pool[(size_t) il];
    return p >= 0 && p < n_pools ? p : 0;
}

void llama_hot_expert_cache::try_promote(int il, layer_state & ls, int32_t expert_id, uint64_t count, bool vram_resident,
                                         bool apply_hysteresis) {
    if (disk_stage != nullptr) {
        // the disk decode cache is the RAM tier: the pool owns the resident
        // slots, so a hot layer can hold more of them than a cold one
        if (!vram_resident) {
            try_promote_pool(il, ls, expert_id, count, apply_hysteresis);
        }
        return;
    }
    if (!ls.tensors_are_host || n_pin <= 0 || !llama_mlock::SUPPORTED) {
        return;  // stats-only mode, nothing to pin
    }

    if (vram_resident) {
        return;  // served by the VRAM tier; no mlock needed
    }

    // below the usage floor: a one-off route (count 1) is routing noise, not a
    // real signal, and mlock'ing it (a page-in) just to evict it later is churn
    if (count < min_pin_count) {
        n_min_count_holds++;
        return;
    }

    // the flat pin-state mirror is the hot-path membership check (the pin map
    // and the inflight set below are the heavyweight containers)
    if (expert_id < 0 || (uint32_t) expert_id >= ls.n_experts || (pin_state_at(ls, expert_id) & (PIN_RESIDENT | PIN_INFLIGHT | PIN_BASE)) != 0) {
        return;  // base, already pinned, or a pin for it is already queued
    }

    expert_key key{ il, expert_id };

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
    // pinned expert. The ordered-set keys are refreshed only at decay
    // boundaries, so a routed expert's key can lag its true count (it never
    // exceeds it). Instead of rebuilding the whole set per candidate, walk up
    // from the cold end, healing stale keys and dropping VRAM-takeover ghosts,
    // until an exact bottom key proves the true coldest: every other key is >=
    // it and <= its own true count. Several takeovers in one token then share
    // one heal pass over the cold end instead of one O(pinned) rebuild each.
    // The munlock of the victim is cheap (no page-in) and stays inline; only
    // the mlock of the newcomer is deferred to the worker.
    std::tuple<uint64_t, int, int32_t> victim{};
    for (;;) {
        const auto it = pinned_rank_pool[0].begin();
        if (it == pinned_rank_pool[0].end()) {
            return;  // nothing pinned (all capacity in flight or released)
        }
        const uint64_t key_count = std::get<0>(*it);
        if (count <= key_count) {
            return;  // count <= every key <= every true count: no takeover
        }
        // hysteresis: on the routing path a takeover needs a real lead, not the
        // one-count overtake that near-equal experts at the cold edge cross by
        // pure noise (the steady churn in the reports). Stale-low keys only make
        // this stricter. The VRAM eviction handoff skips it: that path already
        // re-decided the whole set and must stay prompt
        if (apply_hysteresis && count < key_count + takeover_min_lead) {
            n_hysteresis_holds++;
            return;
        }
        const int     mem_layer = std::get<1>(*it);
        const int32_t mem_id    = std::get<2>(*it);
        if (pinned.count(expert_key{ mem_layer, mem_id }) == 0) {
            pinned_rank_pool[0].erase(it);  // ghost entry from a VRAM takeover; drop it
            continue;
        }
        const uint64_t true_count = count_of(mem_layer, mem_id);
        if (key_count != true_count) {
            pinned_rank_pool[0].erase(it);  // stale-low key: heal it and re-check the bottom
            pinned_rank_pool[0].insert({ true_count, mem_layer, mem_id });
            continue;
        }
        victim = *it;  // exact bottom key: the true coldest pinned expert
        break;
    }

    const int      evict_layer = std::get<1>(victim);
    const int32_t  evict_id    = std::get<2>(victim);

    // hysteresis: an expert that was itself just evicted stays out for
    // evict_grace_tokens decode tokens, or the takeover victim would climb right
    // back on its next routes and take the slot from its own replacement (A/B
    // ping-pong). Its count keeps rising while it is out, so the guard only
    // spaces genuine re-takeovers apart
    if (apply_hysteresis) {
        auto ev = evicted_at.find(key);
        if (ev != evicted_at.end()) {
            if (n_content_tokens - ev->second < evict_grace_tokens) {
                n_hysteresis_holds++;
                return;
            }
            evicted_at.erase(ev);  // grace expired: admission allowed again
        }
    }

    // if the worker queue is already full the replacement cannot be queued, so
    // evicting the victim would only waste a resident expert: skip the takeover
    if (pin_queue.size() >= pin_queue_max ||
            (!pin_queue.empty() && n_bytes_reserved + job.expected_bytes > pin_queue_max_bytes)) {
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

    pinned_rank_pool[0].erase(victim);

    auto & evict_ls = layers[evict_layer];
    if (!evict_ls.resolved_tensors) {
        resolve_tensors(evict_layer, evict_ls);
    }
    unpin_expert(evict_layer, evict_ls, evict_id);

    // remember the eviction: the victim must not immediately take a slot back
    // (the grace guard above holds it out for evict_grace_tokens decode tokens)
    evicted_at[expert_key{ evict_layer, evict_id }] = n_content_tokens;

    if (!enqueue_pin(job)) {
        // only possible if the queue filled or the budget was exhausted between
        // the checks above and the reservation; the slot stays empty and the
        // next promotion refills it
        LLAMA_LOG_DEBUG("%s: pin of layer %d expert %d dropped after evicting %d/%d\n", __func__, il, expert_id,
                        evict_layer, evict_id);
    }
}

void llama_hot_expert_cache::try_promote_pool(int il, layer_state & ls, int32_t expert_id, uint64_t count,
                                               bool apply_hysteresis) {
    // caller holds mu; promotion only reserves a slot, fill_cache() reads the
    // bytes in the next time this expert is routed (so it is read once, not twice)
    if (n_pin <= 0 || il < 0 || il >= (int) n_pinned_layer.size()) {
        return;
    }
    // a layer without a decode cache has no pool to promote into; without this
    // the pool-full path below would evict a resident of pool 0 for nothing
    if (layer_pool.empty() || il >= (int) layer_pool.size() || layer_pool[(size_t) il] < 0) {
        return;
    }
    if (count < min_pin_count) {
        n_min_count_holds++;
        return;
    }
    if (expert_id < 0 || (uint32_t) expert_id >= ls.n_experts ||
            (pin_state_at(ls, expert_id) & (PIN_RESIDENT | PIN_BASE)) != 0) {
        return;  // base or already resident
    }

    const size_t bytes = expert_row_bytes(ls, expert_id);
    if (bytes == 0) {
        return;
    }

    // a free slot in the pool: take it without evicting anyone
    if (disk_stage->resident_add(il, expert_id)) {
        complete_disk_pin(il, ls, expert_id, bytes);
        return;
    }

    // the pool is full: take over its coldest resident, with the same lead and
    // grace guards as the global path so count noise cannot swap near-equal
    // experts. Walking up from the cold end heals stale keys and drops ghosts
    // until the bottom key is exact
    const int pool = pool_of_layer(il);
    auto & rank = pinned_rank_pool[(size_t) pool];

    std::tuple<uint64_t, int, int32_t> victim{};
    for (;;) {
        auto it = rank.begin();
        if (it == rank.end()) {
            return;  // nothing pinned in this pool
        }
        const uint64_t key_count = std::get<0>(*it);
        if (count <= key_count) {
            return;
        }
        if (apply_hysteresis && count < key_count + takeover_min_lead) {
            n_hysteresis_holds++;
            return;
        }
        const int     mem_layer = std::get<1>(*it);
        const int32_t mem_id    = std::get<2>(*it);
        if (pinned.count(expert_key{ mem_layer, mem_id }) == 0) {
            rank.erase(it);  // ghost entry from a VRAM takeover; drop it
            continue;
        }
        const uint64_t true_count = count_of(mem_layer, mem_id);
        if (key_count != true_count) {
            rank.erase(it);  // stale-low key: heal it and re-check the bottom
            rank.insert({ true_count, mem_layer, mem_id });
            continue;
        }
        victim = *it;
        break;
    }

    const int     evict_layer = std::get<1>(victim);
    const int32_t evict_id    = std::get<2>(victim);

    // hysteresis: an expert that was itself just evicted stays out for
    // evict_grace_tokens decode tokens, or the takeover victim would climb right
    // back on its next routes and take the slot from its own replacement
    if (apply_hysteresis) {
        auto ev = evicted_at.find(expert_key{ il, expert_id });
        if (ev != evicted_at.end()) {
            if (n_content_tokens - ev->second < evict_grace_tokens) {
                n_hysteresis_holds++;
                return;
            }
            evicted_at.erase(ev);  // grace expired: admission allowed again
        }
    }

    rank.erase(victim);

    auto & evict_ls = layers[evict_layer];
    if (!evict_ls.resolved_tensors) {
        resolve_tensors(evict_layer, evict_ls);
    }
    unpin_expert(evict_layer, evict_ls, evict_id);  // frees the victim's pool slot
    evicted_at[expert_key{ evict_layer, evict_id }] = n_content_tokens;

    if (disk_stage->resident_add(il, expert_id)) {
        complete_disk_pin(il, ls, expert_id, bytes);
    }
}

void llama_hot_expert_cache::readmit_base(int il, int32_t expert_id) {
    // caller holds mu. The expert is PIN_BASE and currently has no RAM slot (its
    // VRAM takeover freed it). Base experts are never in pinned_rank_pool, so a
    // victim taken here is always a dynamic resident
    layer_state & ls = layers[(size_t) il];
    if (pinned.count(expert_key{ il, expert_id }) != 0) {
        return;  // still a resident (race with a VRAM release); nothing to do
    }
    if (!disk_stage->resident_add(il, expert_id)) {
        const int pool  = pool_of_layer(il);
        auto &    rank  = pinned_rank_pool[(size_t) pool];
        bool      freed = false;
        for (;;) {
            auto it = rank.begin();
            if (it == rank.end()) {
                break;
            }
            const int     mem_layer = std::get<1>(*it);
            const int32_t mem_id    = std::get<2>(*it);
            if (pinned.count(expert_key{ mem_layer, mem_id }) == 0) {
                rank.erase(it);  // ghost entry from a VRAM takeover; drop it
                continue;
            }
            const uint64_t true_count = count_of(mem_layer, mem_id);
            if (std::get<0>(*it) != true_count) {
                rank.erase(it);  // stale-low key: heal it and re-check the bottom
                rank.insert({ true_count, mem_layer, mem_id });
                continue;
            }
            rank.erase(it);
            auto & evict_ls = layers[(size_t) mem_layer];
            if (!evict_ls.resolved_tensors) {
                resolve_tensors(mem_layer, evict_ls);
            }
            unpin_expert(mem_layer, evict_ls, mem_id);  // frees the victim's pool slot
            freed = true;
            break;
        }
        if (!freed || !disk_stage->resident_add(il, expert_id)) {
            return;  // no dynamic victim: the next route reads it transiently
        }
    }
    add_base_pin(il, ls, expert_id);
}

void llama_hot_expert_cache::add_base_pin(int il, layer_state & ls, int32_t expert_id) {
    // caller holds mu. Base entries live in pinned so the RAM tier accounting is
    // exact, but they never enter pinned_rank_pool: the eviction policy cannot
    // select them
    expert_key key{ il, expert_id };
    if (pinned.count(key) != 0) {
        return;
    }
    pinned_expert pe;
    pe.nbytes_locked = expert_row_bytes(ls, expert_id);
    n_bytes_locked += pe.nbytes_locked;
    pinned.emplace(key, std::move(pe));
    if (il >= 0 && il < (int) n_pinned_layer.size()) {
        n_pinned_layer[(size_t) il]++;
    }
    evicted_at.erase(key);
}

void llama_hot_expert_cache::demote_base_pin(int il, layer_state & ls, int32_t expert_id) {
    const expert_key key{ il, expert_id };
    ls.pin_state[(size_t) expert_id] &= (uint8_t) ~PIN_BASE;
    if (n_base > 0) {
        n_base--;
    }
    const auto it = pinned.find(key);
    if (it == pinned.end()) {
        return;  // a VRAM takeover freed the RAM slot; normal promotion can re-admit it
    }
    ls.pin_state[(size_t) expert_id] |= PIN_RESIDENT;
    pinned_rank_pool[(size_t) pool_of_layer(il)].insert({ count_of(il, expert_id), il, expert_id });
}

bool llama_hot_expert_cache::admit_base_pin(int il, layer_state & ls, int32_t expert_id) {
    if (!disk_stage->base_add(il, expert_id)) {
        return false;
    }
    const size_t id = (size_t) expert_id;
    if ((ls.pin_state[id] & PIN_BASE) != 0) {
        return true;
    }

    ls.pin_state[id] |= PIN_BASE;
    n_base++;

    // A dynamic resident can become base without changing slots. Remove all
    // stale rank keys for it; its current count is not necessarily the key
    // stored between rank rebuilds.
    auto & rank = pinned_rank_pool[(size_t) pool_of_layer(il)];
    for (auto it = rank.begin(); it != rank.end();) {
        if (std::get<1>(*it) == il && std::get<2>(*it) == expert_id) {
            it = rank.erase(it);
        } else {
            ++it;
        }
    }
    if (disk_stage->resident_slot_held(il, expert_id)) {
        add_base_pin(il, ls, expert_id);
    }
    return true;
}

bool llama_hot_expert_cache::evict_coldest_for_base(int pool) {
    auto & rank = pinned_rank_pool[(size_t) pool];
    for (;;) {
        auto it = rank.begin();
        if (it == rank.end()) {
            return false;
        }
        const int     il = std::get<1>(*it);
        const int32_t id = std::get<2>(*it);
        if (pinned.count(expert_key{ il, id }) == 0) {
            rank.erase(it);
            continue;
        }
        const uint64_t count = count_of(il, id);
        if (std::get<0>(*it) != count) {
            rank.erase(it);
            rank.insert({ count, il, id });
            continue;
        }
        rank.erase(it);
        layer_state & ls = layers[(size_t) il];
        if (!ls.resolved_tensors) {
            resolve_tensors(il, ls);
        }
        unpin_expert(il, ls, id, /* keep_l2 = */ false);
        return true;
    }
}

void llama_hot_expert_cache::set_base_set_for_tools(const std::vector<std::string> & tools) {
    if (disk_stage == nullptr) {
        return;
    }
    request_base_set(disk_stage->select_base_set(tools));
}

bool llama_hot_expert_cache::set_base_set_for_fence() {
    if (disk_stage == nullptr) {
        return false;
    }
    const std::string name = disk_stage->fence_base_set();
    if (name.empty()) {
        return false;
    }
    request_base_set(name);
    return true;
}

void llama_hot_expert_cache::request_base_set(const std::string & name) {
    if (disk_stage->base_set_by_name(name) == nullptr) {
        LLAMA_LOG_WARN("%s: expert base template selected unknown set '%s'\n", __func__, name.c_str());
        return;
    }

    std::lock_guard<std::mutex> lock(mu);
    if (active_base_set == name) {
        have_pending_base_set = false;  // already there; drop any queued change
        return;
    }
    pending_base_set      = name;
    have_pending_base_set = true;
}

void llama_hot_expert_cache::apply_pending_base_set() {
    std::string name;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!have_pending_base_set) {
            return;
        }
        name = pending_base_set;
        pending_base_set.clear();
        have_pending_base_set = false;
    }
    apply_base_set(name);
}

bool llama_hot_expert_cache::apply_base_set(const std::string & name) {
    if (disk_stage == nullptr) {
        return false;
    }
    const auto * target = disk_stage->base_set_by_name(name);
    if (target == nullptr) {
        LLAMA_LOG_WARN("%s: expert base template selected unknown set '%s'\n", __func__, name.c_str());
        return false;
    }

    std::lock_guard<std::mutex> lock(mu);
    if (disk_stage->active_base_set() == name) {
        return true;
    }

    // First demote experts no longer in the target. Their slots stay resident
    // and join the normal rank; if the pool needs room below, normal ranking
    // chooses which of them (or other dynamic residents) to evict.
    for (int il = 0; il < (int) layers.size() && il < (int) target->size(); ++il) {
        layer_state & ls = layers[(size_t) il];
        if (!ls.resolved_tensors) {
            resolve_tensors(il, ls);
        }
        for (uint32_t id = 0; id < ls.n_experts; ++id) {
            if ((ls.pin_state[(size_t) id] & PIN_BASE) == 0 ||
                    std::binary_search((*target)[(size_t) il].begin(), (*target)[(size_t) il].end(), (int32_t) id)) {
                continue;
            }
            disk_stage->base_remove(il, (int32_t) id);
            demote_base_pin(il, ls, (int32_t) id);
            n_base_released++;
        }
    }

    // Promote target residents first, so forced admission cannot evict an
    // expert that is itself part of the new target.
    for (int il = 0; il < (int) layers.size() && il < (int) target->size(); ++il) {
        layer_state & ls = layers[(size_t) il];
        if (!ls.resolved_tensors) {
            resolve_tensors(il, ls);
        }
        for (const int32_t id : (*target)[(size_t) il]) {
            if (id >= 0 && (uint32_t) id < ls.n_experts && disk_stage->resident_slot_held(il, id) &&
                    (ls.pin_state[(size_t) id] & PIN_BASE) == 0) {
                if (admit_base_pin(il, ls, id)) {
                    n_base_admitted++;
                }
            }
        }
    }

    // New base experts are mandatory. If a pool is full, evict its coldest
    // dynamic resident without the usual heat margin/grace and retry.
    size_t refused = 0;
    for (int il = 0; il < (int) layers.size() && il < (int) target->size(); ++il) {
        layer_state & ls = layers[(size_t) il];
        if (!ls.resolved_tensors) {
            resolve_tensors(il, ls);
        }
        for (const int32_t id : (*target)[(size_t) il]) {
            if (id < 0 || (uint32_t) id >= ls.n_experts ||
                    (ls.pin_state[(size_t) id] & PIN_BASE) != 0) {
                continue;
            }
            bool admitted = admit_base_pin(il, ls, id);
            while (!admitted && evict_coldest_for_base(pool_of_layer(il))) {
                admitted = admit_base_pin(il, ls, id);
            }
            if (admitted) {
                n_base_admitted++;
            } else {
                n_base_refused++;
                refused++;
                LLAMA_LOG_WARN("%s: could not admit base expert layer %d expert %d\n", __func__, il, id);
            }
        }
    }

    disk_stage->set_base_target(name);
    active_base_set = name;
    n_base_switches++;
    rebuild_pinned_rank();
    if (refused > 0) {
        LLAMA_LOG_WARN("%s: %zu base expert(s) of set '%s' have no slot; they will be promoted on demand\n",
                       __func__, refused, name.empty() ? "<base>" : name.c_str());
    }
    LLAMA_LOG_INFO("%s: switched base expert set to '%s' (%d expert(s) permanent)\n",
                   __func__, name.empty() ? "<base>" : name.c_str(), n_base);
    return true;
}

void llama_hot_expert_cache::complete_disk_pin(int il, layer_state & ls, int32_t expert_id, size_t bytes) {
    // caller holds mu; the slot is reserved, its bytes are read by fill_cache()
    expert_key key{ il, expert_id };
    pinned_expert pe;
    pe.nbytes_locked = bytes;
    n_bytes_locked += bytes;
    pinned.emplace(key, std::move(pe));
    if (expert_id >= 0 && (uint32_t) expert_id < ls.n_experts) {
        ls.pin_state[(size_t) expert_id] |= PIN_RESIDENT;
    }
    pinned_rank_pool[(size_t) pool_of_layer(il)].insert({ count_of(il, expert_id), il, expert_id });
    if (il >= 0 && il < (int) n_pinned_layer.size()) {
        n_pinned_layer[il]++;
    }
    evicted_at.erase(key);
}

bool llama_hot_expert_cache::is_vram_resident(int il, int32_t expert_id) const {
    // caller holds mu. Cold path (VRAM eviction re-admission): borrows the
    // residency table instead of copying it
    if (vram_query == nullptr) {
        return false;
    }
    const layer_state * ls = layer_of(il);
    if (ls == nullptr || !ls->resolved_tensors) {
        return false;
    }
    if (expert_id < 0 || (uint32_t) expert_id >= ls->n_experts) {
        return false;
    }
    const uint8_t * flags = vram_query(vram_ud, il);
    return flags != nullptr && flags[(size_t) expert_id] != 0;
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
    if (pin_queue.size() >= pin_queue_max ||
            (!pin_queue.empty() && n_bytes_reserved + job.expected_bytes > pin_queue_max_bytes)) {
        return false;
    }
    if (budget_bytes > 0 && n_bytes_locked + n_bytes_reserved + job.expected_bytes > budget_bytes) {
        return false;
    }

    pin_queue.push_back(job);
    pin_inflight.insert(key);
    if (job.expert_id >= 0 && job.expert_id < (int32_t) layers[job.il].n_experts) {
        pin_state_at(layers[job.il], job.expert_id) |= PIN_INFLIGHT;
    }
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

        layer_state & wls = layers[job.il];
        if (job.expert_id >= 0 && (uint32_t) job.expert_id < wls.n_experts) {
            wls.pin_state[(size_t) job.expert_id] &= (uint8_t) ~PIN_INFLIGHT;
        }

        if (pe.nbytes_locked == 0 || syscall_error) {
            n_pin_failures++;
            LLAMA_LOG_DEBUG("%s: async pin of layer %d expert %d locked nothing\n", __func__, job.il, job.expert_id);
            continue;
        }

        n_bytes_locked += pe.nbytes_locked;

        // the expert's count may have kept climbing (or been decayed) since the
        // job was queued: insert its rank entry with the current value
        const uint64_t c = count_of(job.il, job.expert_id);
        pinned.emplace(key, std::move(pe));
        if (job.expert_id >= 0 && (uint32_t) job.expert_id < wls.n_experts) {
            wls.pin_state[(size_t) job.expert_id] |= PIN_RESIDENT;
        }
        pinned_rank_pool[(size_t) pool_of_layer(job.il)].insert({ c, job.il, job.expert_id });
        evicted_at.erase(key);  // pinned again: the grace bookkeeping is moot
    }
}

void llama_hot_expert_cache::unpin_expert(int il, layer_state & ls, int32_t expert_id, bool keep_l2) {
    expert_key key{ il, expert_id };
    auto       it = pinned.find(key);
    if (it == pinned.end()) {
        return;
    }

    n_bytes_locked -= it->second.nbytes_locked;
    pinned.erase(it);  // pinned_expert's destructor releases the mlock guards
    if (expert_id >= 0 && (uint32_t) expert_id < ls.n_experts) {
        ls.pin_state[(size_t) expert_id] &= (uint8_t) ~PIN_RESIDENT;
    }
    if (disk_stage != nullptr) {
        disk_stage->resident_remove(il, expert_id, keep_l2);
        if (il >= 0 && il < (int) n_pinned_layer.size() && n_pinned_layer[il] > 0) {
            n_pinned_layer[il]--;
        }
    }
}

void llama_hot_expert_cache::rebuild_pinned_rank() {
    // caller holds mu. Base experts are permanent residents and never enter the
    // victim rank
    pinned_rank_pool.assign((size_t) n_pools, std::set<std::tuple<uint64_t, int, int32_t>>());
    for (const auto & kv : pinned) {
        const expert_key & key = kv.first;
        const layer_state * ls = layer_of(key.layer);
        if (ls != nullptr && key.expert_id >= 0 && (uint32_t) key.expert_id < ls->n_experts &&
                (ls->pin_state[(size_t) key.expert_id] & PIN_BASE) != 0) {
            continue;
        }
        pinned_rank_pool[(size_t) pool_of_layer(key.layer)].insert({ count_of(key.layer, key.expert_id), key.layer, key.expert_id });
    }
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
    n_tokens_cur = n_tokens;

    // a mode switch is queued by another thread: apply it here, between graphs,
    // so the decode thread is the only one that touches the layers and the disk
    // stage target
    apply_pending_base_set();

    // the ranking is fed by single-token decode ubatches only, so only they
    // advance its clock: the decay halves the counts (and thereby moves the pin
    // set at the next promotion), so a multi-token batch/prefill ubatch must
    // neither count nor age the ranking - the pinned experts stay exactly as
    // generation left them while prefill runs.
    if (n_tokens != 1) {
        return;
    }

    if (profile_file != nullptr) {
        profile_tokens++;
    }

    n_content_tokens++;

    // periodic decay of the usage counts: keeps the pin set tracking the recent
    // routing mix instead of lifetime leaders. The clock is decode tokens, not
    // ubatches, accumulated here and checked between graph computes.
    if (decay_interval > 0) {
        n_tokens_seen += n_tokens;
        while (n_tokens_seen >= decay_interval) {
            n_tokens_seen -= decay_interval;
            decay_counts();
        }
    }

    // the VRAM tier's recent window ages on its own clock, independent of the
    // long-term ranking the RAM tier keeps
    if (recent_decay_interval > 0) {
        n_recent_tokens += n_tokens;
        while (n_recent_tokens >= recent_decay_interval) {
            n_recent_tokens -= recent_decay_interval;
            decay_recent_counts();
        }
    }
}

void llama_hot_expert_cache::decay_counts() {
    std::lock_guard<std::mutex> lock(mu);

    for (size_t il = 0; il < layers.size(); ++il) {
        auto & ls = layers[il];
        if (!ls.resolved_tensors) {
            continue;
        }
        for (uint64_t & c : ls.counts) {
            const uint64_t new_count = (c + 1) / 2;  // floor at 1, halves everything else
            if (new_count != c) {
                c = new_count;
            }
        }
    }

    // the halving changed the counts the pinned experts are ranked by: rebuild
    // the ordered set once instead of patching one key per pinned expert
    rebuild_pinned_rank();

    n_decays++;
}

void llama_hot_expert_cache::decay_recent_counts() {
    std::lock_guard<std::mutex> lock(mu);

    for (size_t il = 0; il < layers.size(); ++il) {
        auto & ls = layers[il];
        if (!ls.resolved_tensors) {
            continue;
        }
        for (uint64_t & c : ls.recent_counts) {
            c /= 2;  // floor at 0: a stale expert leaves the VRAM ranking
        }
    }

    n_recent_decays++;
}

// JSON string escaper for the profile writer. Byte tokens detokenize to valid
// UTF-8, so only the ASCII escapes and the C0 controls need handling.
static void write_json_string(std::FILE * f, const std::string & s) {
    std::fputc('"', f);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  std::fputs("\\\"", f); break;
            case '\\': std::fputs("\\\\", f); break;
            case '\b': std::fputs("\\b", f);  break;
            case '\f': std::fputs("\\f", f);  break;
            case '\n': std::fputs("\\n", f);  break;
            case '\r': std::fputs("\\r", f);  break;
            case '\t': std::fputs("\\t", f);  break;
            default:
                if (c < 0x20) {
                    std::fprintf(f, "\\u%04x", c);
                } else {
                    std::fputc(c, f);
                }
        }
    }
    std::fputc('"', f);
}

void llama_hot_expert_cache::note_output_token(int32_t token) {
    if (profile_file == nullptr && ngram_file == nullptr && ngram_predict == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu);
    if (profile_file != nullptr) {
        profile_output.push_back(token);
    }
    if (ngram_file != nullptr) {
        ngram_token      = token;
        ngram_have_token = true;
    }
    if (ngram_predict) {
        // predict the experts of the step about to run and hand the disk stage
        // the read list: it reads the non-resident ones into the L2 pool in
        // layer order while the graph computes
        const bool matched = ngram_predict->predict(token, ngram_predicted);
        if (matched) {
            n_ngram_matched++;
        } else {
            for (std::vector<int32_t> & v : ngram_predicted) {
                v.clear();
            }
        }
        if (disk_stage != nullptr && disk_stage->is_active()) {
            disk_stage->prefetch(ngram_predicted, ngram_prefetch_max);
        }
    }
}

void llama_hot_expert_cache::write_profile() {
    if (profile_file == nullptr || profile_routes == 0) {
        return;
    }

    std::fprintf(profile_file,
                 "{\"schema\":\"llama.expert_profile.v1\",\"profile_id\":%" PRIu64
                 ",\"scope\":\"decode\",\"model_arch\":\"%s\",\"n_layers\":%" PRId64
                 ",\"n_experts\":%" PRId64 ",\"n_experts_used\":%" PRId64
                 ",\"decode_tokens\":%" PRIu64 ",\"route_selections\":%" PRIu64 ",\"output\":",
                 profile_id++, model.arch_name().c_str(), (int64_t) model.hparams.n_layer(),
                 (int64_t) model.hparams.n_expert, (int64_t) model.hparams.n_expert_used(),
                 profile_tokens, profile_routes);
    write_json_string(profile_file, model.vocab.detokenize(profile_output, true));
    std::fputs(",\"layers\":[", profile_file);

    bool first_layer = true;
    for (size_t il = 0; il < layers.size(); ++il) {
        const auto & counts = layers[il].profile_counts;
        bool has_counts = false;
        for (uint64_t count : counts) {
            if (count != 0) {
                has_counts = true;
                break;
            }
        }
        if (!has_counts) {
            continue;
        }

        std::fprintf(profile_file, "%s{\"layer\":%zu,\"experts\":[", first_layer ? "" : ",", il);
        first_layer = false;
        bool first_expert = true;
        for (size_t expert = 0; expert < counts.size(); ++expert) {
            if (counts[expert] == 0) {
                continue;
            }
            std::fprintf(profile_file, "%s{\"expert\":%zu,\"count\":%" PRIu64 "}",
                         first_expert ? "" : ",", expert, counts[expert]);
            first_expert = false;
        }
        std::fputs("]}", profile_file);
    }
    std::fputs("]}\n", profile_file);
    std::fflush(profile_file);

    for (auto & layer : layers) {
        std::fill(layer.profile_counts.begin(), layer.profile_counts.end(), 0);
    }
    profile_output.clear();
    profile_tokens = 0;
    profile_routes = 0;
}

void llama_hot_expert_cache::write_ngram_header() {
    const uint32_t version  = NGRAM_REC_VERSION;
    const uint32_t n_layer  = (uint32_t) model.hparams.n_layer();
    const uint32_t n_expert = (uint32_t) model.hparams.n_expert;
    const uint32_t n_used   = (uint32_t) model.hparams.n_expert_used();
    std::fwrite(NGRAM_REC_MAGIC, 1, sizeof(NGRAM_REC_MAGIC), ngram_file);
    std::fwrite(&version,  sizeof(version),  1, ngram_file);
    std::fwrite(&n_layer,  sizeof(n_layer),  1, ngram_file);
    std::fwrite(&n_expert, sizeof(n_expert), 1, ngram_file);
    std::fwrite(&n_used,   sizeof(n_used),   1, ngram_file);
    std::fwrite(&ngram_model_fp,     sizeof(ngram_model_fp),     1, ngram_file);
    std::fwrite(&ngram_tokenizer_fp, sizeof(ngram_tokenizer_fp), 1, ngram_file);
    ngram_records = 0;
}

void llama_hot_expert_cache::write_ngram_step() {
    if (ngram_file == nullptr) {
        return;
    }
    if (!ngram_have_token) {
        ngram_tokens_missed++;
        return;
    }

    uint16_t n_layer_obs = 0;
    for (int il = 0; il < (int) obs_off.size(); ++il) {
        if (obs_off[il] >= 0 && obs_cnt[il] > 0) {
            n_layer_obs++;
        }
    }

    std::fwrite(&NGRAM_REC_DECODE_STEP, sizeof(NGRAM_REC_DECODE_STEP), 1, ngram_file);
    std::fwrite(&ngram_token, sizeof(ngram_token), 1, ngram_file);
    std::fwrite(&n_layer_obs, sizeof(n_layer_obs), 1, ngram_file);
    for (int il = 0; il < (int) obs_off.size(); ++il) {
        if (obs_off[il] < 0 || obs_cnt[il] <= 0) {
            continue;
        }
        const uint16_t il16 = (uint16_t) il;
        const uint16_t k    = (uint16_t) obs_cnt[il];
        std::fwrite(&il16, sizeof(il16), 1, ngram_file);
        std::fwrite(&k, sizeof(k), 1, ngram_file);
        std::fwrite(obs_ids + obs_off[il], sizeof(int32_t), (size_t) k, ngram_file);
    }

    ngram_records++;
    ngram_have_token = false;
}

void llama_hot_expert_cache::flush_profile() {
    std::lock_guard<std::mutex> lock(mu);

    write_profile();
}

void llama_hot_expert_cache::on_prompt_begin() {
    // A new prompt always starts from the template's default base set: the
    // previous turn's tool choice must not leak into the new one. The streamed
    // tool parser and the Markdown fence switch it again during the decode.
    if (disk_stage != nullptr) {
        request_base_set(disk_stage->select_base_set({}));
    }

    std::lock_guard<std::mutex> lock(mu);

    if (profile_file != nullptr) {
        write_profile();
    }

    // prompt boundary: the offline study must not build n-grams across two
    // unrelated prompts
    if (ngram_file != nullptr) {
        std::fwrite(&NGRAM_REC_PROMPT_BEGIN, sizeof(NGRAM_REC_PROMPT_BEGIN), 1, ngram_file);
        std::fflush(ngram_file);
    }
    if (ngram_predict) {
        ngram_predict->on_turn_begin();
    }

    // fresh epoch for the periodic decay clock: do not fire a scheduled halving
    // right on top of a prompt-start decay
    n_tokens_seen = 0;

    if (!track_rank || prompt_decay <= 1) {
        return;  // prefetch-only mode: no usage counts are maintained
    }

    // A new prompt defines new routing priorities (its decode ubatches start
    // feeding the ranking right after this): divide every usage count by
    // prompt_decay immediately (same floor-at-1 rounding as the periodic
    // halving), so the pin/VRAM sets can re-converge on the new prompt's
    // expert mix instead of letting the previous prompt's lifetime leaders hold
    // their slots.
    for (size_t il = 0; il < layers.size(); ++il) {
        auto & ls = layers[il];
        if (!ls.resolved_tensors) {
            continue;
        }
        for (uint64_t & c : ls.counts) {
            const uint64_t new_count = c / prompt_decay + (c % prompt_decay != 0);  // divide, floor at 1
            if (new_count != c) {
                c = new_count;
            }
        }
        // the recent window is already short, but a new prompt must not carry the
        // turn before it into the VRAM ranking either (floor at 0 so it can empty)
        for (uint64_t & c : ls.recent_counts) {
            c /= prompt_decay;
        }
    }

    // the counts the pinned experts are ranked by changed: rebuild the ordered
    // set once instead of patching one key per pinned expert
    rebuild_pinned_rank();

    n_decays++;
    n_decays_prompt++;
}

bool llama_hot_expert_cache::is_pinned(int il, int32_t expert_id) const {
    std::lock_guard<std::mutex> lock(mu);

    const layer_state * ls = layer_of(il);
    if (ls == nullptr || !ls->resolved_tensors) {
        return false;
    }
    if (expert_id < 0 || (uint32_t) expert_id >= ls->n_experts) {
        return false;
    }
    return (ls->pin_state[(size_t) expert_id] & (PIN_RESIDENT | PIN_BASE)) != 0;
}

void llama_hot_expert_cache::set_vram_query(vram_query_fn fn, void * ud) {
    std::lock_guard<std::mutex> lock(mu);

    vram_query = fn;
    vram_ud    = ud;
}

void llama_hot_expert_cache::vram_stats_snapshot(std::vector<uint64_t> & vram_hit,
                                                 std::vector<uint64_t> & vram_miss,
                                                 uint64_t & route_hit, uint64_t & route_miss) const {
    std::lock_guard<std::mutex> lock(mu);

    vram_hit.assign(layers.size(), 0);
    vram_miss.assign(layers.size(), 0);
    for (size_t il = 0; il < layers.size(); ++il) {
        vram_hit[il]  = layers[il].n_vram_hit;
        vram_miss[il] = layers[il].n_vram_miss;
    }
    route_hit  = n_route_hit;
    route_miss = n_route_miss;
}

void llama_hot_expert_cache::stats_snapshot(llama_expert_stats & out) const {
    std::lock_guard<std::mutex> lock(mu);

    out.routed_experts = n_route_total;
    out.decode_tokens  = n_content_tokens;
    out.experts_seen   = n_distinct;

    out.decode_cache.enabled  = disk_stage != nullptr && n_pin > 0;
    out.decode_cache.active   = out.decode_cache.enabled;
    out.decode_cache.residents = (uint64_t) pinned.size();
    out.decode_cache.capacity  = (uint64_t) std::max(n_pin_total, 0);
    out.decode_cache.resident_bytes = n_bytes_locked;
    out.decode_cache.assigned_routes   = n_route_hit;
    out.decode_cache.unassigned_routes = n_route_miss;
    out.decode_cache.base_routes       = n_route_base;

    for (const auto & layer : layers) {
        if (!layer.resolved_tensors) {
            continue;
        }
        for (uint32_t id = 0; id < layer.n_experts; ++id) {
            if ((layer.pin_state[id] & PIN_BASE) != 0 && layer.counts[id] > 0) {
                out.decode_cache.base_experts_used++;
            }
        }
    }
}

uint64_t llama_hot_expert_cache::content_tokens() const {
    std::lock_guard<std::mutex> lock(mu);
    return n_content_tokens;
}

uint64_t llama_hot_expert_cache::recent_decay_tokens() const {
    return recent_decay_interval;
}

void llama_hot_expert_cache::all_counts(std::vector<std::tuple<int, int32_t, uint64_t>> & out) const {
    std::lock_guard<std::mutex> lock(mu);

    out.clear();
    out.reserve(n_distinct);
    // the VRAM tier asks through this accessor: with a recent window configured
    // it ranks by the VRAM-only counts, so the RAM tier keeps the long-term ones
    const bool use_recent = recent_decay_interval > 0;
    for (size_t il = 0; il < layers.size(); ++il) {
        const auto & ls = layers[il];
        if (!ls.resolved_tensors) {
            continue;
        }
        const std::vector<uint64_t> & table = use_recent ? ls.recent_counts : ls.counts;
        for (uint32_t id = 0; id < ls.n_experts; ++id) {
            const uint64_t c = table[(size_t) id];
            if (c > 0) {
                out.emplace_back((int) il, (int32_t) id, c);
            }
        }
    }
}

int32_t llama_hot_expert_cache::assign_global_capacity(uint64_t budget_bytes,
        const std::vector<size_t> & bytes_per_layer,
        const std::vector<size_t> & fixed_bytes, std::vector<int32_t> & out) const {
    std::lock_guard<std::mutex> lock(mu);

    if (budget_bytes == 0 || n_distinct == 0) {
        return 0;
    }

    // walk the global ranking by count descending; an expert is kept only if its
    // whole cost still fits in the remaining budget (experts are indivisible).
    // Each layer also pays its one-time fixed cost (dummy slot + tables) when
    // its first slot is granted, so the final layout always fits the budget.
    std::vector<std::tuple<uint64_t, int, int32_t>> all;
    all.reserve(n_distinct);
    // a recent window ranks the VRAM layout by the VRAM-only counts (see
    // all_counts): the long-term table keeps driving the RAM/disk tier
    const bool use_recent = recent_decay_interval > 0;
    for (size_t il = 0; il < layers.size(); ++il) {
        const auto & ls = layers[il];
        if (!ls.resolved_tensors) {
            continue;
        }
        const std::vector<uint64_t> & table = use_recent ? ls.recent_counts : ls.counts;
        for (uint32_t id = 0; id < ls.n_experts; ++id) {
            const uint64_t c = table[(size_t) id];
            if (c > 0) {
                all.emplace_back(c, (int) il, (int32_t) id);
            }
        }
    }
    std::sort(all.begin(), all.end(), [](const auto & a, const auto & b) {
        return std::get<0>(a) > std::get<0>(b);
    });

    int32_t            assigned = 0;
    uint64_t           used     = 0;
    std::vector<uint8_t> fixed_charged(out.size(), 0);
    for (const auto & [count, layer, expert] : all) {
        if (layer < 0 || layer >= (int) out.size()) {
            continue;
        }
        const size_t cost = bytes_per_layer[layer];
        if (cost == 0) {
            continue;
        }
        const size_t extra = fixed_charged[layer] ? 0 : fixed_bytes[layer];
        if (used + cost + extra > budget_bytes) {
            continue; // does not fit; a colder expert that does fit may take its place
        }
        out[layer]++;
        used += cost + extra;
        fixed_charged[layer] = 1;
        assigned++;
    }
    return assigned;
}

void llama_hot_expert_cache::vram_takeover(int il, int32_t expert_id) {
    std::lock_guard<std::mutex> lock(mu);

    expert_key key{ il, expert_id };
    if (pinned.count(key) != 0) {
        unpin_expert(il, layers[il], expert_id);
    }
}

void llama_hot_expert_cache::promote_expert(int il, int32_t expert_id) {
    std::lock_guard<std::mutex> lock(mu);

    layer_state & ls = layers[il];
    if (!ls.resolved_tensors) {
        resolve_tensors(il, ls);
    }
    if (!ls.tensors_are_host || n_pin <= 0 || (disk_stage == nullptr && !llama_mlock::SUPPORTED)) {
        return;
    }
    if (is_vram_resident(il, expert_id)) {
        return;  // still served from VRAM (race); nothing to re-pin
    }
    if (expert_id < 0 || (uint32_t) expert_id >= ls.n_experts) {
        return;
    }

    // a base expert is a permanent RAM resident: re-admit it unconditionally,
    // ahead of the coldest dynamic resident and ignoring the usage floor
    if ((ls.pin_state[(size_t) expert_id] & PIN_BASE) != 0) {
        if (disk_stage != nullptr) {
            readmit_base(il, expert_id);
        }
        return;
    }

    const uint64_t c = ls.counts[(size_t) expert_id];
    if (c == 0) {
        return;
    }
    if ((ls.pin_state[(size_t) expert_id] & (PIN_RESIDENT | PIN_INFLIGHT)) != 0) {
        return;  // already resident in RAM (or a pin for it is queued)
    }
    try_promote(il, ls, expert_id, c, false, false);
}
