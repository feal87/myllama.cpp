// llama-kld: real-corpus KLD harness over libllama.
//
// Reproduces the serving shape on one real request: a clean multi-token prefill
// of the prompt (system + user, built with the real chat template), then
// single-token decode of the assistant turn, which is where the disk-stage drop
// is active.
//
// Two arms share one base file:
//   --save-base FILE   the drop-off arm quantizes the log-probs of every scored
//                      position and writes them, no statistics
//   --load-base FILE   a treatment arm reads that file and computes
//                      D_KL(base||arm), the true-token probability change and
//                      top-1 agreement per position
// Both arms must produce the same scored positions (pos, target token); the
// base file stores them so a mismatched arm fails loudly.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "../../src/llama-ext.h" // llama_get_expert_stats

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// ---------------------------------------------------------------------------
// KLD plumbing (same math as tools/perplexity/perplexity.cpp)
// ---------------------------------------------------------------------------

struct kl_stats {
    double sum_nll          = 0.0;
    double sum_nll2         = 0.0;
    double sum_nll_base     = 0.0;
    double sum_nll_base2    = 0.0;
    double sum_nll_nll_base = 0.0;
    double sum_kld          = 0.0;
    double sum_kld2         = 0.0;
    double sum_p_diff       = 0.0;
    double sum_p_diff2      = 0.0;
    double sum_p_diff4      = 0.0;
    float  max_p_diff       = 0.0f;
    size_t n_same_top       = 0;
    size_t count            = 0;
};

struct kld_point {
    double kld      = 0.0;
    float  dp       = 0.0f;
    bool   same_top = false;
};

static inline int nearest_int(float fval) {
    // identical to ggml-quants.c so the base file matches the perplexity format
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// quantize one row of log-probs and return the target's NLL
static double quantize_log_softmax(int n_vocab, const float * logits, uint16_t * log_prob, int tok) {
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp  = log(sum_exp);
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale        = (max_logit - min_logit) / 65535.f;
    float * d = (float *) log_prob;
    d[0] = scale;
    d[1] = min_log_prob;
    log_prob += 4;
    if (scale) {
        const float inv_scale = 1 / scale;
        for (int i = 0; i < n_vocab; ++i) {
            log_prob[i] = logits[i] > min_logit ? nearest_int(inv_scale * (logits[i] - min_logit)) : 0;
        }
    } else {
        memset(log_prob, 0, n_vocab * sizeof(uint16_t));
    }
    return max_logit + log_sum_exp - logits[tok];
}

static kld_point kld_log_softmax(int n_vocab, const float * logits, const uint16_t * base_log_prob, int tok, kl_stats & kld) {
    float max_logit = logits[0];
    int   imax      = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            imax      = i;
        }
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);

    const float * d = (const float *) base_log_prob;
    const float scale        = d[0];
    const float min_log_prob = d[1];
    base_log_prob += 4;

    const float nll = max_logit + log_sum_exp - logits[tok];
    kld.sum_nll  += nll;
    kld.sum_nll2 += nll * nll;

    const float nll_base = -(scale * base_log_prob[tok] + min_log_prob);
    kld.sum_nll_base  += nll_base;
    kld.sum_nll_base2 += nll_base * nll_base;
    kld.sum_nll_nll_base += nll * nll_base;

    max_logit += log_sum_exp;
    double sum = 0.0;
    int   imax_base    = -1;
    float p_log_base_max = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        const float p_log_base = scale * base_log_prob[i] + min_log_prob;
        if (i == 0 || p_log_base > p_log_base_max) {
            p_log_base_max = p_log_base;
            imax_base      = i;
        }
        if (p_log_base > -16.f) {
            const float p_base = expf(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    kld.sum_kld  += sum;
    kld.sum_kld2 += sum * sum;
    ++kld.count;

    const bool same_top = imax == imax_base;
    if (same_top) {
        ++kld.n_same_top;
    }

    const float p_base = expf(-nll_base);
    const float p      = expf(-nll);
    const float p_diff = p - p_base;
    kld.sum_p_diff  += p_diff;
    const double p_diff2 = (double) p_diff * p_diff;
    kld.sum_p_diff2 += p_diff2;
    kld.sum_p_diff4 += p_diff2 * p_diff2;
    kld.max_p_diff   = std::max(kld.max_p_diff, std::fabs(p_diff));

    return { sum, p_diff, same_top };
}

// ---------------------------------------------------------------------------
// base file (LLKLD001)
// ---------------------------------------------------------------------------

static const char * KLB_MAGIC   = "LLKLD001";
static const uint32_t KLB_VER   = 1;

static int klb_nv(int n_vocab) {
    return 2 * ((n_vocab + 1) / 2) + 4;
}

static void write_bytes(std::ofstream & out, const void * p, size_t n) {
    out.write((const char *) p, (std::streamsize) n);
}
static void read_bytes(std::ifstream & in, void * p, size_t n) {
    in.read((char *) p, (std::streamsize) n);
}

static void write_base_header(std::ofstream & out, int n_vocab, uint32_t P, uint32_t D,
                              const std::string & tag, const std::vector<llama_token> & toks) {
    const uint32_t version = KLB_VER;
    const uint32_t nv      = (uint32_t) n_vocab;
    const uint32_t taglen  = (uint32_t) tag.size();
    write_bytes(out, KLB_MAGIC, 8);
    write_bytes(out, &version, 4);
    write_bytes(out, &nv, 4);
    write_bytes(out, &P, 4);
    write_bytes(out, &D, 4);
    write_bytes(out, &taglen, 4);
    write_bytes(out, tag.data(), taglen);
    write_bytes(out, toks.data(), toks.size() * sizeof(toks[0]));
}

// returns false and logs on any mismatch
static bool read_base_header(std::ifstream & in, int n_vocab, uint32_t & P, uint32_t & D,
                             const std::string & tag, const std::vector<llama_token> & toks) {
    char magic[8];
    uint32_t version = 0, nv = 0, taglen = 0;
    read_bytes(in, magic, 8);
    if (!in || memcmp(magic, KLB_MAGIC, 8) != 0) {
        LOG_ERR("%s: not a LLKLD001 base file\n", __func__);
        return false;
    }
    read_bytes(in, &version, 4);
    read_bytes(in, &nv, 4);
    read_bytes(in, &P, 4);
    read_bytes(in, &D, 4);
    read_bytes(in, &taglen, 4);
    if (!in || version != KLB_VER) {
        LOG_ERR("%s: unsupported base version\n", __func__);
        return false;
    }
    if ((int) nv != n_vocab) {
        LOG_ERR("%s: vocabulary mismatch (%u vs %d)\n", __func__, nv, n_vocab);
        return false;
    }
    std::string file_tag(taglen, '\0');
    read_bytes(in, file_tag.data(), taglen);
    if (file_tag != tag) {
        LOG_ERR("%s: config tag mismatch (%s vs %s)\n", __func__, file_tag.c_str(), tag.c_str());
        return false;
    }
    std::vector<llama_token> file_toks(toks.size());
    read_bytes(in, file_toks.data(), file_toks.size() * sizeof(file_toks[0]));
    if (!in) {
        LOG_ERR("%s: truncated base file\n", __func__);
        return false;
    }
    for (size_t i = 0; i < toks.size(); ++i) {
        if (file_toks[i] != toks[i]) {
            LOG_ERR("%s: token stream mismatch at %zu (%d vs %d)\n", __func__, i, file_toks[i], toks[i]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------

struct kld_cli {
    std::string system_file;
    std::string user_file;
    std::string decode_file;
    std::string chat_template_file;
    std::string reasoning_effort = "max";
    std::string prefill_prefix_file;
    std::string dump_positions;
    std::string save_base;
    std::string load_base;
    std::string config_tag;
    int         prefill_batch     = 0;
    int         max_decode_tokens = -1;
};

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERR("%s: failed to open %s\n", __func__, path.c_str());
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool check_logits(const float * lg, int n_vocab, const char * what) {
    if (lg == nullptr) {
        LOG_ERR("%s: null logits for %s\n", __func__, what);
        return false;
    }
    float  mn = lg[0];
    float  mx = lg[0];
    double sum = 0.0;
    int    bad = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float v = lg[i];
        if (!std::isfinite(v)) {
            ++bad;
        }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
    }
    LOG_INF("logits[%s]: min=%.4f max=%.4f mean=%.4f nonfinite=%d\n", what, mn, mx, sum / n_vocab, bad);
    return bad == 0;
}

static float percentile(std::vector<float> v, float fraction) {
    if (v.empty()) {
        return 0.0f;
    }
    if (fraction <= 0) {
        return v.front();
    }
    if (fraction >= 1) {
        return v.back();
    }
    float p = fraction * (v.size() - 1);
    size_t ip = (size_t) p;
    p -= ip;
    return (1 - p) * v[ip] + p * v[std::min(ip + 1, v.size() - 1)];
}

static void print_usage(int, char ** argv) {
    printf("\nllama-kld: real-corpus prefill+decode KLD harness\n\n");
    printf("usage: %s -m model.gguf --system-file S --user-file U --decode-file D [common options]\n\n", argv[0]);
    printf("harness options:\n");
    printf("  --system-file FILE       system-role content (required)\n");
    printf("  --user-file FILE         user-role content (required)\n");
    printf("  --decode-file FILE       assistant turn to teacher-force (required)\n");
    printf("  --chat-template-file F   template override (default: model metadata)\n");
    printf("  --reasoning-effort LEVEL template reasoning_effort (default: max)\n");
    printf("  --prefill-prefix-file F  prepend this text instead of applying the template\n");
    printf("  --prefill-batch N        multi-token prefill ubatch (default: n_batch)\n");
    printf("  --save-base FILE         drop-off arm: write the base log-probs\n");
    printf("  --load-base FILE         treatment arm: read the base log-probs and score\n");
    printf("  --dump-positions FILE    write per-position CSV\n");
    printf("  --max-decode-tokens N    stop after N decode tokens (-1 = all)\n");
    printf("  --config-tag STR         tag written to / checked against the base file\n");
    printf("\nall other options are passed to llama.cpp (model, -c, --load-mode, --disk-stage-*, ...)\n\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argc, argv);
            return 0;
        }
    }

    kld_cli cli;

    auto take = [&](int & i, const char * name, std::string & out) {
        if (strcmp(argv[i], name) != 0) {
            return false;
        }
        if (i + 1 >= argc) {
            LOG_ERR("missing value for %s\n", name);
            exit(1);
        }
        out = argv[++i];
        return true;
    };

    std::vector<std::string> kept;
    kept.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (take(i, "--system-file",         cli.system_file))        continue;
        if (take(i, "--user-file",           cli.user_file))          continue;
        if (take(i, "--decode-file",         cli.decode_file))        continue;
        if (take(i, "--chat-template-file",  cli.chat_template_file)) continue;
        if (take(i, "--reasoning-effort",    cli.reasoning_effort))   continue;
        if (take(i, "--prefill-prefix-file", cli.prefill_prefix_file)) continue;
        if (take(i, "--dump-positions",      cli.dump_positions))     continue;
        if (take(i, "--save-base",           cli.save_base))          continue;
        if (take(i, "--load-base",           cli.load_base))          continue;
        if (take(i, "--config-tag",          cli.config_tag))         continue;
        if (strcmp(argv[i], "--prefill-batch") == 0 && i + 1 < argc) {
            cli.prefill_batch = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "--max-decode-tokens") == 0 && i + 1 < argc) {
            cli.max_decode_tokens = atoi(argv[++i]);
            continue;
        }
        kept.push_back(argv[i]);
    }

    if (cli.system_file.empty() || cli.user_file.empty() || cli.decode_file.empty()) {
        LOG_ERR("--system-file, --user-file and --decode-file are required\n");
        print_usage(argc, argv);
        return 1;
    }
    if (!cli.save_base.empty() && !cli.load_base.empty()) {
        LOG_ERR("--save-base and --load-base are mutually exclusive\n");
        return 1;
    }

    std::vector<char *> cargv;
    cargv.reserve(kept.size());
    for (auto & s : kept) {
        cargv.push_back(s.data());
    }

    common_params params;
    if (!common_params_parse((int) cargv.size(), cargv.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t       n_batch = (int32_t) llama_n_batch(ctx);
    const int           nv      = klb_nv(n_vocab);

    // ---- prompt: real chat template (or a raw prefix override) ----
    const std::string system_text = read_file(cli.system_file);
    const std::string user_text   = read_file(cli.user_file);

    std::string prompt_text;
    if (!cli.prefill_prefix_file.empty()) {
        prompt_text  = read_file(cli.prefill_prefix_file);
        prompt_text += user_text;
    } else {
        const std::string tmpl_src = cli.chat_template_file.empty()
                ? std::string()
                : read_file(cli.chat_template_file);
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, tmpl_src);

        common_chat_templates_inputs inputs;
        common_chat_msg msg_sys;
        msg_sys.role    = "system";
        msg_sys.content = system_text;
        common_chat_msg msg_usr;
        msg_usr.role    = "user";
        msg_usr.content = user_text;

        inputs.messages              = { msg_sys, msg_usr };
        inputs.add_generation_prompt = true;
        inputs.use_jinja             = true;
        inputs.chat_template_kwargs["reasoning_effort"] = "\"" + cli.reasoning_effort + "\"";

        const common_chat_params chat = common_chat_templates_apply(tmpls.get(), inputs);
        prompt_text = chat.prompt;
    }

    std::vector<llama_token> all_tokens = common_tokenize(vocab, prompt_text, false, true);
    const int64_t P = (int64_t) all_tokens.size();
    {
        const std::vector<llama_token> dec = common_tokenize(vocab, read_file(cli.decode_file), false, true);
        all_tokens.insert(all_tokens.end(), dec.begin(), dec.end());
    }
    int64_t D = (int64_t) all_tokens.size() - P;
    if (cli.max_decode_tokens >= 0) {
        D = std::min<int64_t>(D, cli.max_decode_tokens);
        all_tokens.resize(P + D);
    }

    if (P < 2) {
        LOG_ERR("%s: prompt too short (%lld tokens)\n", __func__, (long long) P);
        return 1;
    }
    const int64_t n_ctx = llama_n_ctx(ctx);
    if (P + D > n_ctx) {
        LOG_ERR("%s: P+D=%lld exceeds n_ctx=%lld\n", __func__, (long long) (P + D), (long long) n_ctx);
        return 1;
    }

    LOG_INF("llama-kld: config_tag=%s P=%lld D=%lld n_ctx=%lld n_batch=%d n_vocab=%d\n",
            cli.config_tag.c_str(), (long long) P, (long long) D, (long long) n_ctx, n_batch, n_vocab);
    LOG_INF("template prompt: system=%zu user=%zu prompt=%zu bytes; tokens head %d %d %d %d ... tail %d %d\n",
            system_text.size(), user_text.size(), prompt_text.size(),
            all_tokens[0], all_tokens[1], all_tokens[2], all_tokens[3],
            all_tokens[P - 2], all_tokens[P - 1]);

    // ---- base file streams ----
    std::ofstream  base_out;
    std::ifstream  base_in;
    std::vector<uint16_t> base_block(nv);
    if (!cli.save_base.empty()) {
        base_out.open(cli.save_base, std::ios::binary);
        if (!base_out) {
            LOG_ERR("%s: failed to open %s for writing\n", __func__, cli.save_base.c_str());
            return 1;
        }
        const uint32_t p32 = (uint32_t) P, d32 = (uint32_t) D;
        write_base_header(base_out, n_vocab, p32, d32, cli.config_tag, all_tokens);
    } else if (!cli.load_base.empty()) {
        base_in.open(cli.load_base, std::ios::binary);
        if (!base_in) {
            LOG_ERR("%s: failed to open %s\n", __func__, cli.load_base.c_str());
            return 1;
        }
        uint32_t p32 = 0, d32 = 0;
        if (!read_base_header(base_in, n_vocab, p32, d32, cli.config_tag, all_tokens)) {
            return 1;
        }
        if ((int64_t) p32 != P || (int64_t) d32 != D) {
            LOG_ERR("%s: base P/D mismatch (%u/%u vs %lld/%lld)\n", __func__, p32, d32, (long long) P, (long long) D);
            return 1;
        }
    }

    // ---- per-position stats ----
    kl_stats stats;
    std::vector<float> kld_values;
    std::vector<float> p_diff_values;
    if (D > 0) {
        kld_values.reserve(D);
        p_diff_values.reserve(D);
    }

    std::ofstream pos_csv;
    if (!cli.dump_positions.empty()) {
        pos_csv.open(cli.dump_positions);
        if (cli.load_base.empty()) {
            pos_csv << "pos,token,drop,routed\n";
        } else {
            pos_csv << "pos,token,kld,dp,top1,drop,routed\n";
        }
    }

    auto score = [&](int64_t pos, llama_token target, const float * logits, uint64_t drop_step, uint64_t routed_step) {
        if (base_out.is_open()) {
            double nll = quantize_log_softmax(n_vocab, logits, base_block.data(), target);
            const uint32_t p32 = (uint32_t) pos;
            const uint32_t t32 = (uint32_t) target;
            const float    f32 = (float) nll;
            write_bytes(base_out, &p32, 4);
            write_bytes(base_out, &t32, 4);
            write_bytes(base_out, &f32, 4);
            write_bytes(base_out, base_block.data(), (size_t) nv * 2);
            if (!base_out) {
                LOG_ERR("%s: base write failed at pos %lld\n", __func__, (long long) pos);
                exit(1);
            }
            if (pos_csv.is_open()) {
                pos_csv << pos << "," << target << "," << drop_step << "," << routed_step << "\n";
            }
            return;
        }
        if (base_in.is_open()) {
            uint32_t p32 = 0, t32 = 0;
            float    f32 = 0.0f;
            read_bytes(base_in, &p32, 4);
            read_bytes(base_in, &t32, 4);
            read_bytes(base_in, &f32, 4);
            read_bytes(base_in, base_block.data(), (size_t) nv * 2);
            if (!base_in) {
                LOG_ERR("%s: base read failed at pos %lld\n", __func__, (long long) pos);
                exit(1);
            }
            if ((int64_t) p32 != pos || (llama_token) t32 != target) {
                LOG_ERR("%s: position/target mismatch at pos %lld (%u/%u)\n", __func__, (long long) pos, p32, t32);
                exit(1);
            }
            (void) f32; // base NLL is recomputed inside kld_log_softmax
            const kld_point pt = kld_log_softmax(n_vocab, logits, base_block.data(), target, stats);
            kld_values.push_back((float) pt.kld);
            p_diff_values.push_back(pt.dp);
            if (pos_csv.is_open()) {
                pos_csv << pos << "," << target << "," << pt.kld << "," << pt.dp << ","
                        << (pt.same_top ? 1 : 0) << "," << drop_step << "," << routed_step << "\n";
            }
            return;
        }
        // no base: just the drop/routed record
        if (pos_csv.is_open()) {
            pos_csv << pos << "," << target << "," << drop_step << "," << routed_step << "\n";
        }
    };

    int pb = cli.prefill_batch > 0 ? cli.prefill_batch : n_batch;
    pb = std::min(pb, n_batch);

    llama_memory_clear(llama_get_memory(ctx), true);

    common_batch batch(ctx);

    // ---- prefill: multi-token, must be drop-free ----
    const auto t_pf0 = std::chrono::high_resolution_clock::now();

    const llama_expert_stats st0 = llama_get_expert_stats(ctx);
    uint64_t dropped_seen = st0.decode_cache.dropped_routes;

    int last_batch_index = -1;
    for (int64_t off = 0; off < P; off += pb) {
        const int n = (int) std::min<int64_t>(pb, P - off);
        batch.clear();
        for (int i = 0; i < n; ++i) {
            batch.add(all_tokens[off + i], (llama_pos) (off + i), 0, off + i == P - 1);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: prefill failed at %lld\n", __func__, (long long) off);
            return 1;
        }
        const uint64_t d = llama_get_expert_stats(ctx).decode_cache.dropped_routes;
        if (d != dropped_seen) {
            LOG_ERR("%s: drop fired in prefill at pos %lld (%llu -> %llu)\n",
                    __func__, (long long) off, (unsigned long long) dropped_seen, (unsigned long long) d);
            return 1;
        }
        last_batch_index = n - 1;
    }

    llama_synchronize(ctx);
    const double t_prefill = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_pf0).count();

    const llama_expert_stats st_pf = llama_get_expert_stats(ctx);
    LOG_INF("prefill done: %.1f s, %.1f t/s, routed=%llu dropped=%llu\n",
            t_prefill, P / std::max(0.001, t_prefill),
            (unsigned long long) (st_pf.routed_experts - st0.routed_experts),
            (unsigned long long) (st_pf.decode_cache.dropped_routes - st0.decode_cache.dropped_routes));

    check_logits(llama_get_logits_ith(ctx, last_batch_index), n_vocab, "prefill-tail");

    // ---- decode: single-token, drop active ----
    const auto t_dec0 = std::chrono::high_resolution_clock::now();

    uint64_t prev_drop   = st_pf.decode_cache.dropped_routes;
    uint64_t prev_routed = st_pf.routed_experts;

    // first scored position: the prefill tail predicts decode[0]
    if (D > 0) {
        score(P - 1, all_tokens[P], llama_get_logits_ith(ctx, last_batch_index), 0, 0);
    }

    const int64_t n_steps = D > 0 ? D - 1 : 0; // last output token is a target only
    for (int64_t k = 0; k < n_steps; ++k) {
        batch.clear();
        batch.add(all_tokens[P + k], (llama_pos) (P + k), 0, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: decode failed at %lld\n", __func__, (long long) k);
            return 1;
        }

        const llama_expert_stats st = llama_get_expert_stats(ctx);
        const uint64_t dd = st.decode_cache.dropped_routes - prev_drop;
        const uint64_t rr = st.routed_experts - prev_routed;
        prev_drop   = st.decode_cache.dropped_routes;
        prev_routed = st.routed_experts;

        if (k == 0) {
            check_logits(llama_get_logits(ctx), n_vocab, "decode-0");
        }
        score(P + k, all_tokens[P + k + 1], llama_get_logits(ctx), dd, rr);

        if (k > 0 && k % 2000 == 0) {
            const double t = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_dec0).count();
            LOG_INF("decode %lld/%lld (%.2f t/s)\n", (long long) k, (long long) n_steps, k / std::max(0.001, t));
        }
    }

    llama_synchronize(ctx);
    const double t_dec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_dec0).count();

    const llama_expert_stats st_end = llama_get_expert_stats(ctx);
    const uint64_t routed_dec  = st_end.routed_experts - st_pf.routed_experts;
    const uint64_t dropped_dec = st_end.decode_cache.dropped_routes - st_pf.decode_cache.dropped_routes;

    LOG_INF("decode done: %lld steps, %.1f s, %.2f t/s, routed=%llu dropped=%llu (%.2f%%)\n",
            (long long) n_steps, t_dec, n_steps / std::max(0.001, t_dec),
            (unsigned long long) routed_dec, (unsigned long long) dropped_dec,
            routed_dec ? 100.0 * (double) dropped_dec / (double) routed_dec : 0.0);

    LOG_INF("total: %.1f s (prefill %.1f + decode %.1f)\n", t_prefill + t_dec, t_prefill, t_dec);

    // ---- summary ----
    if (base_in.is_open() && stats.count > 0) {
        std::sort(kld_values.begin(), kld_values.end());
        std::sort(p_diff_values.begin(), p_diff_values.end());

        const double mean_nll  = stats.sum_nll / stats.count;
        const double mean_base = stats.sum_nll_base / stats.count;
        const double ppl_q     = std::exp(mean_nll);
        const double ppl_base  = std::exp(mean_base);
        const double mean_kld  = stats.sum_kld / stats.count;
        const double rms_dp    = std::sqrt(stats.sum_p_diff2 / stats.count);

        LOG_INF("====== KLD summary ======\n");
        LOG_INF("scored        : %zu (routed %llu, dropped %llu, %.2f%%)\n",
                stats.count, (unsigned long long) routed_dec, (unsigned long long) dropped_dec,
                routed_dec ? 100.0 * (double) dropped_dec / (double) routed_dec : 0.0);
        LOG_INF("PPL(Q)        : %.6f\n", ppl_q);
        LOG_INF("PPL(base)     : %.6f\n", ppl_base);
        LOG_INF("PPL(Q)/base   : %.6f\n", ppl_q / ppl_base);
        LOG_INF("Mean KLD      : %.6f\n", mean_kld);
        LOG_INF("Median KLD    : %.6f\n", (kld_values.size() % 2 == 0
                ? 0.5f * (kld_values[kld_values.size() / 2] + kld_values[kld_values.size() / 2 - 1])
                : kld_values[kld_values.size() / 2]));
        LOG_INF("99.9%% KLD     : %.6f\n", percentile(kld_values, 0.999f));
        LOG_INF("99.0%% KLD     : %.6f\n", percentile(kld_values, 0.990f));
        LOG_INF("95.0%% KLD     : %.6f\n", percentile(kld_values, 0.950f));
        LOG_INF("Maximum KLD   : %.6f\n", kld_values.back());
        LOG_INF("Mean dp       : %.6f\n", stats.sum_p_diff / stats.count);
        LOG_INF("RMS dp        : %.6f\n", rms_dp);
        LOG_INF("Same top p    : %.6f\n", 100.0 * stats.n_same_top / stats.count);
    }

    llama_perf_context_print(ctx);

    return 0;
}
