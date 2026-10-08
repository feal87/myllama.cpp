#include "llama-dense-vram.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <regex>
#include <vector>

struct llama_dense_vram::impl {
    ggml_backend_dev_t dev = nullptr;

    std::vector<ggml_tensor *> hosts; // model tensors that get promoted
    std::vector<ggml_tensor *> devs;  // their device copies, bound in buf_dev

    // saved state of the model tensors while promoted
    std::vector<ggml_backend_buffer_t> orig_buffer;
    std::vector<void *>                 orig_data;
    std::vector<void *>                 orig_extra;

    ggml_context *        ctx_dev = nullptr;
    ggml_backend_buffer_t buf_dev = nullptr;

    uint64_t nbytes = 0;

    bool armed  = false; // patterns matched at least one promotable tensor
    bool active = false; // model tensors currently repointed at the copies
    bool failed = false; // a previous promote() could not allocate

    void release() {
        for (size_t i = 0; i < hosts.size(); ++i) {
            hosts[i]->buffer = orig_buffer[i];
            hosts[i]->data   = orig_data[i];
            hosts[i]->extra  = orig_extra[i];
        }
        orig_buffer.clear();
        orig_data.clear();
        orig_extra.clear();

        if (buf_dev != nullptr) {
            ggml_backend_buffer_free(buf_dev);
            buf_dev = nullptr;
        }
        if (ctx_dev != nullptr) {
            ggml_free(ctx_dev);
            ctx_dev = nullptr;
        }
        devs.clear();
        active = false;
    }

    ~impl() {
        release();
    }
};

llama_dense_vram::llama_dense_vram(const llama_model & model, const std::string & patterns, ggml_backend_dev_t dev)
    : pimpl(new impl) {
    auto & p = *pimpl;
    p.dev = dev;

    if (dev == nullptr || patterns.empty()) {
        return;
    }

    std::vector<std::regex> res;
    size_t start = 0;
    while (start <= patterns.size()) {
        const size_t comma = patterns.find(',', start);
        std::string token = patterns.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        // trim leading/trailing blanks so a spaced list works
        const size_t b = token.find_first_not_of(" \t");
        const size_t e = token.find_last_not_of(" \t");
        token = b == std::string::npos ? std::string() : token.substr(b, e - b + 1);
        if (!token.empty()) {
            try {
                res.emplace_back(token);
            } catch (const std::regex_error & err) {
                LLAMA_LOG_WARN("%s: ignoring invalid regex '%s': %s\n", __func__, token.c_str(), err.what());
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (res.empty()) {
        return;
    }

    for (const auto & entry : model.tensors_by_name) {
        ggml_tensor * t = entry.second;
        if (t == nullptr || t->buffer == nullptr) {
            continue;
        }
        // already on the target device (or any device): nothing to promote
        if (!ggml_backend_buffer_is_host(t->buffer)) {
            continue;
        }
        // repointing a view would corrupt the aliasing; only own weights qualify
        if (t->view_src != nullptr) {
            continue;
        }
        bool match = false;
        for (const auto & re : res) {
            if (std::regex_search(entry.first, re)) {
                match = true;
                break;
            }
        }
        if (!match) {
            continue;
        }
        p.hosts.push_back(t);
        p.nbytes += ggml_nbytes(t);
    }

    p.armed = !p.hosts.empty();
}

llama_dense_vram::~llama_dense_vram() = default;

bool llama_dense_vram::is_enabled() const {
    return pimpl->armed;
}

bool llama_dense_vram::is_active() const {
    return pimpl->active;
}

uint64_t llama_dense_vram::bytes() const {
    return pimpl->armed && !pimpl->failed ? pimpl->nbytes : 0;
}

ggml_backend_dev_t llama_dense_vram::device() const {
    return pimpl->dev;
}

bool llama_dense_vram::promote() {
    auto & p = *pimpl;
    if (!p.armed || p.failed || p.active) {
        return p.active;
    }

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (p.hosts.size() + 1),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    p.ctx_dev = ggml_init(ip);
    if (p.ctx_dev == nullptr) {
        p.failed = true;
        return false;
    }

    for (ggml_tensor * h : p.hosts) {
        ggml_tensor * d = ggml_new_tensor(p.ctx_dev, h->type, GGML_MAX_DIMS, h->ne);
        ggml_set_name(d, h->name);
        p.devs.push_back(d);
    }

    p.buf_dev = ggml_backend_alloc_ctx_tensors_from_buft(p.ctx_dev, ggml_backend_dev_buffer_type(p.dev));
    if (p.buf_dev == nullptr) {
        LLAMA_LOG_WARN("%s: failed to allocate %.1f MiB on %s, dense VRAM promotion disabled\n",
                __func__, p.nbytes/(1024.0*1024.0), ggml_backend_dev_name(p.dev));
        ggml_free(p.ctx_dev);
        p.ctx_dev = nullptr;
        p.devs.clear();
        p.failed = true;
        return false;
    }
    // without this the scheduler can assign the matmul by expansion instead of
    // placing it on the weight's backend (see llama-moecache.cpp)
    ggml_backend_buffer_set_usage(p.buf_dev, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    p.orig_buffer.reserve(p.hosts.size());
    p.orig_data.reserve(p.hosts.size());
    p.orig_extra.reserve(p.hosts.size());

    uint64_t uploaded = 0;
    for (size_t i = 0; i < p.hosts.size(); ++i) {
        ggml_tensor * h = p.hosts[i];
        ggml_tensor * d = p.devs[i];

        ggml_backend_tensor_copy(h, d);

        p.orig_buffer.push_back(h->buffer);
        p.orig_data.push_back(h->data);
        p.orig_extra.push_back(h->extra);

        h->buffer = d->buffer;
        h->data   = d->data;
        h->extra  = d->extra;

        uploaded += ggml_nbytes(h);
    }
    p.active = true;

    LLAMA_LOG_INFO("%s: promoted %zu tensor(s), %.1f MiB to %s for decode\n",
            __func__, p.hosts.size(), uploaded/(1024.0*1024.0), ggml_backend_dev_name(p.dev));
    return true;
}

void llama_dense_vram::demote() {
    auto & p = *pimpl;
    if (!p.active) {
        return;
    }
    p.release();
    LLAMA_LOG_INFO("%s: restored %zu dense tensor(s) to host memory\n", __func__, p.hosts.size());
}
