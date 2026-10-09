#include "llama-expert-predict.h"

#include "llama-mmap.h"

#include <cstring>
#include <stdexcept>

// The on-disk layout is defined by scripts/expert-ngram/export_profile.py; keep
// the two in sync. All fields are little-endian.
namespace {

constexpr char     profile_magic[8] = { 'L', 'E', 'N', 'G', 'P', 'R', 'O', 'F' };
constexpr uint32_t profile_version  = 1;
constexpr size_t   header_size      = 68; // 8 magic + 9 u32 + 3 u64
constexpr uint16_t empty_id         = 0xFFFF;

constexpr uint64_t fnv_offset = 0xCBF29CE484222325ULL;
constexpr uint64_t fnv_prime  = 0x100000001B3ULL;
constexpr uint64_t order_salt = 0x9E3779B97F4A7C15ULL;

// FNV-1a over the token values, folded with the order, then a splitmix-style
// finalizer. Must match hash_key() in export_profile.py exactly.
uint64_t hash_ngram(const int32_t * toks, int32_t order) {
    uint64_t h = fnv_offset;
    for (int32_t i = 0; i < order; ++i) {
        h ^= (uint32_t) toks[i];
        h *= fnv_prime;
    }
    h ^= (uint64_t) order * order_salt;
    h *= fnv_prime;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    return h != 0 ? h : 1;
}

uint16_t rd_u16(const uint8_t * p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

uint32_t rd_u32(const uint8_t * p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

uint64_t rd_u64(const uint8_t * p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

} // namespace

struct llama_expert_predict_impl {
    std::unique_ptr<llama_file> file;
    std::unique_ptr<llama_mmap> map;

    const uint8_t * mask  = nullptr;
    const uint8_t * table = nullptr;
    size_t          slots = 0;

    int32_t  n_layer   = 0;
    int32_t  n_expert  = 0;
    int32_t  max_order = 0;
    int32_t  top_m     = 0;
    int32_t  n_enabled = 0;
    uint32_t table_size = 0;
    uint64_t n_entries  = 0;

    // enabled layers in ascending order, i.e. the payload ordinal order
    std::vector<int32_t> enabled;
    // the last max_order input tokens, oldest first
    std::vector<int32_t> history;

    uint64_t n_steps   = 0;
    uint64_t n_matched = 0;

    size_t slot_stride() const {
        return 8 + (size_t) n_enabled * (size_t) top_m * sizeof(uint16_t);
    }
};

llama_expert_predict::llama_expert_predict(const std::string & path, int32_t n_layer, int32_t n_expert)
    : pimpl(std::make_unique<llama_expert_predict_impl>()) {
    llama_expert_predict_impl & p = *pimpl;

    p.file = std::make_unique<llama_file>(path.c_str(), "rb");
    p.map  = std::make_unique<llama_mmap>(p.file.get());

    const uint8_t * base = (const uint8_t *) p.map->addr();
    const size_t    size = p.map->size();
    if (size < header_size || std::memcmp(base, profile_magic, sizeof(profile_magic)) != 0) {
        throw std::runtime_error("expert predict: '" + path + "' is not a LENGPROF table");
    }
    const uint32_t version = rd_u32(base + 8);
    if (version != profile_version) {
        throw std::runtime_error("expert predict: '" + path + "' has unsupported version " +
                                 std::to_string(version));
    }

    p.n_layer    = (int32_t) rd_u32(base + 12);
    p.n_expert   = (int32_t) rd_u32(base + 16);
    p.max_order  = (int32_t) rd_u32(base + 24);
    p.top_m      = (int32_t) rd_u32(base + 28);
    p.table_size = rd_u32(base + 32);
    p.n_enabled  = (int32_t) rd_u32(base + 36);
    const uint32_t mask_bytes = rd_u32(base + 40);
    p.n_entries  = rd_u64(base + 60);

    if (p.n_layer != n_layer || p.n_expert != n_expert) {
        throw std::runtime_error("expert predict: '" + path + "' is for " +
                                 std::to_string(p.n_layer) + " layers / " + std::to_string(p.n_expert) +
                                 " experts, the model has " + std::to_string(n_layer) + " / " +
                                 std::to_string(n_expert));
    }
    if (p.max_order < 1 || p.top_m < 1 || p.n_enabled < 0 || p.table_size == 0 ||
            (p.table_size & (p.table_size - 1)) != 0) {
        throw std::runtime_error("expert predict: '" + path + "' has invalid header fields");
    }

    const size_t mask_off  = header_size;
    const size_t table_off = mask_off + mask_bytes + (size_t) p.n_layer * sizeof(uint32_t);
    const size_t stride    = p.slot_stride();
    const size_t need      = table_off + (size_t) p.table_size * stride;
    if (need > size) {
        throw std::runtime_error("expert predict: '" + path + "' is truncated (" +
                                 std::to_string(size) + " < " + std::to_string(need) + " bytes)");
    }

    p.mask  = base + mask_off;
    p.table = base + table_off;
    p.slots = (size_t) p.table_size * stride;

    for (int32_t il = 0; il < p.n_layer; ++il) {
        if ((p.mask[il >> 3] >> (il & 7)) & 1) {
            p.enabled.push_back(il);
        }
    }
    if ((int32_t) p.enabled.size() != p.n_enabled) {
        throw std::runtime_error("expert predict: '" + path + "' layer mask does not match n_enabled");
    }
    p.history.reserve((size_t) p.max_order);
}

llama_expert_predict::~llama_expert_predict() = default;

void llama_expert_predict::on_turn_begin() {
    pimpl->history.clear();
}

bool llama_expert_predict::predict(int32_t token, std::vector<std::vector<int32_t>> & out) {
    llama_expert_predict_impl & p = *pimpl;
    p.n_steps++;

    out.assign((size_t) p.n_layer, {});
    if (p.n_enabled == 0 || p.table_size == 0) {
        return false;
    }

    p.history.push_back(token);
    if ((int32_t) p.history.size() > p.max_order) {
        p.history.erase(p.history.begin());
    }

    const size_t stride = p.slot_stride();
    for (int32_t order = p.max_order; order >= 1; --order) {
        if ((int32_t) p.history.size() < order) {
            continue;
        }
        const int32_t * toks = p.history.data() + (p.history.size() - (size_t) order);
        const uint64_t  hash = hash_ngram(toks, order);
        uint32_t        idx  = (uint32_t) hash & (p.table_size - 1);
        for (uint32_t probe = 0; probe < p.table_size; ++probe) {
            const uint8_t * slot = p.table + (size_t) idx * stride;
            const uint64_t  key  = rd_u64(slot);
            if (key == 0) {
                break; // empty slot: this order is not in the table
            }
            if (key == hash) {
                const uint8_t * payload = slot + 8;
                for (int32_t o = 0; o < p.n_enabled; ++o) {
                    const int32_t          il  = p.enabled[(size_t) o];
                    std::vector<int32_t> & dst = out[(size_t) il];
                    for (int32_t m = 0; m < p.top_m; ++m) {
                        const uint16_t id = rd_u16(payload + ((size_t) o * p.top_m + m) * sizeof(uint16_t));
                        if (id != empty_id) {
                            dst.push_back((int32_t) id);
                        }
                    }
                }
                p.n_matched++;
                return true;
            }
            idx = (idx + 1) & (p.table_size - 1);
        }
    }
    return false;
}

int32_t llama_expert_predict::n_layer() const   { return pimpl->n_layer; }
int32_t llama_expert_predict::n_enabled() const { return pimpl->n_enabled; }
int32_t llama_expert_predict::max_order() const { return pimpl->max_order; }
int32_t llama_expert_predict::top_m() const     { return pimpl->top_m; }
uint64_t llama_expert_predict::n_steps() const  { return pimpl->n_steps; }
uint64_t llama_expert_predict::n_matched() const { return pimpl->n_matched; }
