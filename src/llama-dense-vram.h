#pragma once

// Decode-only promotion of host-resident model weights into VRAM.
//
// The pool shares the VRAM reclaimed from the prefill compute buffer with the
// MoE expert cache (see llama_context::vram_swap). At the prefill -> decode
// transition the weights matched by the name patterns are copied into a device
// buffer and the model tensors are repointed at the copies, so the decode graph
// runs their matmuls on the GPU; at decode -> prefill the originals are restored
// and the pool freed. Prefill always uses the host originals.
//
// This is exact: the device tensor is a plain copy of the same weights, only the
// backend that runs the op changes.

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_model;
struct ggml_tensor;

class llama_dense_vram {
  public:
    // patterns: comma-separated std::regex matched against model tensor names.
    // Only tensors that are NOT already on `dev` are promoted. dev == null
    // disables the feature
    llama_dense_vram(const llama_model & model, const std::string & patterns, ggml_backend_dev_t dev);
    ~llama_dense_vram();

    llama_dense_vram(const llama_dense_vram &) = delete;
    llama_dense_vram & operator=(const llama_dense_vram &) = delete;

    bool is_enabled() const; // matched at least one promotable tensor
    bool is_active()  const; // device copies currently bound to the model tensors
    uint64_t bytes()  const; // total device bytes of the promoted set (0 when disabled or failed)
    ggml_backend_dev_t device() const;

    // contiguous host byte runs (base, size, 4096-aligned) of the promoted set.
    // The bytes hold the weights until promote() and are free for a decode-only
    // expert L2 pool while the promotion is active. Static so the disk stage can
    // size its L2 pool before the context builds this object
    static std::vector<std::pair<void *, size_t>> find_host_regions(const llama_model & model, const std::string & patterns);
    const std::vector<std::pair<void *, size_t>> & host_regions() const;

    // allocate the device copies, upload and repoint the model tensors. Returns
    // false (and disarms the feature) when the device buffer cannot be allocated
    bool promote();
    // restore the model tensors and free the device copies; no-op when not active
    void demote();

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
