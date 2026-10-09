#pragma once

// Runtime MoE expert prediction (--expert-ngram-profile FILE).
//
// The profile is a LENGPROF v1 table built by scripts/expert-ngram/export_profile.py:
// token n-grams hashed to the top-M experts each enabled MoE layer routed. Per
// single-token decode step the predictor hashes the last `order` input tokens for
// order = max_order .. 1 and stops at the first order present (the same backoff
// the exporter measures). One probe yields every enabled layer's ids, so the
// caller can hand the disk stage the whole step's read list at once.
//
// The table is mmap'd read-only and shared by every context of the model.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_expert_predict_impl;

class llama_expert_predict {
  public:
    // `path` is mmap'd read-only; throws std::runtime_error when the file is not
    // a LENGPROF v1 table or its layer/expert counts do not match the model
    llama_expert_predict(const std::string & path, int32_t n_layer, int32_t n_expert);
    ~llama_expert_predict();
    llama_expert_predict(const llama_expert_predict &) = delete;
    llama_expert_predict & operator=(const llama_expert_predict &) = delete;

    // forget the rolling token history at a turn boundary, so no n-gram spans
    // two prompts (the exporter splits its keys the same way)
    void on_turn_begin();

    // feed the input token of the decode step about to run and write the
    // predicted expert ids per layer into `out` (resized to n_layer and cleared).
    // Returns false when no stored order matched
    bool predict(int32_t token, std::vector<std::vector<int32_t>> & out);

    int32_t n_layer() const;
    int32_t n_enabled() const;
    int32_t max_order() const;
    int32_t top_m() const;

    // decode steps ahead the table was trained for (0 for a v1 table)
    int32_t lead() const;

    // decode steps fed and steps where a stored order matched
    uint64_t n_steps() const;
    uint64_t n_matched() const;

  private:
    std::unique_ptr<llama_expert_predict_impl> pimpl;
};
