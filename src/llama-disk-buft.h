#pragma once

#include "ggml-backend.h"

// Dedicated buffer type for weights that are streamed from disk and are never
// resident: the MoE expert tensors and the PLE table. Tensors of this type do
// not own RAM and are not mapped; their buffer is a zero-size dummy so the
// loader can associate ownership without allocating. The graph substitutes the
// disk-backed tensors (llama_disk_stage staging / decode cache) for them, so
// their data pointer is never dereferenced.
ggml_backend_buffer_type_t llama_disk_buft(void);

bool llama_disk_buft_is(ggml_backend_buffer_type_t buft);

// Reserve `size` bytes of address space with no backing memory. The returned
// pointer is non-null but must never be dereferenced. Used to give streamed
// weight tensors a data pointer that keeps the scheduler from allocating them.
void * llama_disk_buft_reserve(size_t size);
