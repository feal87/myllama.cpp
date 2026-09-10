#include "llama-disk-buft.h"

#include "ggml-backend-impl.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

static const char * disk_buft_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "Disk";
}

static size_t disk_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 4096;
}

static size_t disk_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static bool disk_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static void disk_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
}

static void * disk_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return nullptr;
}

static enum ggml_status disk_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void disk_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    GGML_UNUSED(value);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

// streamed weights are never copied through the buffer; the graph reads them
// from the disk-backed substitute tensors instead
static void disk_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
    GGML_ABORT("disk buffer: set_tensor is not supported");
}

static void disk_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
    GGML_ABORT("disk buffer: get_tensor is not supported");
}

static void disk_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static ggml_backend_buffer_t disk_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    static const struct ggml_backend_buffer_i iface = {
        /* .free_buffer   = */ disk_buffer_free_buffer,
        /* .get_base      = */ disk_buffer_get_base,
        /* .init_tensor   = */ disk_buffer_init_tensor,
        /* .memset_tensor = */ disk_buffer_memset_tensor,
        /* .set_tensor    = */ disk_buffer_set_tensor,
        /* .get_tensor    = */ disk_buffer_get_tensor,
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ nullptr,
        /* .clear         = */ disk_buffer_clear,
        /* .reset         = */ nullptr,
    };
    return ggml_backend_buffer_init(buft, iface, nullptr, size);
}

ggml_backend_buffer_type_t llama_disk_buft(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ disk_buft_get_name,
            /* .alloc_buffer   = */ disk_buft_alloc_buffer,
            /* .get_alignment  = */ disk_buft_get_alignment,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ disk_buft_get_alloc_size,
            /* .is_host        = */ disk_buft_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &buft;
}

bool llama_disk_buft_is(ggml_backend_buffer_type_t buft) {
    return buft == llama_disk_buft();
}

void * llama_disk_buft_reserve(size_t size) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE);
#else
    void * p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}
