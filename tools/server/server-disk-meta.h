#pragma once

// Shared primitives for the versioned metadata sidecar written next to a data
// file. The slot checkpoint store writes a small <data>.meta record that carries
// a settings key, so the format read/write and the atomic commit live here.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <type_traits>

template <typename T>
bool server_disk_meta_write(std::ostream & output, const T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "disk metadata values must be trivially copyable");
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return output.good();
}

template <typename T>
bool server_disk_meta_read(std::istream & input, T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "disk metadata values must be trivially copyable");
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    return input.good();
}

inline bool server_disk_u64_add(uint64_t & value, uint64_t addend) {
    if (addend > std::numeric_limits<uint64_t>::max() - value) {
        return false;
    }
    value += addend;
    return true;
}

inline bool server_disk_u64_mul(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline bool server_disk_range_valid(uint64_t offset, uint64_t size, uint64_t total) {
    return offset <= total && size <= total - offset;
}

// Write meta_path atomically: body() fills a temporary file that replaces the
// target only when it reports success and the stream is still good.
template <typename Fn>
bool server_disk_meta_commit(const std::string & meta_path, Fn && body) {
    const std::string temporary_path = meta_path + ".tmp";

    std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return false;
    }

    bool ok = body(output);
    output.close();
    ok = ok && !output.fail();

    std::error_code ec;
    if (!ok) {
        std::filesystem::remove(temporary_path, ec);
        return false;
    }

    std::filesystem::rename(temporary_path, meta_path, ec);
    if (ec) {
        std::error_code ec2;
        std::filesystem::remove(temporary_path, ec2);
        return false;
    }

    return true;
}
