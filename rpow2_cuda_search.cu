// CUDA nonce search helper for RPOW2.
//
// Searches for:
//   sha256(nonce_prefix || uint64_le(nonce))
// with at least difficulty_bits trailing zero bits in the 32-byte digest.
//
// Final result is printed to stdout as:
//   RESULT {"nonce":"...","hash":"...","hashes":...,"elapsed":...,"rate_mh":...}
//
// Progress is printed to stdout as:
//   progress hashes=... rate=... MH/s

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int MAX_PREFIX_BYTES = 47;  // prefix + 8 nonce must fit one SHA-256 block before padding.

__constant__ uint32_t K256[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

struct SearchResult {
    uint32_t found;
    uint64_t nonce;
    uint32_t state[8];
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32U - n));
}

__device__ __forceinline__ uint32_t choose32(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t majority32(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint8_t digest_byte_from_end(const uint32_t state[8], int offset_from_end) {
    const int word_from_end = offset_from_end >> 2;
    const int byte_from_word_end = offset_from_end & 3;
    const uint32_t word = state[7 - word_from_end];
    return static_cast<uint8_t>((word >> (8 * byte_from_word_end)) & 0xffU);
}

__device__ __forceinline__ bool has_trailing_zero_bits_gpu(const uint32_t state[8], int difficulty_bits) {
    const int full_zero_bytes = difficulty_bits >> 3;
    const int partial_bits = difficulty_bits & 7;

    for (int i = 0; i < full_zero_bytes; ++i) {
        if (digest_byte_from_end(state, i) != 0) {
            return false;
        }
    }

    if (partial_bits == 0) {
        return true;
    }

    const uint8_t byte = digest_byte_from_end(state, full_zero_bytes);
    return (byte & ((1U << partial_bits) - 1U)) == 0;
}

__device__ __forceinline__ void sha256_one_block(const uint8_t* prefix, int prefix_len, uint64_t nonce, uint32_t out[8]) {
    uint8_t block[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        block[i] = 0;
    }

    for (int i = 0; i < prefix_len; ++i) {
        block[i] = prefix[i];
    }

    for (int i = 0; i < 8; ++i) {
        block[prefix_len + i] = static_cast<uint8_t>((nonce >> (8 * i)) & 0xffULL);
    }

    const int msg_len = prefix_len + 8;
    block[msg_len] = 0x80U;
    const uint64_t bit_len = static_cast<uint64_t>(msg_len) * 8ULL;
    for (int i = 0; i < 8; ++i) {
        block[63 - i] = static_cast<uint8_t>((bit_len >> (8 * i)) & 0xffULL);
    }

    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int j = i * 4;
        w[i] = (static_cast<uint32_t>(block[j]) << 24) |
               (static_cast<uint32_t>(block[j + 1]) << 16) |
               (static_cast<uint32_t>(block[j + 2]) << 8) |
               static_cast<uint32_t>(block[j + 3]);
    }

    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = 0x6a09e667U;
    uint32_t b = 0xbb67ae85U;
    uint32_t c = 0x3c6ef372U;
    uint32_t d = 0xa54ff53aU;
    uint32_t e = 0x510e527fU;
    uint32_t f = 0x9b05688cU;
    uint32_t g = 0x1f83d9abU;
    uint32_t h = 0x5be0cd19U;

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint32_t temp1 = h + S1 + choose32(e, f, g) + K256[i] + w[i];
        const uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint32_t temp2 = S0 + majority32(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    out[0] = a + 0x6a09e667U;
    out[1] = b + 0xbb67ae85U;
    out[2] = c + 0x3c6ef372U;
    out[3] = d + 0xa54ff53aU;
    out[4] = e + 0x510e527fU;
    out[5] = f + 0x9b05688cU;
    out[6] = g + 0x1f83d9abU;
    out[7] = h + 0x5be0cd19U;
}

__global__ void search_kernel(
    const uint8_t* prefix,
    int prefix_len,
    int difficulty_bits,
    uint64_t start_nonce,
    uint64_t stride,
    uint32_t iterations,
    SearchResult* result
) {
    const uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + thread_id;
    uint32_t state[8];

    for (uint32_t i = 0; i < iterations; ++i) {
        if (result->found) {
            return;
        }

        sha256_one_block(prefix, prefix_len, nonce, state);
        if (has_trailing_zero_bits_gpu(state, difficulty_bits)) {
            if (atomicCAS(reinterpret_cast<unsigned int*>(&result->found), 0U, 1U) == 0U) {
                result->nonce = nonce;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    result->state[j] = state[j];
                }
            }
            return;
        }

        nonce += stride;
    }
}

static void cuda_check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << context << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

static uint8_t hex_value(char c) {
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(10 + c - 'a');
    if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(10 + c - 'A');
    throw std::runtime_error("invalid hex character");
}

static std::vector<uint8_t> parse_hex(const std::string& hex) {
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("nonce_prefix has odd hex length");
    }
    std::vector<uint8_t> bytes(hex.size() / 2);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>((hex_value(hex[i * 2]) << 4) | hex_value(hex[i * 2 + 1]));
    }
    return bytes;
}

static std::string digest_hex(const uint32_t state[8]) {
    static const char* alphabet = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 8; ++i) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            const uint8_t byte = static_cast<uint8_t>((state[i] >> shift) & 0xffU);
            out.push_back(alphabet[byte >> 4]);
            out.push_back(alphabet[byte & 0x0fU]);
        }
    }
    return out;
}

static int parse_int_arg(int argc, char** argv, const char* name, int default_value) {
    const std::string key = std::string("--") + name;
    for (int i = 3; i + 1 < argc; ++i) {
        if (argv[i] == key) {
            return std::atoi(argv[i + 1]);
        }
    }
    return default_value;
}

static uint64_t parse_u64_arg(int argc, char** argv, const char* name, uint64_t default_value) {
    const std::string key = std::string("--") + name;
    for (int i = 3; i + 1 < argc; ++i) {
        if (argv[i] == key) {
            return std::strtoull(argv[i + 1], nullptr, 10);
        }
    }
    return default_value;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: " << argv[0] << " <nonce_prefix_hex> <difficulty_bits> "
                      << "[--device 0] [--blocks 0] [--threads 256] [--batch-iters 256] "
                      << "[--start 0] [--progress-ms 2000]\n";
            return 2;
        }

        const std::string prefix_hex = argv[1];
        const int difficulty_bits = std::atoi(argv[2]);
        const int device = parse_int_arg(argc, argv, "device", 0);
        int blocks = parse_int_arg(argc, argv, "blocks", 0);
        const int threads = parse_int_arg(argc, argv, "threads", 256);
        const int batch_iters = parse_int_arg(argc, argv, "batch-iters", 256);
        const int progress_ms = parse_int_arg(argc, argv, "progress-ms", 2000);
        uint64_t start_nonce = parse_u64_arg(argc, argv, "start", 0);

        if (difficulty_bits < 0 || difficulty_bits > 256) {
            throw std::runtime_error("difficulty_bits must be between 0 and 256");
        }
        if (threads <= 0 || threads > 1024) {
            throw std::runtime_error("--threads must be between 1 and 1024");
        }
        if (batch_iters <= 0) {
            throw std::runtime_error("--batch-iters must be positive");
        }

        std::vector<uint8_t> prefix = parse_hex(prefix_hex);
        if (prefix.empty() || prefix.size() > MAX_PREFIX_BYTES) {
            throw std::runtime_error("nonce_prefix length unsupported");
        }

        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        if (blocks <= 0) {
            blocks = prop.multiProcessorCount * 8;
        }

        uint8_t* d_prefix = nullptr;
        SearchResult* d_result = nullptr;
        cuda_check(cudaMalloc(&d_prefix, prefix.size()), "cudaMalloc prefix");
        cuda_check(cudaMalloc(&d_result, sizeof(SearchResult)), "cudaMalloc result");
        cuda_check(cudaMemcpy(d_prefix, prefix.data(), prefix.size(), cudaMemcpyHostToDevice), "cudaMemcpy prefix");

        const uint64_t stride = static_cast<uint64_t>(blocks) * static_cast<uint64_t>(threads);
        const uint64_t nonces_per_launch = stride * static_cast<uint64_t>(batch_iters);
        uint64_t total_hashes = 0;
        SearchResult host_result{};
        auto started = std::chrono::steady_clock::now();
        auto last_progress = started;

        while (true) {
            cuda_check(cudaMemset(d_result, 0, sizeof(SearchResult)), "cudaMemset result");
            search_kernel<<<blocks, threads>>>(
                d_prefix,
                static_cast<int>(prefix.size()),
                difficulty_bits,
                start_nonce,
                stride,
                static_cast<uint32_t>(batch_iters),
                d_result
            );
            cuda_check(cudaGetLastError(), "search_kernel launch");
            cuda_check(cudaDeviceSynchronize(), "search_kernel sync");
            cuda_check(cudaMemcpy(&host_result, d_result, sizeof(SearchResult), cudaMemcpyDeviceToHost), "cudaMemcpy result");

            total_hashes += nonces_per_launch;
            if (host_result.found) {
                break;
            }

            start_nonce += nonces_per_launch;
            auto now = std::chrono::steady_clock::now();
            auto since_progress = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress).count();
            if (since_progress >= progress_ms) {
                const double elapsed = std::chrono::duration<double>(now - started).count();
                const double rate_mh = static_cast<double>(total_hashes) / elapsed / 1'000'000.0;
                std::cout << "progress hashes=" << total_hashes << " rate=" << rate_mh << " MH/s" << std::endl;
                last_progress = now;
            }
        }

        auto ended = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(ended - started).count();
        const double rate_mh = static_cast<double>(total_hashes) / elapsed / 1'000'000.0;
        const std::string hash = digest_hex(host_result.state);

        std::cout << "RESULT {"
                  << "\"nonce\":\"" << host_result.nonce << "\","
                  << "\"hash\":\"" << hash << "\","
                  << "\"hashes\":" << total_hashes << ","
                  << "\"elapsed\":" << elapsed << ","
                  << "\"rate_mh\":" << rate_mh
                  << "}" << std::endl;

        cudaFree(d_prefix);
        cudaFree(d_result);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << std::endl;
        return 1;
    }
}
