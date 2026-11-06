#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <vector>

namespace hohlraum {
inline void require(bool condition) {
    if (!condition) throw std::logic_error("cache invariant violated");
}

using BlockId = uint32_t;
using SessionId = uint32_t;
constexpr BlockId kNoBlock = std::numeric_limits<BlockId>::max();

enum class Location { absent, serving, absorption };
enum class Ranking { joint, lru };
enum class Candidates { head_tail, tail };

struct Request {
    double time = 0;
    uint32_t context_tokens = 0;
    SessionId session = 0;
    uint32_t fold = 0;
    std::vector<BlockId> blocks;
    // Prefix made usable by HBM/DRAM plus the one storage-prefetch attempt.
    size_t accessible_blocks = std::numeric_limits<size_t>::max();
};

struct Trace {
    uint32_t block_count = 0;
    uint32_t session_count = 0;
    std::vector<Request> requests;
};

struct Config {
    size_t serving_blocks = 12112;
    size_t absorption_blocks = 5401;
    size_t ssd_blocks = 101725;
    double measure_start = 172800;
    double measure_end = 259200;
    double threshold = 0.7;
    double qg_gap_seconds = 30;
    uint64_t kv_bytes_per_block = 75497472;
    Ranking ranking = Ranking::joint;
    Candidates candidates = Candidates::head_tail;
    uint64_t write_budget_blocks = std::numeric_limits<uint64_t>::max();
    // Explicit scope ablation; production excludes SSD-backed staging copies.
    bool allow_ssd_copies_in_absorption = false;
    bool eager_write = false;
    bool merge_regions_on_budget_fallback = false;
};
} // namespace hohlraum
