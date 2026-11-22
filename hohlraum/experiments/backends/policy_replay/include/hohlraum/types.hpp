#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <vector>
#include "hohlraum/replay_lifecycle_history.hpp"

namespace hohlraum {
inline void require(bool condition) {
    if (!condition) throw std::logic_error("cache invariant violated");
}

constexpr BlockId kNoBlock = std::numeric_limits<BlockId>::max();

enum class Location { absent, serving, absorption };
enum class Ranking { joint, lru };

struct Config {
    size_t serving_blocks = 12112;
    size_t absorption_blocks = 5401;
    size_t ssd_blocks = 101725;
    double measure_end = 259200;
    double threshold = 0.7;
    double qg_gap_seconds = 60;
    Ranking ranking = Ranking::joint;
    uint64_t write_budget_blocks = std::numeric_limits<uint64_t>::max();
    // Explicit scope ablation; production excludes SSD-backed staging copies.
    bool allow_ssd_copies_in_absorption = false;
};
} // namespace hohlraum
