#pragma once

#include "cache.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace hohlraum {

struct CoveragePoint {
    unsigned percent;
    size_t blocks;
    uint64_t hits, previous_hits;
};

struct Coverage {
    uint64_t requests = 0, refs = 0, theory_hits = 0;
    std::vector<CoveragePoint> points;
};

class AccessOrder {
    std::vector<size_t> positions_;
    std::vector<size_t> occupancy_;
    size_t length_ = 0, distinct_ = 0;

    void update(size_t position, bool add) {
        for (; position < occupancy_.size(); position += position & -position) {
            if (add) ++occupancy_[position];
            else --occupancy_[position];
        }
    }

public:
    explicit AccessOrder(size_t references) : occupancy_(references + 1) {}

    size_t rank(Block block) const {
        if (block >= positions_.size() || !positions_[block]) return 0;
        size_t older = 0;
        for (size_t position = positions_[block] - 1; position; position &= position - 1)
            older += occupancy_[position];
        return distinct_ - older;
    }

    void touch(Block block) {
        if (block >= occupancy_.size() || length_ + 1 >= occupancy_.size())
            throw std::runtime_error("reference count or dense page IDs disagree with the replay stream");
        if (block >= positions_.size()) positions_.resize(block + 1);
        auto& position = positions_[block];
        if (position) update(position, false);
        else ++distinct_;
        position = ++length_;
        update(position, true);
    }
};

inline Coverage calibrate_dram(const std::string& path, size_t references) {
    if (!references || references == std::numeric_limits<size_t>::max())
        throw std::runtime_error("invalid reference count for DRAM calibration");
    AccessOrder access(references);
    EventReader events(path);
    Request request;
    Coverage result;
    std::vector<uint64_t> hits(1);
    while (events.next(request)) {
        ++result.requests;
        result.refs += request.blocks.size();
        size_t required = 0;
        for (Block block : request.blocks) {
            const size_t rank = access.rank(block);
            if (!rank) break;
            required = std::max(required, rank);
            if (required >= hits.size()) hits.resize(required + 1);
            ++hits[required];
            ++result.theory_hits;
        }
        // At zero staging, replay retires the oldest read leaf. Reverse
        // recency within one request puts every child behind its parent,
        // which makes eviction by stack rank identical to leaf eviction.
        for (auto it = request.blocks.rbegin(); it != request.blocks.rend(); ++it)
            access.touch(*it);
    }
    if (result.refs != references || !result.theory_hits)
        throw std::runtime_error("missing reference or reuse counts for DRAM calibration");
    const std::array<unsigned, 3> targets{80, 90, 99};
    uint64_t total = 0;
    for (size_t blocks = 1; blocks < hits.size() && result.points.size() < targets.size(); ++blocks) {
        const uint64_t previous = total;
        total += hits[blocks];
        while (result.points.size() < targets.size()) {
            const unsigned percent = targets[result.points.size()];
            const uint64_t target = result.theory_hits / 100 * percent
                + (result.theory_hits % 100 * percent + 99) / 100;
            if (total < target) break;
            result.points.push_back({percent, blocks, total, previous});
        }
    }
    if (result.points.size() != targets.size())
        throw std::runtime_error("DRAM coverage did not reach all calibration targets");
    return result;
}

} // namespace hohlraum
