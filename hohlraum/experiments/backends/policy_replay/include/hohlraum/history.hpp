#pragma once

#include "types.hpp"
#include <array>

namespace hohlraum {
// The policy trace supplies one timestamp per complete request. Its observation
// is at that timestamp and completion is called after the atomic transition.
class ReplayHistory final : public History {
public:
    ReplayHistory(size_t blocks, size_t session_count, double threshold, double qg_gap)
        : History(blocks, session_count, threshold, qg_gap), positions_(session_count) {
        if (!branch_online())
            throw std::runtime_error("the replay requires branch priors");
    }

    void observe(const Request& request) {
        arrive(request.session, request.time);
        History::observe(request);
        observe_positions(request);
    }

    void finish(const Request& request) {
        complete(request.session, request.time);
    }

    double context_risk(BlockId block) const {
        double risk = 0.;
        for (const auto& owner : owners(block)) {
            const auto& session = sessions[owner.session];
            if (owner.request_sequence == session.request_sequence)
                risk = std::max(risk, online_length_risk(session.fold, session.length));
        }
        return risk;
    }

    double position_risk(BlockId block) const {
        double risk = 0.;
        const auto found = current_owners_.find(block);
        if (found == current_owners_.end()) return risk;
        for (const auto& owner : found->second) {
            const auto& count = position_counts_[owner.second];
            if (count.observed) risk = std::max(risk, double(count.lost) / count.observed);
        }
        return risk;
    }

private:
    struct PositionCount {
        uint64_t observed = 0;
        uint64_t lost = 0;
    };

    // Match SLI.RelativePositionRisk. The branch hashes are chained, so
    // after the first mismatching page the previous and current paths cannot
    // rejoin. Only current owners need page-level position metadata.
    static unsigned position_bucket(size_t distance) {
        if (distance == 0) return 0;
        if (distance < 4) return 1;
        if (distance < 16) return 2;
        return 3;
    }

    void observe_positions(const Request& request) {
        auto& previous = positions_[request.session];
        const auto& current = request.blocks;
        const auto same = std::mismatch(previous.begin(), previous.end(),
                                        current.begin(), current.end());
        const size_t shared = same.first - previous.begin();
        for (size_t i = 0; i < previous.size(); ++i) {
            const auto bucket = position_bucket(previous.size() - 1 - i);
            auto& count = position_counts_[bucket];
            ++count.observed;
            if (i >= shared) {
                ++count.lost;
                auto owners = current_owners_.find(previous[i]);
                require(owners != current_owners_.end());
                auto& entries = owners->second;
                const auto owner = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) {
                    return entry.first == request.session;
                });
                require(owner != entries.end());
                entries.erase(owner);
                if (entries.empty()) current_owners_.erase(owners);
            }
        }
        for (size_t i = 0; i < current.size(); ++i) {
            const auto bucket = position_bucket(current.size() - 1 - i);
            if (i >= shared) {
                current_owners_[current[i]].emplace_back(request.session, bucket);
            } else if (bucket != position_bucket(previous.size() - 1 - i)) {
                auto& entries = current_owners_.at(current[i]);
                const auto owner = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) {
                    return entry.first == request.session;
                });
                require(owner != entries.end());
                owner->second = bucket;
            }
        }
        previous = current;
    }

    std::vector<std::vector<BlockId>> positions_;
    std::unordered_map<BlockId, std::vector<std::pair<SessionId, unsigned>>> current_owners_;
    std::array<PositionCount, 4> position_counts_{};
};
} // namespace hohlraum
