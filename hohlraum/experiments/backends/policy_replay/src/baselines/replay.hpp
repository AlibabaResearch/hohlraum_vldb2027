#pragma once
#include "cache.hpp"
#include <algorithm>
#include <list>
#include <optional>
#include <unordered_map>

namespace hohlraum {

struct ReplayStats {
    uint64_t requests = 0, requested = 0, hits = 0, theoretical = 0, writes = 0;
};

class LRUSet {
    std::list<Block> order;
    std::unordered_map<Block, std::list<Block>::iterator> positions;

public:
    bool contains(Block block) const { return positions.count(block) != 0; }
    size_t size() const { return positions.size(); }
    Block oldest() const { return order.front(); }
    Block newest() const { return order.back(); }

    void erase(Block block) {
        auto it = positions.find(block);
        if (it == positions.end()) return;
        order.erase(it->second);
        positions.erase(it);
    }

    void touch(Block block) {
        erase(block);
        order.push_back(block);
        positions[block] = std::prev(order.end());
    }
};

class AccessCounts {
    struct Session {
        double last = 0.;
        unsigned qg = 0;
    };
    struct BlockVisits {
        unsigned max_groups = 0;
        std::unordered_map<SessionId, std::pair<unsigned, unsigned>> by_session;
    };
    std::unordered_map<Block, BlockVisits> visits;
    std::unordered_map<SessionId, Session> sessions;

public:
    void observe(const Request& request) {
        auto& session = sessions[request.session_id];
        if (session.qg == 0 || request.time - session.last >= 60.) ++session.qg;
        session.last = request.time;
        for (Block block : request.blocks) {
            auto& record = visits[block];
            auto& [last_qg, groups] = record.by_session[request.session_id];
            if (last_qg != session.qg) {
                last_qg = session.qg;
                record.max_groups = std::max(record.max_groups, ++groups);
            }
        }
    }

    unsigned max_groups(Block block) const { return visits.at(block).max_groups; }
};

inline ReplayStats replay_baseline(const std::string& path, size_t capacity,
                                   std::optional<double> minimum_groups) {
    EventReader events(path);
    Request request;
    ReplayStats stats;
    LRUSet dram;
    AccessCounts counts;
    std::unordered_set<Block> ever, ssd;
    while (events.next(request)) {
        ++stats.requests;
        bool prefix = true, theoretical_prefix = true;
        for (Block block : request.blocks) {
            ++stats.requested;
            prefix = prefix && (dram.contains(block) || ssd.count(block));
            theoretical_prefix = theoretical_prefix && ever.count(block);
            stats.hits += prefix;
            stats.theoretical += theoretical_prefix;
        }
        if (minimum_groups) counts.observe(request);
        for (Block block : request.blocks) {
            dram.touch(block);
            ever.insert(block);
        }
        while (dram.size() > capacity) {
            const Block victim = dram.oldest();
            dram.erase(victim);
            if (!minimum_groups || counts.max_groups(victim) >= *minimum_groups)
                stats.writes += ssd.insert(victim).second;
        }
    }
    return stats;
}

class LARC {
    std::unordered_set<Block> data;
    LRUSet ghost;
    const size_t capacity;
    double target;
    uint64_t writes_ = 0;

public:
    explicit LARC(size_t ghost_blocks)
        : capacity(std::max(size_t(10), ghost_blocks)), target(.1 * capacity) {}

    bool contains(Block block) const { return data.count(block) != 0; }
    uint64_t writes() const { return writes_; }

    void access(Block block) {
        if (contains(block)) {
            target = std::max(.1 * capacity, target - capacity / (capacity - target));
            return;
        }
        target = std::min(.9 * capacity, target + capacity / target);
        if (ghost.contains(block)) {
            ghost.erase(block);
            data.insert(block);
            ++writes_;
        } else {
            ghost.touch(block);
            if (ghost.size() > capacity) ghost.erase(ghost.oldest());
            if (ghost.size() > target) ghost.erase(ghost.newest());
        }
    }
};

inline ReplayStats replay_larc(const std::string& path, size_t capacity, size_t ghost_blocks) {
    EventReader events(path);
    Request request;
    ReplayStats stats;
    LRUSet dram;
    LARC ssd(ghost_blocks);
    std::unordered_set<Block> ever;
    while (events.next(request)) {
        ++stats.requests;
        bool prefix = true, theoretical_prefix = true;
        for (Block block : request.blocks) {
            ++stats.requested;
            const bool resident = dram.contains(block);
            prefix = prefix && (resident || ssd.contains(block));
            theoretical_prefix = theoretical_prefix && ever.count(block);
            stats.hits += prefix;
            stats.theoretical += theoretical_prefix;
        }
        for (Block block : request.blocks)
            if (!dram.contains(block)) ssd.access(block);
        for (Block block : request.blocks) {
            dram.touch(block);
            if (dram.size() > capacity) dram.erase(dram.oldest());
            ever.insert(block);
        }
    }
    stats.writes = ssd.writes();
    return stats;
}
} // namespace hohlraum
