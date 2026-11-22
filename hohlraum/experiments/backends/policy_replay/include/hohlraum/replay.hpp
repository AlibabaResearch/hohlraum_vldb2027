#pragma once
#include "endpoints.hpp"
#include "history.hpp"
#include <set>

namespace hohlraum {
struct Metrics {
    uint64_t refs = 0, requests = 0, dram_hits = 0;
    uint64_t global_hits = 0, theory_hits = 0, writes = 0;
    uint64_t ssd_evictions = 0, ssd_reads = 0, dram_suffix_drops = 0;
    uint64_t budget_fallbacks = 0;
    uint64_t prefix_writes = 0, prefix_read_releases = 0, discard_on_entry = 0;
    uint64_t transition_writes = 0, transition_returns = 0;
    double duplicate_area = 0;
};

class Replay {
public:
    Replay(uint32_t blocks, uint32_t sessions, Config config);
    void process(const Request& request);

    const Metrics& metrics() const { return metrics_; }

private:
    struct Block {
        Location location = Location::absent;
        bool registered = false, seen = false, ssd = false;
        unsigned live_children = 0, serving_children = 0, ssd_descendants = 0, depth = 0;
        uint64_t recency = 0, ssd_recency = 0;
    };
    using Order = std::pair<uint64_t, BlockId>;
    Config config_;
    std::vector<BlockId> parent_;
    std::vector<Block> blocks_;
    ReplayHistory history_;
    EndpointIndex index_;
    Metrics metrics_;
    size_t serving_size_ = 0, absorption_size_ = 0, ssd_size_ = 0, duplicate_size_ = 0;
    uint64_t clock_ = 0, ssd_clock_ = 0;
    double now_ = 0;
    bool selecting_ = false;
    bool fallback_active_ = false;
    std::set<Order> serving_leaves_, ssd_leaves_;
    std::set<BlockId> staged_;
    std::map<BlockId, unsigned> head_references_, tail_references_;
    std::vector<EndpointIndex::Roles> session_endpoints_;
    std::set<SessionId> touched_sessions_;
    History::WaitMemo waits_;
    std::unordered_map<BlockId, History::JointScore> round_scores_;

    bool uses_head_tail() const { return !fallback_active_; }
    Order order(BlockId b) const { return {blocks_[b].recency, b}; }
    void register_path(const Request& request);
    void propagate_live(BlockId block, int delta);
    void remove_from_dram_queues(BlockId b);
    void add_to_dram_queues(BlockId b);
    void change_absorption(BlockId b, bool active);
    void refresh_endpoints(SessionId session);
    void rebuild_endpoints();
    void remove_ssd_leaf(BlockId b);
    void add_ssd_leaf(BlockId b);
    void touch_ssd(BlockId b);
    void release_absorption(BlockId b);
    void erase_ssd(BlockId b);
    void persist(BlockId b, bool transition = false);
    void activate_budget_fallback();
    void exit_absorption(BlockId b, double admission);
    bool persist_head(BlockId b);
    void release_read(BlockId b);
    void move_to_absorption(BlockId b);
    const History::JointScore& score(BlockId b);
    void measure_access(const Request& request);
    void promote_to_serving(const Request& request);
    void shrink_serving();
    void retire_stage();
};
} // namespace hohlraum
