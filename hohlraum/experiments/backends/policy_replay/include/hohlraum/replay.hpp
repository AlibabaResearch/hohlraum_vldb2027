#pragma once
#include <functional>
#include "endpoints.hpp"
#include "history.hpp"
#include <array>
#include <iomanip>
#include <ostream>
#include <queue>
#include <set>
#include <tuple>

namespace hohlraum {
struct Metrics {
    uint64_t refs = 0, requests = 0, dram_hits = 0, service_hits = 0;
    uint64_t global_hits = 0, theory_hits = 0, writes = 0;
    uint64_t accepted = 0, rejected = 0, first_writes = 0, rewrites = 0;
    uint64_t pending_read_overrides = 0;
    uint64_t ssd_evictions = 0, ssd_reads = 0, bypassed = 0;
    uint64_t dram_suffix_drops = 0, nonleaf_selections = 0, budget_blocked = 0;
    uint64_t absorption_pressure_events = 0;
    uint64_t admission_useful_accepted = 0, admission_useful_rejected = 0;
    uint64_t budget_fallbacks = 0;
    size_t ssd_peak = 0;
    double ssd_area = 0, absorption_area = 0, duplicate_area = 0;
};

class Replay {
public:
    // Optional measurement hooks; never alter cache decisions.
    std::function<void(double, BlockId, double)> on_terminal_candidate;
    std::function<void(double, double)> on_accepted_staging_wait;
    std::function<void(double, BlockId)> on_staging_entry;
    std::function<void(double, BlockId)> on_serving_departure;
    std::function<void(double, BlockId, double, bool)> on_staging_terminal;
    std::function<void(double, BlockId)> on_ssd_read;
    std::function<void(double, BlockId)> on_read_copy_release;
    Replay(uint32_t blocks, uint32_t sessions, Config config, std::vector<Model> models, bool multi_model);
    Replay(uint32_t blocks, uint32_t sessions, Config config, std::array<Model, 2> models);
    void process(const Request& request);
    void acquire_pending_reads(const std::vector<BlockId>& ids);
    void release_pending_reads(const std::vector<BlockId>& ids);
    bool has_pending_read(BlockId b) const { return pending_read_refs_.at(b)>0; }
    uint64_t pending_read_references() const { return pending_read_references_; }
    void set_region_capacities(size_t serving_blocks, size_t absorption_blocks);
    void set_prefetch_reserved(size_t blocks);
    void set_admission_threshold(double threshold);
    void set_joint_ranking(bool enabled);
    // Session clock events from the serving driver (arrival / completion).
    void session_arrive(SessionId session, double time) { history_.arrive(session, time); }
    void session_complete(SessionId session, double time) { history_.complete(session, time); }
    void run(const Trace& trace);
    void write_json(std::ostream& out) const;

    size_t capacity_block_count() const { return blocks_.size(); }
    uint64_t capacity_recency(BlockId b) const { return blocks_.at(b).recency; }
    const Metrics& metrics() const { return metrics_; }
    Location location(BlockId b) const { return blocks_.at(b).location; }
    bool has_ssd(BlockId b) const { return blocks_.at(b).ssd; }
    size_t absorption_size() const { return absorption_size_; }
    size_t serving_size() const { return serving_size_; }
    size_t ssd_size() const { return ssd_size_; }
    size_t serving_capacity() const { return config_.serving_blocks; }
    size_t absorption_capacity() const { return config_.absorption_blocks; }
    const History& history() const { return history_; }
    double now() const { return now_; }
    const std::vector<BlockId>& last_writes() const { return last_writes_; }
    double admission_threshold() const { return config_.threshold; }
    const std::array<uint64_t, 100>& admission_histogram() const { return admission_histogram_; }
    const std::array<uint64_t, 100>& useful_accepted_histogram() const { return useful_accepted_histogram_; }
    const std::array<uint64_t, 100>& useful_rejected_histogram() const { return useful_rejected_histogram_; }
    const std::vector<double>& dwell_access() const { return dwell_access_; }
    const std::vector<double>& dwell_accept() const { return dwell_accept_; }
    const std::vector<double>& dwell_reject() const { return dwell_reject_; }
    const std::vector<double>& dwell_suffix() const { return dwell_suffix_; }
    const std::vector<double>& dwell_resize() const { return dwell_resize_; }
    const std::vector<double>& dwell_bypass() const { return dwell_bypass_; }
    const std::vector<double>& dwell_budget() const { return dwell_budget_; }

private:
    void refresh_pending_recency(const std::vector<BlockId>& ids);
    struct Block {
        Location location = Location::absent;
        bool registered = false, seen = false, ssd = false, written_before = false;
        unsigned dram_children = 0, serving_children = 0, ssd_descendants = 0;
        uint64_t recency = 0, ssd_recency = 0;
        std::vector<BlockId> children;
    };
    using Order = std::pair<uint64_t, BlockId>;
    using Choice = std::tuple<bool, double, int64_t, BlockId>;
    Config config_;
    std::vector<BlockId> parent_;
    std::vector<Block> blocks_;
    History history_;
    EndpointIndex index_;
    Metrics metrics_;
    size_t serving_size_ = 0, absorption_size_ = 0, ssd_size_ = 0, duplicate_size_ = 0;
    size_t dram_blocks_ = 0, prefetch_reserved_ = 0;
    uint64_t clock_ = 0, ssd_clock_ = 0;
    double now_ = 0, last_time_ = 0;
    bool selecting_ = false, finished_ = false;
    bool fallback_active_ = false, fallback_pending_ = false;
    std::set<Order> serving_leaves_, absorption_leaves_, endpoints_, ssd_leaves_;
    std::set<BlockId> clean_absorption_;
    std::vector<std::set<BlockId>> session_endpoints_;
    std::vector<unsigned> endpoint_references_;
    std::set<SessionId> touched_sessions_;
    std::priority_queue<Choice> heap_;
    std::vector<uint64_t> read_bins_;
    std::vector<BlockId> last_writes_;
    std::array<uint64_t, 100> admission_histogram_{};
    // 1: latest evaluated candidate was accepted; -1: rejected; 0: none/resolved.
    std::vector<uint32_t> pending_read_refs_;
    uint64_t pending_read_references_ = 0;
    std::vector<int8_t> pending_admission_;
    std::vector<uint8_t> pending_admission_bin_;
    std::array<uint64_t, 100> useful_accepted_histogram_{};
    std::array<uint64_t, 100> useful_rejected_histogram_{};
    std::vector<double> absorption_enter_time_;
    std::vector<double> dwell_access_, dwell_accept_, dwell_reject_, dwell_suffix_;
    std::vector<double> dwell_resize_, dwell_bypass_, dwell_budget_;
    void record_dwell(BlockId b, std::vector<double>& sink);

    bool uses_head_tail() const { return config_.candidates == Candidates::head_tail; }
    Order order(BlockId b) const { return {blocks_[b].recency, b}; }
    Choice choice(BlockId b);
    void register_path(const Request& request);
    void remove_from_dram_queues(BlockId b);
    void add_to_dram_queues(BlockId b);
    void change_absorption(BlockId b, bool active);
    void refresh_endpoints(SessionId session);
    void rebuild_endpoints();
    void remove_ssd_leaf(BlockId b);
    void add_ssd_leaf(BlockId b);
    void touch_ssd(BlockId b);
    void release_absorption(BlockId b);
    void erase_ssd(BlockId b, bool measured);
    void persist(BlockId b, bool measured);
    void activate_budget_fallback();
    void discard_dram_suffix(BlockId prefix, bool measured);
    void exit_absorption(BlockId b, bool measured, bool bypass);
    void release_existing_copies(bool measured);
    void measure_access(const Request& request, bool measured);
    void promote_to_serving(const Request& request);
    void promote_absorption_for_resize();
    void shrink_serving();
    void shrink_absorption(bool measured);
    void accumulate_area(double end);
    std::array<double, 3> read_bandwidth(size_t seconds) const;
};
} // namespace hohlraum
