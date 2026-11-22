#include "hohlraum/replay.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hohlraum {
Replay::Replay(uint32_t blocks, uint32_t sessions, Config config)
    : config_(config), parent_(blocks, kNoBlock), blocks_(blocks),
      history_(blocks, sessions, config.threshold, config.qg_gap_seconds),
      index_(parent_, sessions), session_endpoints_(sessions) {
    if (!std::isfinite(config.threshold) || config.threshold < 0 || config.threshold > 1 ||
        !std::isfinite(config.measure_end) || config.measure_end <= 0 ||
        !std::isfinite(config.qg_gap_seconds) || config.qg_gap_seconds <= 0)
        throw std::runtime_error("invalid replay configuration");
    if (config_.serving_blocks == 0 || config_.ssd_blocks == 0)
        throw std::runtime_error("replay capacities must preserve read and storage");
    if (config_.write_budget_blocks == 0)
        activate_budget_fallback();
}

void Replay::register_path(const Request& request) {
    BlockId previous = kNoBlock;
    for (auto b : request.blocks) {
        auto& block = blocks_.at(b);
        if (!block.registered) {
            block.registered = true;
            parent_[b] = previous;
            if (previous != kNoBlock) block.depth = blocks_[previous].depth + 1;
        } else if (parent_[b] != previous) {
            throw std::runtime_error("conflicting prefix identity");
        }
        previous = b;
    }
}

void Replay::propagate_live(BlockId b, int delta) {
    // An ancestor tracks children whose subtrees have at least one DRAM page.
    while (b != kNoBlock) {
        auto& node = blocks_[b];
        const bool before = node.live_children || node.location != Location::absent;
        if (delta > 0) ++node.live_children;
        else { require(node.live_children > 0); --node.live_children; }
        const bool after = node.live_children || node.location != Location::absent;
        if (before == after) break;
        b = parent_[b];
    }
}

void Replay::remove_from_dram_queues(BlockId b) {
    if (b != kNoBlock && blocks_[b].location == Location::serving && !blocks_[b].serving_children)
        require(serving_leaves_.erase(order(b)) == 1);
}

void Replay::add_to_dram_queues(BlockId b) {
    if (b != kNoBlock && blocks_[b].location == Location::serving && !blocks_[b].serving_children)
        require(serving_leaves_.insert(order(b)).second);
}

void Replay::change_absorption(BlockId b, bool active) {
    if (!uses_head_tail()) return;
    for (const auto& owner : history_.owners(b)) {
        index_.set_active(owner.session, b, active);
        if (selecting_) refresh_endpoints(owner.session);
    }
}

void Replay::refresh_endpoints(SessionId session) {
    touched_sessions_.insert(session);
    auto current = index_.endpoints(session);
    auto& previous = session_endpoints_[session];
    auto update = [](auto& refs, const auto& before, const auto& after) {
        for (auto b : before) {
            if (after.count(b)) continue;
            auto it = refs.find(b);
            require(it != refs.end());
            if (--it->second == 0) refs.erase(it);
        }
        for (auto b : after)
            if (!before.count(b)) ++refs[b];
    };
    update(head_references_, previous.heads, current.heads);
    update(tail_references_, previous.tails, current.tails);
    previous = std::move(current);
}

void Replay::rebuild_endpoints() {
    head_references_.clear();
    tail_references_.clear();
    for (auto session : touched_sessions_) session_endpoints_[session] = {};
    touched_sessions_.clear();
    for (auto session : index_.active_sessions) refresh_endpoints(session);
}

void Replay::remove_ssd_leaf(BlockId b) {
    const auto& block = blocks_[b];
    if (block.ssd && block.ssd_descendants == 0)
        require(ssd_leaves_.erase({block.ssd_recency, b}) == 1);
}

void Replay::add_ssd_leaf(BlockId b) {
    const auto& block = blocks_[b];
    if (block.ssd && block.ssd_descendants == 0)
        require(ssd_leaves_.insert({block.ssd_recency, b}).second);
}

void Replay::touch_ssd(BlockId b) {
    remove_ssd_leaf(b);
    blocks_[b].ssd_recency = ++ssd_clock_;
    add_ssd_leaf(b);
}


void Replay::release_absorption(BlockId b) {
    auto& block = blocks_[b];
    require(block.location == Location::absorption && staged_.erase(b) == 1);
    change_absorption(b, false);
    if (block.ssd) { require(duplicate_size_); --duplicate_size_; }
    block.location = Location::absent;
    --absorption_size_;
    if (!block.live_children) propagate_live(parent_[b], -1);
}

void Replay::erase_ssd(BlockId b) {
    auto& block = blocks_[b];
    require(block.ssd);
    remove_ssd_leaf(b);
    block.ssd = false;
    --ssd_size_;
    if (block.location == Location::absorption) { require(duplicate_size_); --duplicate_size_; }
    ++metrics_.ssd_evictions;
    for (auto parent = parent_[b]; parent != kNoBlock; parent = parent_[parent]) {
        remove_ssd_leaf(parent);
        require(blocks_[parent].ssd_descendants);
        --blocks_[parent].ssd_descendants;
        add_ssd_leaf(parent);
    }
}

void Replay::persist(BlockId b, bool transition) {
    auto& block = blocks_[b];
    require(!block.ssd);
    require(transition || (!fallback_active_ && metrics_.writes < config_.write_budget_blocks));
    block.ssd = true;
    ++ssd_size_;
    if (block.location == Location::absorption) ++duplicate_size_;
    block.ssd_recency = ++ssd_clock_;
    add_ssd_leaf(b);
    for (auto parent = parent_[b]; parent != kNoBlock; parent = parent_[parent]) {
        remove_ssd_leaf(parent);
        ++blocks_[parent].ssd_descendants;
        add_ssd_leaf(parent);
    }
    if (transition) ++metrics_.transition_writes;
    else ++metrics_.writes;
    while (ssd_size_ > config_.ssd_blocks) {
        require(!ssd_leaves_.empty());
        erase_ssd(ssd_leaves_.begin()->second);
    }
}

void Replay::activate_budget_fallback() {
    if (fallback_active_) return;
    fallback_active_ = true;
    selecting_ = false;
    ++metrics_.budget_fallbacks;
    config_.serving_blocks += config_.absorption_blocks;
    config_.absorption_blocks = 0;

    // In this atomic replay there are no outstanding DMA pins. Restore
    // parents before children, without recording a cache access.
    std::vector<BlockId> waiting(staged_.begin(), staged_.end());
    std::sort(waiting.begin(), waiting.end(), [&](BlockId a, BlockId b) {
        return std::pair{blocks_[a].depth,a} < std::pair{blocks_[b].depth,b};
    });
    for (auto b : waiting) {
        const auto parent = parent_[b];
        if (parent != kNoBlock && blocks_[parent].location != Location::serving) {
            if (!blocks_[b].ssd) persist(b, true);
            release_absorption(b);
            continue;
        }
        auto& block = blocks_[b];
        require(staged_.erase(b) == 1);
        if (block.ssd) { require(duplicate_size_); --duplicate_size_; }
        remove_from_dram_queues(parent);
        if (parent != kNoBlock) ++blocks_[parent].serving_children;
        block.location = Location::serving;
        add_to_dram_queues(parent);
        add_to_dram_queues(b);
        --absorption_size_;
        ++serving_size_;
        ++metrics_.transition_returns;
    }
}

void Replay::release_read(BlockId b) {
    auto& block = blocks_[b];
    require(block.location == Location::serving && block.serving_children == 0);
    remove_from_dram_queues(b);
    const auto parent = parent_[b];
    if (parent != kNoBlock) {
        require(blocks_[parent].serving_children > 0);
        --blocks_[parent].serving_children;
        add_to_dram_queues(parent);
    }
    block.location = Location::absent;
    --serving_size_;
    if (!block.live_children) propagate_live(parent, -1);
}

void Replay::move_to_absorption(BlockId b) {
    auto& block = blocks_[b];
    require(block.location == Location::serving && block.serving_children == 0);
    remove_from_dram_queues(b);
    const auto parent = parent_[b];
    if (parent != kNoBlock) {
        require(blocks_[parent].serving_children > 0);
        --blocks_[parent].serving_children;
        add_to_dram_queues(parent);
    }
    block.location = Location::absorption;
    require(staged_.insert(b).second);
    --serving_size_;
    ++absorption_size_;
    if (block.ssd) ++duplicate_size_;
    change_absorption(b, true);
}

bool Replay::persist_head(BlockId b) {
    std::vector<BlockId> missing;
    for (auto parent = parent_[b]; parent != kNoBlock && !blocks_[parent].ssd;
         parent = parent_[parent]) {
        if (blocks_[parent].location != Location::serving)
            throw std::runtime_error("staging head lacks a read or SSD ancestor");
        missing.push_back(parent);
    }
    if (missing.size() >= config_.write_budget_blocks - metrics_.writes) {
        activate_budget_fallback();
        return false;
    }
    for (auto it = missing.rbegin(); it != missing.rend(); ++it) {
        // These writes are obligations of the selected page, not a second
        // independent admission offer.
        persist(*it);
        ++metrics_.prefix_writes;
    }
    persist(b);
    if (config_.absorption_blocks && metrics_.writes < config_.write_budget_blocks) {
        for (auto prefix : missing) {
            if (blocks_[prefix].ssd && !blocks_[prefix].serving_children) {
                release_read(prefix);
                ++metrics_.prefix_read_releases;
            }
        }
    }
    return true;
}

void Replay::exit_absorption(BlockId b, double admission) {
    if (!blocks_[b].ssd) {
        if (admission >= config_.threshold) {
            require(head_references_.count(b));
            if (!persist_head(b)) return;
        } else {
            require(tail_references_.count(b) && blocks_[b].live_children == 0);
            ++metrics_.dram_suffix_drops;
        }
    }
    release_absorption(b);
    if (metrics_.writes == config_.write_budget_blocks) activate_budget_fallback();
}

void Replay::measure_access(const Request& request) {
    bool service_gap = false, global_prefix = true, theoretical_prefix = true;
    for (auto b : request.blocks) {
        const auto& block = blocks_[b];
        if (block.location != Location::serving) service_gap = true;
        else if (service_gap) throw std::runtime_error("serving prefix hole");
        global_prefix = global_prefix && (block.location != Location::absent || block.ssd);
        theoretical_prefix = theoretical_prefix && block.seen;
        if (global_prefix && block.location == Location::absent) {
            touch_ssd(b);
            ++metrics_.ssd_reads;
        }
        ++metrics_.refs;
        metrics_.dram_hits += global_prefix && block.location != Location::absent;
        metrics_.global_hits += global_prefix;
        metrics_.theory_hits += theoretical_prefix;
    }
    ++metrics_.requests;
}

void Replay::promote_to_serving(const Request& request) {
    for (auto b : request.blocks) {
        auto& block = blocks_[b];
        auto parent = parent_[b];
        remove_from_dram_queues(b);
        if (block.location != Location::serving) {
            if (parent != kNoBlock) {
                require(blocks_[parent].location == Location::serving);
                remove_from_dram_queues(parent);
                ++blocks_[parent].serving_children;
                add_to_dram_queues(parent);
            }
            if (block.location == Location::absorption) {
                require(staged_.erase(b) == 1);
                change_absorption(b, false);
                --absorption_size_;
                if (block.ssd) { require(duplicate_size_); --duplicate_size_; }
                block.location = Location::serving;
            } else {
                block.location = Location::serving;
                if (!block.live_children) propagate_live(parent, 1);
            }
            ++serving_size_;
        }
        block.seen = true;
        block.recency = ++clock_;
        add_to_dram_queues(b);
    }
}

const History::JointScore& Replay::score(BlockId b) {
    const auto result = round_scores_.try_emplace(b);
    if (result.second)
        result.first->second = history_.joint_score(b, now_, &waits_);
    return result.first->second;
}

void Replay::retire_stage() {
    struct Candidate {
        BlockId block;
        double score, wait, risk, weight;
        bool write;
    };
    std::vector<Candidate> waiting, discardable;
    auto consider = [&](BlockId b, bool head, bool tail) {
        // The all-KV comparison retains the same shortlist and endpoint
        // constraints. An existing copy has no finite wait to avoid a write.
        const auto& profile = score(b);
        const bool write = profile.admission >= config_.threshold;
        if ((write && !head) || (!write && (!tail || blocks_[b].live_children)))
            return;
        const double context = history_.context_risk(b);
        const double risk = context + history_.position_risk(b);
        const double weight = risk >= 1. ? 0. : profile.weight * (1. - risk) / (1. - context);
        const double wait = blocks_[b].ssd ? std::numeric_limits<double>::infinity() : profile.wait;
        (write ? waiting : discardable).push_back(
            {b, profile.admission, wait, risk, weight, write});
    };
    for (const auto& [b, count] : head_references_)
        consider(b, true, tail_references_.count(b) != 0);
    for (const auto& [b, count] : tail_references_)
        if (!head_references_.count(b)) consider(b, false, true);
    if (waiting.empty() && discardable.empty())
        throw std::runtime_error("no legal staging departure: " + std::to_string(now_));

    const auto priority = [](const Candidate& c) {
        if (std::isinf(c.wait))
            return std::tuple{c.wait, int(!c.write), c.write ? -c.risk : -c.score, c.weight, c.block};
        return std::tuple{c.wait, 0, -c.score, 0., c.block};
    };
    if (config_.ranking == Ranking::joint) {
        std::sort(waiting.begin(), waiting.end(), [](const Candidate& a, const Candidate& b) {
            return std::pair{-a.weight,a.block} < std::pair{-b.weight,b.block};
        });
        waiting.resize((waiting.size() + 1) / 2);
    }
    discardable.insert(discardable.end(), waiting.begin(), waiting.end());
    const auto selected = std::max_element(discardable.begin(), discardable.end(),
        [&](const Candidate& a, const Candidate& b) {
            if (config_.ranking == Ranking::lru) return order(a.block) > order(b.block);
            return priority(a) < priority(b);
        });
    exit_absorption(selected->block, selected->score);
}

void Replay::shrink_serving() {
    if (serving_size_ <= config_.serving_blocks) return;
    if (uses_head_tail()) {
        rebuild_endpoints();
        selecting_ = true;
    }
    waits_.clear();
    round_scores_.clear();
    while (serving_size_ > config_.serving_blocks) {
        require(!serving_leaves_.empty());
        const auto b = serving_leaves_.begin()->second;
        const auto& block = blocks_[b];
        if (fallback_active_ || (block.ssd && !config_.allow_ssd_copies_in_absorption)) {
            release_read(b);
            continue;
        }
        if (!block.ssd && !block.live_children) {
            const auto admission = score(b).admission;
            if (admission < config_.threshold) {
                ++metrics_.discard_on_entry;
                ++metrics_.dram_suffix_drops;
                release_read(b);
                continue;
            }
        }
        if (absorption_size_ >= std::max<size_t>(1, config_.absorption_blocks)) {
            retire_stage();
            // Copying eligible read ancestors may have satisfied the original
            // reclamation request or changed its next LRU victim.
            continue;
        }
        move_to_absorption(b);
        if (config_.absorption_blocks == 0) retire_stage();
    }
    selecting_ = false;
}

void Replay::process(const Request& request) {
    if (request.time < now_ || request.time >= config_.measure_end)
        throw std::runtime_error("request outside replay time/order");
    metrics_.duplicate_area += duplicate_size_ * (request.time - now_);
    now_ = request.time;
    register_path(request);
    history_.observe(request);
    if (uses_head_tail()) index_.observe(request.session, request.blocks, [&](BlockId b) {
        return blocks_[b].location == Location::absorption;
    });
    measure_access(request);
    promote_to_serving(request);
    shrink_serving();
    history_.finish(request);
    if (!config_.allow_ssd_copies_in_absorption)
        require(duplicate_size_ == 0); // The all-KV scope ablation intentionally permits copies.
    require(serving_size_ <= config_.serving_blocks);
    require(absorption_size_ <= config_.absorption_blocks);
    require(ssd_size_ <= config_.ssd_blocks);
}

} // namespace hohlraum
