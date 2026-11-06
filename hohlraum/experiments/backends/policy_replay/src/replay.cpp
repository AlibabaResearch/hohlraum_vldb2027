#include "hohlraum/replay.hpp"
#include "hohlraum/preference.hpp"
#include <cmath>
#include <stdexcept>

namespace hohlraum {
Replay::Replay(uint32_t blocks, uint32_t sessions, Config config, std::vector<Model> models, bool multi_model)
    : config_(config), parent_(blocks, kNoBlock), blocks_(blocks),
      history_(blocks, sessions, std::move(models), config.threshold, config.qg_gap_seconds),
      index_(parent_, sessions), session_endpoints_(sessions), endpoint_references_(blocks),
      dram_blocks_(config.serving_blocks + config.absorption_blocks), pending_read_refs_(blocks, 0), pending_admission_(blocks, 0), pending_admission_bin_(blocks, 0), absorption_enter_time_(blocks, -1) {
    (void)multi_model;
    if (!std::isfinite(config.threshold) || config.threshold < 0 || config.threshold > 1 ||
        !std::isfinite(config.measure_start) || config.measure_start < 0 || config.measure_end <= config.measure_start ||
        !std::isfinite(config.measure_end) || !std::isfinite(config.qg_gap_seconds) || config.qg_gap_seconds <= 0 || config.kv_bytes_per_block == 0)
        throw std::runtime_error("invalid replay configuration");
    read_bins_.resize(size_t(std::ceil(config.measure_end - config.measure_start)));
    if (config_.write_budget_blocks == 0) {
        fallback_pending_ = true;
        activate_budget_fallback();
    }
}

Replay::Replay(uint32_t blocks, uint32_t sessions, Config config, std::array<Model, 2> models)
    : Replay(blocks, sessions, std::move(config), std::vector<Model>{std::move(models[0]), std::move(models[1])}, true) {}

Replay::Choice Replay::choice(BlockId b) {
    const auto score = history_.score(b, now_);
    // One global residency order. Admission controls the exit action of a
    // selected candidate; it must not override residency in the rank key.
    return {false, -score.residency, 0, b};
}

void Replay::register_path(const Request& request) {
    BlockId previous = kNoBlock;
    for (auto b : request.blocks) {
        auto& block = blocks_.at(b);
        if (!block.registered) {
            block.registered = true;
            parent_[b] = previous;
            if (previous != kNoBlock) blocks_[previous].children.push_back(b);
        } else if (parent_[b] != previous) {
            throw std::runtime_error("conflicting prefix identity");
        }
        previous = b;
    }
}

void Replay::remove_from_dram_queues(BlockId b) {
    if (b == kNoBlock) return;
    const auto& block = blocks_[b];
    clean_absorption_.erase(b);
    if (block.location == Location::serving && block.serving_children == 0)
        require(serving_leaves_.erase(order(b)) == 1);
    if (block.location == Location::absorption && block.dram_children == 0)
        require(absorption_leaves_.erase(order(b)) == 1);
}

void Replay::add_to_dram_queues(BlockId b) {
    if (b == kNoBlock) return;
    const auto& block = blocks_[b];
    if (block.location == Location::absorption && block.ssd) clean_absorption_.insert(b);
    if (block.location == Location::serving && block.serving_children == 0)
        require(serving_leaves_.insert(order(b)).second);
    if (block.location == Location::absorption && block.dram_children == 0) {
        require(absorption_leaves_.insert(order(b)).second);
        if (selecting_ && !uses_head_tail()) heap_.push(choice(b));
    }
}

void Replay::change_absorption(BlockId b, bool active) {
    if (!uses_head_tail()) return;
    for (const auto& association : history_.associations.at(b))
        index_.set_active(association.session, b, active);
}

void Replay::refresh_endpoints(SessionId session) {
    touched_sessions_.insert(session);
    auto current = index_.endpoints(session);
    auto& previous = session_endpoints_[session];
    for (auto b : previous) {
        if (current.count(b)) continue;
        require(endpoint_references_[b]);
        if (--endpoint_references_[b] == 0) require(endpoints_.erase(order(b)) == 1);
    }
    for (auto b : current) {
        if (previous.count(b)) continue;
        if (endpoint_references_[b]++ == 0) {
            require(endpoints_.insert(order(b)).second);
            if (selecting_ && config_.ranking == Ranking::joint) heap_.push(choice(b));
        }
    }
    previous = std::move(current);
}

void Replay::rebuild_endpoints() {
    for (const auto& entry : endpoints_) endpoint_references_[entry.second] = 0;
    endpoints_.clear();
    for (auto session : touched_sessions_) session_endpoints_[session].clear();
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


void Replay::record_dwell(BlockId b, std::vector<double>& sink) {
    const double entered = absorption_enter_time_.at(b);
    if (entered >= 0) {
        sink.push_back(std::max(0.0, now_ - entered));
    }
    absorption_enter_time_[b] = -1;
}

void Replay::release_absorption(BlockId b) {
    auto& block = blocks_[b];
    require(block.location == Location::absorption);
    change_absorption(b, false);
    remove_from_dram_queues(b);
    if (block.ssd) { require(duplicate_size_); --duplicate_size_; }
    block.location = Location::absent;
    --absorption_size_;
    auto parent = parent_[b];
    if (parent != kNoBlock) {
        remove_from_dram_queues(parent);
        require(blocks_[parent].dram_children);
        --blocks_[parent].dram_children;
        add_to_dram_queues(parent);
    }
    if (uses_head_tail() && selecting_)
        for (const auto& a : history_.associations[b]) refresh_endpoints(a.session);
}

void Replay::erase_ssd(BlockId b, bool measured) {
    auto& block = blocks_[b];
    // An accepted write evicted before reuse remains an unused accepted candidate.
    if (pending_admission_[b] > 0) pending_admission_[b] = 0;
    require(block.ssd);
    remove_ssd_leaf(b);
    block.ssd = false;
    --ssd_size_;
    clean_absorption_.erase(b);
    if (block.location == Location::absorption) { require(duplicate_size_); --duplicate_size_; }
    if (measured) ++metrics_.ssd_evictions;
    for (auto parent = parent_[b]; parent != kNoBlock; parent = parent_[parent]) {
        remove_ssd_leaf(parent);
        require(blocks_[parent].ssd_descendants);
        --blocks_[parent].ssd_descendants;
        add_ssd_leaf(parent);
    }
}

void Replay::persist(BlockId b, bool measured) {
    auto& block = blocks_[b];
    require(!block.ssd);
    if (fallback_active_ || metrics_.writes >= config_.write_budget_blocks) {
        if (measured) ++metrics_.budget_blocked;
        fallback_pending_ = true;
        return;
    }
    block.ssd = true;
    ++ssd_size_;
    block.ssd_recency = ++ssd_clock_;
    add_ssd_leaf(b);
    for (auto parent = parent_[b]; parent != kNoBlock; parent = parent_[parent]) {
        remove_ssd_leaf(parent);
        ++blocks_[parent].ssd_descendants;
        add_ssd_leaf(parent);
    }
    if (measured) {
        ++metrics_.writes;
        if (block.written_before) ++metrics_.rewrites;
        else ++metrics_.first_writes;
    }
    block.written_before = true;
    last_writes_.push_back(b);
    if (metrics_.writes >= config_.write_budget_blocks) fallback_pending_ = true;
    while (ssd_size_ > config_.ssd_blocks) {
        require(!ssd_leaves_.empty());
        erase_ssd(ssd_leaves_.begin()->second, measured);
    }
}

void Replay::activate_budget_fallback() {
    if (fallback_active_) return;
    fallback_active_ = true;
    fallback_pending_ = false;
    if (metrics_.budget_fallbacks == 0) ++metrics_.budget_fallbacks;
    // Once no SSD writes remain, stop spending DRAM on an admission-sensitive
    // ranking. Keep the configured regions but use LRU order and bypass SSD writes.
    config_.ranking = Ranking::lru;
    config_.candidates = Candidates::tail;
    selecting_ = false;
    heap_ = std::priority_queue<Choice>();
    if (config_.merge_regions_on_budget_fallback) {
        // Dynamic allocation can reassign all DRAM to direct-hit LRU once
        // no SSD writes remain. Fixed-capacity comparisons retain their split.
        // This nested resize must not erase writes already committed by the
        // enclosing request: the RPC consumer still has to schedule their I/O.
        auto committed_writes = std::move(last_writes_);
        set_region_capacities(dram_blocks_, 0);
        last_writes_.insert(last_writes_.begin(), committed_writes.begin(), committed_writes.end());
        // Fragments with an absent prefix cannot be promoted.
        shrink_absorption(now_ >= config_.measure_start);
    }
}

void Replay::discard_dram_suffix(BlockId prefix, bool measured) {
    // Rejecting a prefix frees its dependent DRAM suffix, never its SSD copies.
    bool was_selecting = selecting_;
    selecting_ = false;
    std::set<SessionId> affected;
    std::vector<BlockId> pending = blocks_[prefix].children;
    while (!pending.empty()) {
        auto b = pending.back(); pending.pop_back();
        auto& block = blocks_[b];
        if (!block.seen) continue;
        pending.insert(pending.end(), block.children.begin(), block.children.end());
        if (block.location == Location::serving)
            throw std::runtime_error("serving suffix below rejected absorption prefix");
        if (block.location == Location::absorption) {
            record_dwell(b, dwell_suffix_);
            if (uses_head_tail())
                for (const auto& a : history_.associations[b]) affected.insert(a.session);
            release_absorption(b);
            if (measured) ++metrics_.dram_suffix_drops;
        }
    }
    selecting_ = was_selecting;
    if (uses_head_tail() && selecting_)
        for (auto session : affected) refresh_endpoints(session);
}

void Replay::refresh_pending_recency(const std::vector<BlockId>& ids) {
    std::vector<BlockId> resident;
    for (auto b : ids)
        if (blocks_.at(b).location != Location::absent) resident.push_back(b);
    // Preserve the relative order within a simultaneous touch, not block-ID order.
    std::sort(resident.begin(), resident.end(), [&](BlockId a, BlockId b) {
        return order(a) < order(b);
    });
    resident.erase(std::unique(resident.begin(), resident.end()), resident.end());
    for (auto b : resident) {
        remove_from_dram_queues(b);
        blocks_[b].recency = ++clock_;
        add_to_dram_queues(b);
    }
}

void Replay::acquire_pending_reads(const std::vector<BlockId>& ids) {
    for(auto b:ids) { ++pending_read_refs_.at(b); ++pending_read_references_; }
    // A queued reuse is an observed reference even before execution. In read-only
    // residency, refresh its existing DRAM copy without inserting or pinning KV.
    if (config_.absorption_blocks == 0) refresh_pending_recency(ids);
}
void Replay::release_pending_reads(const std::vector<BlockId>& ids) {
    for(auto b:ids) {
        if(!pending_read_refs_.at(b)) throw std::runtime_error("pending read reference underflow");
        --pending_read_refs_[b]; --pending_read_references_;
    }
}

void Replay::exit_absorption(BlockId b, bool measured, bool bypass) {
    const bool had_ssd = blocks_[b].ssd;
    const double entered = absorption_enter_time_.at(b);
    const double dwell = entered >= 0 ? std::max(0.0, now_ - entered) : 0.0;
    if (bypass || had_ssd) dwell_bypass_.push_back(dwell);
    else if (fallback_active_) dwell_budget_.push_back(dwell);
    absorption_enter_time_[b] = -1;
    release_absorption(b);
    if (measured && bypass) ++metrics_.bypassed;
    if (bypass || had_ssd) return;
    if (fallback_active_) { if (measured) ++metrics_.budget_blocked; return; }
    const auto decision = history_.score(b, now_);
    const double admission = decision.admission;
    if(on_terminal_candidate) on_terminal_candidate(now_,b,admission);
    if (measured) {
        const auto bin = std::min<size_t>(admission_histogram_.size() - 1,
                                          size_t(admission * admission_histogram_.size()));
        ++admission_histogram_[bin];
    }
    const bool pending_override = admission < config_.threshold && pending_read_refs_.at(b)>0;
    bool accept = admission >= config_.threshold || pending_override;
    if(on_staging_terminal) on_staging_terminal(now_,b,dwell,accept);
    // Reuse the wait already computed for ranking/admission; no new forecast.
    if(accept && on_accepted_staging_wait && !history_.joint.empty())
        on_accepted_staging_wait(now_, admission >= config_.threshold ? -decision.residency : std::numeric_limits<double>::infinity());
    if(measured && pending_override) ++metrics_.pending_read_overrides;
    (accept ? dwell_accept_ : dwell_reject_).push_back(dwell);
    if (measured) {
        if (accept) ++metrics_.accepted;
        else ++metrics_.rejected;
        pending_admission_[b] = accept ? 1 : -1;
        pending_admission_bin_[b] = static_cast<uint8_t>(std::min<size_t>(
            admission_histogram_.size() - 1, size_t(admission * admission_histogram_.size())));
    }
    if (accept) persist(b, measured);
    else discard_dram_suffix(b, measured);
}

void Replay::release_existing_copies(bool measured) {
    if (config_.allow_ssd_copies_in_absorption) return;
    while (!clean_absorption_.empty()) {
        auto b = *clean_absorption_.begin();
        require(blocks_[b].location == Location::absorption && blocks_[b].ssd);
        exit_absorption(b, measured, true);
    }
}

void Replay::measure_access(const Request& request, bool measured) {
    bool service_prefix = true, global_prefix = true, theoretical_prefix = true;
    bool service_gap = false;
    size_t position = 0;
    for (auto b : request.blocks) {
        if (measured && pending_admission_[b] != 0) {
            const auto bin = pending_admission_bin_[b];
            if (pending_admission_[b] > 0) {
                ++metrics_.admission_useful_accepted;
                ++useful_accepted_histogram_[bin];
            } else {
                ++metrics_.admission_useful_rejected;
                ++useful_rejected_histogram_[bin];
            }
            pending_admission_[b] = 0;
        }
        const auto& block = blocks_[b];
        if (block.location != Location::serving) service_gap = true;
        else if (service_gap) throw std::runtime_error("serving prefix hole");
        service_prefix = service_prefix && block.location == Location::serving;
        global_prefix = global_prefix && position < request.accessible_blocks;
        theoretical_prefix = theoretical_prefix && block.seen;
        if (global_prefix && block.location == Location::absent && block.ssd) {
            touch_ssd(b);
            if (measured) {
                ++metrics_.ssd_reads;
                if(on_ssd_read) on_ssd_read(now_,b);
                ++read_bins_.at(size_t(request.time-config_.measure_start));
            }
        }
        if (measured) {
            ++metrics_.refs;
            metrics_.dram_hits += global_prefix && block.location != Location::absent;
            metrics_.service_hits += service_prefix;
            metrics_.global_hits += global_prefix;
            metrics_.theory_hits += theoretical_prefix;
        }
        ++position;
    }
    if (measured) ++metrics_.requests;
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
                if (block.location == Location::absent) ++blocks_[parent].dram_children;
                add_to_dram_queues(parent);
            }
            if (block.location == Location::absorption) {
                record_dwell(b, dwell_access_);
                change_absorption(b, false);
                --absorption_size_;
                if (block.ssd) { require(duplicate_size_); --duplicate_size_; }
            }
            block.location = Location::serving;
            ++serving_size_;
        }
        block.seen = true;
        block.recency = ++clock_;
        add_to_dram_queues(b);
    }
}

void Replay::shrink_serving() {
    while (serving_size_ > config_.serving_blocks) {
        require(!serving_leaves_.empty());
        auto b = serving_leaves_.begin()->second;
        auto parent = parent_[b];
        remove_from_dram_queues(b);
        if (parent != kNoBlock) {
            require(blocks_[parent].location == Location::serving);
            remove_from_dram_queues(parent);
            require(blocks_[parent].serving_children);
            --blocks_[parent].serving_children;
            add_to_dram_queues(parent);
        }
        auto& block = blocks_[b];
        if (block.ssd && !config_.allow_ssd_copies_in_absorption && on_read_copy_release)
            on_read_copy_release(now_, b);
        if(on_serving_departure) on_serving_departure(now_,b);
        block.location = Location::absorption;
        absorption_enter_time_[b] = now_;
        if(on_staging_entry && !block.ssd) on_staging_entry(now_,b);
        --serving_size_; ++absorption_size_;
        change_absorption(b, true);
        if (block.ssd) ++duplicate_size_;
        add_to_dram_queues(b);
    }
}

void Replay::shrink_absorption(bool measured) {
    if (absorption_size_ <= config_.absorption_blocks) return;
    if (measured) ++metrics_.absorption_pressure_events;
    auto has_resident_suffix = [&](BlockId prefix) {
        if (blocks_[prefix].dram_children > 0) return true;
        // An SSD-backed intermediate may be absent from DRAM while deeper
        // descendants remain resident. Check the same subtree that rejection
        // would otherwise cascade through, not just immediate DRAM children.
        std::vector<BlockId> pending = blocks_[prefix].children;
        while (!pending.empty()) {
            auto b = pending.back(); pending.pop_back();
            if (blocks_[b].location != Location::absent) return true;
            pending.insert(pending.end(), blocks_[b].children.begin(),
                           blocks_[b].children.end());
        }
        return false;
    };
    if(!history_.joint.empty() && uses_head_tail()) {
        if(uses_head_tail())rebuild_endpoints();selecting_=true;
        while(absorption_size_>config_.absorption_blocks){
            std::vector<BlockId> candidates;for(const auto& e:endpoints_)candidates.push_back(e.second);
            if(candidates.empty())for(const auto& e:absorption_leaves_)candidates.push_back(e.second);
            if(candidates.empty())throw std::runtime_error("no joint candidate");
            std::unordered_map<BlockId,History::JointScore> scores;
            for(auto b:candidates) {
                auto j=history_.joint_score(b,now_);
                if(blocks_[b].ssd) j.wait=std::numeric_limits<double>::infinity();
                scores.emplace(b,j);
            }
            // A rejected non-leaf is only a ranking representative: keep it and
            // its suffix in DRAM, then choose another candidate. Apply this
            // before the joint-model fast path, not only in the legacy heap path.
            candidates.erase(std::remove_if(candidates.begin(),candidates.end(),
                [&](BlockId b) {
                    return !blocks_[b].ssd && scores.at(b).admission < config_.threshold &&
                           has_resident_suffix(b);
                }),candidates.end());
            if(candidates.empty()) {
                // Keep capacity enforcement live using actual leaves, whose
                // rejection cannot invalidate another resident block.
                for(const auto& e:absorption_leaves_) {
                    auto j=history_.joint_score(e.second,now_);
                    if(blocks_[e.second].ssd) j.wait=std::numeric_limits<double>::infinity();
                    if(!blocks_[e.second].ssd && j.admission < config_.threshold && has_resident_suffix(e.second))
                        continue;
                    candidates.push_back(e.second);
                    scores.emplace(e.second,j);
                }
            }
            if(candidates.empty())throw std::runtime_error("no eligible joint candidate");
            // Apply the age/context-change preference after prefix safety.
            // Within the highest-weight half, preserve the existing T ordering
            // and equal-T admission tie-break to isolate this ablation.
            // Ranking ablation shares candidate construction and prefix safety.
            const BlockId chosen=config_.ranking==Ranking::joint
                ? preferred_exit(candidates,scores)
                : *std::min_element(candidates.begin(),candidates.end(),
                    [&](BlockId a,BlockId b){return order(a)<order(b);});
            if(measured&&blocks_[chosen].dram_children>0)++metrics_.nonleaf_selections;
            exit_absorption(chosen,measured,false);release_existing_copies(measured);
        }
        selecting_=false;return;
    }
    heap_ = std::priority_queue<Choice>();
    if (uses_head_tail()) rebuild_endpoints();
    selecting_ = true;
    auto& eligible = (config_.ranking == Ranking::joint && uses_head_tail()) ? endpoints_ : absorption_leaves_;
    if (config_.ranking == Ranking::joint)
        for (const auto& entry : eligible) heap_.push(choice(entry.second));
    while (absorption_size_ > config_.absorption_blocks) {
        if (eligible.empty()) throw std::runtime_error("no absorption endpoint");
        auto b = eligible.begin()->second;
        if (config_.ranking == Ranking::joint) {
            while (!heap_.empty()) {
                auto candidate = std::get<3>(heap_.top());
                bool valid = blocks_[candidate].location == Location::absorption &&
                    (uses_head_tail() ? endpoint_references_[candidate] > 0 : blocks_[candidate].dram_children == 0);
                if (valid) break;
                heap_.pop();
            }
            if (heap_.empty()) {
                // Endpoint representatives may all become temporarily
                // ineligible after a resize. Fall back to a real absorption
                // leaf so capacity enforcement always makes progress.
                if (absorption_leaves_.empty())
                    throw std::runtime_error("no absorption leaf fallback");
                b = absorption_leaves_.begin()->second;
            } else {
                b = std::get<3>(heap_.top()); heap_.pop();
            }
        }
        // A non-leaf endpoint is only a representative for ranking.  If its
        // current admission decision is reject, retain the whole branch and
        // try another candidate; removing this representative would otherwise
        // invalidate and cascade-delete its DRAM suffix.
        if (config_.ranking == Ranking::joint && has_resident_suffix(b) &&
            history_.score(b, now_).admission < config_.threshold) {
            continue;
        }
        if (measured && blocks_[b].dram_children > 0) ++metrics_.nonleaf_selections;
        exit_absorption(b, measured, false);
        release_existing_copies(measured);
    }
    selecting_ = false;
}

void Replay::accumulate_area(double end) {
    double duration = std::max(0., end-std::max(last_time_, config_.measure_start));
    metrics_.ssd_area += ssd_size_*duration;
    metrics_.absorption_area += absorption_size_*duration;
    metrics_.duplicate_area += duplicate_size_*duration;
}

void Replay::promote_absorption_for_resize() {
    // Only the absorption frontier can become serving without creating a
    // serving-prefix hole. Preserve access recency: migration is not a hit.
    std::priority_queue<Choice> ranked;
    std::priority_queue<Order, std::vector<Order>, std::greater<Order>> lru;
    auto enqueue = [&](BlockId b) {
        if (blocks_[b].location != Location::absorption) return;
        if (config_.ranking == Ranking::joint) ranked.push(choice(b));
        else lru.push(order(b));
    };
    for (BlockId b = 0; b < blocks_.size(); ++b) {
        if (blocks_[b].location != Location::absorption) continue;
        const auto parent = parent_[b];
        if (parent == kNoBlock || blocks_[parent].location == Location::serving)
            enqueue(b);
    }
    while (absorption_size_ > config_.absorption_blocks &&
           serving_size_ < config_.serving_blocks) {
        BlockId selected = kNoBlock;
        if (config_.ranking == Ranking::joint && !ranked.empty()) {
            selected = std::get<3>(ranked.top());
            ranked.pop();
        } else if (config_.ranking == Ranking::lru && !lru.empty()) {
            selected = lru.top().second;
            lru.pop();
        }
        if (selected == kNoBlock)
            break; // Remaining candidates have an absent prefix.
        const auto parent = parent_[selected];
        if (parent != kNoBlock) remove_from_dram_queues(parent);
        remove_from_dram_queues(selected);
        record_dwell(selected, dwell_resize_);
        change_absorption(selected, false);
        auto& block = blocks_[selected];
        block.location = Location::serving;
        --absorption_size_;
        ++serving_size_;
        if (block.ssd) { require(duplicate_size_); --duplicate_size_; }
        if (parent != kNoBlock) {
            ++blocks_[parent].serving_children;
            add_to_dram_queues(parent);
        }
        add_to_dram_queues(selected);
        for (auto child : block.children) enqueue(child);
    }
}

void Replay::set_joint_ranking(bool enabled) {
    if (fallback_active_ && enabled) return;
    config_.ranking = enabled ? Ranking::joint : Ranking::lru;
    config_.candidates = Candidates::head_tail;
    selecting_ = false;
    heap_ = std::priority_queue<Choice>();
}

void Replay::set_admission_threshold(double threshold) {
    if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
        throw std::invalid_argument("invalid admission threshold");
    config_.threshold = threshold;
    history_.set_threshold(threshold);
}

void Replay::set_region_capacities(size_t serving_blocks, size_t absorption_blocks) {
    last_writes_.clear();
    if (serving_blocks + absorption_blocks != dram_blocks_)
        throw std::invalid_argument("region resize must preserve physical DRAM capacity");
    if (serving_blocks <= prefetch_reserved_)
        throw std::invalid_argument("prefetch reservation exhausts service region");
    const bool entering_read_only = config_.absorption_blocks != 0 && absorption_blocks == 0;
    config_.serving_blocks = serving_blocks - prefetch_reserved_;
    config_.absorption_blocks = absorption_blocks;
    // A zero-sized absorption region ends admission-guided residency.  Keep
    // SSD admission active, but restore ordinary LRU residency ordering.
    if (absorption_blocks == 0 && config_.ranking == Ranking::joint) {
        config_.ranking = Ranking::lru;
        config_.candidates = Candidates::tail;
        selecting_ = false;
        heap_ = std::priority_queue<Choice>();
    }
    if (entering_read_only) {
        std::vector<BlockId> queued;
        for (BlockId b = 0; b < blocks_.size(); ++b)
            if (pending_read_refs_[b]) queued.push_back(b);
        refresh_pending_recency(queued);
    }
    // Rebalance cached KV around externally staged prefetch pages. The latter
    // consume physical service-region blocks but are intentionally absent from
    // the policy index until their request is admitted.
    promote_absorption_for_resize();
    shrink_serving();
    release_existing_copies(now_ >= config_.measure_start);
    shrink_absorption(now_ >= config_.measure_start);
}

void Replay::set_prefetch_reserved(size_t blocks) {
    last_writes_.clear();
    const auto serving_target = config_.serving_blocks + prefetch_reserved_;
    if (blocks >= serving_target)
        throw std::invalid_argument("prefetch reservation exhausts service region");
    prefetch_reserved_ = blocks;
    config_.serving_blocks = serving_target - prefetch_reserved_;
    shrink_serving();
    release_existing_copies(now_ >= config_.measure_start);
    shrink_absorption(now_ >= config_.measure_start);
}

void Replay::process(const Request& request) {
    last_writes_.clear();
    if (finished_ || request.time < now_ || request.time >= config_.measure_end)
        throw std::runtime_error("request outside replay time/order");
    now_ = request.time;
    register_path(request);
    history_.observe(request);
    if (uses_head_tail()) index_.observe(request.session, request.blocks, [&](BlockId b) {
        return blocks_[b].location == Location::absorption;
    });
    if (request.time > config_.measure_start) accumulate_area(request.time);
    last_time_ = request.time;
    bool measured = request.time >= config_.measure_start;
    measure_access(request, measured);
    promote_to_serving(request);
    if (config_.eager_write && !fallback_active_) {
        for (auto b : request.blocks) {
            if (blocks_[b].ssd) continue;
            persist(b, measured);
            if (fallback_pending_) break;
        }
    }
    shrink_serving();
    release_existing_copies(measured);
    shrink_absorption(measured);
    if (!config_.allow_ssd_copies_in_absorption)
        require(duplicate_size_ == 0); // The all-KV scope ablation intentionally permits copies.
    if (fallback_pending_) activate_budget_fallback();
    require(serving_size_ <= config_.serving_blocks);
    require(absorption_size_ <= config_.absorption_blocks);
    require(ssd_size_ <= config_.ssd_blocks);
    if (measured) metrics_.ssd_peak = std::max(metrics_.ssd_peak, ssd_size_);
}

void Replay::run(const Trace& trace) {
    for (const auto& request : trace.requests) {
        if (request.time >= config_.measure_end) break;
        process(request);
    }
    accumulate_area(config_.measure_end);
    finished_ = true; // No artificial end-of-trace flush or death decisions.
}

std::array<double, 3> Replay::read_bandwidth(size_t seconds) const {
    std::vector<double> samples;
    for (size_t i = 0; i < read_bins_.size(); i += seconds) {
        uint64_t blocks = 0;
        for (size_t j = i; j < std::min(i+seconds, read_bins_.size()); ++j) blocks += read_bins_[j];
        samples.push_back(blocks*config_.kv_bytes_per_block/1e9/std::min(seconds, read_bins_.size()-i));
    }
    std::sort(samples.begin(), samples.end());
    return {samples[size_t(std::ceil(.95*samples.size()))-1],
            samples[size_t(std::ceil(.99*samples.size()))-1], samples.back()};
}

void Replay::write_json(std::ostream& out) const {
    out << std::setprecision(12) << "{";
    bool first = true;
    auto field = [&](const char* name, auto value) {
        if (!first) out << ',';
        first = false;
        out << '\"' << name << "\":" << value;
    };
    field("head_tail", uses_head_tail());
    field("service_capacity", config_.serving_blocks); field("absorption_capacity", config_.absorption_blocks);
    field("ssd_capacity", config_.ssd_blocks);
    out << ",\"policy\":\"" << (config_.ranking == Ranking::joint ? "joint" : "lru") << "\"";
    field("refs", metrics_.refs); field("requests", metrics_.requests);
    field("dram_hits", metrics_.dram_hits); field("service_hits", metrics_.service_hits);
    field("global_hits", metrics_.global_hits); field("theory_hits", metrics_.theory_hits);
    field("writes", metrics_.writes); field("accepted", metrics_.accepted); field("rejected", metrics_.rejected);
    field("admission_useful_accepted", metrics_.admission_useful_accepted);
    field("admission_useful_rejected", metrics_.admission_useful_rejected);
    field("write_budget_blocks", config_.write_budget_blocks); field("budget_blocked", metrics_.budget_blocked);
    field("budget_fallbacks", metrics_.budget_fallbacks);
    field("first_writes", metrics_.first_writes); field("rewrites", metrics_.rewrites);
    field("ssd_evictions", metrics_.ssd_evictions); field("ssd_reads", metrics_.ssd_reads);
    field("nonleaf_selections", metrics_.nonleaf_selections); field("bypassed", metrics_.bypassed);
    field("cascade_unpersisted_drops", metrics_.dram_suffix_drops);
    field("ssd_end_units", ssd_size_); field("ssd_peak_units", metrics_.ssd_peak);
    double duration = config_.measure_end-config_.measure_start;
    field("ssd_mean_units", metrics_.ssd_area/duration);
    field("mean_abs_units", metrics_.absorption_area/duration);
    field("mean_duplicate_units", metrics_.duplicate_area/duration);
    for (size_t seconds : {1, 10, 60}) {
        auto values = read_bandwidth(seconds);
        out << ",\"read_" << seconds << "s_GBps\":[" << values[0] << ',' << values[1] << ',' << values[2] << ']';
    }
    out << "}\n";
}
} // namespace hohlraum
