#pragma once
#include "cache.hpp"
#include "models.hpp"

// replay.hpp
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <random>
#include <set>
#include <limits>

namespace hohlraum {

// 块写入后立即可见；默认无限容量，也可显式配置有限容量 LRU。
// 不模拟延迟或异步 I/O。无限容量保持原有首次持久化语义。
class SimulatedStorage {
    TierQueue blocks;
    size_t capacity;
    uint64_t access_clock=0;
public:
    explicit SimulatedStorage(size_t cap=std::numeric_limits<size_t>::max()):capacity(cap){}
    // 返回是否新建当前副本；有限 SSD 淘汰后的重新写入也计一次写。
    bool put(Block block) {
        if(!capacity||contains(block))return false;
        blocks.put(block,{++access_clock,0});
        while(blocks.size()>capacity)blocks.erase(blocks.oldest());
        return true;
    }
    void touch(Block block) {if(contains(block))blocks.put(block,{++access_clock,0});}
    bool contains(Block block) const { return blocks.contains(block); }
    size_t size() const { return blocks.size(); }
};

struct ReplayConfig {
    size_t serving=0, absorption=0;
    bool exclusive=false, learned_admission=false, learned_rank=false;
    double threshold=.7;
    // 基线准入模式：random=按概率写入，accesses=引用组数达到门槛才写。
    std::string admission_mode="all"; double admission_param=0;
    size_t ssd_capacity=std::numeric_limits<size_t>::max();
    bool temporal_rank=false;
    bool rule_rank=false, path_rank=false, path_admission=false, session_rank=false;
    // Session admission and session residency share the same empirical QG curve.
    bool session_admission=false;
    unsigned path_mask=7;
};
struct ReplayStats {
    uint64_t requests=0, requested=0, hits=0, dram_hits=0, serving_hits=0, theoretical=0;
    uint64_t dram_served_hits=0, ssd_served_hits=0;
    uint64_t writes=0, rejected=0, evictions=0, flushed=0, backed_bypass=0;
    uint64_t temporal_accept_finite=0, temporal_accept_infinite=0;
    uint64_t temporal_reject_finite=0, temporal_reject_infinite=0;
    std::vector<double> residence;
};
struct SessionState {
    bool initialized=false;
    double last=0, start=0, growth=0;
    unsigned q=0;
    CompactionId compaction_id=0;
    uint64_t length=0;
    SessionPrefixTree prefix_tree;
    SessionPrefixTree::NodeId current_prefix=SessionPrefixTree::root_id;
    // 该 session 在 Absorption 中可逐出的 block 链；tree 只负责路径索引。
    std::list<Block> absorption_chain;
    std::unordered_map<Block,std::list<Block>::iterator> absorption_positions;
    uint64_t absorption_version=0;
};
class Replay {
public:
    // Read-only diagnostic observer: called after choosing the victim; never returns a score.
    using ReleaseObserver = std::function<void(double,const std::vector<std::pair<uint64_t,double>>&)>;
private:
    ReleaseObserver release_observer;
    ReplayConfig cfg;
    const AdmissionModel* admission_model;
    const RetentionModel* retention_model;
    const TemporalModel* temporal_model;
    const PathTimeModel* path_model;
    uint64_t clock=0;
    std::mt19937_64 rng=std::mt19937_64(0x9E3779B97F4A7C15ull);
    std::uniform_real_distribution<double> uniform{0.,1.};
    std::unordered_map<SessionId,SessionState> sessions;
    // 全局 block 对象保存所有关联 session 的路径与 Query-Group 状态。
    BlockMap block_map;
    TierQueue serving, absorption;
    SimulatedStorage ssd;
    std::unordered_set<Block> ever;
    ReplayStats stats;
    Branch state(const SessionState& session,const SessionMembership& membership)const {
        if(membership.current_path)return Branch::current;
        if(membership.compaction_id!=session.compaction_id)return Branch::displaced_by_compaction;
        return Branch::old_suffix;
    }
    double session_claim_score(const SessionState& session,const SessionMembership& membership,
                               double now)const {
        const auto branch=state(session,membership);
        if(branch!=Branch::current&&std::max(0.,now-membership.old_since)>=120)return 0;
        return temporal_model->admission_score(session.q,std::max(0.,now-session.last),cfg.threshold);
    }
    struct ScoreCache {
        std::unordered_map<SessionId,std::array<std::pair<double,double>,8>> sessions;
        std::unordered_map<Block,Candidate> blocks;
    };
    enum class Endpoint { head, tail };
    struct SessionVictim {
        Candidate block;
        SessionId session_id;
        uint64_t version;
        Endpoint endpoint;
    };

    void link_absorption(Block block) {
        for(const auto& [session_id,membership]:block_map.at(block).sessions) {
            (void)membership;
            auto& session=sessions.at(session_id);
            if(session.absorption_positions.count(block))continue;
            session.absorption_chain.push_back(block);
            session.absorption_positions.emplace(block,std::prev(session.absorption_chain.end()));
            session.absorption_version++;
        }
    }
    std::vector<SessionId> unlink_absorption(Block block) {
        std::vector<SessionId> affected;
        for(const auto& [session_id,membership]:block_map.at(block).sessions) {
            (void)membership;
            auto& session=sessions.at(session_id);
            const auto found=session.absorption_positions.find(block);
            if(found==session.absorption_positions.end())continue;
            session.absorption_chain.erase(found->second);
            session.absorption_positions.erase(found);
            session.absorption_version++;
            affected.push_back(session_id);
        }
        return affected;
    }
    void enter_absorption(Block block,CachedBlock entry) {
        absorption.put(block,entry);
        link_absorption(block);
    }

    // 学习策略只评分一个 session 及其链头 block；不会聚合或扫描全部 block。
    Candidate session_candidate(SessionId session_id,Block block,double now,ScoreCache& cache)const {
        const auto& session=sessions.at(session_id);
        const auto& membership=block_map.at(block).sessions.at(session_id);
        if(cfg.session_rank) {
            const auto cached=cache.blocks.find(block);
            if(cached!=cache.blocks.end())return cached->second;
            double decision_score=0,block_wait=0;bool any_current=false,any_old_suffix=false,all_old_mature=true;
            for(const auto& [claim_id,claim]:block_map.at(block).sessions) {
                const auto& owner=sessions.at(claim_id);const auto branch=state(owner,claim);
                any_current=any_current||branch==Branch::current;any_old_suffix=any_old_suffix||branch==Branch::old_suffix;
                const double silence=std::max(0.,now-owner.last);
                const double actual=session_claim_score(owner,claim,now);
                decision_score=std::max(decision_score,actual);
                double wait=temporal_model->wait(write(actual,cfg.threshold),owner.q,silence,cfg.threshold);
                if(branch!=Branch::current) {
                    const double age=std::max(0.,now-claim.old_since);
                    all_old_mature=all_old_mature&&age>=120;
                    if(age<120)wait=std::min(wait,120-age);
                } else all_old_mature=false;
                block_wait=std::max(block_wait,wait);
            }
            if(all_old_mature)block_wait=std::numeric_limits<double>::infinity();
            const Branch aggregate=any_current?Branch::current:any_old_suffix?Branch::old_suffix:Branch::displaced_by_compaction;
            Candidate result{decision_score,block_wait,double(absorption.at(block).recency),block,aggregate};
            cache.blocks.emplace(block,result);return result;
        }
        if(cfg.path_rank) {
            const auto cached=cache.blocks.find(block);
            if(cached!=cache.blocks.end())return cached->second;
            double decision_score=0,block_wait=0;
            for(const auto& [claim_id,claim]:block_map.at(block).sessions) {
                const auto& owner=sessions.at(claim_id);
                const auto branch=state(owner,claim);
                const double silence=std::max(0.,now-owner.last);
                const double old_seconds=branch==Branch::current?0:std::max(0.,now-claim.old_since);
                const unsigned age=owner.q>claim.last_query_group?1:0;
                const double path_score=path_model->score(owner.q,owner.length,silence,branch,old_seconds);
                const double actual_score=cfg.path_admission?path_score:admission_model->predict(
                    silence,owner.q,owner.length,branch,claim.referenced_query_groups,age);
                decision_score=std::max(decision_score,actual_score);
                if(write(actual_score,cfg.threshold))
                    block_wait=std::max(block_wait,cfg.path_admission
                        ? path_model->rejection_wait(owner.q,owner.length,silence,branch,old_seconds,cfg.threshold)
                        : path_model->decision_wait(*admission_model,owner.q,owner.length,silence,branch,
                                                    claim.referenced_query_groups,age,old_seconds,cfg.threshold,cfg.path_mask));
            }
            Candidate result{decision_score,block_wait,double(absorption.at(block).recency),block};
            cache.blocks.emplace(block,result);
            return result;
        }
        if(cfg.rule_rank) {
            const auto cached=cache.blocks.find(block);
            if(cached!=cache.blocks.end())return cached->second;
            double admission_score=-1;
            unsigned dominant_q=1;
            double dominant_length=0,dominant_context_survival=1;
            bool any_current=false,any_old_suffix=false;
            auto block_score=[&](double extra) {
                double maximum=0;
                for(const auto& [claim_id,claim]:block_map.at(block).sessions) {
                    const auto& owner=sessions.at(claim_id);
                    const auto branch=state(owner,claim);
                    maximum=std::max(maximum,admission_model->predict(
                        std::max(0.,now-owner.last)+extra,owner.q,owner.length,branch,
                        claim.referenced_query_groups,owner.q>claim.last_query_group?1:0));
                }
                return maximum;
            };
            for(const auto& [claim_id,claim]:block_map.at(block).sessions) {
                const auto& owner=sessions.at(claim_id);
                const auto branch=state(owner,claim);
                any_current=any_current||branch==Branch::current;
                any_old_suffix=any_old_suffix||branch==Branch::old_suffix;
                const double score=admission_model->predict(std::max(0.,now-owner.last),owner.q,
                    owner.length,branch,claim.referenced_query_groups,
                    owner.q>claim.last_query_group?1:0);
                if(score>admission_score) {
                    admission_score=score;dominant_q=owner.q;dominant_length=owner.length;
                    dominant_context_survival=admission_model->length[owner.q>1].predict(owner.length);
                }
            }
            double wait=0;
            if(write(admission_score,cfg.threshold)) {
                double hi=1;constexpr double maximum=365.*24*3600;
                while(hi<maximum&&write(block_score(hi),cfg.threshold))hi*=2;
                hi=std::min(hi,maximum);
                if(write(block_score(hi),cfg.threshold))wait=std::numeric_limits<double>::infinity();
                else {
                    double lo=0;
                    for(unsigned i=0;i<64;++i) {
                        const double mid=(lo+hi)/2;
                        (write(block_score(mid),cfg.threshold)?lo:hi)=mid;
                    }
                    wait=hi;
                }
            }
            const Branch aggregate=any_current?Branch::current:
                any_old_suffix?Branch::old_suffix:Branch::displaced_by_compaction;
            Candidate result{admission_score,wait,double(absorption.at(block).recency),block,
                             aggregate,dominant_q,dominant_length,dominant_context_survival};
            cache.blocks.emplace(block,result);
            return result;
        }
        auto scores=cache.sessions.find(session_id);
        if(scores==cache.sessions.end()) {
            std::array<std::pair<double,double>,8> values;
            values.fill({-1,-1});
            const double silence=std::max(0.,now-session.last);
            const double growth=session.growth/std::max(1.,session.last-session.start);
            for(unsigned j=0;j<8;++j) {
                if(j>=2 && admission_model->branch.p[0][0]<0)continue;
                const Branch branch=j==0?Branch::current:j==1?Branch::displaced_by_compaction:Branch::old_suffix;
                const unsigned references=j<2?1:1+(j-2)/2;
                const unsigned age=j<2?0:(j-2)%2;
                const double admission_score=admission_model->predict(silence,session.q,session.length,branch,references,age);
                const double rank_value=cfg.rule_rank
                    ? temporal_model->rejection_wait(*admission_model,session.q,session.length,silence,branch,references,age,cfg.threshold)
                    : cfg.temporal_rank
                        ? temporal_model->wait(*admission_model,session.q,session.length,silence,branch,references,age,cfg.threshold)
                        : retention_model->predict(*admission_model,session.q,silence,session.length,growth,branch,references,age,cfg.threshold);
                values[j]={admission_score,rank_value};
            }
            scores=cache.sessions.emplace(session_id,values).first;
        }
        const auto branch=state(session,membership);
        const unsigned age=session.q>membership.last_query_group?1:0;
        const unsigned index=branch==Branch::current?0:branch==Branch::displaced_by_compaction?1:
            2+(std::min(3u,membership.referenced_query_groups)-1)*2+age;
        if(scores->second[index].first<0)throw std::runtime_error("old suffix candidate requires unavailable branch model");
        return {scores->second[index].first,scores->second[index].second,
            double(absorption.at(block).recency),block,branch,session.q,double(session.length)};
    }
    std::vector<SessionVictim> session_victims(SessionId session_id,double now,ScoreCache& cache)const {
        const auto& session=sessions.at(session_id);
        std::vector<SessionVictim> result;
        if(session.absorption_chain.empty())return result;
        // Only the tail participates, including a one-block chain.
        // Rejection releases the selected tail; there is no head protection.
        const Block tail=session.absorption_chain.back();
        result.push_back({session_candidate(session_id,tail,now,cache),session_id,
                          session.absorption_version,Endpoint::tail});
        return result;
    }
    bool admission_accept(Block block,double now) {
        if(ssd.contains(block))return true;
        if(cfg.admission_mode=="random")return uniform(rng)<cfg.admission_param;
        if(cfg.admission_mode=="accesses") {
            unsigned maximum=0;
            for(const auto& [session_id,membership]:block_map.at(block).sessions) {
                (void)session_id;
                maximum=std::max(maximum,membership.referenced_query_groups);
            }
            return maximum>=cfg.admission_param;
        }
        if(cfg.session_admission) {
            double score=0;
            for(const auto& [session_id,membership]:block_map.at(block).sessions)
                score=std::max(score,session_claim_score(sessions.at(session_id),membership,now));
            return write(score,cfg.threshold);
        }
        if(cfg.path_admission) {
            double score=0;
            for(const auto& [session_id,membership]:block_map.at(block).sessions) {
                const auto& session=sessions.at(session_id);
                const auto branch=state(session,membership);
                const double old_seconds=branch==Branch::current?0:std::max(0.,now-membership.old_since);
                score=std::max(score,path_model->score(session.q,session.length,std::max(0.,now-session.last),
                                                       branch,old_seconds));
            }
            return write(score,cfg.threshold);
        }
        if(cfg.learned_admission) {
            double probability=0;
            for(const auto& [session_id,membership]:block_map.at(block).sessions) {
                const auto& session=sessions.at(session_id);
                probability=std::max(probability,admission_model->predict(std::max(0.,now-session.last),session.q,session.length,
                    state(session,membership),membership.referenced_query_groups,session.q>membership.last_query_group?1:0));
            }
            return write(probability,cfg.threshold);
        }
        return true;
    }
    std::vector<SessionId> release(Block block,double now,bool flush,std::optional<bool> decision=std::nullopt) {
        if(release_observer && !flush && !ssd.contains(block)) {
            std::vector<std::pair<uint64_t,double>> current;
            for(const auto& entry:block_map.at(block).sessions)
                if(entry.second.current_path)current.emplace_back(entry.first,sessions.at(entry.first).last);
            release_observer(now,current);
        }
        const auto item=absorption.erase(block);
        auto affected=unlink_absorption(block);
        stats.evictions++;
        if(flush)stats.flushed++;
        else stats.residence.push_back(std::max(0.,now-item.entered));
        // 保留已有 SSD 副本；有限容量淘汰后允许重新持久化。
        if(ssd.contains(block))return affected;
        const bool accept=decision.value_or(admission_accept(block,now));
        if(accept && ssd.put(block))stats.writes++;
        else if(!accept)stats.rejected++;
        return affected;
    }
    void trim_absorption(double now) {
        if(absorption.size()<=cfg.absorption)return;
        if(!cfg.learned_rank&&!cfg.temporal_rank&&!cfg.rule_rank&&!cfg.path_rank&&!cfg.session_rank) {
            while(absorption.size()>cfg.absorption)release(absorption.oldest(),now,false);
            return;
        }
        ScoreCache cache;
        if(cfg.rule_rank||cfg.path_rank) {
            // Bound every rule-based correction to the 16 oldest session-tail
            // candidates.  This keeps LRU as the global temperature order.
            constexpr size_t shortlist_size=16;
            std::vector<SessionVictim> lru_heap;
            auto lru_less=[](const SessionVictim& a,const SessionVictim& b) {
                if(a.block.last_access!=b.block.last_access)
                    return a.block.last_access>b.block.last_access;
                if(a.block.id!=b.block.id)return a.block.id>b.block.id;
                return a.session_id>b.session_id;
            };
            auto push=[&](SessionId session_id) {
                for(const auto& victim:session_victims(session_id,now,cache)) {
                    lru_heap.push_back(victim);
                    std::push_heap(lru_heap.begin(),lru_heap.end(),lru_less);
                }
            };
            auto valid=[&](const SessionVictim& victim) {
                const auto& session=sessions.at(victim.session_id);
                return session.absorption_version==victim.version&&
                    !session.absorption_chain.empty()&&
                    session.absorption_chain.back()==victim.block.id;
            };
            for(const auto& [session_id,session]:sessions){(void)session;push(session_id);}
            while(absorption.size()>cfg.absorption) {
                std::vector<SessionVictim> shortlist;
                while(shortlist.size()<shortlist_size&&!lru_heap.empty()) {
                    std::pop_heap(lru_heap.begin(),lru_heap.end(),lru_less);
                    auto candidate=lru_heap.back();lru_heap.pop_back();
                    if(valid(candidate))shortlist.push_back(candidate);
                }
                if(shortlist.empty())break;
                size_t selected=0;
                for(size_t i=1;i<shortlist.size();++i)
                    if((cfg.path_rank
                        ? path_evict_before(shortlist[i].block,shortlist[selected].block,cfg.threshold)
                        : rule_evict_before(shortlist[i].block,shortlist[selected].block,
                                            cfg.threshold,retention_model->context_reference)))selected=i;
                const auto victim=shortlist[selected];
                for(size_t i=0;i<shortlist.size();++i)if(i!=selected) {
                    lru_heap.push_back(shortlist[i]);
                    std::push_heap(lru_heap.begin(),lru_heap.end(),lru_less);
                }
                const bool accepted=ssd.contains(victim.block.id)||admission_accept(victim.block.id,now);
                for(SessionId affected:release(victim.block.id,now,false,accepted))push(affected);
            }
            return;
        }
        std::vector<SessionVictim> heap;
        std::set<std::pair<SessionId,Block>> protected_heads;
        auto before=[&](const SessionVictim& a,const SessionVictim& b) {
            const auto compare=[&](const Candidate& x,const Candidate& y) {
                if(cfg.rule_rank)return rule_evict_before(x,y,cfg.threshold,retention_model->context_reference);
                return (cfg.temporal_rank||cfg.session_rank)?temporal_evict_before(x,y):evict_before(x,y,cfg.threshold);
            };
            if(compare(a.block,b.block))return true;
            if(compare(b.block,a.block))return false;
            return a.session_id<b.session_id;
        };
        auto heap_less=[&](const SessionVictim& a,const SessionVictim& b) {return before(b,a);};
        auto push=[&](SessionId session_id) {
            for(const auto& victim:session_victims(session_id,now,cache)) {
                if(victim.endpoint==Endpoint::head&&protected_heads.count({session_id,victim.block.id}))continue;
                heap.push_back(victim);
                std::push_heap(heap.begin(),heap.end(),heap_less);
            }
        };
        for(const auto& [session_id,session]:sessions) {
            (void)session;
            push(session_id);
        }
        while(absorption.size()>cfg.absorption) {
            if(heap.empty())break; // all remaining candidates are protected rejected heads
            std::pop_heap(heap.begin(),heap.end(),heap_less);
            const auto victim=heap.back();
            heap.pop_back();
            const auto& session=sessions.at(victim.session_id);
            if(session.absorption_version!=victim.version || session.absorption_chain.empty())continue;
            const Block endpoint=victim.endpoint==Endpoint::head ? session.absorption_chain.front()
                                                                 : session.absorption_chain.back();
            if(endpoint!=victim.block.id)continue;
            const bool accepted=ssd.contains(victim.block.id)||admission_accept(victim.block.id,now);
            if(cfg.temporal_rank) {
                const bool finite=std::isfinite(victim.block.retention_value);
                if(accepted) (finite?stats.temporal_accept_finite:stats.temporal_accept_infinite)++;
                else (finite?stats.temporal_reject_finite:stats.temporal_reject_infinite)++;
            }
            // A rejected head remains the branch prefix in Absorption.  It is
            // not removed and the chain head does not advance.  The tail (and
            // candidates from other sessions) can still satisfy this trim.
            if(victim.endpoint==Endpoint::head&&!accepted &&
               !(cfg.session_rank&&victim.block.branch!=Branch::current&&std::isinf(victim.block.retention_value))) {
                stats.rejected++;
                protected_heads.insert({victim.session_id,victim.block.id});
                continue;
            }
            for(SessionId affected:release(victim.block.id,now,false,accepted))push(affected);
        }
    }
    void trim_serving(double now) {
        while(serving.size()>cfg.serving) {
            const Block block=serving.oldest();
            const auto entry=serving.erase(block);
            if(cfg.exclusive&&ssd.contains(block)){stats.backed_bypass++;continue;}
            enter_absorption(block,{entry.recency,now});
        }
        trim_absorption(now);
    }
public:
    void observe_releases(ReleaseObserver observer) {release_observer=std::move(observer);}
    Replay(ReplayConfig config,const AdmissionModel* a=nullptr,const RetentionModel* r=nullptr,
           const TemporalModel* t=nullptr,const PathTimeModel* p=nullptr):
        cfg(config),admission_model(a),retention_model(r),temporal_model(t),path_model(p),ssd(config.ssd_capacity) {
        probability(cfg.threshold);
        if((cfg.learned_admission||cfg.learned_rank||cfg.temporal_rank||cfg.rule_rank)&&!a)throw std::invalid_argument("missing admission model");
        if((cfg.learned_rank||cfg.rule_rank)&&!r)throw std::invalid_argument("missing retention model");
        if((cfg.temporal_rank||cfg.rule_rank||cfg.session_rank||cfg.session_admission)&&!t)
            throw std::invalid_argument("missing temporal model");
        if((cfg.path_rank||cfg.path_admission)&&!p)throw std::invalid_argument("missing path-time model");
    }
    void access(const Request& r) {
        stats.requests++;
        // 只计完整token块。首个前缀缺失后，后续即使有SSD拷贝也不计本次命中。
        // 先计命中再更新历史/放入重算结果，避免miss当场变hit。
        bool hit=true,dh=true,sh=true,th=true;
        for(Block block:r.blocks) {
            stats.requested++;
            const bool in_s=serving.contains(block),in_d=in_s||absorption.contains(block);
            hit=hit&&(in_d||ssd.contains(block));dh=dh&&in_d;sh=sh&&in_s;th=th&&ever.count(block);
            stats.hits+=hit;stats.dram_hits+=dh;stats.serving_hits+=sh;stats.theoretical+=th;
            if(hit) {
                if(in_d) stats.dram_served_hits++;
                else { stats.ssd_served_hits++; ssd.touch(block); }
            }
        }
        auto& o=sessions[r.session_id];
        if(o.initialized&&r.time<o.last)throw std::runtime_error("session time regressed");
        const bool newq=!o.initialized||r.time-o.last>=30;
        const bool compact=o.initialized&&o.compaction_id!=r.compaction_id;
        if(newq)o.q++;
        if(!o.initialized||compact){o.start=r.time;o.growth=0;}
        else o.growth+=r.length>o.length?r.length-o.length:0;
        if (o.initialized) {
            o.prefix_tree.visit_blocks(o.current_prefix, [&](Block block) {
                auto& membership=block_map.ensure(block).sessions[r.session_id];
                if(membership.current_path)membership.old_since=r.time;
                membership.current_path=false;
            });
        }
        o.initialized=true;o.last=r.time;o.compaction_id=r.compaction_id;o.length=r.length;
        o.current_prefix=o.prefix_tree.upsert(r.blocks);
        for(Block block:r.blocks) {
            auto& m=block_map.ensure(block).sessions[r.session_id];
            if(m.last_query_group!=o.q){m.referenced_query_groups++;m.last_query_group=o.q;}
            m.current_path=true;m.old_since=-1;m.compaction_id=r.compaction_id;
            ever.insert(block);
            // 一次请求原子读取后迁回 Serving；其 session 链上的 Absorption 条目同步移除。
            if(absorption.contains(block)) {
                absorption.erase(block);
                unlink_absorption(block);
            }
            serving.put(block,{++clock,r.time});
        }
        trim_serving(r.time);
    }
    void finish(double now) {
        // 最后统一将 Serving 交给 Absorption/准入；不引入额外的容量压力时刻。
        while(!serving.empty()) {
            const Block block=serving.oldest();
            const auto entry=serving.erase(block);
            if(cfg.exclusive&&ssd.contains(block)){stats.backed_bypass++;continue;}
            enter_absorption(block,{entry.recency,now});
        }
        while(!absorption.empty())release(absorption.oldest(),now,true);
    }
    const ReplayStats& result()const{return stats;}
    size_t dram_size()const{return serving.size()+absorption.size();}
    bool has_ssd(Block block)const{return ssd.contains(block);}
    bool has_dram(Block block)const{return serving.contains(block)||absorption.contains(block);}
    size_t session_count(Block block_id)const {
        const auto* block=block_map.find(block_id);
        return block?block->sessions.size():0;
    }
    size_t stored_blocks()const{return ssd.size();}
};
} // namespace hohlraum


// baselines/lru_set.hpp
#include <cstdint>
#include <list>
#include <limits>
#include <unordered_map>

namespace hohlraum {
// 基线用的单层块 LRU；cap 可设为极大值表示不限容量。
struct LRUSet {
    size_t cap; std::list<uint64_t> q; std::unordered_map<uint64_t,std::list<uint64_t>::iterator> where;
    explicit LRUSet(size_t c):cap(c){}
    bool has(uint64_t b)const{return where.count(b);}
    void touch(uint64_t b){
        auto it=where.find(b);
        if(it!=where.end()){q.erase(it->second);where.erase(it);}
        q.push_back(b);where[b]=std::prev(q.end());
        while(where.size()>cap){where.erase(q.front());q.pop_front();}
    }
    void erase(uint64_t b){auto it=where.find(b);if(it!=where.end()){q.erase(it->second);where.erase(it);}}
};
} // namespace hohlraum


// baselines/larc.hpp
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace hohlraum {
// LARC 适配（图 7 基线）：ghost 历史自适应准入，SSD 不淘汰。
// 移植自旧预算实验树的 larc_finite.cpp，算法不变。
struct LARC {
    LRUSet data,ghost;double cr;uint64_t writes=0,reads=0,hits=0,ghost_hits=0;
    bool enabled;size_t nominal;
    LARC(size_t cap,bool use,bool no_evict=false)
        :data(no_evict?std::numeric_limits<size_t>::max():cap),ghost(cap),cr(.1*cap),enabled(use),nominal(cap){}
    void access(uint64_t b){
        ++reads;
        if(data.has(b)){++hits;data.touch(b);if(enabled)cr=std::max(.1*nominal,cr-nominal/(nominal-cr));return;}
        if(!enabled){data.touch(b);++writes;return;}
        cr=std::min(.9*nominal,cr+nominal/cr);
        if(ghost.has(b)){++ghost_hits;ghost.erase(b);data.touch(b);++writes;}
        else{ghost.touch(b);if(ghost.q.size()>cr)ghost.erase(ghost.q.back());}
    }
};
// LARC 基线回放：DRAM 单层 LRU（总容量）+ LARC SSD；命中为连续前缀口径。
// ghost_blocks 是 ghost 容量（块数），旧实验取 write-all 写入量 × 尺度参数。
inline ReplayStats replay_larc(const std::string& event_path,size_t total,size_t ghost_blocks){
    LRUSet dram(total); LARC ssd(std::max(size_t(10),ghost_blocks),true,true);
    ReplayStats s;
    std::unordered_set<Block> ever;
    EventReader events(event_path);
    Request request;
    while(events.next(request)) {
        ++s.requests;
        bool prefix=true,theory_prefix=true;
        std::vector<Block> missing;
        for(Block block:request.blocks){
            ++s.requested;
            const bool in=dram.has(block);
            prefix=prefix&&(in||ssd.data.has(block));
            theory_prefix=theory_prefix&&ever.count(block);
            s.hits+=prefix; s.theoretical+=theory_prefix;
            if(!in)missing.push_back(block);
        }
        for(Block block:missing)ssd.access(block);
        for(Block block:request.blocks){dram.touch(block);ever.insert(block);}
    }
    s.writes=ssd.writes;
    return s;
}
} // namespace hohlraum
