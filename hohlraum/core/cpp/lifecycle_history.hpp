#pragma once
#include "return_tail.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <queue>
#include <map>
#include <tuple>
#include <cstdlib>
#include <unordered_set>
#include "session_level_index.hpp"


namespace hohlraum {
using BlockId = uint32_t;
using SessionId = uint32_t;
struct Request {
    double time = 0;
    uint32_t context_tokens = 0;
    SessionId session = 0;
    uint32_t fold = 0;
    std::vector<BlockId> blocks;
};
// Lifecycle estimates backed by session-local history ranges.
class History {
public:
    using Association = SessionLevelIndex::State;
    struct Session {
        bool initialized = false;
        double first_seen=std::numeric_limits<double>::quiet_NaN();
        unsigned last_sample_qg=0;
        unsigned qgs = 0, length = 0, fold = 0;
        BlockId last_input_block = 0;
        bool episode_initialized=false;unsigned episode_start=0,episode_end=0;std::vector<bool> risk_credited;
        uint64_t request_sequence = 0;
        double last_access = 0;
        // Session clock: system time spent queued or in service is excluded.
        double offset = 0, busy_since = -1;
        unsigned busy = 0;
        // Open QG-gap sample for online calibration.
        bool sample_completion_observed = false;
        double sample_completed_at = -1;
        uint64_t return_sample_id = UINT64_MAX; // Stable ID of this QG's provisional negative.
    };

    std::vector<Session> sessions;

    SessionLevelIndex index;
    std::vector<Association> owners(BlockId b)const {return index.owners(b);}

    History(size_t blocks, size_t session_count, double threshold, double qg_gap)
        : sessions(session_count),
          threshold_(threshold), qg_gap_(qg_gap) { load_joint(); load_return_counts(); load_online(); load_branch(); }

    struct JointCurve {
        struct Curve {std::vector<double> t,a;};
        std::array<Curve,4> q;std::vector<double> risk;
        double score(unsigned n,double d)const {
            const auto& c=q[std::min(n,4u)-1];auto it=std::upper_bound(c.t.begin(),c.t.end(),d);
            return c.a[it==c.t.begin()?0:size_t(it-c.t.begin()-1)];
        }
        double wait(unsigned n,double d,double threshold)const {
            const auto& c=q[std::min(n,4u)-1];
            for(size_t i=0;i<c.t.size();++i)if(c.t[i]>d&&c.a[i]<threshold)return c.t[i]-d;
            return std::numeric_limits<double>::infinity();
        }
    };
    std::vector<JointCurve> joint;
    void load_joint() {
        const char* root=std::getenv("HOHLRAUM_JOINT_MODELS");if(!root)return;
        unsigned count=std::getenv("HOHLRAUM_MODEL_COUNT")?std::stoul(std::getenv("HOHLRAUM_MODEL_COUNT")):6;
        if(count!=2&&count!=6)throw std::runtime_error("use unified 2-fold AgentX priors or 6-fold Playing priors; typed AgentX priors are unsupported");
        if(count==2){
            std::ifstream schema(std::string(root)+"/model-schema.txt");std::string tag;schema>>tag;
            if(tag!="session-qg-only-v1")throw std::runtime_error("rebuild unified AgentX priors; old typed model files cannot be relabeled");
        }
        for(unsigned i=0;i<count;++i){
            std::ifstream in(std::string(root)+"/"+std::to_string(i)+".txt");JointCurve m;size_t n;
            for(auto& c:m.q){in>>n;for(size_t k=0;k<n;++k){double t,a;in>>t>>a;c.t.push_back(t);c.a.push_back(a);}}
            in>>n;for(size_t k=0;k<n;++k){double r;in>>r;m.risk.push_back(r);}
            if(!in)throw std::runtime_error("invalid joint model");joint.push_back(std::move(m));
        }
        for(const auto& model:joint)ensure_risk_bins(model.risk.size());
    }
    // Hourly calibration uses real cohort counts. The idle grid only tracks
    // survived exposure and mature outcomes, never fixed-strength priors.
    static constexpr double kOnlineHorizon = 3600, kOnlineRatio = 1.15;
    std::vector<double> edges;
    std::vector<std::array<std::vector<double>,4>> prior_hazard, online_curve;
    std::array<std::array<std::vector<double>,4>,3> at_risk{}, events{};
    std::array<std::array<bool,4>,3> dirty{};
    // Length-conditioned risk sets include open (right-censored) gaps.
    using LengthBins = std::array<std::array<std::vector<double>,8>,4>;
    std::array<LengthBins,3> length_at_risk{}, length_events{};
    mutable std::array<LengthBins,6> length_curves{};
    std::array<LengthBins,6> length_prior_hazard{};
    bool has_length_hazards=false;
    mutable std::array<std::array<std::array<bool,8>,4>,6> length_dirty{};
    double statistics_period()const{return statistics_period_;}
    uint64_t statistics_updates=0;
    double last_statistics_update=0;
    bool hourly_statistics()const{return online_enabled_&&statistics_period_>0&&!frozen_priors();}
    bool frozen_priors() const { return frozen_priors_; }
    bool online() const { return online_enabled_ && !joint.empty(); }
    void load_return_counts(){
        if(joint.empty())return;
        std::ifstream counts(std::string(std::getenv("HOHLRAUM_JOINT_MODELS"))+"/return-counts.txt");
        if(!counts)return;
        std::string version;size_t folds=0,buckets=0;
        if(!(counts>>version>>folds>>buckets)||(version!="qg-return-counts-v1"&&version!="qg-return-counts-v2")||folds!=joint.size()||buckets!=4)
            throw std::runtime_error("invalid offline return-counts.txt");
        prior_return_n.resize(folds);prior_return_yes.resize(folds);prior_observed_return_n.resize(folds);
        for(size_t f=0;f<folds;++f)for(unsigned q=0;q<4;++q){
            if(!(counts>>prior_return_n[f][q]>>prior_return_yes[f][q])||prior_return_yes[f][q]>prior_return_n[f][q])
                throw std::runtime_error("invalid offline return counts");
            prior_observed_return_n[f][q]=prior_return_yes[f][q];
            if(version=="qg-return-counts-v2"&&(!(counts>>prior_observed_return_n[f][q])||
               prior_observed_return_n[f][q]>prior_return_yes[f][q]))
                throw std::runtime_error("invalid observed return-interval count");
        }
    }
    void load_online() {
        if(std::getenv("HOHLRAUM_ONLINE_PRIOR"))
            throw std::runtime_error("fixed online prior weight removed; use HOHLRAUM_ONLINE_UPDATES=1");
        const char* enabled=std::getenv("HOHLRAUM_ONLINE_UPDATES");
        if(!enabled||joint.empty())return;
        if(std::string(enabled)!="0"&&std::string(enabled)!="1")
            throw std::runtime_error("HOHLRAUM_ONLINE_UPDATES must be 0 or 1");
        online_enabled_=std::string(enabled)=="1";
        if(!online_enabled_){frozen_priors_=true;return;}
        if(!frozen_priors()&&statistics_period_!=3600)throw std::runtime_error("non-hourly online statistics removed; use hourly updates or frozen priors");
        if(hourly_statistics()){
            if(prior_return_n.empty())throw std::runtime_error("hourly updates require true offline return-counts.txt; rebuild priors");
            std::ifstream risk_counts(std::string(std::getenv("HOHLRAUM_JOINT_MODELS"))+"/context-risk-counts.txt");
            std::string risk_version;size_t risk_folds=0,risk_bins=0;
            if(!(risk_counts>>risk_version>>risk_folds>>risk_bins)||risk_version!="context-risk-counts-v1"||risk_folds!=joint.size()||risk_bins!=risk_bin_count||!risk_bins)
                throw std::runtime_error("hourly risk updates require context-risk-counts.txt; rebuild priors");
            prior_risk_n.resize(risk_folds);prior_risk_yes.resize(risk_folds);
            for(size_t f=0;f<risk_folds;++f){prior_risk_n[f].assign(risk_bins,0);prior_risk_yes[f].assign(risk_bins,0);}
            for(size_t f=0;f<risk_folds;++f)for(size_t i=0;i<risk_bins;++i){
                auto& n=prior_risk_n[f][i];auto& y=prior_risk_yes[f][i];
                if(!(risk_counts>>n>>y)||y>n||i>=joint[f].risk.size()||
                   std::abs(joint[f].risk[i]-(n?double(y)/n:0.))>1e-12)
                    throw std::runtime_error("context risk counts disagree with prior");
            }
        }
        for(double e=qg_gap_;e<kOnlineHorizon;e*=kOnlineRatio)edges.push_back(e);edges.push_back(kOnlineHorizon);
        const size_t bins=edges.size()-1;
        for(auto& p:at_risk)for(auto& v:p)v.assign(bins,0.);
        for(auto& p:events)for(auto& v:p)v.assign(bins,0.);
        for(auto& role:length_at_risk)for(auto& q:role)for(auto& b:q)b.assign(bins,0.);
        for(auto& role:length_events)for(auto& q:role)for(auto& b:q)b.assign(bins,0.);
        for(auto& f:length_dirty)for(auto& q:f)q.fill(true);
    }
    void recompute(unsigned p,unsigned n){dirty[p][n]=false;}
    double curve_score(unsigned fold,unsigned qgs,double d)const{
        return online()?length_score(fold,qgs,0,d):joint.at(fold).score(qgs,d);
    }
    double curve_wait(unsigned fold,unsigned qgs,double d,double threshold)const{
        return online()?joint_wait(fold,qgs,d,0.,true,0):joint.at(fold).wait(qgs,d,threshold);
    }
    void advance(double now){
        observation_now_=now;
        if(frozen_priors())return;
        advance_branch(now);
        if(online()){expire_return_samples(now);publish_statistics(now);}
    }
    double clock(const Session& s,double now)const{
        const double t=(s.busy?s.busy_since:now)-s.offset;
        // Subtracting accumulated busy time can lose a few ulps at zero idle.
        // Clamp only numerical cancellation; genuine reversals remain errors.
        return s.initialized && t<s.last_access && s.last_access-t<=1e-8 ? s.last_access : t;
    }
    void arrive(SessionId id,double now){advance(now);auto& s=sessions.at(id);if(std::isnan(s.first_seen))s.first_seen=now;if(s.busy++==0)s.busy_since=now;session_clock_=true;}
    void complete(SessionId id,double now){
        advance(now);auto& s=sessions.at(id);if(!s.busy||now<s.busy_since)throw std::runtime_error("session completion without arrival");
        if(--s.busy==0){s.offset+=now-s.busy_since;s.busy_since=-1;s.sample_completed_at=now;s.sample_completion_observed=true;
            if(hourly_statistics()&&s.initialized)return_timeouts.push({now+kOnlineHorizon,now,id});
        }
    }
    bool session_clock() const { return session_clock_; }


    // One sample per observed transition to an old suffix; KV references share it.
    struct OldGap {SessionId session;double start;size_t credited=0;};
    std::vector<OldGap> old_gaps;
    std::unordered_set<size_t> open_old;
    using BranchDeadline=std::pair<double,size_t>;
    using BranchDeadlines=std::priority_queue<BranchDeadline,std::vector<BranchDeadline>,std::greater<BranchDeadline>>;
    std::unordered_map<SessionId,BranchDeadlines> branch_deadlines;
    struct EventualBranch {double p=0;uint64_t total=0,returned=0;std::vector<double> gaps;};
    std::vector<EventualBranch> eventual_branch;
    std::vector<double> branch_edges;
    std::vector<std::vector<double>> branch_prior,branch_curves;
    std::array<std::vector<double>,3> branch_risk,branch_events;
    std::array<bool,3> branch_dirty{};
    bool branch_online()const{return !branch_edges.empty();}
    void load_branch(){
        const char* path=std::getenv("HOHLRAUM_BRANCH_PRIOR");if(!path)return;
        if(!online()&&!frozen_priors())throw std::runtime_error("branch calibration requires online updates or frozen priors");
        std::ifstream in(path);size_t n;in>>n;branch_edges.resize(n);for(auto& t:branch_edges)in>>t;
        if(n<2||branch_edges.front()!=0||branch_edges.back()!=kOnlineHorizon)throw std::runtime_error("branch prior grid");
        branch_prior.assign(joint.size(),std::vector<double>(n-1));branch_curves=branch_prior;
        for(auto& f:branch_prior)for(auto& h:f)in>>h;
        if(!in)throw std::runtime_error("branch prior input");
        for(auto& a:branch_risk)a.assign(n-1,0);for(auto& a:branch_events)a.assign(n-1,0);
        for(unsigned provider=0;provider<3;++provider)recompute_branch(provider);
        if(frozen_priors()||hourly_statistics()){
            const std::string root=std::getenv("HOHLRAUM_JOINT_MODELS");
            std::ifstream tag(root+"/prior-target.txt");std::string target;tag>>target;
            if(target!="eventual-return-v1")throw std::runtime_error("frozen scoring requires eventual-return priors");
            std::ifstream source(root+"/branch-eventual.txt");eventual_branch.resize(joint.size());
            for(auto& b:eventual_branch){size_t count=0;if(!(source>>b.p>>count)||b.p<0||b.p>1)throw std::runtime_error("invalid eventual branch prior");b.gaps.resize(count);for(auto& d:b.gaps)source>>d;}
            if(!source)throw std::runtime_error("invalid eventual branch priors");
            if(hourly_statistics()){
                std::ifstream counts(root+"/branch-counts.txt");
                std::string version;size_t folds=0;
                if(!(counts>>version>>folds)||version!="branch-return-counts-v1"||folds!=joint.size())
                    throw std::runtime_error("hourly branch updates require real branch-counts.txt; rebuild priors");
                for(auto& b:eventual_branch){
                    if(!(counts>>b.total>>b.returned)||b.returned>b.total||b.returned!=b.gaps.size()||
                       std::abs(b.p-(b.total?double(b.returned)/b.total:0.))>1e-12)
                        throw std::runtime_error("offline branch counts disagree with return distribution");
                }
            }
        }
    }
    void recompute_branch(unsigned provider){branch_dirty[provider]=false;}
    double branch_score(unsigned fold,double idle)const{
        if(frozen_priors()||hourly_statistics()){
            const auto& b=eventual_branch.at(fold);
            double survive=b.gaps.empty()?0.:double(b.gaps.end()-std::lower_bound(b.gaps.begin(),b.gaps.end(),idle))/b.gaps.size();
            double probability=b.p;
            if(hourly_statistics()){
                unsigned pool=fold/2;double returned=0,tail=0;
                for(size_t k=0;k<published_branch_events[pool].size();++k){
                    returned+=published_branch_events[pool][k];
                    if(branch_edges[k+1]>idle)tail+=published_branch_events[pool][k];
                }
                double negative=published_branch_risk[pool].empty()?0.:published_branch_risk[pool].back()-published_branch_events[pool].back();
                // Pool actual completed cohorts, just as for session return.
                // Offline positives all have observed branch-return intervals.
                // No fixed-strength pseudo-prior or inferred sample weight.
                const double total=double(b.total)+returned+negative;
                const double positive=double(b.returned)+returned;
                const double surviving=double(b.gaps.end()-std::lower_bound(b.gaps.begin(),b.gaps.end(),idle))+tail;
                const double denominator=total-positive+surviving;
                return denominator>0?surviving/denominator:0.;
            }
            double v=probability*survive,denom=1-probability+v;return denom>0?v/denom:0.;
        }
        throw std::runtime_error("non-hourly online branch statistics removed");
    }
    void advance_branch(double now){
        if(!branch_online())return;
        // Each queue uses its session's idle clock, so engine/client busy time
        // still freezes all its observations exactly as before. Only due bins
        // are visited; this does not advance any serving/simulator clock.
        for(auto it=branch_deadlines.begin();it!=branch_deadlines.end();){
            auto& due=it->second;
            const double current=clock(sessions[it->first],now);
            std::vector<BranchDeadline> rounding_boundary;
            while(!due.empty()&&due.top().first<=current){
                const size_t id=due.top().second;due.pop();
                if(!open_old.count(id))continue;
                auto& g=old_gaps[id];unsigned provider=sessions[g.session].fold/2;
                const double idle=std::max(0.,current-g.start);
                while(g.credited+1<branch_edges.size()&&branch_edges[g.credited+1]<=idle){
                    ++branch_risk[provider][g.credited++];branch_dirty[provider]=true;
                }
                if(g.credited+1==branch_edges.size())open_old.erase(id);
                else {
                    // The original subtraction decides boundary membership.
                    // Conservative heap deadlines never delay a rounded sum;
                    // recheck a borderline entry once per call, not in a loop.
                    BranchDeadline next{std::nextafter(g.start+branch_edges[g.credited+1],
                        -std::numeric_limits<double>::infinity()),id};
                    if(next.first<=current)rounding_boundary.push_back(next);
                    else due.push(next);
                }
            }
            for(const auto& entry:rounding_boundary)due.push(entry);
            if(due.empty())it=branch_deadlines.erase(it);else ++it;
        }
        for(unsigned p=0;p<3;++p)if(branch_dirty[p])recompute_branch(p);
    }
    void return_branch(int64_t id,double time){
        if(frozen_priors())return;
        if(id<0||!open_old.erase(size_t(id)))return;
        auto& g=old_gaps[size_t(id)];unsigned provider=sessions[g.session].fold/2;
        // advance_branch has credited fully survived bins; this bin contains the event.
        if(time-g.start<branch_edges.back()){
            ++branch_risk[provider][g.credited];++branch_events[provider][g.credited];recompute_branch(provider);
        }
    }
    struct LengthReturn {double base=0;std::array<double,8> p{};std::vector<double> gaps;mutable ReturnTailFit offline_tail, online_tail;};
    size_t tail_minimum_=std::getenv("HOHLRAUM_TAIL_MIN_SAMPLES")?std::stoul(std::getenv("HOHLRAUM_TAIL_MIN_SAMPLES")):64;
    mutable bool length_loaded=false;
    std::vector<std::array<uint64_t,4>> prior_return_n,prior_return_yes,prior_observed_return_n;
    mutable std::array<std::array<LengthReturn,4>,6> length_returns;
    static unsigned length_bucket(unsigned){return 0;} // QG-only statistics; retained slot index for file compatibility.
    std::array<std::array<std::array<uint64_t,8>,4>,3> length_n{},length_yes{};
    std::array<std::array<std::vector<double>,4>,3> returned_gaps{};
    std::array<std::vector<uint64_t>,3> risk_n{},risk_yes{};
    std::vector<std::vector<uint64_t>> prior_risk_n,prior_risk_yes;
    uint64_t online_length_returns=0,online_length_nonreturns=0,online_context_changes=0;
    // A session becomes eligible one hour after its first arrival. Its closed
    // QGs then each contribute one positive; its final open QG stays uncounted
    // until one idle hour, when it becomes provisionally negative. Late returns
    // correct that same record rather than append another outcome.
    decltype(length_n) published_length_n{},published_length_yes{};
    decltype(returned_gaps) published_returned_gaps{};
    decltype(risk_n) published_risk_n{},published_risk_yes{};
    decltype(branch_risk) published_branch_risk{},published_branch_events{};
    double next_statistics_update=std::numeric_limits<double>::quiet_NaN();
    struct ReturnSample {
        unsigned role,q;
        double gap,eligible_at;
        bool returned,published=false;
    };
    std::vector<ReturnSample> return_samples;
    struct ReturnTimeout {
        double deadline,completed_at;
        SessionId session;
        bool operator<(const ReturnTimeout& x)const {
            return deadline==x.deadline?session>x.session:deadline>x.deadline;
        }
    };
    std::priority_queue<ReturnTimeout> return_timeouts;
    uint64_t online_late_return_corrections=0;
    bool return_statistics_published=false;
    void publish_return_sample(ReturnSample& x){
        if(x.published)throw std::runtime_error("QG sample published twice");
        ++length_n[x.role][x.q][0];
        if(x.returned){
            ++length_yes[x.role][x.q][0];++online_length_returns;
            auto& gaps=returned_gaps[x.role][x.q];
            gaps.insert(std::upper_bound(gaps.begin(),gaps.end(),x.gap),x.gap);
        }else ++online_length_nonreturns;
        x.published=true;
    }
    void expire_return_samples(double now){
        while(!return_timeouts.empty()&&return_timeouts.top().deadline<=now+1e-8){
            const auto t=return_timeouts.top();return_timeouts.pop();
            auto& s=sessions[t.session];
            if(!s.initialized||s.busy||!s.sample_completion_observed||
               s.sample_completed_at!=t.completed_at||s.return_sample_id!=UINT64_MAX)continue;
            if(s.last_sample_qg==s.qgs)throw std::runtime_error("duplicate QG sample");
            s.last_sample_qg=s.qgs;
            s.return_sample_id=return_samples.size();
            return_samples.push_back({s.fold/2,std::min(s.qgs,4u)-1,0.,t.deadline,false});
        }
    }
    void publish_statistics(double now){
        if(!hourly_statistics())return;
        if(std::isnan(next_statistics_update))next_statistics_update=(std::floor(now/statistics_period_)+1)*statistics_period_;
        if(now+1e-8<next_statistics_update)return;
        const double cutoff=std::floor((now+1e-8)/statistics_period_)*statistics_period_;
        for(auto& x:return_samples)if(!x.published&&x.eligible_at<=cutoff+1e-8)
            publish_return_sample(x);
        published_length_n=length_n;published_length_yes=length_yes;
        published_returned_gaps=returned_gaps;
        return_statistics_published=true;
        published_risk_n=risk_n;published_risk_yes=risk_yes;
        published_branch_risk=branch_risk;published_branch_events=branch_events;
        last_statistics_update=now;++statistics_updates;
        do{next_statistics_update+=statistics_period_;}while(next_statistics_update<=now+1e-8);
    }
    double observation_now_=0;
    void record_qg_return(Session& s,double gap){
        if(s.return_sample_id==UINT64_MAX){
            if(s.last_sample_qg==s.qgs)throw std::runtime_error("duplicate QG sample");
            s.last_sample_qg=s.qgs;
            // Gate on session age, not on the individual QG's completion age.
            return_samples.push_back({s.fold/2,std::min(s.qgs,4u)-1,gap,
                                      std::max(s.first_seen+kOnlineHorizon,observation_now_),true});
            return;
        }
        auto& x=return_samples.at(s.return_sample_id);
        if(x.returned)throw std::runtime_error("QG negative corrected twice");
        x.returned=true;x.gap=gap;
        if(x.published){
            if(!online_length_nonreturns||length_yes[x.role][x.q][0]>=length_n[x.role][x.q][0])
                throw std::runtime_error("missing provisional negative");
            ++length_yes[x.role][x.q][0];--online_length_nonreturns;++online_length_returns;
            auto& gaps=returned_gaps[x.role][x.q];
            gaps.insert(std::upper_bound(gaps.begin(),gaps.end(),gap),gap);
        }else{
            // Mature but not yet ordinarily published: reveal this corrected
            // cohort immediately, without publishing unrelated pending cohorts.
            publish_return_sample(x);
            ++published_length_n[x.role][x.q][0];
        }
        ++published_length_yes[x.role][x.q][0];
        auto& gaps=published_returned_gaps[x.role][x.q];
        gaps.insert(std::upper_bound(gaps.begin(),gaps.end(),gap),gap);
        return_statistics_published=true;
        s.return_sample_id=UINT64_MAX;++online_late_return_corrections;
    }
    // Length-risk coverage follows the model and grows with observed contexts.
    // Preserve cohort counts and credited flags; newly reached bins have no
    // evidence until an episode crosses them. No tail risk is fabricated.
    size_t risk_bin_count=0;
    void ensure_risk_bins(size_t bins){
        if(bins<=risk_bin_count)return;
        risk_bin_count=bins;
        for(auto& model:joint)model.risk.resize(bins,0.);
        for(auto& values:risk_n)values.resize(bins,0);
        for(auto& values:risk_yes)values.resize(bins,0);
        for(auto& values:published_risk_n)values.resize(bins,0);
        for(auto& values:published_risk_yes)values.resize(bins,0);
        for(auto& values:prior_risk_n)values.resize(bins,0);
        for(auto& values:prior_risk_yes)values.resize(bins,0);
        for(auto& session:sessions)session.risk_credited.resize(bins,false);
    }
    void update_length_risk(Session& s,unsigned L,bool changed){
        ensure_risk_bins(size_t(L)/4096+1);
        // Bridge-created sessions may be added after the model table was sized.
        s.risk_credited.resize(risk_bin_count,false);
        if(frozen_priors()){s.episode_initialized=true;return;}
        if(!s.episode_initialized){s.episode_initialized=true;s.episode_start=s.episode_end=L;std::fill(s.risk_credited.begin(),s.risk_credited.end(),false);return;}
        if(!changed)s.episode_end=std::max(s.episode_end,L);
        unsigned role=s.fold/2;
        for(size_t i=0;i<risk_bin_count;++i){unsigned pos=i*4096;
            if(s.risk_credited[i]||s.episode_start>pos||s.episode_end<=pos)continue;
            if(changed||s.episode_end>pos+8192){++risk_n[role][i];risk_yes[role][i]+=changed&&s.episode_end<=pos+8192;s.risk_credited[i]=true;}
        }
        if(changed){++online_context_changes;s.episode_start=s.episode_end=L;std::fill(s.risk_credited.begin(),s.risk_credited.end(),false);}
    }
    double online_length_risk(unsigned fold,unsigned length)const{
        if(hourly_statistics()){
            unsigned i=length/4096;if(i>=joint[fold].risk.size())return 0;
            const double n=double(prior_risk_n.at(fold)[i])+published_risk_n[fold/2][i];
            return n?(double(prior_risk_yes.at(fold)[i])+published_risk_yes[fold/2][i])/n:0.;
        }
        if(frozen_priors()||!online())return length/4096<joint.at(fold).risk.size()?joint.at(fold).risk[length/4096]:0.;
        unsigned i=length/4096;if(i>=joint[fold].risk.size())return 0;
        throw std::runtime_error("non-hourly online risk statistics removed");
    }
    double fitted_return_tail(const LengthReturn& q,const std::vector<double>& online,double d,bool merged,double offline_weight=1.)const {
        const double cutoff=merged?d-1e-8:d;
        const double empirical=offline_weight*double(q.gaps.end()-std::lower_bound(q.gaps.begin(),q.gaps.end(),cutoff))+
                               double(online.end()-std::lower_bound(online.begin(),online.end(),cutoff));
        auto& fit=merged?q.online_tail:q.offline_tail;
        fit.update(q.gaps,online,tail_minimum_,qg_gap_,offline_weight,merged&&!online.empty());
        if(merged&&!online.empty()){
            const double last=std::max(q.gaps.empty()?0.:q.gaps.back(),online.back());
            if(d<=last+1e-8)return empirical;
            const double last_mass=offline_weight*double(q.gaps.end()-std::lower_bound(q.gaps.begin(),q.gaps.end(),last-1e-8))+
                                   double(online.end()-std::lower_bound(online.begin(),online.end(),last-1e-8));
            // Extrapolation may spend only mass surviving the last observed
            // interval; it cannot raise the curve above that empirical value.
            return std::min(last_mass,fit.tail(d,empirical));
        }
        return fit.tail(d,empirical);
    }
    void load_length_returns()const {
        if(length_loaded)return;
        std::ifstream in(std::getenv("HOHLRAUM_LENGTH_CURVES"));
        for(size_t fi=0;fi<joint.size();++fi)for(auto& q:length_returns[fi]){
            in>>q.base;for(auto& v:q.p)in>>v;size_t n;in>>n;
            q.gaps.resize(n);for(auto& v:q.gaps)in>>v;
        }
        if(!in)throw std::runtime_error("length curve read");
        length_loaded=true;
    }
    double published_return_survival(unsigned fold,unsigned qi,double d)const {
        if(d<qg_gap_)return 1.;
        load_length_returns();
        const auto& q=length_returns[fold][qi];
        const auto& gaps=published_returned_gaps[fold/2][qi];
        const auto offline_n=prior_return_n[fold][qi],offline_yes=prior_return_yes[fold][qi];
        if(q.gaps.size()!=prior_observed_return_n[fold][qi] ||
           (offline_n&&std::abs(q.p[0]-double(offline_yes)/offline_n)>1e-10))
            throw std::runtime_error("offline return counts disagree with return distribution");
        const double timed_positive=double(offline_yes)+gaps.size();
        // Untimed offline positives use the observed return distribution.
        // Provisional negatives contribute no fabricated return intervals.
        if(!q.gaps.empty()){
            const double tail=fitted_return_tail(q,gaps,d,true,double(offline_yes)/q.gaps.size());
            return std::clamp(timed_positive?tail/timed_positive:1.,0.,1.);
        }
        if(!gaps.empty())return std::clamp(fitted_return_tail(q,gaps,d,true)/gaps.size(),0.,1.);
        return 1.; // No observed intervals: preserve the untimed-positive convention.
    }
    bool use_published_returns()const {return hourly_statistics()&&return_statistics_published;}

    uint64_t return_sample_count(unsigned fold,unsigned bucket)const {
        return prior_return_n[fold][bucket]+
            (use_published_returns()?published_length_n[fold/2][bucket][0]:0);
    }

    unsigned return_bucket(unsigned fold,unsigned qgs)const {
        const unsigned bucket=std::min(qgs,4u)-1;
        if(prior_return_n.empty()||return_sample_count(fold,bucket))return bucket;
        // Borrow the nearest supported QG cohort in the same training fold.
        // A tie prefers the lower QG and never creates or recounts samples.
        for(unsigned distance=1;distance<4;++distance){
            if(distance<=bucket&&return_sample_count(fold,bucket-distance))return bucket-distance;
            if(bucket+distance<4&&return_sample_count(fold,bucket+distance))return bucket+distance;
        }
        return bucket;
    }

    double length_score(unsigned fold,unsigned qgs,unsigned length,double d)const {
        load_length_returns();
        const unsigned qi=return_bucket(fold,qgs),b=length_bucket(length),role=fold/2;
        if(!prior_return_n.empty()&&!return_sample_count(fold,qi))return 0.;
        const auto& q=length_returns.at(fold).at(qi);
        if(!use_published_returns()){
            static const std::vector<double> empty;
            const double survival=q.gaps.empty()?1.:fitted_return_tail(q,empty,d,false)/q.gaps.size();
            return return_posterior(q.p[b],survival);
        }
        const double total=return_sample_count(fold,qi);
        const double yes=double(prior_return_yes[fold][qi])+published_length_yes[role][qi][b];
        return return_posterior(yes/total,published_return_survival(fold,qi,d));
    }
    double joint_wait(unsigned fold,unsigned qgs,double ds,double db,bool current=false,unsigned length=0)const{
        const unsigned qi=return_bucket(fold,qgs);
        if(length_score(fold,qgs,length,ds)*(current?1.:branch_score(fold,db))<threshold_)return 0.;
        if(tail_minimum_>=2&&(frozen_priors()||hourly_statistics())) {
            auto score=[&](double dt){return length_score(fold,qgs,length,ds+dt)*
                (current?1.:branch_score(fold,db+dt));};
            const double initial=score(0.);
            (void)length_score(fold,qgs,length,std::max(ds,qg_gap_));
            const auto& q=length_returns.at(fold).at(qi);
            const auto& fit=hourly_statistics()&&!frozen_priors()&&return_statistics_published?q.online_tail:q.offline_tail;
            if(fit.usable) {
                if(initial<threshold_)return 0.;
                if(threshold_<=0)return std::numeric_limits<double>::infinity();
                // Bracket the monotone crossing, including beyond observed gaps.
                double lo=0.,hi=std::max(1.,qg_gap_-ds);
                while(score(hi)>=threshold_&&hi<1e15)hi*=2.;
                if(score(hi)>=threshold_)return std::numeric_limits<double>::infinity();
                for(unsigned i=0;i<48;++i){double mid=(lo+hi)/2;
                    if(score(mid)<threshold_)hi=mid;else lo=mid;}
                return hi;
            }
        }
        std::vector<double> times;
        if(ds<qg_gap_)times.push_back(qg_gap_-ds);
        if(frozen_priors()||hourly_statistics()){
            for(double t:length_returns.at(fold).at(qi).gaps)if(t>=ds)times.push_back(t-ds+1e-6);
            if(hourly_statistics())for(double t:published_returned_gaps[fold/2][qi])if(t>=ds)times.push_back(t-ds+1e-6);
            if(hourly_statistics()&&!current)for(double t:branch_edges)if(t>=db)times.push_back(t-db+1e-6);
            if(!current)for(double t:eventual_branch.at(fold).gaps)if(t>=db)times.push_back(t-db+1e-6);
        }
        else if(online()){for(double t:edges)if(t>ds)times.push_back(t-ds);}
        else for(double t:joint[fold].q[std::min(qgs,4u)-1].t)if(t>ds)times.push_back(t-ds);
        if(!current&&!frozen_priors()&&!hourly_statistics())for(double t:branch_edges)if(t>db)times.push_back(t-db);
        std::sort(times.begin(),times.end());
        for(double dt:times)if(length_score(fold,qgs,length,ds+dt)*(current?1.:branch_score(fold,db+dt))<threshold_)return dt;
        return std::numeric_limits<double>::infinity();
    }
    struct JointScore {double admission=0,weight=0,wait=0;bool pending=false,mature=true;};
    using WaitKey = std::tuple<unsigned,unsigned,double,double,bool,unsigned>;
    using WaitMemo = std::map<WaitKey,double>;
    JointScore joint_score(BlockId b,double now,WaitMemo* waits=nullptr)const {
        JointScore out;double risk=0,idle_min=std::numeric_limits<double>::infinity();bool current=false;
        for(const auto& a:owners(b)){
            const auto& s=sessions[a.session];const auto& model=joint.at(s.fold);
            const double t=clock(s,now);
            const bool cur=a.request_sequence==s.request_sequence;current=current||cur;
            const bool old_mature=branch_online()?!cur:(!cur&&a.old_since>=0&&t-a.old_since>=120);
            out.mature=out.mature&&old_mature;
            out.pending=out.pending||(!cur&&!old_mature);
            idle_min=std::min(idle_min,std::max(0.,t-a.last_access));
            const double ds=std::max(0.,t-s.last_access),db=std::max(0.,t-a.old_since);
            // Only a current branch is protected while its QG may still be active.
            const double protection=cur?std::max(0.,qg_gap_-ds):0.;
            const double v=protection>0.?1.:
                (branch_online()?length_score(s.fold,s.qgs,s.length,ds)*(cur?1.:branch_score(s.fold,db)):
                 (old_mature?0.:curve_score(s.fold,s.qgs,ds)));
            out.admission=std::max(out.admission,v);
            if(cur&&s.length/4096<model.risk.size())risk=std::max(risk,online_length_risk(s.fold,s.length));
            if(v>=threshold_){
                // Measure the next-QG wait starting at the end of the hold.
                const double scoring_idle=protection>0.?qg_gap_:ds;
                double w;
                if(branch_online()) {
                    if(waits) {
                        const WaitKey key{s.fold,s.qgs,scoring_idle,cur?0.:db,cur,s.length};
                        auto it=waits->find(key);
                        if(it==waits->end())it=waits->emplace(key,joint_wait(s.fold,s.qgs,scoring_idle,db,cur,s.length)).first;
                        w=it->second;
                    } else w=joint_wait(s.fold,s.qgs,scoring_idle,db,cur,s.length);
                }
                else {
                    w=curve_score(s.fold,s.qgs,scoring_idle)<threshold_?0.:
                        curve_wait(s.fold,s.qgs,scoring_idle,threshold_);
                    if(!cur)w=std::min(w,std::max(0.,120-(t-a.old_since)));
                }
                out.wait=std::max(out.wait,protection+w);
            }
        }
        // The one-sided waiting objective applies only while admission accepts.
        // Already-rejected candidates have no further accept-to-reject event to
        // wait for, so they share the +infinity eviction priority.
        if(out.admission < threshold_) out.wait=std::numeric_limits<double>::infinity();
        out.pending=branch_online()?(!current&&out.admission>=threshold_):(out.pending&&!current);
        // A QG-sized age baseline preserves context-risk discrimination for recent KV.
        out.weight=(qg_gap_+(std::isfinite(idle_min)?idle_min:std::max(0.,now)))*(1-risk);
        return out;
    }
    void observe(const Request& input) {
        Request request=input;
        if(joint.size()==2){
            if(request.fold>=4)throw std::runtime_error("invalid AgentX fold");
            request.fold%=2; // Discard legacy role bit; retain held-out root fold.
        }
        advance(request.time);
        observation_now_=request.time;
        auto& s = sessions.at(request.session);
        if(std::isnan(s.first_seen))s.first_seen=request.time;
        const double time = clock(s, request.time);
        const auto ordered=index.locate(request.session,request.blocks);
        const auto next=SessionLevelIndex::normalize(ordered);
        auto& table=index.tables[request.session];
        const auto removed=SessionLevelIndex::subtract(table.current,next);
        if(!joint.empty()){
            // Visit state ranges in request order, preserving return-event order.
            if(branch_online())for(auto span:ordered)table.visit(span,[&](const Association& a){return_branch(a.old_gap,time);});
            // A terminal-page replacement is a positional/branch event, not
            // a new global context-length episode. Preserve all branch updates.
            const bool terminal_only = removed.size()==1 &&
                removed[0].second==removed[0].first+1 &&
                table.positions.at(s.last_input_block)==removed[0].first;
            if(s.initialized)update_length_risk(s,request.context_tokens,
                                               !removed.empty() && !terminal_only);
            int64_t gap=-1;
            if(branch_online()&&!frozen_priors()&&!removed.empty()){
                gap=int64_t(old_gaps.size());old_gaps.push_back({request.session,time,0});open_old.insert(size_t(gap));
                branch_deadlines[request.session].push({std::nextafter(time+branch_edges[1],
                    -std::numeric_limits<double>::infinity()),size_t(gap)});
            }
            for(auto span:removed){table.mark_old(span,time,gap);++index.range_updates;}
        }
        table.current=next;
        if (s.initialized && time < s.last_access)
            throw std::runtime_error("session time reversal");
        if(!s.episode_initialized){s.fold=request.fold;update_length_risk(s,request.context_tokens,false);}
        if(hourly_statistics()&&s.initialized&&s.sample_completion_observed){
            const double gap=std::max(0.,time-s.last_access);
            if(gap>=qg_gap_)record_qg_return(s,gap);
        }
        if (!s.initialized || time - s.last_access >= qg_gap_) ++s.qgs;
        s.initialized = true;
        s.last_access = time;
        s.length = request.context_tokens;
        s.last_input_block = request.blocks.empty()?0:request.blocks.back();
        s.fold = request.fold;
        ++s.request_sequence;
        s.sample_completed_at=request.time;s.sample_completion_observed=!s.busy;
        if(hourly_statistics()&&!s.busy)return_timeouts.push({request.time+kOnlineHorizon,request.time,request.session});


        const Association value{request.session,time,-1,s.request_sequence,-1};
        for(auto span:next){table.assign(span,value);++index.range_updates;}
    }

    void set_threshold(double threshold) {
        if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
            throw std::invalid_argument("invalid admission threshold");
        threshold_ = threshold;
    }

private:
    double threshold_, qg_gap_;
    bool online_enabled_ = false;
    double statistics_period_ = std::getenv("HOHLRAUM_STATISTICS_PERIOD_S") ? std::stod(std::getenv("HOHLRAUM_STATISTICS_PERIOD_S")) : 3600.;
    bool frozen_priors_ = std::getenv("HOHLRAUM_FREEZE_STATISTICS") && std::string(std::getenv("HOHLRAUM_FREEZE_STATISTICS"))=="1";
    bool session_clock_ = false;
};
} // namespace hohlraum
