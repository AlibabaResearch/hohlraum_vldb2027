#pragma once
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
        unsigned qgs = 0, length = 0, fold = 0;
        bool episode_initialized=false;unsigned episode_start=0,episode_end=0;std::array<bool,65> risk_credited{};
        uint64_t request_sequence = 0;
        double last_access = 0;
        // Session clock: system time spent queued or in service is excluded.
        double offset = 0, busy_since = -1;
        unsigned busy = 0;
        // Open QG-gap sample for online calibration.
        bool gap_open = false;
        unsigned gap_bucket = 0, gap_credited = 0;
    };

    std::vector<Session> sessions;
    SessionLevelIndex index;
    std::vector<Association> owners(BlockId b)const {return index.owners(b);}

    History(size_t blocks, size_t session_count, double threshold, double qg_gap)
        : sessions(session_count),
          threshold_(threshold), qg_gap_(qg_gap) { load_joint(); load_online(); load_branch(); }

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
        if(count!=4&&count!=6)throw std::runtime_error("model count must be4 or6");
        for(unsigned i=0;i<count;++i){
            std::ifstream in(std::string(root)+"/"+std::to_string(i)+".txt");JointCurve m;size_t n;
            for(auto& c:m.q){in>>n;for(size_t k=0;k<n;++k){double t,a;in>>t>>a;c.t.push_back(t);c.a.push_back(a);}}
            in>>n;for(size_t k=0;k<n;++k){double r;in>>r;m.risk.push_back(r);}
            if(!in)throw std::runtime_error("invalid joint model");joint.push_back(std::move(m));
        }
    }
    // Online calibration of the QG return curves. Discrete-time hazards on a
    // geometric idle grid; offline curves enter as a per-bin Beta prior of
    // strength online_prior_. Counts are pooled per provider (fold/2).
    static constexpr double kOnlineHorizon = 3600, kOnlineRatio = 1.15;
    std::vector<double> edges;
    std::vector<std::array<std::vector<double>,4>> prior_hazard, online_curve;
    std::array<std::array<std::vector<double>,4>,3> at_risk{}, events{};
    std::array<std::array<bool,4>,3> dirty{};
    bool online() const { return online_prior_ > 0 && !joint.empty(); }
    void load_online() {
        const char* prior=std::getenv("HOHLRAUM_ONLINE_PRIOR");if(!prior||joint.empty())return;
        online_prior_=std::stod(prior);if(!(online_prior_>0))throw std::runtime_error("invalid online prior");
        for(double e=qg_gap_;e<kOnlineHorizon;e*=kOnlineRatio)edges.push_back(e);edges.push_back(kOnlineHorizon);
        const size_t bins=edges.size()-1;
        for(auto& p:at_risk)for(auto& v:p)v.assign(bins,0.);
        for(auto& p:events)for(auto& v:p)v.assign(bins,0.);
        prior_hazard.resize(joint.size());online_curve.resize(joint.size());
        for(size_t f=0;f<joint.size();++f)for(unsigned n=0;n<4;++n){
            const auto& c=joint[f].q[n];
            auto before=[&](double t){auto it=std::lower_bound(c.t.begin(),c.t.end(),t);
                return std::min(1-1e-12,c.a[it==c.t.begin()?0:size_t(it-c.t.begin()-1)]);};
            auto& h=prior_hazard[f][n];h.assign(bins,0.);
            for(size_t k=0;k<bins;++k){const double a0=before(edges[k]),a1=k+1==bins?0.:before(edges[k+1]);
                h[k]=std::clamp(1-(1-a0)/(1-a1),0.,1.);}
            online_curve[f][n].assign(bins,0.);
        }
        for(unsigned p=0;p<3;++p)for(unsigned n=0;n<4;++n)recompute(p,n);
    }
    void recompute(unsigned p,unsigned n){
        for(size_t f=0;f<joint.size();++f){if(f/2!=p)continue;
            double survive=1;auto& a=online_curve[f][n];
            for(size_t k=a.size();k-->0;){
                const double h=(events[p][n][k]+online_prior_*prior_hazard[f][n][k])/(at_risk[p][n][k]+online_prior_);
                survive*=1-h;a[k]=1-survive;}
        }
        dirty[p][n]=false;
    }
    double curve_score(unsigned fold,unsigned qgs,double d)const{
        if(!online())return joint.at(fold).score(qgs,d);
        const auto& a=online_curve.at(fold)[std::min(qgs,4u)-1];
        if(d>=edges.back())return 0.;if(d<edges[0])return a[0];
        return a[size_t(std::upper_bound(edges.begin(),edges.end(),d)-edges.begin()-1)];
    }
    double curve_wait(unsigned fold,unsigned qgs,double d,double threshold)const{
        if(!online())return joint.at(fold).wait(qgs,d,threshold);
        const auto& a=online_curve.at(fold)[std::min(qgs,4u)-1];
        for(size_t k=0;k<a.size();++k)if(edges[k]>d&&a[k]<threshold)return edges[k]-d;
        return edges.back()>d?edges.back()-d:std::numeric_limits<double>::infinity();
    }
    // Credit every bin an open gap has fully survived; idle beyond the horizon
    // closes the sample as a non-return.
    void credit(Session& s,double idle){
        const unsigned p=s.fold/2,n=s.gap_bucket;const size_t bins=edges.size()-1;
        while(s.gap_credited<bins&&edges[s.gap_credited+1]<=idle){++at_risk[p][n][s.gap_credited++];dirty[p][n]=true;}
        if(s.gap_credited==bins){length_outcome(s,idle,false);s.gap_open=false;}
    }
    void advance(double now){
        advance_branch(now);
        if(!online())return;
        for(auto& s:sessions)if(s.gap_open&&!s.busy)credit(s,clock(s,now)-s.last_access);
        for(unsigned p=0;p<3;++p)for(unsigned n=0;n<4;++n)if(dirty[p][n])recompute(p,n);
    }
    double clock(const Session& s,double now)const{
        const double t=(s.busy?s.busy_since:now)-s.offset;
        // Subtracting accumulated busy time can lose a few ulps at zero idle.
        // Clamp only numerical cancellation; genuine reversals remain errors.
        return s.initialized && t<s.last_access && s.last_access-t<=1e-8 ? s.last_access : t;
    }
    void arrive(SessionId id,double now){auto& s=sessions.at(id);if(s.busy++==0)s.busy_since=now;session_clock_=true;}
    void complete(SessionId id,double now){
        auto& s=sessions.at(id);if(!s.busy||now<s.busy_since)throw std::runtime_error("session completion without arrival");
        if(--s.busy==0){s.offset+=now-s.busy_since;s.busy_since=-1;}
    }
    bool session_clock() const { return session_clock_; }
    double online_prior() const { return online_prior_; }


    // One sample per observed transition to an old suffix; KV references share it.
    struct OldGap {SessionId session;double start;size_t credited=0;};
    std::vector<OldGap> old_gaps;
    std::unordered_set<size_t> open_old;
    std::vector<double> branch_edges;
    std::vector<std::vector<double>> branch_prior,branch_curves;
    std::array<std::vector<double>,3> branch_risk,branch_events;
    std::array<bool,3> branch_dirty{};
    bool branch_online()const{return !branch_edges.empty();}
    void load_branch(){
        const char* path=std::getenv("HOHLRAUM_BRANCH_PRIOR");if(!path)return;
        if(!online())throw std::runtime_error("branch calibration requires existing online prior");
        std::ifstream in(path);size_t n;in>>n;branch_edges.resize(n);for(auto& t:branch_edges)in>>t;
        if(n<2||branch_edges.front()!=0||branch_edges.back()!=kOnlineHorizon)throw std::runtime_error("branch prior grid");
        branch_prior.assign(joint.size(),std::vector<double>(n-1));branch_curves=branch_prior;
        for(auto& f:branch_prior)for(auto& h:f)in>>h;
        if(!in)throw std::runtime_error("branch prior input");
        for(auto& a:branch_risk)a.assign(n-1,0);for(auto& a:branch_events)a.assign(n-1,0);
        for(unsigned provider=0;provider<3;++provider)recompute_branch(provider);
    }
    void recompute_branch(unsigned provider){
        for(size_t f=0;f<joint.size();++f)if(f/2==provider){
            double survive=1;auto& curve=branch_curves[f];
            for(size_t k=curve.size();k-->0;){
                double h=(branch_events[provider][k]+online_prior_*branch_prior[f][k])/(branch_risk[provider][k]+online_prior_);
                survive*=1-h;curve[k]=1-survive;
            }
        }
        branch_dirty[provider]=false;
    }
    double branch_score(unsigned fold,double idle)const{
        if(idle>=branch_edges.back())return 0;
        size_t k=std::upper_bound(branch_edges.begin(),branch_edges.end(),std::max(0.,idle))-branch_edges.begin()-1;
        return branch_curves.at(fold).at(k);
    }
    void advance_branch(double now){
        if(!branch_online())return;
        for(auto it=open_old.begin();it!=open_old.end();){
            auto& g=old_gaps[*it];const auto& s=sessions[g.session];unsigned provider=s.fold/2;
            double idle=std::max(0.,clock(s,now)-g.start);
            while(g.credited+1<branch_edges.size()&&branch_edges[g.credited+1]<=idle){
                ++branch_risk[provider][g.credited++];branch_dirty[provider]=true;
            }
            if(g.credited+1==branch_edges.size())it=open_old.erase(it);else ++it;
        }
        for(unsigned p=0;p<3;++p)if(branch_dirty[p])recompute_branch(p);
    }
    void return_branch(int64_t id,double time){
        if(id<0||!open_old.erase(size_t(id)))return;
        auto& g=old_gaps[size_t(id)];unsigned provider=sessions[g.session].fold/2;
        // advance_branch has credited fully survived bins; this bin contains the event.
        if(time-g.start<branch_edges.back()){
            ++branch_risk[provider][g.credited];++branch_events[provider][g.credited];recompute_branch(provider);
        }
    }
    struct LengthReturn {double base=0;std::array<double,8> p{};std::vector<double> gaps;};
    mutable bool length_loaded=false;
    mutable std::array<std::array<LengthReturn,4>,6> length_returns;
    static unsigned length_bucket(unsigned length){const unsigned boundaries[]={32768,65536,98304,131072,196608,262144,524288};unsigned b=0;while(b<7&&length>=boundaries[b])++b;return b;}
    std::array<std::array<std::array<uint64_t,8>,4>,3> length_n{},length_yes{};
    std::array<std::array<std::vector<double>,4>,3> returned_gaps{};
    std::array<std::array<uint64_t,65>,3> risk_n{},risk_yes{};
    uint64_t online_length_returns=0,online_length_nonreturns=0,online_context_changes=0;
    void length_outcome(const Session& s,double gap,bool returned){
        auto role=s.fold/2,q=std::min(s.qgs,4u)-1,b=length_bucket(s.length);
        ++length_n[role][q][b];length_yes[role][q][b]+=returned;
        if(returned){auto&g=returned_gaps[role][q];g.insert(std::upper_bound(g.begin(),g.end(),gap),gap);++online_length_returns;}
        else ++online_length_nonreturns;
    }
    void update_length_risk(Session& s,unsigned L,bool changed){
        if(!s.episode_initialized){s.episode_initialized=true;s.episode_start=s.episode_end=L;s.risk_credited.fill(false);return;}
        if(!changed)s.episode_end=std::max(s.episode_end,L);
        unsigned role=s.fold/2;
        for(unsigned i=0;i<65;++i){unsigned pos=i*4096;
            if(s.risk_credited[i]||s.episode_start>pos||s.episode_end<=pos)continue;
            if(changed||s.episode_end>pos+8192){++risk_n[role][i];risk_yes[role][i]+=changed&&s.episode_end<=pos+8192;s.risk_credited[i]=true;}
        }
        if(changed){++online_context_changes;s.episode_start=s.episode_end=L;s.risk_credited.fill(false);}
    }
    double online_length_risk(unsigned fold,unsigned length)const{
        unsigned i=length/4096;if(i>=joint[fold].risk.size())return 0;
        return (20*joint[fold].risk[i]+risk_yes[fold/2][i])/(20+risk_n[fold/2][i]);
    }
    double length_score(unsigned fold,unsigned qgs,unsigned length,double d)const {
        if(!length_loaded){std::ifstream in(std::getenv("HOHLRAUM_LENGTH_CURVES"));for(size_t fi=0;fi<joint.size();++fi)for(auto&q:length_returns[fi]){in>>q.base;for(auto&v:q.p)in>>v;size_t n;in>>n;q.gaps.resize(n);for(auto&v:q.gaps)in>>v;}if(!in)throw std::runtime_error("length curve read");length_loaded=true;}
        if(d>=kOnlineHorizon)return 0;
        const auto&q=length_returns.at(fold).at(std::min(qgs,4u)-1);
        const unsigned boundaries[]={32768,65536,98304,131072,196608,262144,524288};unsigned bucket=0;while(bucket<7&&length>=boundaries[bucket])++bucket;
        const unsigned role=fold/2,qi=std::min(qgs,4u)-1;
        const double p=(20*q.p[bucket]+length_yes[role][qi][bucket])/(20+length_n[role][qi][bucket]);
        const double prior_survival=q.gaps.empty()?0.:double(q.gaps.end()-std::upper_bound(q.gaps.begin(),q.gaps.end(),d))/q.gaps.size();
        const auto&g=returned_gaps[role][qi];
        const double survival=(20*prior_survival+double(g.end()-std::upper_bound(g.begin(),g.end(),d)))/(20+g.size());
        const double numerator=p*survival,denominator=1-p+numerator;
        return denominator>0?numerator/denominator:0.;
    }
    double joint_wait(unsigned fold,unsigned qgs,double ds,double db,bool current=false,unsigned length=0)const{
        std::vector<double> times;
        if(online()){for(double t:edges)if(t>ds)times.push_back(t-ds);}
        else for(double t:joint[fold].q[std::min(qgs,4u)-1].t)if(t>ds)times.push_back(t-ds);
        if(!current)for(double t:branch_edges)if(t>db)times.push_back(t-db);
        std::sort(times.begin(),times.end());
        for(double dt:times)if(length_score(fold,qgs,length,ds+dt)*(current?1.:branch_score(fold,db+dt))<threshold_)return dt;
        return std::numeric_limits<double>::infinity();
    }
    struct JointScore {double admission=0,weight=0,wait=0;bool pending=false,mature=true;};
    JointScore joint_score(BlockId b,double now)const {
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
            const double v=branch_online()?length_score(s.fold,s.qgs,s.length,ds)*(cur?1.:branch_score(s.fold,db)):
                (old_mature?0.:curve_score(s.fold,s.qgs,ds));
            out.admission=std::max(out.admission,v);
            if(cur&&s.length/4096<model.risk.size())risk=std::max(risk,online_length_risk(s.fold,s.length));
            if(v>=threshold_){
                double w;
                if(branch_online())w=joint_wait(s.fold,s.qgs,ds,db,cur,s.length);
                else {w=curve_wait(s.fold,s.qgs,ds,threshold_);if(!cur)w=std::min(w,std::max(0.,120-(t-a.old_since)));}
                out.wait=std::max(out.wait,w);
            }
        }
        // The one-sided waiting objective applies only while admission accepts.
        // Already-rejected candidates have no further accept-to-reject event to
        // wait for, so they share the +infinity eviction priority.
        if(out.admission < threshold_) out.wait=std::numeric_limits<double>::infinity();
        out.pending=branch_online()?(!current&&out.admission>=threshold_):(out.pending&&!current);
        out.weight=(std::isfinite(idle_min)?idle_min:std::max(0.,now))*(1-risk);
        return out;
    }
    void observe(const Request& request) {
        advance(request.time);
        auto& s = sessions.at(request.session);
        const double time = clock(s, request.time);
        const auto ordered=index.locate(request.session,request.blocks);
        const auto next=SessionLevelIndex::normalize(ordered);
        auto& table=index.tables[request.session];
        const auto removed=SessionLevelIndex::subtract(table.current,next);
        if(!joint.empty()){
            // Visit state ranges in request order, preserving return-event order.
            if(branch_online())for(auto span:ordered)table.visit(span,[&](const Association& a){return_branch(a.old_gap,time);});
            if(s.initialized)update_length_risk(s,request.context_tokens,!removed.empty());
            int64_t gap=-1;
            if(branch_online()&&!removed.empty()){
                gap=int64_t(old_gaps.size());old_gaps.push_back({request.session,time,0});open_old.insert(size_t(gap));
            }
            for(auto span:removed){table.mark_old(span,time,gap);++index.range_updates;}
        }
        table.current=next;
        if (s.initialized && time < s.last_access)
            throw std::runtime_error("session time reversal");
        if(!s.episode_initialized){s.fold=request.fold;update_length_risk(s,request.context_tokens,false);}
        if(online()&&s.gap_open){
            const double gap=time-s.last_access;credit(s,gap);
            if(s.gap_open&&gap>=qg_gap_){length_outcome(s,gap,true);const unsigned p=s.fold/2,n=s.gap_bucket;
                ++at_risk[p][n][s.gap_credited];++events[p][n][s.gap_credited];dirty[p][n]=true;recompute(p,n);}
            s.gap_open=false;
        }
        if (!s.initialized || time - s.last_access >= qg_gap_) ++s.qgs;
        s.initialized = true;
        s.last_access = time;
        s.length = request.context_tokens;
        s.fold = request.fold;
        ++s.request_sequence;
        if(online()){s.gap_open=true;s.gap_bucket=std::min(s.qgs,4u)-1;s.gap_credited=0;}
        const Association value{request.session,time,-1,s.request_sequence,-1};
        for(auto span:next){table.assign(span,value);++index.range_updates;}
    }

    void set_threshold(double threshold) {
        if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
            throw std::invalid_argument("invalid admission threshold");
        threshold_ = threshold;
    }

private:
    double threshold_, qg_gap_, online_prior_ = 0;
    bool session_clock_ = false;
};
} // namespace hohlraum
