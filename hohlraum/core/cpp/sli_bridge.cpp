#include "lifecycle_history.hpp"
#include <string>
#include <map>
#include <tuple>
using namespace hohlraum;
static thread_local std::string error;
extern "C" {
const char* hh_error(){return error.c_str();}
void* hh_create_joint(double threshold){
 try {return new History(1,1,threshold,30.);}
 catch(const std::exception& e){error=e.what();return nullptr;}
}
unsigned hh_sli_version(){return 1;}
unsigned long long hh_sli_ranges(void* p){return static_cast<History*>(p)->index.range_count();}
unsigned long long hh_sli_updates(void* p){return static_cast<History*>(p)->index.range_updates;}
void hh_destroy(void* p){delete static_cast<History*>(p);}
int hh_clock(void* p,unsigned sid,double t,int complete){try{auto& h=*static_cast<History*>(p);if(h.sessions.size()<=sid)h.sessions.resize(sid+1);if(complete)h.complete(sid,t);else h.arrive(sid,t);return 0;}catch(const std::exception& e){error=e.what();return -1;}}
int hh_observe(void* p,unsigned sid,unsigned fold,unsigned length,double t,const unsigned* ids,unsigned n){try{auto& h=*static_cast<History*>(p);if(h.sessions.size()<=sid)h.sessions.resize(sid+1);Request r;r.time=t;r.session=sid;r.fold=fold;r.context_tokens=length;r.blocks.assign(ids,ids+n);h.observe(r);return 0;}catch(const std::exception& e){error=e.what();return -1;}}
int hh_advance(void* p,double t){try{static_cast<History*>(p)->advance(t);return 0;}catch(const std::exception& e){error=e.what();return -1;}}
int hh_threshold(void* p,double v){try{static_cast<History*>(p)->set_threshold(v);return 0;}catch(const std::exception& e){error=e.what();return -1;}}
int hh_query(void* p,double t,const unsigned* ids,unsigned n,double* out){try{auto& h=*static_cast<History*>(p);using Sig=std::vector<std::tuple<SessionId,double,double,uint64_t,int64_t>>;
std::map<Sig,History::JointScore> memo;
for(unsigned i=0;i<n;++i){Sig sig;const auto owners=h.owners(ids[i]);sig.reserve(owners.size());
for(const auto& a:owners)sig.emplace_back(a.session,a.last_access,a.old_since,a.request_sequence,a.old_gap);
auto it=memo.find(sig);if(it==memo.end())it=memo.emplace(std::move(sig),h.joint_score(ids[i],t)).first;
const auto& s=it->second;out[5*i]=s.admission;out[5*i+1]=s.wait;out[5*i+2]=s.weight;out[5*i+3]=s.pending;out[5*i+4]=s.mature;}return 0;}catch(const std::exception& e){error=e.what();return -1;}}
}
