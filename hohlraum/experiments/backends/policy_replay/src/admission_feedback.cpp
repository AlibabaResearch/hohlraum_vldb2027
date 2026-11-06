#include "hohlraum/replay.hpp"
#include "admission_pressure.hpp"
#include <algorithm>
#include <chrono>
#include <deque>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
using namespace hohlraum;
int main(int argc,char**argv){
 if(argc!=15){std::cerr<<"usage: feedback outdir initial_threshold events blocks sessions mode workload variant replay_end budget dram staging ssd budget_period_seconds\n";return 2;}
 std::string outdir=argv[1];const double initial=std::stod(argv[2]);const uint64_t blocks=std::stoull(argv[4]);const unsigned sessions=std::stoul(argv[5]);const bool dynamic=std::string(argv[6])=="dynamic";
 // The starting threshold is the restoration ceiling; no separate ROI target.
 const double normal=initial;
 if(!std::isfinite(initial)||initial<0||initial>1||!std::isfinite(normal)||normal<0||normal>1)
   throw std::invalid_argument("initial and normal thresholds must be in [0,1]");
 const bool playing=std::string(argv[7])=="playing";const std::string variant=argv[8];
 const double horizon=std::stod(argv[9]),period=300,window=1800;
 // Single accounting period starting at logical time zero. The replay may
 // end earlier; its endpoint is not the controller's budget deadline.
 const double budget_period_seconds=std::stod(argv[14]);
 if(!std::isfinite(budget_period_seconds) || budget_period_seconds<=0 ||
    !std::isfinite(horizon) || horizon<=0 || horizon>budget_period_seconds)
   throw std::invalid_argument("require 0 < replay_end <= budget_period_seconds; multi-period renewal is not implemented");
 Config config;config.measure_start=0;config.measure_end=horizon;config.qg_gap_seconds=30;config.threshold=initial;
 config.absorption_blocks=std::stoull(argv[12]);config.serving_blocks=std::stoull(argv[11])-config.absorption_blocks;config.ssd_blocks=std::stoull(argv[13]);config.allow_ssd_copies_in_absorption=false;
 config.write_budget_blocks=std::stoull(argv[10]);
 config.candidates=Candidates::head_tail;config.ranking=Ranking::joint;
 // Portable paths supplied by the artifact runner; policy/control logic unchanged.
 auto required=[](const char* key){const char* v=std::getenv(key);if(!v||!*v)throw std::runtime_error(std::string("missing ")+key);return std::string(v);};
 const std::string root=required("HOHLRAUM_JOINT_MODELS");
 const std::string legacy=required("HOHLRAUM_LEGACY_MODELS");
 required("HOHLRAUM_MODEL_COUNT");required("HOHLRAUM_BRANCH_PRIOR");required("HOHLRAUM_LENGTH_CURVES");
 setenv("HOHLRAUM_ONLINE_PRIOR","20",1);
 std::vector<Model> models;
 for(int i=0;i<6;++i)models.push_back(Model::load(legacy+"/legacy-interface-"+std::to_string(i)+".model"));
 auto begin=std::chrono::steady_clock::now();
 Replay actual(blocks,sessions,config,models,true);
 std::deque<std::pair<double,uint64_t>> history{{0,0}};
 std::ofstream log(outdir+"/trajectory.jsonl");log<<std::setprecision(17);
 std::unordered_map<BlockId,unsigned> written_bins;
 std::unordered_map<BlockId,double> candidate_scores;
 struct RecentOffer {double time;unsigned bin;double score;};
 std::unordered_map<BlockId,RecentOffer> recent_offers;
 std::set<std::pair<double,BlockId>> offer_expiry;
 std::unordered_set<BlockId> pending_offers;
 // Period-to-date workload reference. One latest score per unique offered block;
 // repeated eviction attempts do not manufacture new write demand.
 std::unordered_map<BlockId,unsigned> cumulative_offer_bins;
 std::array<uint64_t,21> cumulative_counts{};
 uint64_t candidate_events=0,repeated_candidate_events=0;
 std::array<uint64_t,21> observed_writes{},observed_hits{},offer_counts{};
 std::array<double,21> write_times{},exposure{};
 auto score_bin=[](double score){return unsigned(std::clamp(int(std::floor(score*20)),0,20));};
 auto settle=[&](double now){
  for(unsigned j=0;j<=20;++j)exposure[j]=std::max(0.,now*observed_writes[j]-write_times[j]);
  while(!offer_expiry.empty()&&offer_expiry.begin()->first<=now-window){
   auto b=offer_expiry.begin()->second;auto it=recent_offers.find(b);
   if(it==recent_offers.end())throw std::runtime_error("offer expiry missing record");
   --offer_counts[it->second.bin];recent_offers.erase(it);pending_offers.erase(b);offer_expiry.erase(offer_expiry.begin());
  }
 };
 double first_candidate=-1;
 actual.on_terminal_candidate=[&](double time,BlockId b,double score){
  if(first_candidate<0)first_candidate=time;
  ++candidate_events;auto j=score_bin(score);
  auto old=cumulative_offer_bins.find(b);
  if(old!=cumulative_offer_bins.end())--cumulative_counts[old->second];
  cumulative_offer_bins[b]=j;++cumulative_counts[j];
  auto it=recent_offers.find(b);
  if(it!=recent_offers.end()){++repeated_candidate_events;--offer_counts[it->second.bin];offer_expiry.erase({it->second.time,b});}
  recent_offers[b]={time,j,score};offer_expiry.insert({time,b});++offer_counts[j];
  pending_offers.insert(b);candidate_scores[b]=score;
 };
 actual.on_ssd_read=[&](double time,BlockId b){auto it=written_bins.find(b);if(it==written_bins.end())throw std::runtime_error("read without write record");++observed_hits[it->second];};
 std::ofstream yieldtrace(outdir+"/yield-trajectory.jsonl");
 std::ofstream scoretrace(outdir+"/score-trajectory.jsonl");
 int streak=0;double pending_target=-1;unsigned switches=0;double next=period,exhausted=-1;
 auto control=[&](double now){
  settle(now);
  uint64_t recent_count=0;for(auto count:offer_counts)recent_count+=count;
  if(recent_count!=recent_offers.size())throw std::runtime_error("dedup size mismatch");
  for(auto b:pending_offers)if(written_bins.count(b))throw std::runtime_error("written KV remains pending");
  yieldtrace<<std::setprecision(17)<<"{\"t\":"<<now<<",\"unique_recent_count\":"<<recent_offers.size()<<",\"pending_count\":"<<pending_offers.size()<<",\"candidate_events\":"<<candidate_events<<",\"repeated_candidate_events\":"<<repeated_candidate_events<<",\"offers\":[";
  for(unsigned j=0;j<=20;++j){if(j)yieldtrace<<",";yieldtrace<<offer_counts[j];}yieldtrace<<"],\"observed_writes\":[";
  for(unsigned j=0;j<=20;++j){if(j)yieldtrace<<",";yieldtrace<<observed_writes[j];}yieldtrace<<"],\"observed_hits\":[";
  for(unsigned j=0;j<=20;++j){if(j)yieldtrace<<",";yieldtrace<<observed_hits[j];}yieldtrace<<"],\"exposure_block_seconds\":[";
  for(unsigned j=0;j<=20;++j){if(j)yieldtrace<<",";yieldtrace<<exposure[j];}yieldtrace<<"]}\n";
  const auto&m=actual.metrics();uint64_t used=m.writes;
  uint64_t remaining=used<config.write_budget_blocks?config.write_budget_blocks-used:0;
  history.emplace_back(now,used);
  while(history.size()>1 && history[1].first<=now-window)history.pop_front();
  double span=now-std::max(history.front().first,first_candidate);
  uint64_t window_writes=used-history.front().second;
  double ratio=-1,long_normal_projected=-1,long_current_projected=-1,long_span=first_candidate<0?0:now-first_candidate;
  double before=actual.admission_threshold();std::string reason="hold";
  if(now>=budget_period_seconds)reason="budget_period_end";
  else if(!remaining)reason="budget_exhausted";
  else if(!dynamic)reason="fixed";
  else if(first_candidate<0 || span<period)reason="warmup";
  else{
   ratio=double(window_writes)/span*(budget_period_seconds-now)/remaining;
   std::vector<double> scores; scores.reserve(recent_offers.size());
   for(const auto& item:recent_offers)scores.push_back(item.second.score);
   std::sort(scores.begin(),scores.end());
   auto estimate=[&](double threshold){
    double count=scores.end()-std::lower_bound(scores.begin(),scores.end(),threshold);
    return std::tuple<double,double,bool>(count,0,count>0);
   };
   scoretrace<<std::setprecision(17)<<"{\"t\":"<<now<<",\"scores\":[";
   for(size_t i=0;i<scores.size();){if(i)scoretrace<<",";auto end=std::upper_bound(scores.begin()+i,scores.end(),scores[i]);scoretrace<<"["<<scores[i]<<","<<(end-(scores.begin()+i))<<"]";i=end-scores.begin();}scoretrace<<"]}\n";
   int direction=0;double target=before;
   // Long-run offered demand represents current load/workload pressure.
   // Surplus permits relaxation; pressure restores only toward the starting value.
   // Recent quiet windows cannot erase it; recent write overload can veto relaxing.
   std::array<double,21> rates{};uint64_t suffix=0;
   for(int j=20;j>=0;--j){suffix+=cumulative_counts[j];rates[j]=double(suffix)/std::max(1.,long_span);}
   const auto decision=admission_pressure::select(before,normal,rates,remaining,budget_period_seconds-now,ratio);
   long_normal_projected=decision.normal_projected;
   long_current_projected=rates[unsigned(std::clamp(int(std::ceil(before*20-1e-9)),0,20))]*(budget_period_seconds-now);
   if(long_span<window){reason="warmup_window";}
   else{target=decision.target;reason=decision.pressure?"approach_normal_pending":"long_demand_surplus_pending";}
   if(target<before-1e-12)direction=-1;
   else if(target>before+1e-12)direction=1;
   else reason="hold_target";
   if(direction==0){streak=0;pending_target=-1;}
   else{
    streak=(pending_target>=0 && ((pending_target>before)==(target>before)))?streak+1:1;
    pending_target=target;
    if(streak>=2){
     if(target<0 || target>std::max(initial,normal)+1e-12)throw std::runtime_error("threshold outside target bounds");
     if(target!=before){actual.set_admission_threshold(target);++switches;reason=decision.pressure?"approach_normal":"long_demand_surplus";}
     streak=0;pending_target=-1;
    }
   }
  }
  log<<"{\"t\":"<<now<<",\"threshold_before\":"<<before<<",\"threshold\":"<<actual.admission_threshold()<<",\"reason\":\""<<reason<<"\",\"budget_ratio\":"<<ratio<<",\"long_normal_projected\":"<<long_normal_projected<<",\"long_current_projected\":"<<long_current_projected<<",\"long_span\":"<<long_span<<",\"unique_cumulative_candidates\":"<<cumulative_offer_bins.size()<<",\"window_writes\":"<<window_writes<<",\"window_span\":"<<span<<",\"writes\":"<<used<<",\"remaining\":"<<remaining<<",\"refs\":"<<m.refs<<",\"global_hits\":"<<m.global_hits<<",\"ssd_hits\":"<<m.ssd_reads<<"}\n";log.flush();
 };
 std::ifstream input(argv[3]);
 if(!input)throw std::runtime_error("cannot open trace");
 std::string line;
 while(std::getline(input,line)){
  std::istringstream in(line);std::string ts,ss,fs,ns,ids;
  std::getline(in,ts,'\t');std::getline(in,ss,'\t');std::getline(in,fs,'\t');std::getline(in,ns,'\t');std::getline(in,ids,'\t');
  Request q;q.time=std::stod(ts);if(q.time>=horizon)break;
  while(next<=q.time&&next<=horizon){control(next);next+=period;}
  q.session=std::stoul(ss);q.fold=std::stoul(fs);q.context_tokens=std::stoul(ns);
  std::istringstream blocks(ids);std::string id;
  while(std::getline(blocks,id,','))if(!id.empty())q.blocks.push_back(std::stoul(id));
  q.accessible_blocks=0;
  for(auto b:q.blocks){if(actual.location(b)==Location::absent&&!actual.has_ssd(b))break;++q.accessible_blocks;}
  settle(q.time);
  actual.process(q);
  for(auto b:actual.last_writes()){
   auto it=candidate_scores.find(b);if(it==candidate_scores.end())throw std::runtime_error("write without terminal score");
   auto j=score_bin(it->second);if(written_bins.count(b))throw std::runtime_error("unexpected rewrite");
   written_bins[b]=j;++observed_writes[j];write_times[j]+=q.time;pending_offers.erase(b);
  }
  candidate_scores.clear();

  if(exhausted<0&&actual.metrics().writes>=config.write_budget_blocks)exhausted=q.time;
 }
 while(next<=horizon){control(next);next+=period;}
 std::ofstream yieldlog(outdir+"/yield-bins.json");yieldlog<<"[";
 for(unsigned j=0;j<=20;++j){if(j)yieldlog<<",";yieldlog<<"{\"bin\":"<<j<<",\"observed_writes\":"<<observed_writes[j]<<",\"observed_ssd_hits\":"<<observed_hits[j]<<"}";}yieldlog<<"]";
 const auto&m=actual.metrics();
 uint64_t tracked_writes=0,tracked_hits=0;for(unsigned j=0;j<=20;++j){tracked_writes+=observed_writes[j];tracked_hits+=observed_hits[j];}
 if(tracked_writes!=m.writes||tracked_hits!=m.ssd_reads)throw std::runtime_error("yield accounting mismatch");
 std::cout<<std::setprecision(17)<<"{\"initial_threshold\":"<<initial<<",\"final_threshold\":"<<actual.admission_threshold()<<",\"budget_blocks\":"<<config.write_budget_blocks<<",\"exhausted_at\":"<<exhausted<<",\"switches\":"<<switches<<",\"requests\":"<<m.requests<<",\"refs\":"<<m.refs<<",\"global_hits\":"<<m.global_hits<<",\"dram_hits\":"<<m.dram_hits<<",\"ssd_reads\":"<<m.ssd_reads<<",\"theory_hits\":"<<m.theory_hits<<",\"writes\":"<<m.writes<<",\"ssd_evictions\":"<<m.ssd_evictions<<",\"suffix_drops\":"<<m.dram_suffix_drops<<",\"fallbacks\":"<<m.budget_fallbacks<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<"}\n";
}
