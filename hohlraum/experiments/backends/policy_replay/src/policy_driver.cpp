#include "hohlraum/replay.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
using namespace hohlraum;
int main(int argc,char**argv){
 if(argc!=11){std::cerr<<"usage: pure mode threshold events blocks sessions serving absorption model_dir legacy_dir\n";return 2;}
 Config c;c.measure_start=0;c.measure_end=std::stod(argv[10]);c.qg_gap_seconds=30;c.threshold=std::stod(argv[2]);c.serving_blocks=std::stoull(argv[6]);c.absorption_blocks=std::stoull(argv[7]);c.ssd_blocks=std::stoull(argv[4]);c.write_budget_blocks=std::numeric_limits<uint64_t>::max();std::string mode=argv[1];c.candidates=Candidates::head_tail;c.ranking=(mode=="lru_nocopy"||mode=="lru_copy")?Ranking::lru:Ranking::joint;c.allow_ssd_copies_in_absorption=(mode=="full_copy"||mode=="lru_copy");
 setenv("HOHLRAUM_JOINT_MODELS",argv[8],1);if(!std::getenv("HOHLRAUM_MODEL_COUNT")) throw std::runtime_error("set model count");setenv("HOHLRAUM_ONLINE_PRIOR","20",1);
 if(!std::getenv("HOHLRAUM_BRANCH_PRIOR")) throw std::runtime_error("set branch prior");if(!std::getenv("HOHLRAUM_LENGTH_CURVES")) throw std::runtime_error("set workload length curves");
 std::vector<Model> models;for(int i=0;i<6;++i)models.push_back(Model::load(std::string(argv[9])+"/legacy-interface-"+std::to_string(i)+".model"));
 Replay replay(std::stoul(argv[4]),std::stoul(argv[5]),c,std::move(models),true);
 std::ifstream input(argv[3]);if(!input)throw std::runtime_error("cannot open events");std::string line;size_t rows=0;
 while(std::getline(input,line)){
  std::istringstream in(line);std::string ts,ss,fs,ns,ids;std::getline(in,ts,'\t');std::getline(in,ss,'\t');std::getline(in,fs,'\t');std::getline(in,ns,'\t');std::getline(in,ids,'\t');
  Request q;q.time=std::stod(ts);if(q.time>c.measure_end)break;q.session=std::stoul(ss);q.fold=std::stoul(fs);q.context_tokens=std::stoul(ns);
  std::istringstream bi(ids);std::string item;while(std::getline(bi,item,','))if(!item.empty())q.blocks.push_back(std::stoul(item));
  q.accessible_blocks=0;for(auto b:q.blocks){if(replay.location(b)==Location::absent&&!replay.has_ssd(b))break;++q.accessible_blocks;}
  replay.process(q);++rows;
  if(rows%10000==0)std::cerr<<"processed "<<rows<<"\n";
 }
 const auto&m=replay.metrics();std::cerr<<"duplicate_area "<<m.duplicate_area<<"\n";std::cout<<"{\"mode\":\""<<argv[1]<<"\",\"threshold\":"<<c.threshold<<",\"requests\":"<<m.requests<<",\"refs\":"<<m.refs<<",\"global_hits\":"<<m.global_hits<<",\"theory_hits\":"<<m.theory_hits<<",\"written_blocks\":"<<m.writes<<",\"ssd_reads\":"<<m.ssd_reads<<",\"dram_hits\":"<<m.dram_hits<<",\"ssd_evictions\":"<<m.ssd_evictions<<",\"suffix_drops\":"<<m.dram_suffix_drops<<",\"fallbacks\":"<<m.budget_fallbacks<<"}\n";
}
