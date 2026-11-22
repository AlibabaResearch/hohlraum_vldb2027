#include "hohlraum/replay.hpp"
#include "trace_input.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
using namespace hohlraum;
int main(int argc,char**argv){
 if(argc!=10){std::cerr<<"usage: pure mode threshold events blocks sessions serving absorption model_dir end_seconds\n";return 2;}
 Config c;c.measure_end=std::stod(argv[9]);c.qg_gap_seconds=60;c.threshold=std::stod(argv[2]);c.serving_blocks=std::stoull(argv[6]);c.absorption_blocks=std::stoull(argv[7]);c.ssd_blocks=std::stoull(argv[4]);c.write_budget_blocks=std::numeric_limits<uint64_t>::max();std::string mode=argv[1];c.ranking=(mode=="lru_nocopy"||mode=="lru_copy")?Ranking::lru:Ranking::joint;c.allow_ssd_copies_in_absorption=(mode=="full_copy"||mode=="lru_copy");
 setenv("HOHLRAUM_JOINT_MODELS",argv[8],1);if(!std::getenv("HOHLRAUM_MODEL_COUNT")) throw std::runtime_error("set model count");setenv("HOHLRAUM_ONLINE_UPDATES","1",1);
 if(!std::getenv("HOHLRAUM_BRANCH_PRIOR")) throw std::runtime_error("set branch prior");if(!std::getenv("HOHLRAUM_LENGTH_CURVES")) throw std::runtime_error("set workload length curves");
 Replay replay(std::stoul(argv[4]),std::stoul(argv[5]),c);
 std::ifstream input(argv[3]);if(!input)throw std::runtime_error("cannot open events");size_t rows=0;
 Request q;
 while(read_request(input,c.measure_end,q)){
  replay.process(q);++rows;
  if(rows%10000==0)std::cerr<<"processed "<<rows<<"\n";
 }
 const auto&m=replay.metrics();std::cerr<<"duplicate_area "<<m.duplicate_area<<"\n";std::cout<<"{\"mode\":\""<<argv[1]<<"\",\"threshold\":"<<c.threshold<<",\"requests\":"<<m.requests<<",\"refs\":"<<m.refs<<",\"global_hits\":"<<m.global_hits<<",\"theory_hits\":"<<m.theory_hits<<",\"written_blocks\":"<<m.writes<<",\"ssd_reads\":"<<m.ssd_reads<<",\"dram_hits\":"<<m.dram_hits<<",\"ssd_evictions\":"<<m.ssd_evictions<<",\"suffix_drops\":"<<m.dram_suffix_drops<<",\"discard_on_entry\":"<<m.discard_on_entry<<",\"prefix_writes\":"<<m.prefix_writes<<",\"prefix_read_releases\":"<<m.prefix_read_releases<<",\"fallbacks\":"<<m.budget_fallbacks<<"}\n";
}
