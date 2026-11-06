#include "replay.hpp"
#include <iomanip>
#include <iostream>
#include <string>
using namespace hohlraum;
int main(int argc,char**argv){
 if(argc!=4){std::cerr<<"usage: baseline EVENTS all|access:N|larc:G DRAM_BLOCKS\n";return 2;}
 const std::string path=argv[1],mode=argv[2];const size_t cap=std::stoull(argv[3]);
 if(mode.rfind("larc:",0)==0){auto s=replay_larc(path,cap,std::stoull(mode.substr(5)));std::cout<<std::setprecision(12)<<"{\"mode\":\""<<mode<<"\",\"refs\":"<<s.requested<<",\"global_hits\":"<<s.hits<<",\"theory_hits\":"<<s.theoretical<<",\"written_blocks\":"<<s.writes<<"}\n";return 0;}
 ReplayConfig c;c.serving=cap;c.absorption=0;c.exclusive=false;c.admission_mode=mode=="all"?"all":"accesses";if(mode.rfind("access:",0)==0)c.admission_param=std::stod(mode.substr(7));else if(mode!="all")return 2;
 AdmissionModel a;RetentionModel v;TemporalModel t;PathTimeModel p;Replay replay(c,&a,&v,&t,&p);EventReader events(path);Request q{};
 while(events.next(q))replay.access(q);
 const auto&s=replay.result();std::cout<<std::setprecision(12)<<"{\"mode\":\""<<mode<<"\",\"refs\":"<<s.requested<<",\"global_hits\":"<<s.hits<<",\"theory_hits\":"<<s.theoretical<<",\"written_blocks\":"<<s.writes<<"}\n";
}
