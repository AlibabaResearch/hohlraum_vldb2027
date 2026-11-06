#pragma once
#include <array>
#include <algorithm>
#include <cmath>
namespace admission_pressure {
struct Decision { double target; bool pressure; double normal_projected; };
inline Decision select(double before,double normal,const std::array<double,21>& rates,
                       double remaining,double seconds_left,double recent_ratio) {
 const unsigned k=unsigned(std::clamp(int(std::ceil(normal*20-1e-9)),0,20));
 const double projected=rates[k]*seconds_left;
 const bool pressure=recent_ratio>1.2 || projected>remaining;
 double target=before;
 if(pressure){
  if(before<normal)target=std::min(normal,before+.05);
  else if(before>normal)target=std::max(normal,before-.05);
 }else{
  target=normal;
  for(unsigned j=0;j<=20 && j*.05<=normal+1e-12;++j)
   if(rates[j]*seconds_left<=remaining){target=j*.05;break;}
 }
 return {target,pressure,projected};
}
}
