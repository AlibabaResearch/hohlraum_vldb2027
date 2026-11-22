#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <stdexcept>

namespace hohlraum {

inline double return_posterior(double probability,double survival) {
    // A cohort with no negative observations has no evidence of termination.
    if(probability>=1.)return 1.-1e-12;
    const double mass=probability*survival;
    return std::min(1.-1e-12,mass/(1.-probability+mass));
}


// Empirical positive-return counts above the join; an anchored log-normal
// survival tail below it. Negative/non-return counts are never changed.
struct ReturnTailFit {
    size_t count=std::numeric_limits<size_t>::max(), minimum=0;
    double join=0, anchor=0, mu=0, sigma=0, at_join=0, mean=0, cached_weight=-1;
    bool usable=false, exponential=false, cached_stable_intervals=false;
    void update(const std::vector<double>& offline,const std::vector<double>& online,
                size_t min_tail,double qg_gap,double offline_weight=1.,bool stable_intervals=false) {
        const size_t n=offline.size()+online.size();
        if(count==n&&minimum==min_tail&&cached_weight==offline_weight&&cached_stable_intervals==stable_intervals)return;
        count=n;minimum=min_tail;cached_weight=offline_weight;cached_stable_intervals=stable_intervals;usable=false;
        if(min_tail<2||n<2)return;
        std::vector<double> gaps;gaps.reserve(n);
        gaps.insert(gaps.end(),offline.begin(),offline.end());
        gaps.insert(gaps.end(),online.begin(),online.end());
        std::sort(gaps.begin(),gaps.end());
        join=n>min_tail?gaps[n-min_tail]:qg_gap;
        join=std::max(join,qg_gap);
        // Clock subtraction may split identical gaps by a few ulps.
        const double boundary=stable_intervals?join-1e-8:join;
        anchor=offline_weight*double(offline.end()-std::lower_bound(offline.begin(),offline.end(),boundary))+
               double(online.end()-std::lower_bound(online.begin(),online.end(),boundary));
        double sum=0,sum2=0,total=0;mean=0;
        auto accumulate=[&](const std::vector<double>& values,double weight){
            for(double d:values){if(!(d>0)||!std::isfinite(d))return false;
                const double z=std::log(d);sum+=weight*z;sum2+=weight*z*z;
                mean+=weight*d;total+=weight;}
            return true;
        };
        if(!accumulate(offline,offline_weight)||!accumulate(online,1.)||!(total>0))return;
        mu=sum/total;sigma=std::sqrt(std::max(0.,sum2/total-mu*mu));mean/=total;
        exponential=sigma<1e-9;
        at_join=exponential?1.:survival(join);
        if(!(at_join>0)){exponential=true;at_join=1.;}
        usable=anchor>0;
    }
    double survival(double d)const {
        return .5*std::erfc((std::log(d)-mu)/(sigma*std::sqrt(2.)));
    }
    double tail(double d,double empirical)const {
        if(!usable||d<=join)return empirical;
        const double ratio=exponential?std::exp(-(d-join)/std::max(1.,mean)):survival(d)/at_join;
        return std::clamp(anchor*ratio,0.,anchor);
    }
};
}
