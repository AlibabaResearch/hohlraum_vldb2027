#pragma once
#include <algorithm>
#include <vector>
#include <stdexcept>
namespace hohlraum {
// Preference shortlist first; existing wait/admission ordering second.
// Odd cardinalities round upward so at least one candidate remains.
template<class Id, class Scores>
Id preferred_exit(std::vector<Id> candidates, const Scores& scores) {
    if(candidates.empty()) throw std::runtime_error("empty preference candidates");
    std::sort(candidates.begin(),candidates.end(),[&](Id a,Id b){
        if(scores.at(a).weight != scores.at(b).weight)
            return scores.at(a).weight > scores.at(b).weight;
        return a < b;
    });
    candidates.resize((candidates.size()+1)/2);
    return *std::max_element(candidates.begin(),candidates.end(),[&](Id a,Id b){
        if(scores.at(a).wait != scores.at(b).wait)
            return scores.at(a).wait < scores.at(b).wait;
        if(scores.at(a).admission != scores.at(b).admission)
            return scores.at(a).admission > scores.at(b).admission;
        return a < b;
    });
}
}
