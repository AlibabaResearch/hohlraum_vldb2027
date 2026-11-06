#pragma once
#include <algorithm>
#include <map>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>

namespace hohlraum {
// Session-local stable positions are lookup handles, not per-KV history.
// Equal-history positions share one range record. No CLOCK or snapshots yet.
class SessionLevelIndex {
public:
    struct State {
        uint32_t session=0;
        double last_access=0, old_since=-1;
        uint64_t request_sequence=0;
        int64_t old_gap=-1;
        bool operator==(const State& b)const {
            return session==b.session && last_access==b.last_access &&
                old_since==b.old_since && request_sequence==b.request_sequence && old_gap==b.old_gap;
        }
    };
    using Span=std::pair<size_t,size_t>;
    struct Location {uint32_t session;size_t position;};
    struct BranchTable {
        std::unordered_map<uint32_t,size_t> positions;
        std::map<size_t,State> ranges;
        std::vector<Span> current;
        size_t size=0;
        auto split(size_t p) {
            auto it=ranges.lower_bound(p);
            if(it!=ranges.end()&&it->first==p)return it;
            State value=it==ranges.begin()?State{}:std::prev(it)->second;
            return ranges.emplace_hint(it,p,value);
        }
        template<class F> void visit(Span span,F fn)const {
            auto it=std::prev(ranges.upper_bound(span.first));
            while(it!=ranges.end()&&it->first<span.second){fn(it->second);++it;}
        }
        void assign(Span span,const State& value) {
            split(span.second);auto begin=split(span.first);
            ranges.erase(begin,ranges.lower_bound(span.second));
            auto it=ranges.emplace(span.first,value).first;
            auto after=std::next(it);
            if(after!=ranges.end()&&after->first<size&&after->second==it->second)ranges.erase(after);
            if(it!=ranges.begin()&&std::prev(it)->second==it->second)ranges.erase(it);
        }
        void mark_old(Span span,double time,int64_t gap) {
            split(span.second);auto it=split(span.first);
            for(;it!=ranges.end()&&it->first<span.second;++it){it->second.old_since=time;it->second.old_gap=gap;}
        }
    };
    std::vector<BranchTable> tables;
    std::unordered_map<uint32_t,std::vector<Location>> locations;
    uint64_t range_updates=0;
    static std::vector<Span> normalize(std::vector<Span> spans) {
        std::sort(spans.begin(),spans.end());std::vector<Span> out;
        for(auto x:spans){if(!out.empty()&&x.first<=out.back().second)out.back().second=std::max(out.back().second,x.second);else out.push_back(x);}
        return out;
    }
    std::vector<Span> locate(uint32_t session,const std::vector<uint32_t>& blocks) {
        if(tables.size()<=session)tables.resize(session+1);
        auto& t=tables[session];std::vector<Span> spans;
        for(auto block:blocks){
            auto [it,fresh]=t.positions.emplace(block,t.size);
            if(fresh){locations[block].push_back({session,t.size});t.ranges[t.size]=State{session};++t.size;}
            const size_t pos=it->second;
            if(!spans.empty()&&spans.back().second==pos)++spans.back().second;
            else spans.push_back({pos,pos+1});
        }
        return spans;
    }
    static std::vector<Span> subtract(const std::vector<Span>& old,const std::vector<Span>& next) {
        std::vector<Span> out;size_t j=0;
        for(auto [start,end]:old){
            while(j<next.size()&&next[j].second<=start)++j;
            for(size_t k=j;k<next.size()&&next[k].first<end;++k){
                if(start<next[k].first)out.push_back({start,std::min(end,next[k].first)});
                start=std::max(start,next[k].second);if(start>=end)break;
            }
            if(start<end)out.push_back({start,end});
        }
        return out;
    }
    std::vector<State> owners(uint32_t block)const {
        std::vector<State> out;
        auto it=locations.find(block);if(it==locations.end())throw std::out_of_range("SLI unknown KV handle");
        for(auto loc:it->second){const auto& t=tables[loc.session];out.push_back(std::prev(t.ranges.upper_bound(loc.position))->second);}
        return out;
    }
    size_t range_count()const {size_t n=0;for(const auto& t:tables)for(const auto& x:t.ranges)n+=x.first<t.size;return n;}
};
}
