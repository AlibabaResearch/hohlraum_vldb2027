#pragma once

// admission/admission.hpp
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace hohlraum {
inline void probability(double p) {
    if (!std::isfinite(p) || p < 0 || p > 1)
        throw std::invalid_argument("invalid probability");
}
inline double logistic(double x) {
    if (x >= 0) return 1 / (1 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1 + e);
}
struct Observation {
    double silence_seconds;     // 当前时刻减去末次请求到达时间，单位秒。
    unsigned query_groups;     // 已经观测到的QG数；不是最终QG总数。
    double conditional_growth; // 训练曲线给出的r_z(L)，不是当前样本真实剩余长度。
};
struct SessionModel {
    // 两行对应Q=1、Q>1；四列为截距、log静默、logQ、条件长度概率。
    // training.hpp负责拟合，训练结果需显式加载，绝不查找历史.fit文件。
    std::array<std::array<double, 4>, 2> beta;
    double predict(const Observation& o) const {
        if (!std::isfinite(o.silence_seconds) || o.silence_seconds < 0 || !o.query_groups)
            throw std::invalid_argument("invalid observed history");
        probability(o.conditional_growth);
        const auto& b = beta[o.query_groups > 1];
        return logistic(b[0] + b[1] * std::log1p(o.silence_seconds / 30)
            + b[2] * std::log1p(o.query_groups) + b[3] * o.conditional_growth);
    }
};
// 这是当前状态，不是永久死亡标签；重新回到路径时应恢复为current。
enum class Branch { current, displaced_by_compaction, old_suffix };
inline double session_admission(double session_return, Branch state, double branch_return) {
    probability(session_return);
    probability(branch_return);
    const double q = state == Branch::current ? 1 :
        state == Branch::displaced_by_compaction ? 0 : branch_return;
    return session_return * q;
}
inline double admission(const std::vector<double>& session_scores) {
    // 任一 session 仍有较高复用概率，就不能因其他 session 拒写而丢掉共享数据。
    if (session_scores.empty()) throw std::invalid_argument("missing session claims");
    for (double p : session_scores) probability(p);
    return *std::max_element(session_scores.begin(), session_scores.end());
}
inline bool write(double p, double threshold) {
    // 全部模块复用同一边界约定：恰好等于阈值仍写入SSD。
    probability(p); probability(threshold);
    return p >= threshold;
}
} // namespace hohlraum


// training/logistic.hpp
#include <numeric>
#include <ostream>
#include <istream>

namespace hohlraum {
// 所有训练行的 x 都不含截距。weight 是采样权重，不是块数。
// 这里不读取 trace，更不会把测试样本或未来标签变成在线特征。
struct LogisticRow { std::vector<double> x; double y, weight; };
struct LogisticFit {
    std::vector<double> beta; // 已还原到原始特征尺度，beta[0] 为截距。
    double predict(const std::vector<double>& x) const {
        if (beta.size() != x.size()+1) throw std::invalid_argument("feature dimension");
        double v=beta[0];
        for (size_t i=0;i<x.size();++i) {
            if (!std::isfinite(x[i])) throw std::invalid_argument("nonfinite feature");
            v+=beta[i+1]*x[i];
        }
        return logistic(v);
    }
    void save(std::ostream& out) const {
        out<<beta.size(); for(double v:beta) out<<' '<<v; out<<'\n';
    }
    static LogisticFit load(std::istream& in) {
        size_t n=0; in>>n;
        if(!in || n<1 || n>100) throw std::runtime_error("invalid fit size");
        LogisticFit f; f.beta.resize(n);
        for(double& v:f.beta) {
            in>>v; if(!in || !std::isfinite(v)) throw std::runtime_error("invalid coefficient");
        }
        return f;
    }
};

// 加权 BCE + 非截距 L2。训练集内标准化，梯度下降配回溯线搜索。
// 不搜索测试准确率，不用固定 beta 冒充训练结果。
inline LogisticFit fit_logistic(const std::vector<LogisticRow>& rows,
                                double l2=0.001, unsigned iterations=2000) {
    if(rows.empty() || l2<0 || !std::isfinite(l2)) throw std::invalid_argument("empty/invalid training");
    const size_t d=rows[0].x.size();
    std::vector<double> mean(d), scale(d), b(d+1);
    double mass=0, positive=0;
    for(const auto& r:rows) {
        probability(r.y);
        if(r.x.size()!=d || !std::isfinite(r.weight) || r.weight<=0)
            throw std::invalid_argument("training row");
        mass+=r.weight; positive+=r.weight*r.y;
        for(size_t j=0;j<d;++j) {
            if(!std::isfinite(r.x[j])) throw std::invalid_argument("nonfinite training feature");
            mean[j]+=r.weight*r.x[j];
        }
    }
    for(double& v:mean) v/=mass;
    for(const auto& r:rows) for(size_t j=0;j<d;++j)
        scale[j]+=r.weight*std::pow(r.x[j]-mean[j],2);
    for(double& v:scale) {v=std::sqrt(v/mass); if(v<1e-12) v=1;}
    auto standardized=rows;
    for(auto& r:standardized) for(size_t j=0;j<d;++j) r.x[j]=(r.x[j]-mean[j])/scale[j];
    const double prior=std::clamp(positive/mass,1e-6,1-1e-6);
    b[0]=std::log(prior/(1-prior));
    auto loss=[&](const std::vector<double>& v) {
        double total=0;
        for(const auto& r:standardized) {
            double z=v[0]; for(size_t j=0;j<d;++j) z+=v[j+1]*r.x[j];
            total+=r.weight*(std::max(z,0.)-r.y*z+std::log1p(std::exp(-std::abs(z))));
        }
        total/=mass; for(size_t j=1;j<=d;++j) total+=.5*l2*v[j]*v[j];
        return total;
    };
    for(unsigned it=0;it<iterations;++it) {
        std::vector<double> grad(d+1);
        for(const auto& r:standardized) {
            double z=b[0]; for(size_t j=0;j<d;++j) z+=b[j+1]*r.x[j];
            const double e=r.weight*(logistic(z)-r.y)/mass;
            grad[0]+=e; for(size_t j=0;j<d;++j) grad[j+1]+=e*r.x[j];
        }
        double norm=0; for(size_t j=0;j<=d;++j) {if(j)grad[j]+=l2*b[j]; norm+=grad[j]*grad[j];}
        if(norm<1e-14) break;
        const double before=loss(b); double step=1; bool accepted=false;
        for(unsigned search=0;search<30;++search) {
            auto candidate=b;
            for(size_t j=0;j<=d;++j) candidate[j]-=step*grad[j];
            if(loss(candidate)<=before-1e-4*step*norm) {b=candidate; accepted=true; break;}
            step*=.5;
        }
        if(!accepted) break;
    }
    // 将标准化模型还原，推理时无需隐式全局均值或历史模型文件。
    for(size_t j=0;j<d;++j) {b[j+1]/=scale[j]; b[0]-=b[j+1]*mean[j];}
    return {b};
}
} // namespace hohlraum


// admission/training.hpp
#include <map>
#include <string>
#include <cstdint>

namespace hohlraum {
// 标签只供离线拟合：returned=完整随访窗口内有返回，grew=同窗口同段再增长至少32K。
// 调用者必须排除标签未知的右删失行；采样权重由采样器决定，默认每表样本等权。
// task 必须是隔离单元：父子agent归同任务，关联小时记录归同一个单元。
struct AdmissionRow {
    std::string task;
    unsigned query_groups;
    double silence, length, returned, grew, weight;
};
struct Curve {
    std::vector<double> x, p;
    double predict(double length) const {
        if(x.empty() || !std::isfinite(length) || length<0) throw std::invalid_argument("length curve input");
        if(length<=x.front()) return p.front();
        if(length>=x.back()) return p.back();
        auto hi=std::upper_bound(x.begin(),x.end(),length)-x.begin();
        const double t=(length-x[hi-1])/(x[hi]-x[hi-1]);
        return p[hi-1]+t*(p[hi]-p[hi-1]);
    }
    void save(std::ostream& out) const {
        out<<x.size()<<'\n'; for(size_t i=0;i<x.size();++i) out<<x[i]<<' '<<p[i]<<'\n';
    }
    static Curve load(std::istream& in) {
        size_t n=0; in>>n; if(!in || n<1 || n>8) throw std::runtime_error("curve size");
        Curve c;
        for(size_t i=0;i<n;++i) {
            double x,p; in>>x>>p; if(!in || !std::isfinite(x) || x<0 || (!c.x.empty()&&x<=c.x.back()))
                throw std::runtime_error("curve point");
            probability(p); c.x.push_back(x); c.p.push_back(p);
        }
        return c;
    }
};

// 8个加权分位桶；相同长度不跨桶。先按长度聚合，再按累计权重切分。
// 各桶正例率向组内整体率收缩，m=10；不强加单调，允许多个终止区域。
inline Curve fit_curve(const std::vector<AdmissionRow>& rows) {
    if(rows.empty()) throw std::invalid_argument("empty length group/fold");
    std::map<double,std::pair<double,double>> points;
    double total=0, yes=0;
    for(const auto& r:rows) {
        probability(r.grew); probability(r.returned);
        if(!r.query_groups || !std::isfinite(r.silence) || r.silence<0 || !std::isfinite(r.length)||r.length<0 ||
           !std::isfinite(r.weight)||r.weight<=0) throw std::invalid_argument("admission sample");
        auto& p=points[r.length]; p.first+=r.weight; p.second+=r.weight*r.grew;
        total+=r.weight; yes+=r.weight*r.grew;
    }
    Curve c; double mass=0, y=0, lx=0, cumulative=0;
    for(auto it=points.begin();it!=points.end();++it) {
        const double w=it->second.first;
        mass+=w; lx+=w*it->first; y+=it->second.second; cumulative+=w;
        if((cumulative>=total*(c.x.size()+1)/8 && c.x.size()<7) || std::next(it)==points.end()) {
            c.x.push_back(lx/mass); c.p.push_back((y+10*yes/total)/(mass+10));
            mass=y=lx=0;
        }
    }
    return c;
}
struct BranchRow { unsigned survived_qgs, age; double restored, weight; };
struct BranchModel {
    // 三类经历数 x 两桶缺席年龄（0=仍在离开当组，1=已缺席满一个组）。
    std::array<std::array<double,2>,3> p{};
    double predict(unsigned k,unsigned age) const {
        if(p[0][0]<0)throw std::runtime_error("old suffix encountered but branch model has no training samples");
        if(k==0) throw std::invalid_argument("block has no query history");
        return p[std::min(k,3u)-1][age?1:0];
    }
    static BranchModel fit(const std::vector<BranchRow>& rows) {
        if(rows.empty()) { BranchModel m; for(auto& r:m.p)r.fill(-1); return m; } // unavailable, not a probability
        double w=0,y=0; std::array<std::array<double,2>,3> ws{},ys{};
        for(const auto& r:rows) {
            probability(r.restored);
            if(!r.survived_qgs||r.age>1||!std::isfinite(r.weight)||r.weight<=0) throw std::invalid_argument("branch sample");
            const auto k=std::min(r.survived_qgs,3u)-1; const auto b=r.age?1:0;
            w+=r.weight; y+=r.weight*r.restored; ws[k][b]+=r.weight; ys[k][b]+=r.weight*r.restored;
        }
        BranchModel m;
        for(size_t k=0;k<3;++k)for(size_t b=0;b<2;++b)
            m.p[k][b]=(ys[k][b]+10*y/w)/(ws[k][b]+10);
        return m;
    }
};
struct AdmissionModel {
    std::array<Curve,2> length;
    SessionModel session;
    BranchModel branch;
    double predict(double silence,unsigned Q,double L,Branch state,unsigned k,unsigned age) const {
        const double p=session.predict({silence,Q,length[Q>1].predict(L)});
        return session_admission(p,state,state==Branch::old_suffix?branch.predict(k,age):1);
    }
    static AdmissionModel fit(const std::vector<AdmissionRow>& rows,const std::vector<BranchRow>& branches,
                              const std::vector<AdmissionRow>* length_rows=nullptr) {
        AdmissionModel model;
        for(unsigned z=0;z<2;++z) {
            std::vector<AdmissionRow> all;
            for(const auto& r:rows) if(unsigned(r.query_groups>1)==z) {
                all.push_back(r);
            }
            std::vector<AdmissionRow> curve_rows;
            for(const auto& r:(length_rows?*length_rows:rows)) if(unsigned(r.query_groups>1)==z) {
                curve_rows.push_back(r);
            }
            // 仅使用训练集合：先估计长度曲线，再拟合准入，不另拆内部任务折。
            model.length[z]=fit_curve(curve_rows);
            std::vector<LogisticRow> fitrows;
            for(const auto& r:all) fitrows.push_back({{
                std::log1p(r.silence/30), z?std::log1p(r.query_groups):0,
                model.length[z].predict(r.length)},r.returned,r.weight});
            auto fit=fit_logistic(fitrows);
            std::copy(fit.beta.begin(),fit.beta.end(),model.session.beta[z].begin());
            // Q=1的Q项显式为0，常数效应只由截距承担。
            if(!z)model.session.beta[z][2]=0;
        }
        model.branch=BranchModel::fit(branches);
        return model;
    }
    void save(std::ostream& out) const {
        out<<"ADMISSION_2\n";
        for(unsigned z=0;z<2;++z) {length[z].save(out); for(double b:session.beta[z])out<<b<<' '; out<<'\n';}
        for(const auto& r:branch.p)for(double q:r)out<<q<<' '; out<<'\n';
    }
    static AdmissionModel load(std::istream& in) {
        std::string tag; in>>tag; if(tag!="ADMISSION_2")throw std::runtime_error("admission version");
        AdmissionModel m;
        for(unsigned z=0;z<2;++z) {
            m.length[z]=Curve::load(in);
            for(double& b:m.session.beta[z]) {in>>b;if(!in||!std::isfinite(b))throw std::runtime_error("admission coefficient");}
        }
        bool missing=false;
        for(auto& r:m.branch.p)for(double& q:r) {
            in>>q;if(!in)throw std::runtime_error("branch model");
            if(q==-1)missing=true; else probability(q);
        }
        if(missing)for(const auto& r:m.branch.p)for(double q:r)if(q!=-1)throw std::runtime_error("mixed missing branch model");
        return m;
    }
};
} // namespace hohlraum


// eviction/eviction.hpp
namespace hohlraum {
// 此函数只合成两条互斥路径。training.hpp中的RetentionModel::predict
// 才负责由历史状态计算a、复用准入得到silent_admission、预测c。
// c是给定有请求的条件概率，故需乘(1-a)，不能直接相加。
inline double value(double no_arrival, double silent_admission,
                    double conditional_compaction_rejection, double threshold) {
    probability(no_arrival); probability(silent_admission);
    probability(conditional_compaction_rejection); probability(threshold);
    const double v=no_arrival * !write(silent_admission, threshold)
        + (1 - no_arrival) * conditional_compaction_rejection;
    return std::clamp(v,0.,1.);
}
inline double shared_value(const std::vector<double>& session_values) {
    // 这是排序代理，不能解释为所有 session 都可拒写的联合概率。
    if (session_values.empty()) throw std::invalid_argument("missing session values");
    for (double p : session_values) probability(p);
    return *std::min_element(session_values.begin(), session_values.end());
}
struct Candidate {
    // 必须在同一物理时刻刷新。比较器不查询未来，也不自行更新模型。
    double admission_score, retention_value, last_access;
    unsigned long long id;
    Branch branch=Branch::current;
    unsigned query_groups=1;
    double context_length=0;
    double context_survival=1;
};
// Rule-based residency uses block-level state aggregated across every session
// sharing the physical KV.  A block is old only when no session still places
// it on the current path.  Current blocks whose dominant claim has one QG are
// protected until admission rejects them or another QG changes their state;
// protected blocks closer to rejection are retained longer.  Others use LRU.
inline bool rule_evict_before(const Candidate& a,const Candidate& b,double threshold,
                              double context_reference) {
    if(!std::isfinite(context_reference)||context_reference<=0)
        throw std::invalid_argument("context reference");
    const bool a_displaced=a.branch==Branch::displaced_by_compaction;
    const bool b_displaced=b.branch==Branch::displaced_by_compaction;
    if(a_displaced!=b_displaced)return a_displaced;
    const bool a_old=a.branch!=Branch::current,b_old=b.branch!=Branch::current;
    const bool a_write=write(a.admission_score,threshold),b_write=write(b.admission_score,threshold);
    const bool a_protected=!a_old&&a_write&&
        (a.query_groups==1||a.context_survival< threshold);
    const bool b_protected=!b_old&&b_write&&
        (b.query_groups==1||b.context_survival< threshold);
    if(a_protected!=b_protected)return !a_protected;
    if(a_protected&&a.retention_value!=b.retention_value)
        return a.retention_value>b.retention_value;
    if(a.last_access!=b.last_access)return a.last_access<b.last_access;
    return a.id<b.id;
}

inline bool path_evict_before(const Candidate& a,const Candidate& b,double threshold) {
    const bool ar= !write(a.admission_score,threshold);
    const bool br= !write(b.admission_score,threshold);
    if(ar!=br)return ar;
    if(!ar&&a.retention_value!=b.retention_value)return a.retention_value>b.retention_value;
    if(a.last_access!=b.last_access)return a.last_access<b.last_access;
    return a.id<b.id;
}
inline bool evict_before(const Candidate& a, const Candidate& b, double) {
    // 排序只比较驻留价值；当前已拒写的候选由模型赋值为0。
    // admission_score只供候选被选中后的写入/丢弃动作使用。
    if (a.retention_value != b.retention_value)
        return a.retention_value < b.retention_value;
    if (a.last_access != b.last_access) return a.last_access < b.last_access;
    return a.id < b.id;
}
} // namespace hohlraum


// eviction/training.hpp
#include <limits>

namespace hohlraum {
// 标签必须有完整未来窗口：no_arrival=窗口内没有请求，不是永久死亡。length 仅供 λ 拟合使用。
struct SilenceRow { unsigned query_groups; double silence, horizon, length, no_arrival, weight; };
struct SilenceModel {
    // logit(pi), log(tau), log(k)。使用变换保证pi∈(0,1)、tau,k>0。
    std::array<std::array<double,3>,2> parameters{};
    static double log_survival(double t,const std::array<double,3>& p) {
        if(!std::isfinite(t)||t<0)throw std::invalid_argument("silence time");
        if(t==0)return 0;
        const double pi=logistic(p[0]), k=std::exp(p[2]);
        const double hazard=std::exp(std::min(50.,k*(std::log(t)-p[1])));
        const double a=std::log(pi),b=std::log1p(-pi)-hazard,m=std::max(a,b);
        return m+std::log(std::exp(a-m)+std::exp(b-m));
    }
    static double predict(double s,double h,const std::array<double,3>& p) {
        if(!std::isfinite(h)||h<=0)throw std::invalid_argument("observation horizon");
        return std::clamp(std::exp(log_survival(s+h,p)-log_survival(s,p)),0.,1.);
    }
    double predict(unsigned Q,double s,double h)const {
        if(!Q)throw std::invalid_argument("Q must be positive");
        return predict(s,h,parameters[Q>1]);
    }
    static SilenceModel fit(const std::vector<SilenceRow>& rows) {
        SilenceModel model;
        for(unsigned z=0;z<2;++z) {
            std::vector<SilenceRow> group;
            double mass=0;
            for(const auto& r:rows) if(unsigned(r.query_groups>1)==z) {
                probability(r.no_arrival);
                if(!r.query_groups||!std::isfinite(r.silence)||r.silence<0||!std::isfinite(r.horizon)||r.horizon<=0||
                   !std::isfinite(r.length)||r.length<0||!std::isfinite(r.weight)||r.weight<=0)throw std::invalid_argument("silence sample");
                group.push_back(r);mass+=r.weight;
            }
            if(group.empty())throw std::invalid_argument("empty silence group");
            auto loss=[&](const std::array<double,3>& p) {
                double result=0;
                for(const auto& r:group) {
                    const double a=std::clamp(predict(r.silence,r.horizon,p),1e-12,1-1e-12);
                    result-=r.weight*(r.no_arrival*std::log(a)+(1-r.no_arrival)*std::log1p(-a));
                }
                return result/mass;
            };
            // 三维确定性坐标搜索；只按训练BCE选择。不是测试集网格选收益。
            // 有限窗口不能证明真实永不返回：pi仅是参数化尾部的拟合项。
            double best=std::numeric_limits<double>::infinity();
            for(double tau:{30.,300.,3000.}) {
                std::array<double,3> p{-1.,std::log(tau),0};
                double current=loss(p),step=1;
                for(unsigned it=0;it<300&&step>1e-5;++it) {
                    bool improved=false;
                    for(unsigned j=0;j<3;++j) for(double sign:{-1.,1.}) {
                        auto candidate=p;
                        const double lo=j==0?-12.:j==1?-10.:-4.;
                        const double hi=j==0?12.:j==1?25.:4.;
                        candidate[j]=std::clamp(p[j]+sign*step,lo,hi);
                        const double l=loss(candidate);
                        if(l<current-1e-12){p=candidate;current=l;improved=true;}
                    }
                    if(!improved)step*=.5;
                }
                if(current<best){best=current;model.parameters[z]=p;}
            }
        }
        return model;
    }
};
// 此表只允许arrival=true且完整观察h的训练点。
// label为窗口末可拒写的移除比例，不能用“第一次压缩移除过”替代。
struct CompactionRow { double length, growth_per_second, silence, removed_fraction, weight; };
struct RetentionModel {
    double horizon=0, context_reference=0;
    SilenceModel silence;
    LogisticFit compaction;
    static std::vector<double> features(double L,double g,double s,double h,double C) {
        if(!std::isfinite(L)||L<0||!std::isfinite(g)||g<0||!std::isfinite(s)||s<0||
           !std::isfinite(h)||h<=0||!std::isfinite(C)||C<=0)throw std::invalid_argument("compaction state");
        return {(L-C)/C,std::log1p(g*h/C),std::log1p(s/30)};
    }
    static RetentionModel fit(const std::vector<SilenceRow>& rows,const std::vector<CompactionRow>& cs,
                              double h,double C) {
        features(0,0,0,h,C);
        for(const auto& r:rows)if(r.horizon!=h)throw std::invalid_argument("mixed horizons require separate models");
        RetentionModel m; m.horizon=h;m.context_reference=C;m.silence=SilenceModel::fit(rows);
        std::vector<LogisticRow> fitrows;
        for(const auto& r:cs)fitrows.push_back({features(r.length,r.growth_per_second,r.silence,h,C),r.removed_fraction,r.weight});
        m.compaction=fit_logistic(fitrows);
        return m;
    }
    double predict(const AdmissionModel& admission_model,unsigned Q,double s,double L,double g,
                   Branch state,unsigned k,unsigned age,double threshold)const {
        // 当前已被准入拒绝的候选没有继续观察的省写价值，直接记为0。
        const double current_admission=admission_model.predict(s,Q,L,state,k,age);
        if(!write(current_admission,threshold))return 0;
        // 对当前会被写入的候选，估计继续驻留后转为拒写的概率。
        // 模型输入都来自同一当前状态；无请求路径只将s推进到s+h。
        const double p_silent=admission_model.predict(s+horizon,Q,L,state,k,age);
        const double a=silence.predict(Q,s,horizon);
        const double c=state==Branch::current ? compaction.predict(features(L,g,s,horizon,context_reference)) : 0;
        return value(a,p_silent,c,threshold);
    }
    void save(std::ostream& out)const {
        out<<"RETENTION_3 "<<horizon<<' '<<context_reference<<'\n';
        for(const auto& p:silence.parameters){for(double x:p)out<<x<<' ';out<<'\n';}
        compaction.save(out);
    }
    static RetentionModel load(std::istream& in) {
        RetentionModel m;std::string tag;in>>tag>>m.horizon>>m.context_reference;
        if(!in||tag!="RETENTION_3")throw std::runtime_error("retention model version");
        features(0,0,0,m.horizon,m.context_reference);
        for(auto& p:m.silence.parameters)for(double& x:p){in>>x;if(!in||!std::isfinite(x))throw std::runtime_error("silence parameter");}
        m.compaction=LogisticFit::load(in);return m;
    }
};
} // namespace hohlraum


// eviction/temporal.hpp
#include <limits>
#include <map>
#include <vector>

namespace hohlraum {

// One fully observed state at a QG head or tail. returned always means a
// later QG, never another request inside the same QG.
struct SessionPathRow {
    unsigned query_groups;
    double silence;
    bool returned;
    double remaining;
    double weight;
};

// Direct empirical QG curves. No parametric distribution and no context
// feature enters this model.
struct TemporalModel {
    static constexpr unsigned q_groups=4;
    struct Curve {
        std::vector<double> silence,return_probability,mean_remaining;
        bool step=false;
        mutable double cached_threshold=-1;
        mutable std::vector<size_t> next_reject;
        // Per silence bucket: remaining times and weighted cumulative return mass.
        // Non-return observations remain in the denominator.
        std::vector<std::vector<std::pair<double,double>>> return_cdf;
        double cumulative(double d,double t)const {
            if(return_cdf.size()!=silence.size())throw std::runtime_error("CDF samples unavailable");
            auto at=[&](size_t i){const auto& v=return_cdf[i];
                auto it=std::upper_bound(v.begin(),v.end(),t,[](double x,const auto& y){return x<y.first;});
                return it==v.begin()?0.:std::prev(it)->second;};
            double result;
            if(d<=silence.front())result=at(0);
            else if(d>=silence.back())result=at(silence.size()-1);
            else {size_t hi=std::upper_bound(silence.begin(),silence.end(),d)-silence.begin();
                double a=(d-silence[hi-1])/(silence[hi]-silence[hi-1]);result=at(hi-1)*(1-a)+at(hi)*a;}
            return std::min(result,probability(d));
        }
        double interpolate(const std::vector<double>& y,double d)const {
            if(silence.empty()||!std::isfinite(d)||d<0)throw std::invalid_argument("session curve input");
            if(step){auto i=std::upper_bound(silence.begin(),silence.end(),d);return i==silence.begin()?y.front():y[size_t(i-silence.begin()-1)];}
            if(d<=silence.front())return y.front();
            if(d>=silence.back())return y.back();
            const size_t hi=std::upper_bound(silence.begin(),silence.end(),d)-silence.begin();
            const double a=(d-silence[hi-1])/(silence[hi]-silence[hi-1]);
            return y[hi-1]*(1-a)+y[hi]*a;
        }
        double probability(double d)const{return interpolate(return_probability,d);}
        double remaining(double d)const{return interpolate(mean_remaining,d);}
    };
    std::array<Curve,q_groups> curves;
    bool cumulative_cutoff=false;
    double cutoff_time(unsigned q,double threshold)const {
        probability(threshold);const auto& c=curves[q_group(q)];
        auto it=std::lower_bound(c.return_probability.begin(),c.return_probability.end(),threshold);
        return it==c.return_probability.end()?std::numeric_limits<double>::infinity():c.silence[size_t(it-c.return_probability.begin())];
    }
    double admission_score(unsigned q,double silence,double threshold)const {
        if(cumulative_cutoff)return silence<cutoff_time(q,threshold)?1.:0.;
        return return_probability(q,silence);
    }
    static unsigned q_group(unsigned q){if(!q)throw std::invalid_argument("Q must be positive");return std::min(q,4u)-1;}
    double return_probability(unsigned q,double silence)const{return curves[q_group(q)].probability(silence);}
    double rejection_wait(unsigned q,double silence,double threshold)const {
        probability(threshold);const auto& c=curves[q_group(q)];
        if(cumulative_cutoff)return std::max(0.,cutoff_time(q,threshold)-silence);
        if(c.probability(silence)<threshold)return 0;
        if(c.step){
            if(c.cached_threshold!=threshold){
                c.cached_threshold=threshold;c.next_reject.resize(c.silence.size());size_t next=c.silence.size();
                for(size_t i=c.silence.size();i-->0;){if(c.return_probability[i]<threshold)next=i;c.next_reject[i]=next;}
            }
            const size_t after=std::upper_bound(c.silence.begin(),c.silence.end(),silence)-c.silence.begin();
            if(after==c.silence.size()||c.next_reject[after]==c.silence.size())return std::numeric_limits<double>::infinity();
            return c.silence[c.next_reject[after]]-silence;
        }
        if(c.return_probability.back()>=threshold)return std::numeric_limits<double>::infinity();
        double lo=silence,hi=c.silence.back();
        for(unsigned i=0;i<64;++i){const double mid=(lo+hi)/2;(c.probability(mid)>=threshold?lo:hi)=mid;}
        return std::max(0.,hi-silence);
    }
    double return_wait(unsigned q,double silence,double threshold)const {
        probability(threshold);const auto& c=curves[q_group(q)];
        if(threshold==0)return 0;
        if(c.cumulative(silence,std::numeric_limits<double>::infinity())<threshold)
            return std::numeric_limits<double>::infinity();
        std::vector<double> times;
        for(const auto& bucket:c.return_cdf)for(const auto& point:bucket)times.push_back(point.first);
        std::sort(times.begin(),times.end());times.erase(std::unique(times.begin(),times.end()),times.end());
        if(times.empty()||c.cumulative(silence,times.back())<threshold)return std::numeric_limits<double>::infinity();
        size_t lo=0,hi=times.size()-1;
        while(lo<hi){size_t mid=(lo+hi)/2;if(c.cumulative(silence,times[mid])>=threshold)hi=mid;else lo=mid+1;}
        return times[lo];
    }
    double wait(bool accepted,unsigned q,double silence,double threshold)const {
        // Accepted claims: retain shorter time-to-rejection first.
        // Rejected claims: no return prediction; lowest retention priority.
        return accepted?rejection_wait(q,silence,threshold):std::numeric_limits<double>::infinity();
    }
    // Compatibility for historical modes; the new session mode calls the
    // empirical overload above directly.
    double rejection_wait(const AdmissionModel& admission,unsigned q,double length,double silence,
                          Branch state,unsigned references,unsigned age,double threshold)const {
        auto score=[&](double extra){return admission.predict(silence+extra,q,length,state,references,age);};
        if(!write(score(0),threshold))return 0;
        double hi=1;constexpr double maximum=365.*24*3600;
        while(hi<maximum&&write(score(hi),threshold))hi*=2;
        hi=std::min(hi,maximum);
        if(write(score(hi),threshold))return std::numeric_limits<double>::infinity();
        double lo=0;for(unsigned i=0;i<64;++i){const double mid=(lo+hi)/2;(write(score(mid),threshold)?lo:hi)=mid;}
        return hi;
    }
    double wait(const AdmissionModel& admission,unsigned q,double length,double silence,
                Branch state,unsigned references,unsigned age,double threshold)const {
        const double current=admission.predict(silence,q,length,state,references,age);
        return write(current,threshold)?rejection_wait(admission,q,length,silence,state,references,age,threshold)
                                       :return_wait(q,silence,threshold);
    }
    static TemporalModel fit(const std::vector<SessionPathRow>& rows) {
        if(rows.empty())throw std::invalid_argument("empty session-path samples");TemporalModel m;
        for(unsigned g=0;g<q_groups;++g) {
            struct Aggregate{double weight=0,returned=0,remaining=0;};
            std::map<double,Aggregate> points;
            for(const auto& r:rows)if(q_group(r.query_groups)==g) {
                if(!std::isfinite(r.silence)||r.silence<0||!std::isfinite(r.weight)||r.weight<=0||
                   (r.returned&&(!std::isfinite(r.remaining)||r.remaining<0))||(!r.returned&&r.remaining!=0))
                    throw std::invalid_argument("session-path sample");
                auto& a=points[r.silence];a.weight+=r.weight;
                if(r.returned){a.returned+=r.weight;a.remaining+=r.weight*r.remaining;}
            }
            if(points.size()<2)throw std::invalid_argument("too few session-path silence points");
            auto& c=m.curves[g];
            for(const auto& [d,a]:points) {
                c.silence.push_back(d);c.return_probability.push_back(a.returned/a.weight);
                c.mean_remaining.push_back(a.returned? a.remaining/a.returned:std::numeric_limits<double>::infinity());
            }
            for(double d:c.silence) {
                std::map<double,double> events;double total=0;
                for(const auto& r:rows)if(q_group(r.query_groups)==g&&r.silence==d){
                    total+=r.weight;if(r.returned)events[r.remaining]+=r.weight;}
                std::vector<std::pair<double,double>> points;double mass=0;
                for(const auto& [t,w]:events){mass+=w;points.emplace_back(t,mass/total);}
                c.return_cdf.push_back(std::move(points));
            }
            // Empirical no-return evidence must not weaken as silence grows.
            for(size_t i=1;i<c.return_probability.size();++i)
                c.return_probability[i]=std::min(c.return_probability[i],c.return_probability[i-1]);
            // Fill undefined conditional means from the nearest earlier/later observed return bucket.
            double last=std::numeric_limits<double>::quiet_NaN();
            for(double& x:c.mean_remaining){if(std::isfinite(x))last=x;else if(std::isfinite(last))x=last;}
            last=std::numeric_limits<double>::quiet_NaN();
            for(auto i=c.mean_remaining.rbegin();i!=c.mean_remaining.rend();++i){if(std::isfinite(*i))last=*i;else if(std::isfinite(last))*i=last;}
            for(double x:c.mean_remaining)if(!std::isfinite(x))throw std::invalid_argument("QG group has no returning sample");
        }
        return m;
    }
    void save(std::ostream& out)const {
        out<<(cumulative_cutoff?"TEMPORAL_9\n":curves[0].step?"TEMPORAL_8\n":"TEMPORAL_7\n");
        for(const auto& c:curves){out<<c.silence.size()<<'\n';for(size_t i=0;i<c.silence.size();++i)
            out<<c.silence[i]<<' '<<c.return_probability[i]<<' '<<c.mean_remaining[i]<<'\n';}
        for(const auto& c:curves)for(const auto& bucket:c.return_cdf){out<<bucket.size()<<'\n';for(const auto& [t,p]:bucket)out<<t<<' '<<p<<'\n';}
    }
    static TemporalModel load(std::istream& in) {
        TemporalModel m;std::string tag;in>>tag;if(!in||(tag!="TEMPORAL_6"&&tag!="TEMPORAL_7"&&tag!="TEMPORAL_8"&&tag!="TEMPORAL_9"))throw std::runtime_error("temporal model version");
        m.cumulative_cutoff=tag=="TEMPORAL_9";
        for(auto& c:m.curves){c.step=tag=="TEMPORAL_8"||m.cumulative_cutoff;size_t n=0;in>>n;if(!in||n<1||n>1000000)throw std::runtime_error("session curve size");
            for(size_t i=0;i<n;++i){double d,p,r;in>>d>>p>>r;probability(p);
                if(!in||!std::isfinite(d)||d<0||!std::isfinite(r)||r<0||(!c.silence.empty()&&d<=c.silence.back()))throw std::runtime_error("session curve point");
                c.silence.push_back(d);c.return_probability.push_back(p);c.mean_remaining.push_back(r);}
            for(size_t i=1;i<n;++i)if(!c.step&&c.return_probability[i]>c.return_probability[i-1]+1e-12)throw std::runtime_error("non-monotone session curve");}
        if(m.cumulative_cutoff)for(const auto& c:m.curves)for(size_t i=1;i<c.return_probability.size();++i)
            if(c.return_probability[i]<c.return_probability[i-1])throw std::runtime_error("decreasing CDF");
        if(tag=="TEMPORAL_7")for(auto& c:m.curves)for(size_t i=0;i<c.silence.size();++i){
            size_t n;in>>n;std::vector<std::pair<double,double>> bucket;
            for(size_t j=0;j<n;++j){double t,p;in>>t>>p;probability(p);if(!in||t<0)throw std::runtime_error("CDF point");bucket.emplace_back(t,p);}
            c.return_cdf.push_back(std::move(bucket));}
        return m;
    }
};

inline bool temporal_evict_before(const Candidate& a,const Candidate& b) {
    if(a.retention_value!=b.retention_value)return a.retention_value>b.retention_value;
    if(a.last_access!=b.last_access)return a.last_access<b.last_access;
    return a.id<b.id;
}
} // namespace hohlraum


// eviction/path_time.hpp
#include <limits>
#include <map>

namespace hohlraum {

struct MonotoneCurve {
    std::vector<double> x,p;
    double predict(double t)const {
        if(x.empty()||!std::isfinite(t)||t<0)throw std::invalid_argument("path curve input");
        if(t<=x.front())return p.front();
        if(t>=x.back())return p.back();
        const size_t hi=std::upper_bound(x.begin(),x.end(),t)-x.begin();
        const double a=(t-x[hi-1])/(x[hi]-x[hi-1]);
        return p[hi-1]+a*(p[hi]-p[hi-1]);
    }
    double rejection_wait(double silence,double threshold)const {
        probability(threshold);
        if(predict(silence)<threshold)return 0;
        if(p.back()>=threshold)return std::numeric_limits<double>::infinity();
        double lo=silence,hi=x.back();
        for(unsigned i=0;i<64;++i) {
            const double mid=(lo+hi)/2;
            (predict(mid)>=threshold?lo:hi)=mid;
        }
        return std::max(0.,hi-silence);
    }
    void save(std::ostream& out)const {
        out<<x.size()<<'\n';
        for(size_t i=0;i<x.size();++i)out<<x[i]<<' '<<p[i]<<'\n';
    }
    static MonotoneCurve load(std::istream& in) {
        MonotoneCurve c;size_t n=0;in>>n;
        if(!in||!n||n>1024)throw std::runtime_error("path curve size");
        for(size_t i=0;i<n;++i){double x,p;in>>x>>p;probability(p);
            if(!in||!std::isfinite(x)||x<0||(!c.x.empty()&&x<=c.x.back()))throw std::runtime_error("path curve point");
            c.x.push_back(x);c.p.push_back(p);
        }
        for(size_t i=1;i<c.p.size();++i)if(c.p[i]>c.p[i-1]+1e-12)throw std::runtime_error("non-monotone path curve");
        return c;
    }
};

struct PathTimeModel {
    static constexpr unsigned q_groups=4,length_buckets=8;
    std::array<MonotoneCurve,q_groups> session;
    std::array<std::array<MonotoneCurve,length_buckets>,q_groups> context;
    std::array<std::array<double,length_buckets>,q_groups> length_centers{};
    BranchModel branch;

    static unsigned q_group(unsigned q){if(!q)throw std::invalid_argument("Q must be positive");return std::min(q,4u)-1;}
    double session_score(unsigned q,double silence)const{return session[q_group(q)].predict(silence);}
    double context_score(unsigned q,double length,double silence)const {
        const unsigned g=q_group(q);const auto& centers=length_centers[g];
        if(length<=centers.front())return context[g][0].predict(silence);
        if(length>=centers.back())return context[g][length_buckets-1].predict(silence);
        const unsigned hi=std::upper_bound(centers.begin(),centers.end(),length)-centers.begin();
        const unsigned lo=hi-1;const double a=(length-centers[lo])/(centers[hi]-centers[lo]);
        return context[g][lo].predict(silence)*(1-a)+context[g][hi].predict(silence)*a;
    }
    static constexpr double old_branch_timeout=120;
    double branch_score(Branch state,double old_seconds)const {
        if(state==Branch::current)return 1;
        if(!std::isfinite(old_seconds)||old_seconds<0)throw std::invalid_argument("old branch age");
        return old_seconds<old_branch_timeout?1:0;
    }
    double score(unsigned q,double length,double silence,Branch state,double old_seconds)const {
        return std::min({session_score(q,silence),context_score(q,length,silence),branch_score(state,old_seconds)});
    }
    double decision_wait(const AdmissionModel& admission,unsigned q,double length,double silence,
                         Branch state,unsigned references,unsigned age,double old_seconds,double threshold,unsigned mask=7)const {
        const double current=admission.predict(silence,q,length,state,references,age);
        if(!write(current,threshold))return 0;
        auto wait_until_reject=[&](auto score) {
            double lo=0,hi=1;constexpr double maximum=365.*24*3600;
            while(hi<maximum&&write(score(hi),threshold))hi*=2;
            hi=std::min(hi,maximum);
            if(write(score(hi),threshold))return std::numeric_limits<double>::infinity();
            for(unsigned i=0;i<64;++i){const double mid=(lo+hi)/2;(write(score(mid),threshold)?lo:hi)=mid;}
            return hi;
        };
        // Session path: only silence advances; QG, length, and branch stay observed.
        const double session_wait=wait_until_reject([&](double extra){
            return admission.predict(silence+extra,q,length,state,references,age);
        });
        // Context path: silence selects an empirical head/tail context-survival value,
        // which replaces only the length-survival input of the existing admission model.
        const double branch_probability=state==Branch::current?1:
            state==Branch::displaced_by_compaction?0:admission.branch.predict(references,age);
        const double context_wait=wait_until_reject([&](double extra){
            const double session_probability=admission.session.predict(
                {silence,q,context_score(q,length,silence+extra)});
            return session_probability*branch_probability;
        });
        const double branch_wait=state==Branch::current?std::numeric_limits<double>::infinity():
            std::max(0.,old_branch_timeout-old_seconds);
        return std::min({(mask&1)?session_wait:std::numeric_limits<double>::infinity(),
                         (mask&2)?context_wait:std::numeric_limits<double>::infinity(),
                         (mask&4)?branch_wait:std::numeric_limits<double>::infinity()});
    }
    double rejection_wait(unsigned q,double length,double silence,Branch state,double old_seconds,double threshold)const {
        if(score(q,length,silence,state,old_seconds)<threshold)return 0;
        const unsigned g=q_group(q);
        const double sw=session[g].rejection_wait(silence,threshold);
        double lo=0,hi=1;constexpr double maximum=365.*24*3600;
        auto cp=[&](double extra){return context_score(q,length,silence+extra);};
        while(hi<maximum&&cp(hi)>=threshold)hi*=2;
        hi=std::min(hi,maximum);
        double cw=std::numeric_limits<double>::infinity();
        if(cp(hi)<threshold){for(unsigned i=0;i<64;++i){const double mid=(lo+hi)/2;(cp(mid)>=threshold?lo:hi)=mid;}cw=hi;}
        const double bw=state==Branch::current?std::numeric_limits<double>::infinity():
            std::max(0.,old_branch_timeout-old_seconds);
        return std::min({sw,cw,bw});
    }

    static MonotoneCurve fit_curve(const std::vector<std::tuple<double,double,double>>& rows) {
        if(rows.empty())throw std::invalid_argument("empty path group");
        std::map<double,std::pair<double,double>> points;
        for(const auto& [x,y,w]:rows){probability(y);if(!std::isfinite(x)||x<0||!std::isfinite(w)||w<=0)throw std::invalid_argument("path sample");auto& a=points[x];a.first+=w;a.second+=w*y;}
        struct Pool{double xw=0,yw=0,w=0;unsigned n=0;};std::vector<Pool> pools;
        for(const auto& [x,v]:points){Pool a{x*v.first,v.second,v.first,1};pools.push_back(a);
            while(pools.size()>1){auto& b=pools.back();auto& a=pools[pools.size()-2];if(a.yw/a.w+1e-15>=b.yw/b.w)break;
                a.xw+=b.xw;a.yw+=b.yw;a.w+=b.w;a.n+=b.n;pools.pop_back();}}
        MonotoneCurve c;
        for(const auto& a:pools){c.x.push_back(a.xw/a.w);c.p.push_back(std::clamp(a.yw/a.w,0.,1.));}
        return c;
    }
    static PathTimeModel fit(const std::vector<AdmissionRow>& rows,const std::vector<BranchRow>& branches) {
        if(rows.empty())throw std::invalid_argument("empty path samples");PathTimeModel m;m.branch=BranchModel::fit(branches);
        for(unsigned g=0;g<q_groups;++g){
            std::vector<AdmissionRow> group;for(const auto& r:rows)if(q_group(r.query_groups)==g)group.push_back(r);
            if(group.size()<length_buckets)throw std::invalid_argument("too few path samples");
            std::vector<AdmissionRow> ordered=group;std::sort(ordered.begin(),ordered.end(),[](const auto&a,const auto&b){return a.length<b.length;});
            std::array<std::vector<AdmissionRow>,length_buckets> buckets;
            for(size_t i=0;i<ordered.size();++i)buckets[std::min<unsigned>(length_buckets-1,i*length_buckets/ordered.size())].push_back(ordered[i]);
            std::vector<std::tuple<double,double,double>> sr;
            for(const auto&r:group)sr.emplace_back(r.silence,r.returned,r.weight);
            m.session[g]=fit_curve(sr);
            for(unsigned b=0;b<length_buckets;++b){double lw=0,w=0;std::vector<std::tuple<double,double,double>> cr;
                for(const auto&r:buckets[b]){lw+=r.length*r.weight;w+=r.weight;cr.emplace_back(r.silence,r.grew,r.weight);}m.length_centers[g][b]=lw/w;m.context[g][b]=fit_curve(cr);}
        }
        return m;
    }
    void save(std::ostream& out)const {
        out<<"PATH_TIME_1\n";
        for(const auto& c:session)c.save(out);
        for(unsigned g=0;g<q_groups;++g){for(double x:length_centers[g])out<<x<<' ';out<<'\n';for(const auto& c:context[g])c.save(out);}
        for(const auto&r:branch.p)for(double p:r)out<<p<<' ';out<<'\n';
    }
    static PathTimeModel load(std::istream& in){PathTimeModel m;std::string tag;in>>tag;if(!in||tag!="PATH_TIME_1")throw std::runtime_error("path time version");
        for(auto&c:m.session)c=MonotoneCurve::load(in);for(unsigned g=0;g<q_groups;++g){double prev=-1;for(double&x:m.length_centers[g]){in>>x;if(!in||!std::isfinite(x)||x<prev)throw std::runtime_error("path length centers");prev=x;}for(auto&c:m.context[g])c=MonotoneCurve::load(in);}for(auto&r:m.branch.p)for(double&p:r){in>>p;if(!in)throw std::runtime_error("path branch");if(p!=-1)probability(p);}return m;}
};

} // namespace hohlraum
