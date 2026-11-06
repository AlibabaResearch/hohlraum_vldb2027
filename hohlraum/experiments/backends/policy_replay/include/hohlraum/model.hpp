#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hohlraum {
// One admission function is reused by actual decisions and future projections.
// Model inputs contain no KV position, compression state, or future trace labels.
class Model {
public:
    struct Forecast {
        double arrival_probability;
        double future_silence;
        double future_length;
        double current_length;
        double current_session;
        double silent_session;
        double returning_session;
    };
    struct Evaluation {
        double admission;
        double branch_return_probability;
        double no_visit;
        double skipped_branch;
        double reused_branch;
        double residency;
    };

    double window_seconds = 600;
    double growth_tokens = 0;
    std::array<double, 2> qg_remaining_seconds{};
    std::vector<double> length_knots, length_rates;
    std::array<double, 4> session_coefficients{};
    std::array<std::array<double, 3>, 2> survival_coefficients{};
    std::array<double, 4> branch_coefficients{};

    static double sigmoid(double x) {
        return x >= 0 ? 1 / (1 + std::exp(-x)) : std::exp(x) / (1 + std::exp(x));
    }

    static Model load(const std::string& path) {
        std::ifstream in(path);
        auto expect = [&](const char* expected) {
            std::string token;
            if (!(in >> token) || token != expected)
                throw std::runtime_error("model field missing: " + std::string(expected));
        };
        auto number = [&]() {
            double value;
            if (!(in >> value) || !std::isfinite(value))
                throw std::runtime_error("invalid model number");
            return value;
        };
        expect("HOHLRAUM_MODEL");
        int version;
        if (!(in >> version) || version != 2) throw std::runtime_error("model version");
        Model m;
        expect("window_seconds"); m.window_seconds = number();
        expect("growth_tokens"); m.growth_tokens = number();
        expect("qg_remaining_seconds");
        for (auto& x : m.qg_remaining_seconds) x = number();
        expect("length_table");
        unsigned n;
        if (!(in >> n) || n == 0 || n > 10000) throw std::runtime_error("length table size");
        for (unsigned i = 0; i < n; ++i) {
            m.length_knots.push_back(number()); m.length_rates.push_back(number());
            if (i && m.length_knots[i] <= m.length_knots[i - 1])
                throw std::runtime_error("length knots not increasing");
        }
        expect("session_coefficients");
        for (auto& x : m.session_coefficients) x = number();
        expect("survival_single_qg");
        for (auto& x : m.survival_coefficients[0]) x = number();
        expect("survival_multiple_qg");
        for (auto& x : m.survival_coefficients[1]) x = number();
        expect("branch_coefficients");
        for (auto& x : m.branch_coefficients) x = number();
        std::string extra;
        if (in >> extra) throw std::runtime_error("unexpected model field: " + extra);
        if (m.window_seconds <= 0 || m.growth_tokens < 0 ||
            m.qg_remaining_seconds[0] < 0 || m.qg_remaining_seconds[1] < 0)
            throw std::runtime_error("invalid prediction times/growth");
        return m;
    }

    double length_statistic(double length) const {
        if (length <= length_knots.front()) return length_rates.front();
        if (length >= length_knots.back()) return length_rates.back();
        size_t j = std::upper_bound(length_knots.begin(), length_knots.end(), length)
                   - length_knots.begin();
        double t = (length - length_knots[j - 1]) / (length_knots[j] - length_knots[j - 1]);
        return length_rates[j - 1] + t * (length_rates[j] - length_rates[j - 1]);
    }

    double session_score(double silence, unsigned qgs) const {
        if (!qgs || silence < 0) throw std::runtime_error("session score input");
        double t = std::log1p(silence / 30), q = std::log(double(qgs));
        const auto& b = session_coefficients;
        return sigmoid(b[0] + b[1]*t + b[2]*q + b[3]*t*q);
    }

    double branch_score(bool old, double silence, double length) const {
        if (silence < 0) throw std::runtime_error("branch score input");
        const auto& b = branch_coefficients;
        return sigmoid(b[0] + b[1]*old + b[2]*(old ? std::log1p(silence / 30) : 0) + b[3]*length_statistic(length));
    }

    double admission(double ds, unsigned qgs, double length, bool old, double db) const {
        return session_score(ds, qgs) * branch_score(old, db, length);
    }

    double no_visit_probability(unsigned qgs, double silence, double window) const {
        const auto& b = survival_coefficients[qgs > 1];
        return std::clamp(std::exp(log_survival(silence + window, b) -
                                   log_survival(silence, b)), 0., 1.);
    }

    Forecast forecast(unsigned qgs, double silence, double length) const {
        Forecast f{};
        f.arrival_probability = 1 - no_visit_probability(qgs, silence, window_seconds);
        double return_time = window_seconds / 2;
        if (f.arrival_probability > 1e-12) {
            double low = 0, high = window_seconds;
            double target = 1 - f.arrival_probability / 2;
            for (int i = 0; i < 32; ++i) {
                double middle = (low + high) / 2;
                if (no_visit_probability(qgs, silence, middle) > target) low = middle;
                else high = middle;
            }
            return_time = (low + high) / 2;
        }
        f.future_silence = std::max(0., window_seconds - return_time - qg_remaining_seconds[qgs > 1]);
        f.current_length = length;
        f.future_length = length + growth_tokens;
        f.current_session = session_score(silence, qgs);
        f.silent_session = session_score(silence + window_seconds, qgs);
        f.returning_session = session_score(f.future_silence, qgs + 1);
        return f;
    }

    Evaluation evaluate(const Forecast& f, bool old, double silence, double threshold) const {
        if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
            throw std::runtime_error("invalid admission threshold");
        Evaluation e{};
        e.branch_return_probability = branch_score(old, silence, f.current_length);
        e.admission = f.current_session * e.branch_return_probability;
        e.no_visit = f.silent_session * branch_score(old, silence + window_seconds, f.current_length);
        e.skipped_branch = f.returning_session * branch_score(true, silence + window_seconds, f.future_length);
        e.reused_branch = f.returning_session * branch_score(false, f.future_silence, f.future_length);
        // Apply the actual decision before taking the expectation. Equality is accepted.
        e.residency = (1-f.arrival_probability)*(e.no_visit < threshold) +
            f.arrival_probability*((1-e.branch_return_probability)*(e.skipped_branch < threshold) +
                                  e.branch_return_probability*(e.reused_branch < threshold));
        return e;
    }

private:
    static double log_survival(double time, const std::array<double, 3>& b) {
        if (time == 0) return 0;
        double pi = sigmoid(b[0]), shape = std::exp(b[2]);
        double hazard = std::exp(std::min(50., shape * (std::log(time) - b[1])));
        double a = std::log(pi), c = std::log1p(-pi) - hazard, m = std::max(a, c);
        return m + std::log(std::exp(a-m) + std::exp(c-m));
    }
};
} // namespace hohlraum
