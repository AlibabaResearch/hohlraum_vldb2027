#include "replay.hpp"
#include "dram_coverage.hpp"
#include <iomanip>
#include <iostream>
#include <string>

using namespace hohlraum;

void print_coverage(const Coverage& coverage) {
    std::cout << "{\"requests\":" << coverage.requests
              << ",\"refs\":" << coverage.refs << ",\"theory_hits\":" << coverage.theory_hits
              << ",\"levels\":{";
    for (size_t i = 0; i < coverage.points.size(); ++i) {
        const auto& point = coverage.points[i];
        if (i) std::cout << ',';
        std::cout << '\"' << point.percent << "\":{\"capacity_blocks\":" << point.blocks
                  << ",\"hits\":" << point.hits << ",\"previous_hits\":" << point.previous_hits << '}';
    }
    std::cout << "}}\n";
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: baseline EVENTS all|access:N|larc:G DRAM_BLOCKS\n"
                  << "       baseline EVENTS coverage REFERENCES\n";
        return 2;
    }
    const std::string path = argv[1], mode = argv[2];
    const size_t capacity = std::stoull(argv[3]);
    if (mode == "coverage") {
        print_coverage(calibrate_dram(path, capacity));
        return 0;
    }
    ReplayStats stats;
    if (mode == "all")
        stats = replay_baseline(path, capacity, std::nullopt);
    else if (mode.rfind("access:", 0) == 0)
        stats = replay_baseline(path, capacity, std::stod(mode.substr(7)));
    else if (mode.rfind("larc:", 0) == 0)
        stats = replay_larc(path, capacity, std::stoull(mode.substr(5)));
    else
        return 2;

    std::cout << std::setprecision(12)
              << "{\"mode\":\"" << mode << "\",\"refs\":" << stats.requested
              << ",\"global_hits\":" << stats.hits
              << ",\"theory_hits\":" << stats.theoretical
              << ",\"written_blocks\":" << stats.writes << "}\n";
}
