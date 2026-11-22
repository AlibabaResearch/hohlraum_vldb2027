#pragma once

#include <cstdint>

namespace hohlraum {

using Block = uint64_t;
using SessionId = uint64_t;

} // namespace hohlraum


#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace hohlraum {

struct Request {
    double time;
    SessionId session_id;
    std::vector<Block> blocks;
};

class EventReader {
public:
    explicit EventReader(const std::string& path) : input(path) {
        if (!input) throw std::runtime_error("cannot read events");
    }

    bool next(Request& request) {
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;

            Request parsed{};
            size_t block_count;
            uint64_t compaction_id, length;
            std::istringstream row(line);
            row >> parsed.time >> parsed.session_id >> compaction_id >> length >> block_count;
            if (!row || !std::isfinite(parsed.time) || parsed.time < 0 || block_count > length / 1024)
                throw std::runtime_error("invalid event header/tail");
            if (has_previous && parsed.time < previous_time)
                throw std::runtime_error("events must be ordered by arrival time");

            std::unordered_set<Block> unique;
            parsed.blocks.reserve(block_count);
            for (size_t index = 0; index < block_count; ++index) {
                Block block;
                row >> block;
                if (!row || !unique.insert(block).second)
                    throw std::runtime_error("invalid/repeated block");
                parsed.blocks.push_back(block);
            }

            // The event stream may include API time and TTFT after the blocks.
            double timing;
            if (row >> timing) row >> timing;
            std::string extra;
            if (row >> extra) throw std::runtime_error("event trailing fields");
            previous_time = parsed.time;
            has_previous = true;
            request = std::move(parsed);
            return true;
        }
        return false;
    }

private:
    std::ifstream input;
    double previous_time=0;
    bool has_previous=false;
};

} // namespace hohlraum
