#pragma once
#include "hohlraum/types.hpp"
#include <istream>
#include <sstream>
#include <string>

namespace hohlraum {

// The replay interval is half-open. Events at the end are outside measurement.
inline bool read_request(std::istream& input, double end, Request& request) {
    std::string line;
    if (!std::getline(input, line)) return false;
    std::istringstream fields(line);
    std::string time, session, fold, context, ids;
    std::getline(fields, time, '\t');
    request.time = std::stod(time);
    if (request.time >= end) return false;
    std::getline(fields, session, '\t');
    std::getline(fields, fold, '\t');
    std::getline(fields, context, '\t');
    std::getline(fields, ids, '\t');
    request.session = std::stoul(session);
    request.fold = std::stoul(fold);
    request.context_tokens = std::stoul(context);
    request.blocks.clear();
    std::istringstream blocks(ids);
    std::string id;
    while (std::getline(blocks, id, ','))
        if (!id.empty()) request.blocks.push_back(std::stoul(id));
    return true;
}

} // namespace hohlraum
