#pragma once
#include "types.hpp"
#include <cassert>
#include <set>
#include <unordered_map>

namespace hohlraum {
// Only requests already processed are registered. A branch is an observed
// root-to-leaf path. Its first/last absorption blocks are candidate endpoints.
class EndpointIndex {
public:
    struct Node {
        std::vector<BlockId> children;
        unsigned active_subtree = 0;
        bool active = false;
    };
    struct Tree {
        std::unordered_map<BlockId, Node> nodes;
        Tree() { nodes.emplace(kNoBlock, Node{}); }
    };
    const std::vector<BlockId>& parent;
    std::vector<Tree> trees;
    std::set<SessionId> active_sessions;

    EndpointIndex(const std::vector<BlockId>& parents, size_t sessions)
        : parent(parents), trees(sessions) {}

    void set_active(SessionId session, BlockId block, bool active) {
        auto& tree = trees.at(session);
        auto& node = tree.nodes.at(block);
        require(node.active != active);
        node.active = active;
        for (auto current = block;; current = parent[current]) {
            auto& count = tree.nodes.at(current).active_subtree;
            if (active) ++count;
            else { require(count); --count; }
            if (current == kNoBlock) break;
        }
        if (tree.nodes.at(kNoBlock).active_subtree) active_sessions.insert(session);
        else active_sessions.erase(session);
    }

    template<class IsAbsorption>
    void observe(SessionId session, const std::vector<BlockId>& blocks, IsAbsorption is_absorption) {
        auto& tree = trees.at(session);
        BlockId previous = kNoBlock;
        for (auto block : blocks) {
            require(parent[block] == previous);
            if (tree.nodes.emplace(block, Node{}).second) {
                tree.nodes.at(previous).children.push_back(block);
                if (is_absorption(block)) set_active(session, block, true);
            }
            previous = block;
        }
    }

    std::set<BlockId> endpoints(SessionId session) const {
        const auto& tree = trees.at(session);
        std::set<BlockId> result;
        struct Visit { BlockId block, last_active; };
        std::vector<Visit> pending{{kNoBlock, kNoBlock}};
        while (!pending.empty()) {
            auto [block, last] = pending.back();
            pending.pop_back();
            const auto& node = tree.nodes.at(block);
            if (!node.active_subtree) continue;
            if (node.active) {
                if (last == kNoBlock) result.insert(block);
                last = block;
            }
            if (node.children.empty() && last != kNoBlock) result.insert(last);
            for (auto child : node.children) {
                if (tree.nodes.at(child).active_subtree) pending.push_back({child, last});
                else if (last != kNoBlock) result.insert(last);
            }
        }
        return result;
    }
};
} // namespace hohlraum
