#pragma once

// common/types.hpp
#include <cstdint>

namespace hohlraum {

using Block = uint64_t;
using SessionId = uint64_t;
using CompactionId = uint64_t;
using RequestId = uint64_t;

} // namespace hohlraum


// common/events.hpp
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
    CompactionId compaction_id;
    uint64_t length;
    RequestId request_id;
    std::vector<Block> blocks;
    double api_time;  // API call total time (seconds)
    double ttft;      // Time to first token (seconds)
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
            std::istringstream row(line);
            row >> parsed.time >> parsed.session_id >> parsed.compaction_id >> parsed.length >> block_count;
            if (!row || !std::isfinite(parsed.time) || parsed.time < 0 || block_count > parsed.length / 1024)
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

            // Read api_time and ttft (optional, for backward compatibility)
            parsed.api_time = 0.0;
            parsed.ttft = 0.0;
            if (row >> parsed.api_time) {
                if (!(row >> parsed.ttft)) {
                    parsed.ttft = 0.0;
                }
            }

            std::string extra;
            if (row >> extra) throw std::runtime_error("event trailing fields");
            parsed.request_id = next_request_id++;
            previous_time = parsed.time;
            has_previous = true;
            request = std::move(parsed);
            return true;
        }
        return false;
    }

private:
    std::ifstream input;
    RequestId next_request_id=0;
    double previous_time=0;
    bool has_previous=false;
};

} // namespace hohlraum


// storage/block_map.hpp
#include <stdexcept>
#include <unordered_map>

namespace hohlraum {

// 一个 session 对某个全局 block 的逻辑关联；前缀树只保存 block ID，
// 该对象保存反向关联所需的路径和 Query-Group 状态。
struct SessionMembership {
    bool current_path=false;
    CompactionId compaction_id=0;
    unsigned referenced_query_groups=0;
    unsigned last_query_group=0;
    double old_since=-1;
};

// 全局 block 元数据。物理数据后续由 tier queue 持有，本对象保留稳定身份
// 以及所有关联 session 的状态，供准入和逐出优先级计算。
struct PhysicalBlock {
    Block id;
    std::unordered_map<SessionId,SessionMembership> sessions;
};

class BlockMap {
public:
    PhysicalBlock& ensure(Block id) {
        auto [entry, inserted] = blocks.try_emplace(id, PhysicalBlock{id, {}});
        (void)inserted;
        return entry->second;
    }

    const PhysicalBlock* find(Block id) const {
        const auto found = blocks.find(id);
        return found == blocks.end() ? nullptr : &found->second;
    }

    const PhysicalBlock& at(Block id) const {
        const auto* block = find(id);
        if (!block) throw std::out_of_range("unknown physical block");
        return *block;
    }

    PhysicalBlock& at(Block id) {
        const auto found = blocks.find(id);
        if (found == blocks.end()) throw std::out_of_range("unknown physical block");
        return found->second;
    }

    size_t size() const { return blocks.size(); }

private:
    std::unordered_map<Block,PhysicalBlock> blocks;
};

} // namespace hohlraum


// storage/tier_queue.hpp
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace hohlraum {

struct CachedBlock {
    uint64_t recency;
    double entered;
};

// 物理 block 的 LRU 队列。front() 永远是下一个 LRU 淘汰对象，
// 因而 LRU 路径不需要扫描或排序全部 block。
class TierQueue {
public:
    bool contains(Block block) const { return entries.count(block) != 0; }
    bool empty() const { return entries.empty(); }
    size_t size() const { return entries.size(); }

    const CachedBlock& at(Block block) const { return find(block).entry; }

    void put(Block block, CachedBlock entry) {
        const auto found = entries.find(block);
        if (found != entries.end()) {
            lru.erase(found->second.position);
            entries.erase(found);
        }
        lru.push_back(block);
        entries.emplace(block, Entry{entry, std::prev(lru.end())});
    }

    Block oldest() const {
        if (lru.empty()) throw std::out_of_range("empty storage tier");
        return lru.front();
    }

    CachedBlock erase(Block block) {
        const auto found = entries.find(block);
        if (found == entries.end()) throw std::out_of_range("block not in storage tier");
        const CachedBlock entry = found->second.entry;
        lru.erase(found->second.position);
        entries.erase(found);
        return entry;
    }

private:
    struct Entry {
        CachedBlock entry;
        std::list<Block>::iterator position;
    };

    const Entry& find(Block block) const {
        const auto found = entries.find(block);
        if (found == entries.end()) throw std::out_of_range("block not in storage tier");
        return found->second;
    }

    std::list<Block> lru;
    std::unordered_map<Block, Entry> entries;
};

} // namespace hohlraum


// replay/session_prefix_tree.hpp
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hohlraum {

// 每个 session 独占一棵压缩前缀树。节点保存相对父节点新增的一段连续 block，
// 而不是只保存一个 block；分支落在节点中间时，会按共同前缀拆分节点。
class SessionPrefixTree {
public:
    using NodeId = size_t;
    static constexpr NodeId root_id = 0;

    struct Node {
        NodeId parent;
        std::vector<Block> blocks;
        std::vector<NodeId> children;
    };

    SessionPrefixTree() : nodes(1, {no_parent, {}, {}}) {}

    NodeId upsert(const std::vector<Block>& prefix_blocks) {
        NodeId parent = root_id;
        size_t offset = 0;

        while (offset < prefix_blocks.size()) {
            const NodeId child = child_starting_with(parent, prefix_blocks[offset]);
            if (child == no_node)
                return append_child(parent, {prefix_blocks.begin() + offset, prefix_blocks.end()});

            const size_t common = common_prefix(nodes[child].blocks, prefix_blocks, offset);
            if (common == nodes[child].blocks.size()) {
                parent = child;
                offset += common;
                continue;
            }

            // 当前请求在已有压缩节点的中间分叉：拆出共同前缀作为新父节点。
            const NodeId split = make_node(parent, {nodes[child].blocks.begin(), nodes[child].blocks.begin() + common});
            replace_child(parent, child, split);
            nodes[child].parent = split;
            nodes[child].blocks.erase(nodes[child].blocks.begin(), nodes[child].blocks.begin() + common);
            nodes[split].children.push_back(child);
            offset += common;

            if (offset == prefix_blocks.size()) return split;
            return append_child(split, {prefix_blocks.begin() + offset, prefix_blocks.end()});
        }

        return parent;
    }

    const Node& node(NodeId node_id) const {
        if (node_id >= nodes.size()) throw std::out_of_range("unknown prefix node");
        return nodes[node_id];
    }

    size_t size() const { return nodes.size(); }

    template<class Visitor>
    void visit_blocks(NodeId leaf, Visitor visit) const {
        if (leaf >= nodes.size()) throw std::out_of_range("unknown prefix node");
        std::vector<NodeId> lineage;
        for (NodeId current = leaf; current != root_id; current = nodes[current].parent)
            lineage.push_back(current);
        for (auto it = lineage.rbegin(); it != lineage.rend(); ++it)
            for (Block block : nodes[*it].blocks) visit(block);
    }

    std::vector<Block> expand(NodeId leaf) const {
        if (leaf >= nodes.size()) throw std::out_of_range("unknown prefix node");
        std::vector<NodeId> lineage;
        for (NodeId current = leaf; current != root_id; current = nodes[current].parent)
            lineage.push_back(current);

        std::vector<Block> prefix;
        for (auto it = lineage.rbegin(); it != lineage.rend(); ++it)
            prefix.insert(prefix.end(), nodes[*it].blocks.begin(), nodes[*it].blocks.end());
        return prefix;
    }

private:
    static constexpr NodeId no_node = std::numeric_limits<NodeId>::max();
    static constexpr NodeId no_parent = std::numeric_limits<NodeId>::max();

    std::vector<Node> nodes;

    NodeId make_node(NodeId parent, std::vector<Block> blocks) {
        if (blocks.empty()) throw std::invalid_argument("prefix node needs blocks");
        nodes.push_back({parent, std::move(blocks), {}});
        return nodes.size() - 1;
    }

    NodeId append_child(NodeId parent, std::vector<Block> blocks) {
        const NodeId child = make_node(parent, std::move(blocks));
        nodes[parent].children.push_back(child);
        return child;
    }

    NodeId child_starting_with(NodeId parent, Block first_block) const {
        for (NodeId child : nodes[parent].children)
            if (nodes[child].blocks.front() == first_block) return child;
        return no_node;
    }

    static size_t common_prefix(const std::vector<Block>& node_blocks,
                                const std::vector<Block>& request_blocks,
                                size_t request_offset) {
        size_t common = 0;
        while (common < node_blocks.size() && request_offset + common < request_blocks.size() &&
               node_blocks[common] == request_blocks[request_offset + common])
            ++common;
        return common;
    }

    void replace_child(NodeId parent, NodeId old_child, NodeId new_child) {
        auto& children = nodes[parent].children;
        const auto found = std::find(children.begin(), children.end(), old_child);
        if (found == children.end()) throw std::logic_error("missing prefix child");
        *found = new_child;
    }
};

} // namespace hohlraum
