/* The registry of native entry points: every group of native/src/run/nodes/all_nodes.inc, built once. */
#include "engine.hpp"

#include <stdexcept>

#define CN_RUN_NODES(group) CN_RUN_NODES_DECLARE(group);
#include "nodes/all_nodes.inc"
#undef CN_RUN_NODES

namespace cn::run {
namespace {

class EngineRegistry final : public Registry {
public:
    void add(std::string_view schema_id, NativeRun run) override { insert(schema_id, Entry{run, nullptr}); }
    void add_generator(std::string_view schema_id, NativeGeneratorRun run) override
    {
        insert(schema_id, Entry{nullptr, run});
    }

    std::map<std::string, Entry, std::less<>> entries;

private:
    void insert(std::string_view schema_id, Entry entry)
    {
        if (!(entry.run || entry.generator)) throw std::logic_error("A native entry point is null: " + std::string(schema_id));
        if (!entries.emplace(schema_id, entry).second)
            throw std::logic_error("A schema has two native entry points: " + std::string(schema_id));
    }
};

}  // namespace

const std::map<std::string, Entry, std::less<>> &entries()
{
    static const std::map<std::string, Entry, std::less<>> built = [] {
        EngineRegistry registry;
#define CN_RUN_NODES(group) cn_run_nodes_##group(registry);
#include "nodes/all_nodes.inc"
#undef CN_RUN_NODES
        return std::move(registry.entries);
    }();
    return built;
}

const Entry *find_entry(std::string_view schema_id)
{
    const auto &all = entries();
    auto found = all.find(schema_id);
    return found == all.end() ? nullptr : &found->second;
}

}  // namespace cn::run
