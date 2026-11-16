// tests/model/adapter_registry_test.cpp - the adapter registry: lookup by architecture, the replaced guard and DeepSeek
// the deepseek41 cases went with that adapter (Halocline, 2026-11-16). Host only: no model and no runtime are linked.
#include "../core/gguf_fixture.hpp"
#include "basalt/model/adapter.hpp"
#include "meta_adapters.hpp"

#include <cstdio>
#include <filesystem>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
}  // namespace

int main() {
    using namespace basalt;
    test::register_meta_adapters();
    check(model::find_adapter("qwen4exp") != nullptr, "the qwen4exp adapter is registered");
    check(model::find_adapter("llama") == nullptr, "an unknown architecture has no adapter");

    const auto dir = std::filesystem::temp_directory_path() / "basalt_adapter_registry_test";
    std::filesystem::create_directories(dir);
    using namespace fixture;

    write(dir / "other.gguf", {str("general.architecture", "llama")}, {});
    check(model::check_architecture(GgufFile((dir / "other.gguf").string())).find("no model adapter") != std::string::npos,
          "an unregistered architecture is refused by name");

    write(dir / "qwen.gguf", {str("general.architecture", "qwen4exp")}, {});
    check(model::check_architecture(GgufFile((dir / "qwen.gguf").string())).find("block_count") != std::string::npos,
          "qwen4exp without its keys is refused");

    std::filesystem::remove_all(dir);
    return failures ? 1 : 0;
}
