// include/halo/log.hpp: the line each level renders. The writes themselves are one call each to stderr.
#include "halo/log.hpp"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void is(const std::string& got, const std::string& want, const char* what) {
    const bool ok = got == want;
    std::printf("log %-52s %s\n", what, ok ? "ok" : "FAILED   <-- WRONG");
    if (!ok) {
        std::printf("   got  %s\n   want %s\n", got.c_str(), want.c_str());
        ++failures;
    }
}

}  // namespace

int main() {
    using halo::log::Level;
    using halo::log::render;
    is(render(Level::info, "halo generate", "session is up (engine %s)", "1.2"),
       "halo generate: session is up (engine 1.2)\n", "info keeps the tag and has no marker");
    is(render(Level::warning, "halo serve", "RAM %d MiB low", 512), "halo serve: warning: RAM 512 MiB low\n",
       "warning marker after the tag");
    is(render(Level::error, "halo generate", "cannot open %s", "x.basalt"),
       "halo generate: error: cannot open x.basalt\n", "error marker after the tag");
    is(render(Level::error, "halo", "kernel: %s", "bad"), "halo: error: kernel: bad\n", "a line of no area");
    is(render(Level::warning, "halo-bench", "NVML is unavailable"), "halo-bench: warning: NVML is unavailable\n",
       "a tool's own tag");
    is(render(Level::info, "halo load", "plan %s%s %llu", "copy ", "cuda1", 4096ull),
       "halo load: plan copy cuda1 4096\n", "a parsed progress line is unchanged");
    is(render(Level::info, "halo serve", "%.0f%% of %lld", 12.0, 7ll), "halo serve: 12% of 7\n", "percent signs");
    is(render(Level::info, "halo", "%s", ""), "halo: \n", "an empty message");
    const std::string longer(1500, 'x');
    is(render(Level::error, "halo generate", "%s", longer.c_str()), "halo generate: error: " + longer + "\n",
       "a message longer than the first buffer");
    is(render(Level::info, "halo serve", "two\nlines"), "halo serve: two\nlines\n", "an inner newline is kept");
    is(halo::log::line(Level::warning, "halo mtp", "text"), "halo mtp: warning: text", "line() has no newline");

    // parse() reads back what render() wrote, and the older forms
    const auto parsed = [](const std::string& l) {
        halo::log::Parsed p;
        if (!halo::log::parse(l, p)) return std::string("not a log line");
        return std::string(p.level == Level::error ? "error" : p.level == Level::warning ? "warning" : "info") + "|" +
               p.tag + "|" + p.text;
    };
    is(parsed("halo generate: error: no GPU"), "error|halo generate|no GPU", "parse: error");
    is(parsed("halo serve: warning: low RAM"), "warning|halo serve|low RAM", "parse: warning");
    is(parsed("halo load: cuda0 1 2"), "info|halo load|cuda0 1 2", "parse: info");
    is(parsed("halo: error: kernel: bad"), "error|halo|kernel: bad", "parse: no area");
    is(parsed("halo-bench: warning: NVML"), "warning|halo-bench|NVML", "parse: a tool's tag");
    is(parsed("halo error: old form"), "error|halo|old form", "parse: older error form");
    is(parsed("halo warning: old form"), "warning|halo|old form", "parse: older warning form");
    is(parsed("halo serve: no error here"), "info|halo serve|no error here", "parse: the word error is no marker");
    is(parsed("CUDA error: out of memory"), "not a log line", "parse: a line of another writer");
    is(parsed("halo (ms/window): x"), "not a log line", "parse: odd characters in the tag");
    return failures == 0 ? 0 : 1;
}
