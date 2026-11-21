// tests/program/pretok_gpt4o_test.cpp - the GPT-4o pre-tokenizer against the fixture made by gen_gpt4o_fixture.py.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "halo/program/gpt4o_pretok.hpp"

namespace {

int failures = 0;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// \\, \n, \r, \t and \xHH, the spelling escape() in gen_gpt4o_fixture.py writes.
bool unescape(const std::string& in, std::string& out) {
    out.clear();
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] != '\\') { out += in[i]; continue; }
        if (++i >= in.size()) return false;
        switch (in[i]) {
            case '\\': out += '\\'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'x': {
                if (i + 2 >= in.size()) return false;
                const int hi = hex_digit(in[i + 1]);
                const int lo = hex_digit(in[i + 2]);
                if (hi < 0 || lo < 0) return false;
                out += (char) (hi * 16 + lo);
                i += 2;
                break;
            }
            default: return false;
        }
    }
    return true;
}

// A readable, ASCII-safe spelling for a failure message.
std::string show(const std::string& bytes) {
    std::string out;
    for (unsigned char c : bytes) {
        if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c >= 0x20 && c < 0x7F) out += (char) c;
        else {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02X", c);
            out += buf;
        }
    }
    return out;
}

void check_case(size_t index, const std::string& text, const std::vector<std::string>& expected) {
    std::vector<size_t> ends;
    halo::program::tokenizer::pretok_gpt4o(text.data(), text.size(), ends);
    std::vector<std::string> got;
    size_t start = 0;
    for (size_t end : ends) {
        if (end <= start || end > text.size()) {
            std::fprintf(stderr, "FAIL case %zu: bad end %zu (start %zu, size %zu) for %s\n",
                         index, end, start, text.size(), show(text).c_str());
            ++failures;
            return;
        }
        got.push_back(text.substr(start, end - start));
        start = end;
    }
    if (start != text.size()) {
        std::fprintf(stderr, "FAIL case %zu: pieces stop at %zu of %zu for %s\n",
                     index, start, text.size(), show(text).c_str());
        ++failures;
        return;
    }
    if (got != expected) {
        std::fprintf(stderr, "FAIL case %zu: %s\n  expected:", index, show(text).c_str());
        for (const auto& p : expected) std::fprintf(stderr, " %s", show(p).c_str());
        std::fprintf(stderr, "\n       got:");
        for (const auto& p : got) std::fprintf(stderr, " %s", show(p).c_str());
        std::fprintf(stderr, "\n");
        ++failures;
    }
}

void run(const std::string& path) {
    std::ifstream f(path);
    CHECK(f.good());
    if (!f.good()) return;

    size_t index = 0;
    bool have_case = false;
    std::string text;
    std::vector<std::string> expected;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') {
            if (have_case) {
                check_case(index++, text, expected);
                have_case = false;
            }
            continue;
        }
        std::istringstream ls(line);
        std::string tag, rest;
        ls >> tag;
        std::getline(ls, rest);
        if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
        std::string value;
        if (!unescape(rest, value)) {
            std::fprintf(stderr, "FAIL %s: bad escape on line: %s\n", path.c_str(), line.c_str());
            ++failures;
            continue;
        }
        if (tag == "CASE") {
            if (have_case) check_case(index++, text, expected);
            have_case = true;
            text = value;
            expected.clear();
        } else if (tag == "PIECE") {
            CHECK(have_case);
            if (have_case) expected.push_back(value);
        } else {
            std::fprintf(stderr, "FAIL %s: unknown line: %s\n", path.c_str(), line.c_str());
            ++failures;
        }
    }
    if (have_case) check_case(index++, text, expected);
    if (index == 0) {
        std::fprintf(stderr, "FAIL %s: no cases\n", path.c_str());
        ++failures;
    }
    std::fprintf(stderr, "pretok_gpt4o: %zu cases, %d failures\n", index, failures);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <fixture>\n", argv[0]);
        return 2;
    }
    run(argv[1]);
    return failures ? 1 : 0;
}
