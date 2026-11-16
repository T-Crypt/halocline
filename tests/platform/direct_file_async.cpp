#include "basalt/platform/direct_file.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/mapped_file.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
using basalt::platform::DirectFile;
using basalt::platform::Completion;

static void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct FileCloser { void operator()(std::FILE* f) const { std::fclose(f); } };

/// Reports a call that names its own failure. Handing `error.c_str()` straight to `check` would read the string
/// before the call that fills it, and that write can move the buffer, so the message is taken after it returned.
static void call_ok(bool ok, const std::string& error, const char* what) {
    if (!ok) throw std::runtime_error(error.empty() ? std::string(what) + ": failed" : error);
}

/// A path no other run of this test holds. mkstemp is POSIX; a name unique to this process and this call is what
/// the test needs. The build directory is where CMake points it, since a temporary directory is not always
/// writable for a test run from a build tree.
static std::string temp_path() {
#if defined(BASALT_TEST_TMPDIR)
    const std::string base = BASALT_TEST_TMPDIR;
#else
    const std::string base = "/tmp";
#endif
    const unsigned long owner = (unsigned long) getpid();
    static std::atomic<unsigned> seq{0};
    return base + "/basalt-direct-test-" + std::to_string(owner) + "-" + std::to_string(seq++) + ".bin";
}

/// A test file of `page_count` alignment-sized pages and an aligned buffer with spare pages for reads that start
/// past the end of it. Both are released whichever way the test leaves.
struct Fixture {
    static constexpr unsigned spare = 8;
    std::string name;
    void* memory = nullptr;
    unsigned page_count = 0;

    Fixture(unsigned pages, char value) : page_count(pages) {
        const uint32_t page = DirectFile::alignment();
        name = temp_path();
        std::ofstream f(name, std::ios::binary);
        const std::string content((size_t) page_count * page, value);
        f.write(content.data(), (std::streamsize) content.size());
        check(f.is_open() && (bool) f, ("cannot write the test file to " + name).c_str());
        memory = DirectFile::alloc_aligned((size_t) (page_count + spare) * page);
        check(memory != nullptr, "buffer allocation");
    }
    ~Fixture() {
        DirectFile::free_aligned(memory);
        std::remove(name.c_str());
    }
    /// Page `i` of the buffer. Pages at or past `page_count` are scratch for reads beyond the end of the file, so
    /// a write that belongs to no completion cannot be mistaken for the data a real read must have left.
    char* at(size_t i) const { return static_cast<char*>(memory) + i * DirectFile::alignment(); }
};

/// Two opens of one file with a bounded issuing pool: reads stay positional, stay on the open that queued them,
/// and a close leaves neither an in-flight read nor a completion behind.
static void test_bounded_opens() {
    using namespace basalt::platform;
    const uint32_t page = DirectFile::alignment();
    constexpr unsigned pages = 12;
    Fixture fx(pages, 0);
    {
        std::ofstream output(fx.name, std::ios::binary | std::ios::trunc);
        for (unsigned i = 0; i < pages; ++i) {
            const std::string content(page, (char) ('a' + i));
            output.write(content.data(), content.size());
        }
        check((bool) output, "cannot write the test contents");
    }
    DirectFile first, second;
    std::string error;
    call_ok(first.open(fx.name, error, 1), error, "open with one worker");
    call_ok(second.open(fx.name, error, 2), error, "open with two workers");
    check(first.size() == (uint64_t) pages * page && second.size() == first.size(), "size mismatch");
    // More outstanding requests than either pool has workers, at descending offsets: an open that shared a file
    // position shows up as the data of another tag, a pool that serialized as a completion that never comes.
    for (unsigned i = 0; i < pages; ++i) {
        DirectFile& f = i % 2 ? second : first;
        call_ok(f.submit((uint64_t) (pages - 1 - i) * page, fx.at(i), page, i, error), error, "submit");
    }
    std::vector<bool> seen(pages, false);
    for (DirectFile* f : {&first, &second}) {
        unsigned received = 0;
        while (received < pages / 2) {
            Completion batch[3];
            const int n = f->wait(batch, 3, 5000);
            check(n > 0, "completion timed out");
            for (int j = 0; j < n; ++j) {
                const auto& c = batch[j];
                check(c.tag < pages && !seen[c.tag], "completion repeated or wrong");
                check(c.ok && c.bytes == page, "completion failed");
                check((c.tag % 2 == 0) == (f == &first), "completion reached the wrong open");
                seen[c.tag] = true;
                ++received;
                for (unsigned k = 0; k < page; ++k)
                    check(fx.at(c.tag)[k] == (char) ('a' + pages - 1 - c.tag), "data mismatch");
            }
        }
    }
    // The expert source reads the pack through its own handle at the same time; these reads must not move it.
    std::unique_ptr<std::FILE, FileCloser> plain(open_file_utf8(fx.name, "rb"));
    check(plain != nullptr, "cannot open the file for positional reads");
    check(fseek64(plain.get(), page + 23) == 0, "cannot position the positional reader");
    call_ok(first.submit(0, fx.at(0), page, 100, error), error, "submit beside a positional reader");
    Completion c;
    check(first.wait(&c, 1, 5000) == 1 && c.ok && c.tag == 100 && c.bytes == page && fx.at(0)[0] == 'a',
          "read beside a positional reader failed");
    check(std::ftell(plain.get()) == (long) page + 23, "the reads moved the positional reader");
    plain.reset();
    call_ok(second.submit(0, fx.at(1), page, 101, error), error, "read after the other handle closed");
    check(second.wait(&c, 1, 5000) == 1 && c.ok && c.tag == 101 && c.bytes == page && fx.at(1)[0] == 'a',
          "closing the other handle invalidated this open");
    for (unsigned i = 0; i < pages; ++i)
        call_ok(second.submit((uint64_t) i * page, fx.at(i), page, i, error), error, "submit before drain");
    second.close();
    std::memset(fx.memory, 0, (size_t) pages * page);
    call_ok(second.open(fx.name, error, 1), error, "reopen");
    check(second.wait(&c, 1, 0) == 0, "close left stale completions");
    call_ok(second.submit((pages - 1) * page, fx.at(0), page, 200, error), error, "reopened submit");
    check(second.wait(&c, 1, 5000) == 1 && c.ok && c.tag == 200 && c.bytes == page && fx.at(0)[0] == 'l',
          "reopened read returned incorrect data");
    second.close();
    check(!second.submit(0, fx.at(0), page, 0, error), "closed file accepted a read that cannot complete");
    check(!second.open("basalt-direct-test-no-such-file.bin", error, 1) && !second.is_open(),
          "open of a missing file accepted");
    check(!second.open(fx.name, error, -1) && !second.is_open(), "negative worker count accepted");
    check(!second.open(fx.name, error, 257) && !second.is_open(), "excessive worker count accepted");
}

/// A file the process also maps and reads through a cached descriptor, read with either open: every read completes
/// once with the file's bytes, including the short last page and reads past the end.
static void test_mapped_elsewhere() {
    using namespace basalt::platform;
    const uint32_t page = DirectFile::alignment();
    constexpr unsigned pages = 40, tail = 123, past_end = 4;
    Fixture fx(pages + 1, 0);
    {
        std::ofstream output(fx.name, std::ios::binary | std::ios::trunc);
        for (unsigned i = 0; i < pages; ++i) {
            const std::string content(page, (char) ('A' + i % 26));
            output.write(content.data(), content.size());
        }
        const std::string last(tail, 'z');
        output.write(last.data(), last.size());
        check((bool) output, "cannot write the test contents");
    }
    const uint64_t file_bytes = (uint64_t) pages * page + tail;
    // The other users: a view whose pages are touched, and a cached descriptor that has read and stays open.
    std::string why;
    std::optional<MappedFile> view = MappedFile::open_ro(fx.name, &why);
    check(view.has_value() && view->size() == file_bytes, "cannot map the test file");
    unsigned sum = 0;
    for (uint64_t i = 0; i < file_bytes; i += page) sum += view->data()[i];
    check(sum > 0, "the mapped view reads as zeros");
    struct Fd { int fd; ~Fd() { close_fd(fd); } } cached{open_fd_utf8(fx.name)};
    check(cached.fd >= 0, "cannot open a cached descriptor");
    char head[64];
    check(pread_all(cached.fd, head, sizeof head, page + 5) && head[0] == 'B', "the cached read failed");
    for (bool mapped : {false, true}) {
        DirectFile f;
        std::string error;
        call_ok(f.open(fx.name, error, 2, mapped), error, "open beside a mapping");
        check(f.size() == file_bytes, "size mismatch beside a mapping");
        const unsigned reads = pages + 1 + past_end;   // the short last page, whole pages, then past the end
        std::memset(fx.memory, 0, (size_t) reads * page);
        for (unsigned i = 0; i < reads; ++i) {
            const unsigned at = i <= pages ? pages - i : pages + 1 + (i - pages);
            call_ok(f.submit((uint64_t) at * page, fx.at(i), page, i, error), error, "submit beside a mapping");
        }
        std::vector<bool> seen(reads, false);
        unsigned received = 0;
        while (received < reads) {
            Completion batch[8];
            const int n = f.wait(batch, 8, 5000);
            check(n > 0, "completion timed out beside a mapping");
            for (int j = 0; j < n; ++j) {
                const Completion& c = batch[j];
                check(c.tag < reads && !seen[c.tag], "completion repeated or wrong beside a mapping");
                seen[c.tag] = true;
                ++received;
                check(c.ok, "read failed beside a mapping");
                if (c.tag == 0) {
                    check(c.bytes == tail && fx.at(0)[0] == 'z' && fx.at(0)[tail - 1] == 'z', "short last page wrong");
                } else if (c.tag <= pages) {
                    const char want = (char) ('A' + (pages - c.tag) % 26);
                    check(c.bytes == page && fx.at(c.tag)[0] == want && fx.at(c.tag)[page - 1] == want,
                          "page data wrong beside a mapping");
                } else {
                    check(c.bytes == 0, "a read past the end returned bytes");
                }
            }
        }
        Completion extra;
        check(f.wait(&extra, 1, 0) == 0, "extra completion beside a mapping");
        f.close();
    }
    check(view->data()[file_bytes - 1] == 'z', "the view changed under the reads");
}


int main() {
    const std::string name = temp_path();
    void* memory = DirectFile::alloc_aligned(4096 * 18);
    try {
        check(memory != nullptr, "buffer allocation");
        {
            std::ofstream f(name, std::ios::binary);
            const std::string bytes(4096 * 16 + 17, 'x');
            f.write(bytes.data(), (std::streamsize) bytes.size());
            check(f.is_open() && (bool) f, ("cannot write the test file to " + name).c_str());
        }
        DirectFile f;
        std::string error;
        call_ok(f.open(name, error), error, "open");
        check(!f.submit(1, memory, 4096, 0, error), "unaligned request accepted");
        for (unsigned i = 0; i < 17; ++i)
            call_ok(f.submit(i * 4096, static_cast<char*>(memory) + i * 4096, 4096, i, error), error, "submit");
        std::vector<bool> seen(17);
        unsigned count = 0;
        while (count < 17) {
            Completion c[17]; int n = f.wait(c, 17, 5000);
            check(n > 0, "completion timeout");
            for (int i = 0; i < n; ++i) {
                check(c[i].tag < 17 && !seen[c[i].tag], "wrong or repeated completion");
                seen[c[i].tag] = true; ++count;
                check(c[i].ok && c[i].bytes == (c[i].tag == 16 ? 17u : 4096u), "incorrect full/EOF completion");
                check(static_cast<char*>(memory)[c[i].tag * 4096] == 'x', "read data mismatch");
            }
        }
        std::jthread waker([&] { f.wake(); });
        Completion c; check(f.wait(&c, 1, 5000) == 1 && c.tag == DirectFile::WAKE_TAG, "wake lost");
        waker.join();
        call_ok(f.submit(0, memory, 4096, 99, error), error, "submit before close");
        f.close(); // Must drain outstanding buffer writes before returning.
        std::memset(memory, 0, 4096);
        call_ok(f.open(name, error), error, "reopen");
        check(f.wait(&c, 1, 0) == 0, "stale completion on reopen");
        call_ok(f.submit(0, memory, 4096, 100, error), error, "submit after reopen");
        check(f.wait(&c, 1, 5000) == 1 && c.ok && c.tag == 100, "reopened read failed");
        f.close();
        test_bounded_opens();
        test_mapped_elsewhere();
        DirectFile::free_aligned(memory); std::remove(name.c_str());
        std::cout << "direct_file_async: passed queued reads, short EOF, wake, close drain, reopen, beside a view\n";
        return 0;
    } catch (const std::exception& e) {
        DirectFile::free_aligned(memory); std::remove(name.c_str());
        std::cerr << e.what() << '\n'; return 1;
    }
}
