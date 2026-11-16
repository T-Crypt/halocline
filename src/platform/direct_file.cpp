// src/platform/direct_file.cpp - see include/basalt/platform/direct_file.hpp.
#include "basalt/platform/direct_file.hpp"

#include "basalt/platform/file.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace basalt::platform {

double now_us() {
    using namespace std::chrono;
    return (double) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

void* DirectFile::alloc_aligned(size_t bytes) {
    void* p = nullptr;
    return posix_memalign(&p, alignment(), bytes) == 0 ? p : nullptr;
}

void DirectFile::free_aligned(void* p) {
    if (p == nullptr) return;
    std::free(p);
}

namespace {
/// A queued read: `submit` only queues it; the pool's threads issue it .  One thread
/// issuing every read would limit the n-gram table's prompt reads to the kernel time per read, while an NVMe
/// drive serves several times that at depth.
struct Pending {
    uint64_t offset;
    void* buffer;
    uint32_t length;
    uint64_t tag;
};

bool valid_workers(int workers, std::string& err) {
    if (workers >= 0 && workers <= 256) return true;
    err = "DirectFile: workers must be 0 (default) or 1..256";
    return false;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ POSIX
// A pool of threads, each doing blocking O_DIRECT preads, so reads run in parallel (the thread
// count is the queue depth).
struct DirectFile::Impl {
    int fd = -1;
    uint64_t size = 0;
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    std::deque<Pending> queue;
    std::deque<Completion> done;
    bool stop = false;
    std::vector<std::thread> pool;

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv_work.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            lk.unlock();
            const ssize_t got = pread(fd, p.buffer, p.length, (off_t) p.offset);
            lk.lock();
            // only a waiter that found the list empty is blocked, so only the first completion needs to wake it
            const bool was_empty = done.empty();
            done.push_back(Completion{p.tag, got < 0 ? 0u : (uint32_t) got, got >= 0});
            if (was_empty) cv_done.notify_all();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_work.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() { close(); delete impl_; }

bool DirectFile::open(const std::string& path, std::string& err, int workers, bool mapped_elsewhere) {
    close();
    if (!valid_workers(workers, err)) return false;
    (void) mapped_elsewhere;   // a mapping of the file does not hold up O_DIRECT reads of it here
    impl_->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (impl_->fd < 0) { err = "DirectFile: cannot open " + path; return false; }
    return finish_open(path, err, workers);
}

bool DirectFile::finish_open(const std::string& path, std::string& err, int workers) {
    struct stat st;
    if (fstat(impl_->fd, &st) != 0) { err = "DirectFile: cannot size " + path; close(); return false; }
    impl_->size = (uint64_t) st.st_size;
    // one blocking read per thread: a drive serves random pages at its full rate only with a deep queue
    const int n = workers == 0 ? 256 : workers;
    try {
        impl_->pool.reserve(n);
        for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { impl_->worker(); });
    } catch (const std::exception& e) {
        close();
        err = "DirectFile: cannot start issuing threads: " + std::string(e.what());
        return false;
    }
    return true;
}

void DirectFile::close() {
    impl_->stop_pool();
    if (impl_->fd >= 0) ::close(impl_->fd);
    impl_->fd = -1;
    impl_->size = 0;
    impl_->done.clear();
    impl_->queue.clear();
}

bool DirectFile::is_open() const { return impl_->fd >= 0; }
bool DirectFile::bypasses_cache() const { return is_open(); }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (!is_open()) { err = "DirectFile: not open"; return false; }
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
    }
    impl_->cv_work.notify_one();
    return true;
}

void DirectFile::wake() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->done.push_back(Completion{WAKE_TAG, 0, true});
    impl_->cv_done.notify_all();
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    if (max <= 0 || !is_open()) return 0;
    std::unique_lock<std::mutex> lk(impl_->mu);
    if (impl_->done.empty() && timeout_ms != 0) {
        auto ready = [&] { return !impl_->done.empty(); };
        if (timeout_ms < 0) impl_->cv_done.wait(lk, ready);
        else impl_->cv_done.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
    }
    int n = 0;
    while (n < max && !impl_->done.empty()) {
        out[n++] = impl_->done.front();
        impl_->done.pop_front();
    }
    return n;
}

}  // namespace basalt::platform
