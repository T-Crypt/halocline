// src/platform/memory.cpp - see include/halo/platform/memory.hpp.
#include "halo/platform/memory.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <mutex>
#include <string>
#include <thread>
#include <fcntl.h>
#include <malloc.h>   // malloc_trim
#include <linux/mman.h>   // MAP_HUGE_2MB
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include "halo/log.hpp"
#include "halo/platform/system.hpp"

namespace halo::platform {


LockResult lock_resident(void* p, uint64_t bytes, const LockProgress& progress) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (!progress) {
        if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
        r.ok = true;
        r.locked_bytes = bytes;
        r.note = "mlock";
        return r;
    }
    const uint64_t chunk = 256ull << 20;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (mlock((uint8_t*) p + off, (size_t) n) != 0) {
            r.note = "mlock failed (raise ulimit -l)";
            r.ok = off > 0;   // what it did lock stays locked; the rest of the region is merely pageable
            return r;
        }
        r.locked_bytes = off + n;
        progress(r.locked_bytes);
    }
    r.ok = true;
    r.note = "mlock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    munlock(p, bytes);
}

uint64_t total_physical_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

namespace {
// One "<key>: <n> kB" line of /proc/meminfo in bytes (0 when absent).
uint64_t meminfo_bytes(const char* key) {
    std::ifstream in("/proc/meminfo");
    std::string line;
    const std::string k = std::string(key) + ":";
    while (std::getline(in, line))
        if (line.compare(0, k.size(), k) == 0) return std::strtoull(line.c_str() + k.size(), nullptr, 10) * 1024;
    return 0;
}

// The OS's own reading, before a container limit lowers it (available_memory).
uint64_t os_available_memory() { return meminfo_bytes("MemAvailable"); }
}  // namespace

namespace {

bool read_cgroup_memory_stat(const std::filesystem::path& path, uint64_t current,
                             CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;
    std::ostringstream text;
    text << input.rdbuf();
    return text.good() && parse_cgroup_memory_stat(text.str(), current, stat);
}

// cgroup v1's memory.stat: the group's hierarchical counters (total_*), the same allowance as v2's.  False when the
// file or total_inactive_file is missing (the usage is then taken as it is).
bool read_cgroup_v1_memory_stat(const std::filesystem::path& path, uint64_t usage, CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;
    bool inactive_file = false;
    std::string key;
    uint64_t value = 0, active_file = 0;
    while (input >> key >> value) {
        if (key == "total_inactive_file") { stat.inactive_file = value; inactive_file = true; }
        else if (key == "total_active_file") active_file = value;
        else if (key == "total_dirty") stat.file_dirty = value;
        else if (key == "total_writeback") stat.file_writeback = value;
    }
    if (!inactive_file) return false;
    stat.file = stat.inactive_file + active_file;
    stat.current = usage;
    stat.valid = true;
    return true;
}

}  // namespace

bool parse_cgroup_memory_stat(const std::string& text, uint64_t current, CgroupMemoryStat& stat) {
    stat = CgroupMemoryStat{};
    bool inactive_file = false, file_dirty = false, file_writeback = false, file = false, shmem = false;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        if (!(fields >> key >> value)) return false;
        fields >> std::ws;
        if (!fields.eof()) return false;

        auto take = [&](bool& seen, uint64_t& out) {
            if (seen) return false;
            seen = true;
            out = value;
            return true;
        };
        if (key == "file") { if (!take(file, stat.file)) return false; }
        else if (key == "shmem") { if (!take(shmem, stat.shmem)) return false; }
        else if (key == "inactive_file") { if (!take(inactive_file, stat.inactive_file)) return false; }
        else if (key == "file_dirty") { if (!take(file_dirty, stat.file_dirty)) return false; }
        else if (key == "file_writeback") { if (!take(file_writeback, stat.file_writeback)) return false; }
    }
    if (!inactive_file || !file_dirty || !file_writeback) return false;
    stat.current = current;
    stat.valid = true;
    return true;
}

bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;

    // Clean file pages are reclaimable.  The page cache is the active and inactive lists together (`file`; the
    // inactive list alone when the file total is not reported), because the container's own reads move pages to
    // the active list.  The counters can race memory.current, so bound them to charged usage first.
    // `file` counts shared memory too (pinned host buffers, tmpfs), which reclaim never frees: it is left out.
    const uint64_t page_cache = stat.file > stat.shmem ? stat.file - stat.shmem : 0;
    uint64_t reclaimable = std::min(std::max(page_cache, stat.inactive_file), stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;
    // Keep a margin: part of the cache is mapped or locked and does not come back on demand.
    reclaimable -= std::min(reclaimable / 2, limit / 64);

    // Reclaiming clean file pages reduces usage; saturating subtraction also handles a transient over-limit read.
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

bool host_available_memory(HostMemory& m, const std::string& meminfo_arg, const std::string& self_cgroup_arg,
                           const std::string& cgroup_root_arg) {
    m = HostMemory{};
    const std::string meminfo = meminfo_arg.empty() ? halo::platform::meminfo_path() : meminfo_arg;
    const std::string self_cgroup = self_cgroup_arg.empty() ? halo::platform::self_cgroup_path() : self_cgroup_arg;
    const std::string cgroup_root = cgroup_root_arg.empty() ? halo::platform::cgroup_root() : cgroup_root_arg;
    if (meminfo.empty()) {
        // No memory accounting files and no cgroup hierarchy here: the OS reading is the whole answer, and there is
        // no container limit tighter than it.
        const uint64_t avail = os_available_memory();
        if (avail == 0) return false;
        m.available = avail;
        m.cgroup_limit = ~0ull;
        return true;
    }
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    std::ifstream info(meminfo);
    std::string line;
    uint64_t bytes = 0;
    while (std::getline(info, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (fields >> key >> value >> unit && key == "MemAvailable:" && unit == "kB" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024) bytes = value * 1024;
    }
    if (bytes == 0) return false;
    // The tightest cgroup limit of this process's group and its ancestors.  A group whose path is not visible here
    // (a container's own cgroup namespace shows its group as the mount's root) is read at that root; a group without
    // a readable limit has none of its own.  No cgroup file or line: MemAvailable alone.
    // This is a point-in-time guard, not a reservation against concurrent allocations.
    const std::filesystem::path root(cgroup_root);
    bool v2 = false;
    std::string v2_path, v1_path;   // the unified group; the v1 memory controller's group
    if (std::ifstream groups(self_cgroup); groups) {
        while (std::getline(groups, line)) {
            if (line.rfind("0::/", 0) == 0) {
                v2 = true;
                v2_path = line.substr(4);
                continue;
            }
            // v1: "N:controller[,controller]:/path"
            const size_t c1 = line.find(':'), c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            std::istringstream ctl(line.substr(c1 + 1, c2 - c1 - 1));
            for (std::string c; std::getline(ctl, c, ',');)
                if (c == "memory") v1_path = line.substr(c2 + 1);
        }
    }
    // the group's directory under `top`, or `top` itself when the group is not visible there
    auto visible_group = [](const std::filesystem::path& top, const std::string& rel) {
        const std::string sub = rel.substr(rel.rfind('/', 0) == 0 ? 1 : 0);
        if (sub.empty()) return top;
        auto path = (top / sub).lexically_normal();
        if (path.string().rfind(top.string(), 0) != 0 || !std::filesystem::is_directory(path)) return top;
        return path;
    };
    if (!v1_path.empty()) {
        // cgroup v1 (also the hybrid layout, where the unified line carries no memory controller): the memory
        // controller's memory.limit_in_bytes - memory.usage_in_bytes, with the clean inactive file cache counted as
        // free as for v2.  An unlimited group says a number near 2^63 (rounded to its page size).
        const std::filesystem::path mroot = root / "memory";
        for (auto path = visible_group(mroot, v1_path); path.string().rfind(mroot.string(), 0) == 0;
             path = path.parent_path()) {
            std::ifstream limit_file(path / "memory.limit_in_bytes"), usage_file(path / "memory.usage_in_bytes");
            uint64_t cap = 0, usage = 0;
            if (limit_file >> cap && usage_file >> usage && cap < (1ull << 62)) {
                uint64_t room = usage < cap ? cap - usage : 0;
                CgroupMemoryStat stat;
                if (read_cgroup_v1_memory_stat(path, usage, stat) && !cgroup_available_bytes(cap, stat, room)) return false;
                bytes = std::min(bytes, room);
                m.cgroup_limit = std::min(m.cgroup_limit, cap);
            }
            if (path == mroot) break;
        }
    } else if (v2) {
        for (auto path = visible_group(root, v2_path); path.string().rfind(root.string(), 0) == 0;
             path = path.parent_path()) {
            std::ifstream limit_file(path / "memory.max"), current_file(path / "memory.current");
            std::string limit;
            uint64_t current = 0;
            // the host's root group, and a group whose parent does not enable the memory controller, have no limit
            if (limit_file >> limit && current_file >> current && limit != "max") {
                try {
                    size_t consumed = 0;
                    const uint64_t cap = std::stoull(limit, &consumed);
                    if (consumed != limit.size()) return false;
                    CgroupMemoryStat stat;
                    if (!read_cgroup_memory_stat(path, current, stat)) return false;
                    uint64_t cgroup_available = 0;
                    if (!cgroup_available_bytes(cap, stat, cgroup_available)) return false;
                    bytes = std::min(bytes, cgroup_available);
                    m.cgroup_limit = std::min(m.cgroup_limit, cap);
                } catch (...) { return false; }
            }
            if (path == root) break;
        }
    }
    m.available = bytes;
    return true;
}

uint64_t available_memory() {
    HostMemory m;
    return host_available_memory(m) ? m.available : os_available_memory();
}

namespace {
std::atomic<uint64_t> g_ram_reserve{kDefaultRamReserveBytes};
}  // namespace

uint64_t ram_reserve() { return g_ram_reserve.load(); }

void set_ram_reserve(uint64_t bytes) { g_ram_reserve = bytes; }

uint64_t ram_budget(bool* known) {
    const uint64_t avail = available_memory();
    if (known) *known = avail > 0;
    if (avail == 0) return UINT64_MAX;
    const uint64_t reserve = ram_reserve();
    return avail > reserve ? avail - reserve : 0;
}

namespace {
std::atomic<bool> g_watch_stop{false};
std::atomic<bool> g_watch_on{false};
std::atomic<bool> g_resident_mode{false};
std::atomic<uint64_t> g_resident_copy{0};
}  // namespace

std::string ram_exhausted_message(uint64_t avail, bool resident, uint64_t copy_bytes, uint64_t reserve) {
    const unsigned long long mib = (unsigned long long) (avail >> 20);
    char buf[384];
    if (!resident) {
        std::snprintf(buf, sizeof buf, "halo error: RAM ran out loading (%llu MiB free); use "
                                       "--resident-experts or --mmap-experts", mib);
    } else if (copy_bytes < (1ull << 30)) {
        // a budget of 1 GiB or more would not shrink a copy this small, so it is not offered
        std::snprintf(buf, sizeof buf, "halo error: RAM ran out loading (%llu MiB free); the resident copy is only "
                                       "%llu MiB, so something else is holding the memory (other programs, or file "
                                       "pages read while loading)", mib, (unsigned long long) (copy_bytes >> 20));
    } else {
        // the copy shrunk by what the reserve is short of leaves the reserve free again
        const uint64_t keep = copy_bytes - std::min(copy_bytes, reserve > avail ? reserve - avail : 0);
        std::snprintf(buf, sizeof buf, "halo error: RAM ran out loading (%llu MiB free); use --resident-budget-gib %llu",
                      mib, (unsigned long long) std::max<uint64_t>(1, keep >> 30));
    }
    return buf;
}

void watch_resident_ram(bool resident, uint64_t copy_bytes) {
    g_resident_mode = resident;
    g_resident_copy = copy_bytes;
}

void start_memory_watchdog(uint64_t floor_bytes) {
    if (g_watch_on.exchange(true)) return;
    g_watch_stop = false;
    std::thread([floor_bytes] {
        while (!g_watch_stop.load()) {
            const uint64_t avail = available_memory();
            if (avail > 0 && avail < floor_bytes) {
                halo::log::error("halo", "%s", ram_exhausted_message(avail, g_resident_mode.load(), g_resident_copy.load(),
                                                                   ram_reserve()).c_str());
                _exit(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();   // never joined: the process may end while it sleeps
}

void stop_memory_watchdog() {
    g_watch_stop = true;
    g_watch_on = false;
}

MemoryStalls parse_memory_stalls(const std::string& text) {
    MemoryStalls m;
    bool some = false, full = false;
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(at, nl - at);
        at = nl + 1;
        const size_t k = line.find("avg10=");
        if (k == std::string::npos) continue;
        const double v = std::strtod(line.c_str() + k + 6, nullptr);
        if (line.compare(0, 5, "some ") == 0) { m.some = v; some = true; }
        else if (line.compare(0, 5, "full ") == 0) { m.full = v; full = true; }
    }
    m.known = some && full;
    return m;
}

MemoryStalls memory_stalls() {
    std::ifstream in("/proc/pressure/memory");
    if (!in) return {};
    std::string text, line;
    while (std::getline(in, line)) text += line + "\n";
    return parse_memory_stalls(text);
}

std::string ram_stop_message(uint64_t avail, uint64_t floor, uint64_t hard, const MemoryStalls& stalls,
                             uint64_t others, uint64_t own) {
    const unsigned long long mib = (unsigned long long) (avail >> 20);
    char head[256];
    if (avail < hard)
        std::snprintf(head, sizeof head, "RAM critical (%llu MiB free < %llu MiB hard floor): request stopped", mib,
                      (unsigned long long) (hard >> 20));
    else if (stalls.known)
        std::snprintf(head, sizeof head, "RAM low (%llu MiB free < %llu MiB guard) and the system stalls on memory "
                                         "(some %.0f%%, full %.0f%%): request stopped",
                      mib, (unsigned long long) (floor >> 20), stalls.some, stalls.full);
    else
        std::snprintf(head, sizeof head, "RAM low (%llu MiB free < %llu MiB guard): request stopped", mib,
                      (unsigned long long) (floor >> 20));
    char tail[160] = "";
    if (own > others)
        std::snprintf(tail, sizeof tail, "; lower --resident-budget-gib");
    else if (others >= (1ull << 20))
        std::snprintf(tail, sizeof tail, "; other programs took %llu MiB since the engine was ready",
                      (unsigned long long) (others >> 20));
    return std::string(head) + tail;
}

void release_free_heap() {
#if   defined(__GLIBC__)
    (void) malloc_trim(0);
#endif
}

namespace {
std::atomic<bool> g_guard_on{false};
std::atomic<RamLevel> g_guard_level{RamLevel::Ok};
std::atomic<uint64_t> g_guard_floor{0};
std::atomic<uint64_t> g_guard_drops{0};
std::atomic<bool> g_guard_paused{false};
std::mutex g_guard_mu;   // the reading behind the level
uint64_t g_guard_avail = 0;
MemoryStalls g_guard_stalls;
std::mutex g_judge_mu;   // one judgement at a time: the guard's thread, or an unpause
RamLevel g_judged = RamLevel::Ok;   // the level the last judgement left (under g_judge_mu)
BelowFloorClock g_below;            // how long the reading has been below the floor (under g_judge_mu)

// Judges the reading `avail` against the floor: sets the level, counts a drop below the floor, logs a change.
void guard_judge(uint64_t avail) {
    std::lock_guard<std::mutex> judge(g_judge_mu);
    const uint64_t floor = g_guard_floor.load();
    // the stall reading only matters below the floor
    const MemoryStalls stalls = ram_below_floor(avail, floor) ? memory_stalls() : MemoryStalls{};
    const uint64_t below_ms = g_below.update(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(),
        ram_below_floor(avail, floor));
    const RamLevel level = ram_level(avail, floor, kRamHardFloor, stalls, below_ms);
    const RamLevel was = g_judged;
    {
        std::lock_guard<std::mutex> lock(g_guard_mu);
        g_guard_avail = avail;
        g_guard_stalls = stalls;
    }
    if (was == RamLevel::Ok && level != RamLevel::Ok) g_guard_drops.fetch_add(1);
    g_guard_level = level;
    const unsigned long long mib = (unsigned long long) (avail >> 20);
    if (was == RamLevel::Ok && level != RamLevel::Ok)
        halo::log::warn("halo", "RAM low: %llu MiB free < %llu MiB guard", mib, (unsigned long long) (floor >> 20));
    if (level == RamLevel::Critical && was != RamLevel::Critical && stalls.known)
        halo::log::warn("halo", "RAM critical: %llu MiB free, hard floor %llu MiB, memory stalls some "
                          "%.0f%% full %.0f%%", mib, (unsigned long long) (kRamHardFloor >> 20), stalls.some, stalls.full);
    if (level == RamLevel::Critical && was != RamLevel::Critical && !stalls.known && avail >= kRamHardFloor)
        halo::log::warn("halo", "RAM critical: %llu MiB free, below the %llu MiB guard for %llu s", mib,
                          (unsigned long long) (floor >> 20), (unsigned long long) (below_ms / 1000));
    if (level == RamLevel::Low && was == RamLevel::Critical)
        halo::log::info("halo", "RAM low, no longer critical: %llu MiB free", mib);
    if (level == RamLevel::Ok && was != RamLevel::Ok) halo::log::info("halo", "RAM back: %llu MiB free", mib);
    g_judged = level;
}
}  // namespace

void ram_guard_reclaim_view(const void* base, uint64_t bytes) {
    (void) base, (void) bytes;   // MemAvailable already counts a view's pages
}

void start_ram_guard(uint64_t floor_bytes) {
    g_guard_floor = floor_bytes;
    if (g_guard_on.exchange(true)) return;
    std::thread([] {
        for (;;) {
            if (g_guard_paused.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            uint64_t avail = available_memory();
            guard_judge(avail);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();   // runs for the life of the process
}

RamLevel ram_guard_level(uint64_t* avail, uint64_t* floor, MemoryStalls* stalls) {
    if (avail || stalls) {
        std::lock_guard<std::mutex> lock(g_guard_mu);
        if (avail) *avail = g_guard_avail;
        if (stalls) *stalls = g_guard_stalls;
    }
    if (floor) *floor = g_guard_floor.load();
    return g_guard_level.load(std::memory_order_relaxed);
}

uint64_t ram_guard_drops() { return g_guard_drops.load(); }

void ram_guard_pause(bool paused) {
    if (paused) {
        g_guard_paused = true;
        return;
    }
    if (g_guard_paused.load()) {
        std::lock_guard<std::mutex> judge(g_judge_mu);
        g_below = BelowFloorClock{};   // a sleep is not time spent below the floor
    }
    if (g_guard_paused.exchange(false) && g_guard_on.load()) guard_judge(available_memory());
}

namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() { return true; }

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
}

void* alloc_aligned(size_t bytes, size_t alignment) {
    if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) return nullptr;
    void* p = nullptr;
    if (posix_memalign(&p, alignment, bytes) != 0) return nullptr;
    return p;
}

void free_aligned(void* p) {
    if (p == nullptr) return;
    std::free(p);
}

bool drop_mapped_view(const void* base, uint64_t view_bytes, uint64_t off, uint64_t bytes) {
    if (base == nullptr || bytes == 0) return true;
    // Only whole pages can be given back: a page the range shares with a neighbour still holds bytes the caller may
    // read. So the range is widened to the page boundaries and then clipped to the view.
    const long ps = sysconf(_SC_PAGE_SIZE);
    if (ps <= 0) return false;
    const uint64_t page = (uint64_t) ps;
    const uint64_t end = off + bytes;
    if (end > view_bytes) return false;
    const uint64_t start = off - off % page;
    const uint64_t remainder = end % page;
    const uint64_t extra = remainder == 0 ? 0 : page - remainder;
    const uint64_t wide = extra > view_bytes - end ? view_bytes : end + extra;
    if (wide <= start) return true;
    return madvise((void*) ((const uint8_t*) base + (size_t) start), (size_t) (wide - start), MADV_DONTNEED) == 0;
}


// ---------------------------------------------------------------------------------------------------- host regions

#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif

namespace {
constexpr uint64_t kSharedArenaHeaderBytes = 4096;
constexpr char kSharedArenaMagic[16] = "BASALT-ARENA-V1";

struct SharedArenaHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t arena_bytes;
    uint64_t arena_hash;
    uint64_t reserved[4];
};
static_assert(sizeof(SharedArenaHeader) <= kSharedArenaHeaderBytes);
}  // namespace

static void* alloc_region_raw(uint64_t bytes, RegionBacking& got, std::string& note, const std::string& shared_file,
                              uint64_t shared_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    mapping_base = nullptr;
    mapping_bytes = 0;
    if (!shared_file.empty()) {
        if (shared_hash == 0) {
            note = "shared arena requires a nonzero model hash";
            return nullptr;
        }
        if (bytes > UINT64_MAX - kSharedArenaHeaderBytes) {
            note = "shared arena size overflows its header";
            return nullptr;
        }
        const uint64_t file_bytes = kSharedArenaHeaderBytes + bytes;
        const int fd = open(shared_file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd < 0) {
            note = "cannot open shared arena " + shared_file + ": " + std::strerror(errno);
            return nullptr;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            const int e = errno;
            close(fd);
            note = "cannot stat shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }

        const bool fresh = st.st_size == 0;
        if (fresh) {
            if (ftruncate(fd, (off_t) file_bytes) != 0) {
                const int e = errno;
                close(fd);
                note = "cannot size shared arena " + shared_file + ": " + std::strerror(e);
                return nullptr;
            }
            SharedArenaHeader hdr{};
            std::memcpy(hdr.magic, kSharedArenaMagic, sizeof hdr.magic);
            hdr.version = 1;
            hdr.header_bytes = (uint32_t) kSharedArenaHeaderBytes;
            hdr.arena_bytes = bytes;
            hdr.arena_hash = shared_hash;
            const ssize_t written = ::pwrite(fd, &hdr, sizeof hdr, 0);
            if (written != (ssize_t) sizeof hdr) {
                const int e = errno;
                close(fd);
                note = "cannot write shared arena header " + shared_file + ": " +
                       (written < 0 ? std::string(std::strerror(e)) : std::string("short write"));
                return nullptr;
            }
        } else {
            if (st.st_size < 0 || (uint64_t) st.st_size != file_bytes) {
                close(fd);
                note = "shared arena " + shared_file + " is " +
                       std::to_string((unsigned long long) st.st_size) + " B, expected " +
                       std::to_string((unsigned long long) file_bytes) + " B including its header";
                return nullptr;
            }
            SharedArenaHeader hdr{};
            const ssize_t got_header = ::pread(fd, &hdr, sizeof hdr, 0);
            if (got_header != (ssize_t) sizeof hdr ||
                std::memcmp(hdr.magic, kSharedArenaMagic, sizeof hdr.magic) != 0 ||
                hdr.version != 1 || hdr.header_bytes != kSharedArenaHeaderBytes || hdr.arena_bytes != bytes) {
                close(fd);
                note = "shared arena " + shared_file + " has an incompatible or missing header";
                return nullptr;
            }
            if (hdr.arena_hash != shared_hash) {
                close(fd);
                char b[256];
                std::snprintf(b, sizeof b,
                              "shared arena %s was written for model hash %016llx, expected %016llx",
                              shared_file.c_str(), (unsigned long long) hdr.arena_hash,
                              (unsigned long long) shared_hash);
                note = b;
                return nullptr;
            }
        }

        void* map = mmap(nullptr, (size_t) file_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const int e = errno;
        close(fd);
        if (map == MAP_FAILED) {
            note = "cannot map shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }
        mapping_base = map;
        mapping_bytes = file_bytes;
        got = RegionBacking::NormalPages;
        note = "MAP_SHARED file " + shared_file + " (model hash checked)";
        return (uint8_t*) map + kSharedArenaHeaderBytes;
    }

    {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) {
            got = RegionBacking::LargePages;
            note = "hugetlb 2 MB pages";
            return p;
        }
        // MAP_HUGETLB is all-or-nothing: a pool smaller than the mapping fails exactly like an absent
        // one, and the old message guessed "no hugetlb pool configured" either way. Name the shortfall:
        // how many 2 MiB pages the mapping needs against what vm.nr_hugepages actually holds.
        const unsigned long long need = ((unsigned long long) bytes + (1ull << 21) - 1) / (1ull << 21);
        unsigned long long pool = 0;
        bool have_pool = false;
        if (std::FILE* f = std::fopen("/proc/sys/vm/nr_hugepages", "r")) {
            have_pool = std::fscanf(f, "%llu", &pool) == 1;
            std::fclose(f);
        }
        note = "MAP_HUGETLB unavailable (needed " + std::to_string(need) + " 2 MiB pages, vm.nr_hugepages=" +
               (have_pool ? std::to_string(pool) : std::string("?")) + "); using 4 KB pages";
    }
    // No hugetlb mapping: a 2 MiB-aligned anonymous mapping with MADV_HUGEPAGE, so transparent huge pages back
    // it where THP is "madvise" (the common distro default). The CPU expert pool streams whole experts out of
    // this arena; with 4 KB pages every 3 MB expert costs ~750 TLB misses.
    constexpr uint64_t kAlign = 2ull << 20;
    const uint64_t padded = bytes + kAlign;
    void* raw = mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    got = RegionBacking::NormalPages;
    if (raw == MAP_FAILED) return nullptr;
    const uintptr_t start = (uintptr_t) raw;
    const uintptr_t aligned = (start + kAlign - 1) & ~(uintptr_t) (kAlign - 1);
    if (aligned > start) munmap(raw, aligned - start);
    const uintptr_t end = aligned + bytes, raw_end = start + padded;
    if (raw_end > end) munmap((void*) end, raw_end - end);
    void* p = (void*) aligned;
    if (madvise(p, bytes, MADV_HUGEPAGE) == 0) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages requested (MADV_HUGEPAGE)";
    }
    return p;
}

void free_region(void* p, uint64_t bytes) {
    if (!p) return;
    munmap(p, bytes);
}

bool pin_budget(uint64_t&, std::string&, const char*, uint32_t, bool) { return false; }   // no OS-side budget below the driver's own limit

void* alloc_region(uint64_t bytes, RegionBacking& got, std::string& note, const std::string& shared_file,
                   uint64_t shared_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    void* p = alloc_region_raw(bytes, got, note, shared_file, shared_hash, mapping_base, mapping_bytes);
    if (p != nullptr && mapping_base == nullptr) {   // not a file mapping: the region is what gets released
        mapping_base = p;
        mapping_bytes = bytes;
    }
    return p;
}


uint64_t system_page_size() {
    const long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? (uint64_t) page : 0;
}

}  // namespace halo::platform
