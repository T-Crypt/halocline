// src/platform/file.cpp - see include/basalt/platform/file.hpp.
//
// One implementation per platform behind one spelling per call. The Linux half is the plain POSIX call the engine
// made there before; the Windows half is the wide-call form of it, plus what reading a CRT descriptor needs there.
#include "basalt/platform/file.hpp"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>  // std::getenv
#include <filesystem>
#include <string>
#include <system_error>

#include <fcntl.h>    // O_RDONLY, posix_fadvise
#include <sys/stat.h>  // ::stat
#include <unistd.h>   // ::open, ::close, ::pread, ::access, ::chdir, ::readlink, fileno

namespace basalt::platform {

// ----------------------------------------------------------------------------------------------- POSIX

std::FILE* open_file_utf8(const std::string& path, const std::string& mode) {
    return std::fopen(path.c_str(), mode.c_str());
}

int open_fd_utf8(const std::string& path) {
    return ::open(path.c_str(), O_RDONLY);
}

void close_fd(int fd) {
    if (fd >= 0) ::close(fd);
}

bool absolute_path(const std::string& path, std::string& out) {
    char real[4096];
    if (::realpath(path.c_str(), real) == nullptr) return false;
    out = real;
    return true;
}

int set_working_dir(const std::string& path) {
    return ::chdir(path.c_str());
}

bool path_executable(const std::string& path) {
    return ::access(path.c_str(), X_OK) == 0;
}

std::string module_path() {
    char exe[4096];
    const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return std::string();
    exe[n] = 0;
    return std::string(exe);
}

std::string temp_dir() {
    // $TMPDIR as POSIX names it, so a host whose /tmp is a small tmpfs can point scratch files at disk
    const char* env = std::getenv("TMPDIR");
    if (env != nullptr && env[0] == '/') {
        std::string dir(env);
        while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
        struct stat st{};
        if (::stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) return dir;
    }
    return "/tmp";
}

int64_t pread(int fd, void* dst, size_t n, uint64_t off) {
    return (int64_t) ::pread(fd, dst, n, (off_t) off);
}

bool file_size(const std::string& path, uint64_t& out) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) return false;
    out = (uint64_t) st.st_size;
    return true;
}

int fseek64(std::FILE* f, int64_t off, int whence) {
    return fseeko(f, (off_t) off, whence);
}

int file_fd(std::FILE* f) {
    return fileno(f);
}

void advise_dontneed(int fd, uint64_t off, uint64_t bytes) {
    // A hint, and its own failure is not the caller's problem: the pages it declines to release are reclaimed
    // under pressure. A `bytes` of 0 is the rest of the file, spelled the same way here.
    (void) posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_DONTNEED);
}

int open_unbuffered(int fd) {
    if (fd < 0) return -1;
    // The descriptor's own path, reopened: /proc/self/fd names the file a descriptor stands for, so this is a second
    // handle on it with the unbuffered flag, and the first keeps its mapping and its cached reads.
    const std::string self = "/proc/self/fd/" + std::to_string(fd);
    return ::open(self.c_str(), O_RDONLY | O_DIRECT);
}

int open_unbuffered(const std::string& path) {
    // The same flag taken from the path directly. A file system that refuses it answers -1, and the caller reads the
    // bytes cached, which is the copy the flag exists to avoid.
    return ::open(path.c_str(), O_RDONLY | O_DIRECT);
}

int open_write_utf8(const std::string& path) {
    return ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
}

int open_write_unbuffered(const std::string& path) {
    // A file system that refuses the flag answers -1, and the caller writes through the cache.
    return ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_DIRECT, 0600);
}

bool pwrite_all(int fd, const void* src, uint64_t n, uint64_t off) {
    const uint8_t* p = (const uint8_t*) src;
    while (n > 0) {
        const ssize_t put = ::pwrite(fd, p, (size_t) (n < (1ull << 30) ? n : (1ull << 30)), (off_t) off);
        if (put <= 0) {
            if (put < 0 && errno == EINTR) continue;
            return false;
        }
        p += put;
        off += (uint64_t) put;
        n -= (uint64_t) put;
    }
    return true;
}

bool sync_fd(int fd) {
    return ::fdatasync(fd) == 0;
}

namespace {
std::filesystem::path fs_path(const std::string& utf8) { return std::filesystem::path(utf8); }
std::string fs_name(const std::filesystem::path& p) { return p.filename().string(); }
}  // namespace

bool make_dirs(const std::string& path) {
    // each missing directory is made owner-only; existing ones keep their mode
    std::error_code ec;
    std::filesystem::path cur;
    for (const auto& part : fs_path(path)) {
        cur /= part;
        if (std::filesystem::is_directory(cur, ec)) continue;
        if (::mkdir(cur.c_str(), 0700) != 0 && errno != EEXIST) return false;
    }
    return std::filesystem::is_directory(fs_path(path), ec);
}


bool list_files(const std::string& dir, std::vector<DirFile>& out) {
    out.clear();
    std::error_code ec;
    std::filesystem::directory_iterator it(fs_path(dir), ec), end;
    if (ec) return false;
    for (; it != end; it.increment(ec)) {
        if (ec) return false;
        std::error_code fe;
        if (!it->is_regular_file(fe) || fe) continue;
        DirFile f;
        f.name = fs_name(it->path());
        f.bytes = (uint64_t) it->file_size(fe);
        if (fe) continue;
        const auto t = it->last_write_time(fe);
        if (fe) continue;
        f.mtime = (int64_t) std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
        out.push_back(std::move(f));
    }
    return true;
}

bool rename_replace(const std::string& from, const std::string& to) {
    std::error_code ec;
    std::filesystem::rename(fs_path(from), fs_path(to), ec);
    return !ec;
}

bool remove_file(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(fs_path(path), ec);
    return !std::filesystem::exists(fs_path(path), ec);
}

bool touch_file(const std::string& path) {
    std::error_code ec;
    std::filesystem::last_write_time(fs_path(path), std::filesystem::file_time_type::clock::now(), ec);
    return !ec;
}

int64_t file_time_now() {
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
               std::filesystem::file_time_type::clock::now().time_since_epoch()).count();
}

uint64_t disk_free_bytes(const std::string& path) {
    std::error_code ec;
    const auto s = std::filesystem::space(fs_path(path), ec);
    return ec ? 0 : (uint64_t) s.available;
}

bool pread_all(int fd, void* dst, uint64_t n, uint64_t off) {
    uint8_t* p = (uint8_t*) dst;
    while (n > 0) {
        const int64_t got = pread(fd, p, (size_t) (n < (1ull << 30) ? n : (1ull << 30)), off);
        if (got <= 0) {
            // A signal can interrupt a read before it has transferred anything; that one is restarted.
            if (got < 0 && errno == EINTR) continue;
            return false;
        }
        p += got;
        off += (uint64_t) got;
        n -= (uint64_t) got;
    }
    return true;
}

std::string install_root() {
    std::string dir = module_path();
    for (int up = 0; up < 2; ++up) {   // the executable's name, then its directory
        const size_t cut = dir.find_last_of("/\\");
        if (cut == std::string::npos || cut == 0) return std::string();
        dir.resize(cut);
    }
    return dir;
}

}  // namespace basalt::platform
