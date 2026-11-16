// src/platform/mapped_file.cpp - see include/basalt/platform/mapped_file.hpp.
#include "basalt/platform/mapped_file.hpp"

#include "basalt/platform/file.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace basalt::platform {

// --------------------------------------------------------------------------------------------- Linux

std::optional<MappedFile> MappedFile::open_ro(const std::string& path, std::string* err, const MapOptions& opt) {
    auto fail = [&](const std::string& why) -> std::optional<MappedFile> {
        if (err) *err = why + " " + path;
        return std::nullopt;
    };
    const int fd = open_fd_utf8(path);
    if (fd < 0) return fail("cannot open");
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close_fd(fd);
        return fail("cannot stat");
    }
    const uint64_t fsize = (uint64_t) st.st_size;
    if (opt.offset > fsize || (opt.bytes != 0 && opt.bytes > fsize - opt.offset)) {
        close_fd(fd);
        return fail("cannot map a range past the end of");
    }
    const uint64_t len = opt.bytes != 0 ? opt.bytes : fsize - opt.offset;
    void* p = mmap(nullptr, (size_t) len, PROT_READ, opt.shared ? MAP_SHARED : MAP_PRIVATE, fd, (off_t) opt.offset);
    close_fd(fd);   // the mapping holds the file; the descriptor is not needed
    if (p == MAP_FAILED) return fail("cannot map");
    MappedFile m;
    m.region_ = p;
    m.region_bytes_ = len;
    m.data_ = (const uint8_t*) p;
    m.size_ = len;
    m.file_size_ = fsize;
    return m;
}

void MappedFile::reset() {
    if (region_ != nullptr) munmap(region_, (size_t) region_bytes_);
    data_ = nullptr;
    size_ = file_size_ = region_bytes_ = 0;
    region_ = nullptr;
}

void MappedFile::prefetch(uint64_t off, uint64_t len) const {
    if (data_ == nullptr || off >= size_) return;
    const uint64_t start = off & ~(uint64_t) 4095;
    madvise((uint8_t*) data_ + start, (size_t) (off + len - start), MADV_WILLNEED);
}

// Only the pages wholly inside the range: a page shared with a neighbour the CPU may still read stays. MADV_DONTNEED
// unmaps them from this process; they stay in the page cache as clean, unmapped pages - the first the kernel takes
// back under pressure, and a minor fault away when they are needed again. Dropping them from the cache too
// (POSIX_FADV_DONTNEED) would make every release a re-read from the disk.
uint64_t MappedFile::release(uint64_t off, uint64_t len) const {
    if (data_ == nullptr || off >= size_) return 0;
    const uint64_t start = (off + 4095) & ~(uint64_t) 4095;
    const uint64_t end = (off + len) & ~(uint64_t) 4095;
    if (end <= start) return 0;
    if (madvise((uint8_t*) data_ + start, (size_t) (end - start), MADV_DONTNEED) != 0) return 0;
    return end - start;
}

LockResult lock_view(void* p, uint64_t bytes, const LockProgress& progress) {
    if (p != nullptr && bytes != 0) madvise(p, (size_t) bytes, MADV_WILLNEED);
    return lock_resident(p, bytes, progress);
}


}  // namespace basalt::platform
