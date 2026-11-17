// Selftest for the optional MAP_SHARED backing of PinnedArena.
#include "halo/core/pinned.hpp"
#include "halo/platform/file.hpp"
#include "halo/platform/system.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    // the arena file is made by the arena itself: a path that does not exist yet is the fresh case
    const std::string path_str = halo::platform::temp_dir() + "/halo-shared-arena-" +
                                 std::to_string(halo::platform::process_id());
    const char* path = path_str.c_str();
    std::remove(path);

    int fail = 0;
    const uint64_t bytes = 64u << 10;
    const uint64_t arena_hash = 0x0123456789abcdefull;
    const std::vector<uint64_t> no_bounds;
    {
        halo::core::PinnedArena first(bytes, no_bounds, path, arena_hash);
        if (!first.valid()) {
            std::fprintf(stderr, "first shared arena failed: %s\n", first.note.c_str());
            fail = 1;
        } else {
            first.data()[123] = 0x5a;

            halo::core::PinnedArena second(bytes, no_bounds, path, arena_hash);
            if (!second.valid()) {
                std::fprintf(stderr, "second shared arena failed: %s\n", second.note.c_str());
                fail = 1;
            } else if (second.data()[123] != 0x5a) {
                std::fprintf(stderr, "shared mapping did not expose the same bytes\n");
                fail = 1;
            }

            halo::core::PinnedArena wrong_hash(bytes, no_bounds, path, arena_hash + 1);
            if (wrong_hash.valid() || wrong_hash.note.find("model hash") == std::string::npos) {
                std::fprintf(stderr, "hash mismatch was not refused: %s\n", wrong_hash.note.c_str());
                fail = 1;
            }

            halo::core::PinnedArena wrong_size(bytes * 2, no_bounds, path, arena_hash);
            if (wrong_size.valid() || wrong_size.note.find("expected") == std::string::npos) {
                std::fprintf(stderr, "size mismatch was not refused: %s\n", wrong_size.note.c_str());
                fail = 1;
            }
        }
    }

    if (std::remove(path) != 0) {
        std::perror("remove");
        fail = 1;
    }
    std::printf("pinned_shared_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
