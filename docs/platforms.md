# Platforms

Linux x64, NVIDIA Ada (sm_89). The engine checks the compute capability at start and refuses anything else, and the
build refuses any `CMAKE_CUDA_ARCHITECTURES` except 89.

| | Linux |
| --- | --- |
| OS | x86-64, AVX2 (AVX-512 used when present) |
| Compiler, build | GCC with C++20 support (16.2.1 on the reference machine), CMake 3.24+, Ninja |
| CUDA, Python | 13.0 or newer, 3.10+ |
| Driver | 615 branch or newer (615.78.08 and CUDA 13.4 on the reference machine) |

## Build

`BUILD_TESTING` is off by default, so give it when you want the suites:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=ON
ninja -C build
ctest --test-dir build -L cpu -j 16 --output-on-failure
```

`bin/halocline` (with a `halo` symlink), `bin/halocline-bench`, `bin/halocline-probe` and `bin/halocline-vision` land
in `bin/`. `ctest -L gpu` runs the kernel suite and needs the card to itself; [`docs/testing.md`](testing.md) lists
what each suite covers.

## Container image

`Dockerfile` builds the engine, the server and the server's venv into one image on a CUDA 13 base: the release tree,
with the venv already installed. Building needs no GPU; the image is compiled for sm_89, the CMake default.

```sh
docker build -t halo .
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$PWD/appdata/halo:/data" -v "$PWD/models:/models" -p 8080:8080 halo
```

`/data` holds `halo_config.json` (v2; `tools/migrations/config_v1_to_v2.py` converts an older file) and takes the
server's folder per model under `~/.cache/halo`, where `HOME` points: its engine log and, when the conversation disk
tier is on, that tier. A pack is read where it lies, so the packs go on `/models` and the config names them by their
mount path (`/models/<pack>.halo`); paths in the config are relative to the config file. The entrypoint is
`halocline serve --config /data/halo_config.json --host 0.0.0.0`, and arguments after the image name follow it
(`--api-key KEY`). The served port is `server.port` (8080) and the mapped host port has to match it; `--gpu` picks the
card the container sees.

The container path is untested: no `halocline-bench` run and no KLD run has been measured inside the image, so every
number in this repo describes native runs.

## Platform code

OS code lives only in `src/platform/` and `include/halo/platform/`. Anywhere else there is no `_WIN32`, `_MSC_VER`,
`__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS call (mmap, madvise, pread, pthread/affinity,
fork/exec/pipe, dlopen, /proc or /sys reads); `tools/check_platform_boundary.sh` enforces it against an allowlist that
may only shrink.

The seams have one implementation each, Linux. The Windows bodies were carried from Basalt v0.4.1 (`d3b75742`) and
removed in `575a92c0`. What remains is `src/platform/halo.exe.manifest` and an empty `halo_embed_manifest()` in
`CMakeLists.txt`, which change nothing on Linux. Halocline does not build or test on Windows; the README lists the
leftover Windows and Docker files as open work.

## FFmpeg for video

Video input loads FFmpeg's shared libraries at run time: `libavformat.so.N`, `libavcodec.so.N`, `libavutil.so.N` and
`libswscale.so.N` of release 6, 7, 8 or 9, newest first. Nothing is linked or shipped, and the libraries must
all be from one release. On Linux the distribution's runtime packages provide them
(for example `libavformat.so.62` for FFmpeg 8), found by the system loader. Without them video is off and the server
says why.

GPU decoding needs an FFmpeg build with NVDEC and a card with room for its surfaces; a stream NVDEC refuses, or one
whose surfaces do not fit, decodes on the CPU with the same pixels.
