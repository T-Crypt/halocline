# Platforms

> [!WARNING]
> Rewritten 2026-11-17 from Basalt's page. Halocline is built and tested on Linux x64 only, on Ada (sm_89). The Windows
> code paths and the container image came with the snapshot, are untested here, and the Windows paths are scheduled for
> removal.

Linux x64, NVIDIA Ada (sm_89). The engine checks the compute capability at start and refuses anything else, and the
build refuses any `CMAKE_CUDA_ARCHITECTURES` except 89.

| | Linux |
| --- | --- |
| OS | x86-64, AVX2 (AVX-512 used when present) |
| Compiler, build | GCC 13+, CMake 3.24+, Ninja |
| CUDA, Python | 13, 3.10+ |
| Driver | 615 branch or newer (CUDA 13.4 on the reference machine) |

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
ninja -C build
ctest --test-dir build -L cpu -j 16 --output-on-failure
```

`bin/halo` and the other binaries land in `bin/`. A GPU suite runs with `ctest -L gpu`; it needs the card to itself and
about 20 GB of VRAM.

## Container image

`Dockerfile` builds the engine, the server and the server's venv into one image on a CUDA 13 base: the release tree, with
the venv already installed. Building needs no GPU; the image is compiled for sm_89, the CMake default.

```sh
docker build -t halo .
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$PWD/appdata/halo:/data" -v "$PWD/models:/models" -p 8080:8080 halo
```

`/data` holds `halo_config.json` (v2; `tools/migrations/config_v1_to_v2.py` converts an older file) and takes the
server's folder per model under `~/.cache/halo`, where `HOME` points: its engine log and, when the conversation disk
tier is on, that tier. A pack is read where it lies, so the packs go on `/models` and the config names them by their
mount path (`/models/<pack>.basalt`); paths in the config are relative to the config file. The entrypoint is
`halo serve --config /data/halo_config.json --host 0.0.0.0`, and arguments after the image name follow it
(`--api-key KEY`). The served port is `server.port` (8080) and the mapped host port has to match it; `--gpu` names the
cards the container sees.

The container path is untested: no `halo-bench` run and no KLD run has been measured inside the image, so every number in
this repo describes native runs.

## Platform code

OS code lives only in `src/platform/` and `include/halo/platform/`, and in CMake `if(WIN32)` blocks. Anywhere else there
is no `_WIN32`, `_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS call (mmap, madvise,
pread, pthread/affinity, fork/exec/pipe, dlopen, /proc or /sys reads). `tools/check_platform_boundary.sh` enforces it.
The Windows bodies of each seam are carried over from Basalt and untested since the rename.

## FFmpeg for video

Video input loads FFmpeg's shared libraries at run time (libavformat, libavcodec, libavutil and libswscale of release 6,
7, 8 or 9; nothing is linked or shipped). On Linux the distribution's runtime packages provide them (`libavformat62` and
its siblings on Ubuntu 26.04, `libavformat60` on 24.04), found by the system loader. Without them video is off and the
server says why. GPU decoding needs an FFmpeg build with NVDEC (the distribution packages have it); without it frames
decode on the CPU.

## Windows

The Windows code paths came with the snapshot. Halocline does not build or test on Windows, the CMake default is sm_89
and the refusals apply there too, and the Windows files are open work in the README. Do not treat anything under
`src/platform/win32` or the `if(WIN32)` blocks as supported.
