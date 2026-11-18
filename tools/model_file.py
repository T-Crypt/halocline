"""Read and write the .halo model file (the layout in include/halo/core/model_format.hpp).

    w = ModelWriter(path)
    w.add_bytes("meta", json_bytes)
    w.add_file("experts", "/path/experts.bin")        # streamed and hashed while copied
    w.add_chunks("dense", chunk_iterator)             # streamed
    w.close()

    m = ModelFile(path)                               # version 2 only; versions=(1,) for the migration's input
    m.names(); m.section("meta") -> (offset, bytes); m.read("meta") -> bytes; m.verify("dense") -> bool

    append_section(path, "name", "/path/source", new_meta, undo_dir)   # no other section moves or changes
    undo_append(path, undo_dir)

    python tools/model_file.py verify PACK [--section NAME ...]      # every section against its SHA-256
"""
from __future__ import annotations

import hashlib
import os
import struct
import sys
import time

MAGIC = b"HALOCLMF"
LEGACY_MAGIC = b"BASALTMF"      # every file written before the rename: the reader accepts it, the writer does not write it
VERSION = 2
MIGRATION = "tools/migrations/halo_v1_to_v2.py"
SECTION_ALIGN = 2 << 20
HEADER_BYTES = 64 << 10
TENSOR_ALIGN = 4096                            # inside the `gguf` section
HEADER = struct.Struct("<8sIIQ40s")           # 64 bytes
ENTRY = struct.Struct("<64sQQ32s16s")         # 128 bytes
MAX_SECTIONS = (HEADER_BYTES - HEADER.size) // ENTRY.size
COPY_CHUNK = 64 << 20


def _align(n: int) -> int:
    return (n + SECTION_ALIGN - 1) // SECTION_ALIGN * SECTION_ALIGN


class Stream:
    """A part of `add_parts` produced on the fly: `size` bytes yielded by the iterable `chunks`."""

    def __init__(self, size: int, chunks):
        self.size, self.chunks = size, chunks


class ModelWriter:
    """Sections are appended in order; the header and table are written by close()."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "wb")
        self.f.write(b"\0" * HEADER_BYTES)
        self.entries: list[tuple[str, int, int, bytes]] = []

    def _start(self, name: str) -> int:
        if len(name.encode()) >= 64:
            raise ValueError(f"section name too long: {name}")
        if any(e[0] == name for e in self.entries):
            raise ValueError(f"duplicate section: {name}")
        if len(self.entries) >= MAX_SECTIONS:
            raise ValueError("too many sections")
        off = _align(self.f.tell())
        self.f.write(b"\0" * (off - self.f.tell()))
        return off

    def add_bytes(self, name: str, data: bytes) -> None:
        off = self._start(name)
        self.f.write(data)
        self.entries.append((name, off, len(data), hashlib.sha256(data).digest()))

    def add_file(self, name: str, src: str, offset: int = 0, size: int | None = None) -> None:
        """Copy `size` bytes of `src` from `offset` (the whole rest of the file by default)."""
        off = self._start(name)
        h = hashlib.sha256()
        left = (os.path.getsize(src) - offset) if size is None else size
        n = 0
        with open(src, "rb") as s:
            s.seek(offset)
            while left > 0:
                b = s.read(min(COPY_CHUNK, left))
                if not b:
                    raise IOError(f"{src}: short read")
                h.update(b)
                self.f.write(b)
                left -= len(b)
                n += len(b)
        self.entries.append((name, off, n, h.digest()))

    def add_chunks(self, name: str, chunks) -> int:
        """One section from an iterator of byte chunks (streamed, hashed as written).  Returns its size."""
        off = self._start(name)
        h = hashlib.sha256()
        n = 0
        for b in chunks:
            h.update(b)
            self.f.write(b)
            n += len(b)
        self.entries.append((name, off, n, h.digest()))
        return n

    def add_parts(self, name: str, parts, align: int = TENSOR_ALIGN) -> list[int]:
        """One section of several parts, each at a multiple of `align` (zero padded). A part is bytes, a Stream or
        (src_path, offset, size); returns each part's offset. The hash covers the whole section."""
        off = self._start(name)
        h = hashlib.sha256()
        pos, offsets = 0, []

        def put(b: bytes) -> None:
            nonlocal pos
            h.update(b)
            self.f.write(b)
            pos += len(b)

        for part in parts:
            pad = (-pos) % align
            if pad:
                put(b"\0" * pad)
            offsets.append(pos)
            if isinstance(part, (bytes, bytearray)):
                put(bytes(part))
                continue
            if isinstance(part, Stream):
                n0 = pos
                for b in part.chunks:
                    put(b)
                if pos - n0 != part.size:
                    raise IOError(f"stream part: {pos - n0} bytes, expected {part.size}")
                continue
            src, o, size = part
            left = size
            with open(src, "rb") as s:
                s.seek(o)
                while left > 0:
                    b = s.read(min(COPY_CHUNK, left))
                    if not b:
                        raise IOError(f"{src}: short read")
                    put(b)
                    left -= len(b)
        self.entries.append((name, off, pos, h.digest()))
        return offsets

    def close(self) -> None:
        end = self.f.tell()
        self.f.seek(0)
        self.f.write(HEADER.pack(MAGIC, VERSION, len(self.entries), end, b"\0" * 40))
        for name, off, n, digest in self.entries:
            self.f.write(ENTRY.pack(name.encode(), off, n, digest, b"\0" * 16))
        self.f.close()


def _sync(f) -> None:
    f.flush()
    os.fsync(f.fileno())


def _write_zeros(f, n: int) -> None:
    while n > 0:
        k = min(COPY_CHUNK, n)
        f.write(b"\0" * k)
        n -= k


def append_section(path: str, name: str, src: str, new_meta: bytes, undo_dir: str) -> None:
    """Adds a section to a file without moving or changing any other: the section and `new_meta` go past the file's end,
    then one write replaces the header and table, then the old meta's bytes, now a gap, are zeroed. Until that one write the
    file reads as it did. `undo_dir` (new) keeps what undo_append restores."""
    m = ModelFile(path)
    if name in m.sections or len(name.encode()) >= 64:
        raise ValueError(f"{path}: cannot add a section named {name}")
    with open(path, "r+b") as f:
        head = f.read(HEADER_BYTES)
        magic, version, n, file_bytes, reserved = HEADER.unpack_from(head, 0)
        if n + 1 > MAX_SECTIONS:
            raise ValueError(f"{path}: the section table is full")
        if os.path.getsize(path) != file_bytes:
            raise ValueError(f"{path}: {os.path.getsize(path)} bytes, its table says {file_bytes}")
        if not m.verify("meta"):
            raise ValueError(f"{path}: its meta section does not match its hash")
        old_off, old_size = m.section("meta")
        os.makedirs(undo_dir)
        with open(os.path.join(undo_dir, "header.bin"), "wb") as u:
            u.write(head)
            _sync(u)
        with open(os.path.join(undo_dir, "meta.bin"), "wb") as u:
            u.write(m.read("meta"))
            _sync(u)

        off = _align(file_bytes)
        f.seek(file_bytes)
        _write_zeros(f, off - file_bytes)
        h = hashlib.sha256()
        size = 0
        with open(src, "rb") as s:
            while b := s.read(COPY_CHUNK):
                h.update(b)
                f.write(b)
                size += len(b)
        meta_off = _align(off + size)
        _write_zeros(f, meta_off - (off + size))
        f.write(new_meta)
        end = meta_off + len(new_meta)
        _sync(f)

        table = [(k, v[0], v[1], v[2]) for k, v in m.sections.items() if k != "meta"]
        table += [(name, off, size, h.digest()), ("meta", meta_off, len(new_meta), hashlib.sha256(new_meta).digest())]
        new_head = HEADER.pack(magic, version, len(table), end, reserved)
        new_head += b"".join(ENTRY.pack(k.encode(), o, b, d, b"\0" * 16) for k, o, b, d in table)
        f.seek(0)
        f.write(new_head + b"\0" * (HEADER_BYTES - len(new_head)))
        _sync(f)

        f.seek(old_off)
        _write_zeros(f, old_size)
        _sync(f)


def undo_append(path: str, undo_dir: str) -> None:
    """Restores the file append_section changed, bit for bit."""
    with open(os.path.join(undo_dir, "header.bin"), "rb") as u:
        head = u.read()
    with open(os.path.join(undo_dir, "meta.bin"), "rb") as u:
        old_meta = u.read()
    _, _, n, file_bytes, _ = HEADER.unpack_from(head, 0)
    old_off = None
    for i in range(n):
        name, off, _, _, _ = ENTRY.unpack_from(head, HEADER.size + i * ENTRY.size)
        if name.rstrip(b"\0") == b"meta":
            old_off = off
    if old_off is None or len(head) != HEADER_BYTES:
        raise ValueError(f"{undo_dir}: not what append_section kept")
    with open(path, "r+b") as f:
        f.seek(old_off)
        f.write(old_meta)
        _sync(f)
        f.seek(0)
        f.write(head)
        _sync(f)
        f.truncate(file_bytes)
        _sync(f)


class ModelFile:
    def __init__(self, path: str, versions: tuple[int, ...] = (VERSION,)):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(HEADER_BYTES)
        if len(head) < HEADER.size:
            raise ValueError(f"{path}: too short to be a model file")
        magic, version, n, file_bytes, _ = HEADER.unpack_from(head, 0)
        if magic not in (MAGIC, LEGACY_MAGIC):
            raise ValueError(f"{path}: not a Halocline model file")
        if version not in versions:
            hint = f"; upgrade it with {MIGRATION}" if version < VERSION else ""
            raise ValueError(f"{path}: Halocline model file version {version}, this build reads version {VERSION}{hint}")
        self.version = version
        if os.path.getsize(path) < file_bytes:
            raise ValueError(f"{path}: truncated ({os.path.getsize(path)} of {file_bytes} bytes)")
        self.sections: dict[str, tuple[int, int, bytes]] = {}
        for i in range(n):
            name, off, size, digest, _ = ENTRY.unpack_from(head, HEADER.size + i * ENTRY.size)
            self.sections[name.rstrip(b"\0").decode()] = (off, size, digest)

    def names(self) -> list[str]:
        return list(self.sections)

    def section(self, name: str) -> tuple[int, int]:
        off, size, _ = self.sections[name]
        return off, size

    def read(self, name: str) -> bytes:
        off, size = self.section(name)
        with open(self.path, "rb") as f:
            f.seek(off)
            return f.read(size)

    def verify(self, name: str) -> bool:
        off, size, digest = self.sections[name]
        got, n = hash_range(self.path, off, size)
        return n == size and got == digest


# ------------------------------------------------------------------ verify a pack (the CLI)
def hash_range(path: str, off: int, size: int, progress=None, name: str = "") -> tuple[bytes, int]:
    """SHA-256 of `size` bytes at `off`, one COPY_CHUNK at a time so a section is never held in memory.
    `progress(name, done, total)` is called after each chunk. Returns (digest, bytes hashed)."""
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        f.seek(off)
        while done < size:
            b = f.read(min(COPY_CHUNK, size - done))
            if not b:
                break
            h.update(b)
            done += len(b)
            if progress is not None:
                progress(name, done, size)
    return h.digest(), done


class _Progress:
    """One rewritten line on stderr while a large section is hashed; nothing when stderr is not a terminal, so a
    piped run gets only the section lines."""
    STEP = 1 << 30

    def __init__(self, stream):
        self.stream, self.name, self.next = stream, None, 0

    def __call__(self, name: str, done: int, total: int) -> None:
        if not self.stream.isatty():
            return
        if name != self.name:
            self.name, self.next = name, 0
        if done < self.next and done != total:
            return
        self.next = done + self.STEP
        self.stream.write(f"\r  {name} {done / (1 << 30):.1f} of {total / (1 << 30):.1f} GiB")
        self.stream.flush()

    def clear(self) -> None:
        if self.stream.isatty():
            self.stream.write("\r" + " " * 78 + "\r")
            self.stream.flush()
        self.name = None


def check_pack(path: str, only: list[str] | None = None, out=None) -> int:
    """Print the header checks ModelFile::open makes, then one line per section: its size and whether its bytes hash to
    the SHA-256 in the directory. The hash streams, so a 17 GB pack is never held in memory.

    `only` limits the sections checked; the header is checked either way. Returns 0 when everything matches, 1 when
    anything does not, 2 when the file or a requested section name is wrong."""
    out = out or sys.stdout
    t0 = time.monotonic()
    try:
        actual = os.path.getsize(path)
    except OSError as e:
        print(f"verify: {e}", file=sys.stderr)
        return 2
    out.write(f"{path}: {actual:,} bytes ({actual / (1 << 30):.2f} GiB)\n")
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
    if len(head) < HEADER_BYTES:
        out.write(f"header: BAD: {len(head)} bytes, a pack starts with a {HEADER_BYTES}-byte header\n")
        return 1
    magic, version, n, file_bytes, _ = HEADER.unpack_from(head, 0)
    problems = []
    if magic == MAGIC:
        how = ""
    elif magic == LEGACY_MAGIC:
        how = " (the spelling files written before the rename carry)"
    else:
        problems.append(f"magic {magic!r} is neither {MAGIC!r} nor {LEGACY_MAGIC!r}")
        how = ""
    if version != VERSION:
        problems.append(f"version {version}, this build reads version {VERSION}")
    if n > MAX_SECTIONS:
        problems.append(f"{n} sections, the directory holds {MAX_SECTIONS}")
    if actual < file_bytes:
        problems.append(f"truncated: it declares {file_bytes:,} bytes")
    out.write(f"header: magic {magic.decode('ascii', 'replace')}{how}, version {version}, {n} sections, "
              f"{file_bytes:,} bytes declared, {actual:,} on disk")
    out.write("  ok\n" if not problems else "  BAD: " + "; ".join(problems) + "\n")
    if actual > file_bytes:
        out.write(f"note: {actual - file_bytes:,} bytes after the declared end (the reader ignores them)\n")
    if problems:
        out.write("verify: FAIL, the header is not one this engine opens\n")
        return 1

    entries = []
    for i in range(n):
        name_b, off, size, digest, _ = ENTRY.unpack_from(head, HEADER.size + i * ENTRY.size)
        entries.append((name_b.rstrip(b"\0").decode("utf-8", "replace"), off, size, digest))
    names = [e[0] for e in entries]
    if only:
        unknown = [s for s in only if s not in names]
        if unknown:
            print(f"verify: no section named {', '.join(unknown)}; the pack has {', '.join(names)}", file=sys.stderr)
            return 2
        entries = [e for e in entries if e[0] in only]

    prog = _Progress(sys.stderr)
    bad, hashed = 0, 0
    for name, off, size, digest in entries:
        if off % SECTION_ALIGN or off < HEADER_BYTES or off > file_bytes or size > file_bytes - off:
            prog.clear()
            out.write(f"  {name:<44} {size:>14,} bytes  BAD: outside the declared file or off the "
                      f"{SECTION_ALIGN >> 20} MiB alignment\n")
            bad += 1
            continue
        got, n_read = hash_range(path, off, size, prog, name)
        prog.clear()
        hashed += n_read
        ok = n_read == size and got == digest
        bad += 0 if ok else 1
        out.write(f"  {name:<44} {size:>14,} bytes  {'ok' if ok else 'BAD'}")
        if not ok:
            out.write(f": SHA-256 {got.hex()[:16]}..., directory {digest.hex()[:16]}..." if n_read == size else
                      f": short read, {n_read:,} of {size:,} bytes")
        out.write("\n")
        out.flush()

    secs = time.monotonic() - t0
    what = f"{len(entries)} of {len(names)} sections" if only else f"{len(names)} sections"
    out.write(f"verify: {what}, {hashed:,} bytes hashed in {secs:.1f} s, {bad} bad"
              f" -> {'OK' if not bad else 'FAIL'}\n")
    return 0 if not bad else 1


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="model_file.py", description="Read, write and verify .halo model files.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="hash every section and compare it with the directory")
    v.add_argument("pack")
    v.add_argument("--section", action="append", metavar="NAME", help="check only this section (repeatable)")
    a = ap.parse_args(argv)
    return check_pack(a.pack, a.section)


if __name__ == "__main__":
    raise SystemExit(main())

