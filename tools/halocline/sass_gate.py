#!/usr/bin/env python3
"""SASS identity gate for the Halocline scope removals.

  sass_gate.py hash BUILD_DIR > gates/sass-<name>.txt
  sass_gate.py compare OLD.txt NEW.txt

`hash` lists every CUDA object ninja builds in BUILD_DIR (`ninja -t targets all`, the `.cu.o` outputs), runs
`cuobjdump -sass` on each and prints `<hash> <object>`, sorted by object. The hash is the first 16 hex digits of the
SHA-256 of the SASS text with the fatbin header block removed (the `Fatbin ...` banners, their `=` rules and the
arch/code version/host/compile_size/compressed/ptxasOptions/identifier lines) and nvcc's anonymous-namespace tag
(`_GLOBAL__N__<8 hex>_`, derived from the source path, so two checkouts in different directories would otherwise differ
on every object with an anonymous namespace) replaced by a fixed one. It changes only when an instruction, a function
name or a section does. An object with no device code still hashes (to the empty-text value).

`compare` prints objects only in OLD (no longer built), only in NEW (newly built) and in both with different hashes.
It exits 1 when any object in both differs, 0 otherwise. Removed or added objects are reported, not failures: the
caller decides whether they were meant to go.

  sass_gate.py funcs BUILD_DIR > gates/sass-<name>.funcs.txt
  sass_gate.py compare-funcs OLD.funcs.txt NEW.funcs.txt

The same per kernel: `funcs` prints `<hash> <object> <function>` for every `Function :` block of every object, the
hash over that block's text alone, with the whole anonymous-namespace tag (`_GLOBAL__N__<8 hex>_<file>_<8 hex>`, whose
last part may follow the file's content) replaced by a fixed one, in the name and in the body, and every run of
whitespace collapsed to one space (cuobjdump pads its columns to the widest text in the object, so removing a kernel
re-pads the ones left; gates/*.funcs.txt before 2b8 were hashed without this). Use it when a step
removes kernels from an object: the object's hash then changes, while every kernel left in it must not.
`compare-funcs` reports functions only in OLD, only in NEW and in both with different hashes, and exits 1 on any
difference in a function both have.
"""
import hashlib
import re
import subprocess
import sys

HEADER = re.compile(r"^\s*(Fatbin \w+ code:|=+$|arch = |code version = |host = |compile_size = |compressed$|"
                    r"ptxasOptions = |identifier = )")


ANON = re.compile(r"_GLOBAL__N__[0-9a-f]{8}_")


def cuda_objects(build):
    out = subprocess.run(["ninja", "-C", build, "-t", "targets", "all"], check=True, capture_output=True,
                         text=True).stdout
    return sorted({line.split(":", 1)[0] for line in out.splitlines() if line.split(":", 1)[0].endswith(".cu.o")})


def sass_hash(path):
    text = subprocess.run(["cuobjdump", "-sass", path], check=True, capture_output=True, text=True).stdout
    kept = "".join(ANON.sub("_GLOBAL__N__x_", line) + "\n" for line in text.splitlines() if not HEADER.match(line))
    return hashlib.sha256(kept.encode()).hexdigest()[:16]


ANON_FULL = re.compile(r"_GLOBAL__N__[0-9a-f]{8}_\w*?_[0-9a-f]{8}(?=\d)|_GLOBAL__N__[0-9a-f]{8}_")
FUNC = re.compile(r"^\s*Function : (\S+)")


def func_hashes(path):
    text = subprocess.run(["cuobjdump", "-sass", path], check=True, capture_output=True, text=True).stdout
    out, name, body = {}, None, []

    def flush():
        if name is not None:
            out[name] = hashlib.sha256("".join(body).encode()).hexdigest()[:16]

    for line in text.splitlines():
        if HEADER.match(line):
            continue
        m = FUNC.match(line)
        if m:
            flush()
            name, body = ANON_FULL.sub("_ANON_", m.group(1)), []
            continue
        if name is not None:
            body.append(" ".join(ANON_FULL.sub("_ANON_", line).split()) + "\n")
    flush()
    return out


def read_funcs(path):
    rows = {}
    with open(path) as f:
        for line in f:
            h, obj, fn = line.split()
            rows[(obj, fn)] = h
    return rows


def read(path):
    rows = {}
    with open(path) as f:
        for line in f:
            h, obj = line.split(None, 1)
            rows[obj.strip()] = h
    return rows


def main(argv):
    if len(argv) == 3 and argv[1] == "hash":
        build = argv[2]
        for obj in cuda_objects(build):
            print(f"{sass_hash(build + '/' + obj)} {obj}")
        return 0
    if len(argv) == 4 and argv[1] == "compare":
        old, new = read(argv[2]), read(argv[3])
        gone = sorted(set(old) - set(new))
        added = sorted(set(new) - set(old))
        diff = sorted(o for o in set(old) & set(new) if old[o] != new[o])
        for o in gone:
            print(f"only in old: {o}")
        for o in added:
            print(f"only in new: {o}")
        for o in diff:
            print(f"DIFFERS: {o} {old[o]} -> {new[o]}")
        print(f"{len(set(old) & set(new)) - len(diff)} identical, {len(diff)} differ, {len(gone)} gone, {len(added)} new")
        return 1 if diff else 0
    if len(argv) == 3 and argv[1] == "funcs":
        build = argv[2]
        for obj in cuda_objects(build):
            for fn, h in sorted(func_hashes(build + "/" + obj).items()):
                print(f"{h} {obj} {fn}")
        return 0
    if len(argv) == 4 and argv[1] == "compare-funcs":
        old, new = read_funcs(argv[2]), read_funcs(argv[3])
        gone = sorted(set(old) - set(new))
        added = sorted(set(new) - set(old))
        diff = sorted(k for k in set(old) & set(new) if old[k] != new[k])
        for o, f in gone:
            print(f"only in old: {o} {f}")
        for o, f in added:
            print(f"only in new: {o} {f}")
        for o, f in diff:
            print(f"DIFFERS: {o} {f}")
        print(f"{len(set(old) & set(new)) - len(diff)} identical, {len(diff)} differ, {len(gone)} gone, {len(added)} new")
        return 1 if diff else 0
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
