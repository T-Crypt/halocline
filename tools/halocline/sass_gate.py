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
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
