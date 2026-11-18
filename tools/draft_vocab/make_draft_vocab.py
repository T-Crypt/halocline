"""Rank the vocabulary by how often a corpus writes each token: the draft vocabulary of --draft-vocab.

    make_draft_vocab.py --pack PACK --corpus TEXT [--corpus DIR ...] --n 131072 \
        --out tools/draft_vocab/data/qwen38-27b-wikitext103-code-n131072.bin \
        --provenance tools/draft_vocab/data/qwen38-27b-wikitext103-code-n131072.json

The output is little-endian int32 token ids, strictly ascending: row r of the draft head is a copy of the head's row
`ids[r]`, and the engine refuses any other shape by name.  The N chosen are the N most frequent tokens of the corpus,
ties to the lower id, with the pack's chat control tokens always in (a drafter that cannot propose an end of turn
cannot draft one).  A DIR corpus contributes every regular file under it, in path order, with a newline after each.

The tokenizer is the pack's own (`tokenizer/` section, taken from the GGUF metadata the pack carries), so the counts
are ids the engine and the pack already agree on.  `--provenance` writes what a re-run needs: the pack and its
tokenizer, each corpus with its file count, bytes and sha256, N and the corpus coverage.

Why these corpora: wikitext-103's train text is the repository's own gate corpus (tools/depth_kld cut the 40 MB file
the depth KLD bases were scored on), and the tree's sources are the code the served workloads write (JSON tool calls,
edits).  Neither is tuned per request; the head is one fixed ranking, and the verify pass still scores every row, so
the ranking can only move the acceptance rate.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

FORMAT = "int32 little-endian, strictly ascending token ids (tools/draft_vocab)"


def read_pack_tokenizer(pack: str):
    """The pack's tokenizer, from the GGUF metadata its `meta` section carries."""
    from model_file import ModelFile  # tools/model_file.py
    from tokenizer import Tokenizer  # tools/tokenizer.py (needs the `regex` package)

    meta = json.loads(ModelFile(pack).read("meta"))
    kv = meta.get("gguf_kv") or {}
    tok = Tokenizer.from_metadata(kv)
    return meta, tok


def corpus_files(path: pathlib.Path, exclude: tuple[pathlib.Path, ...] = ()) -> list[pathlib.Path]:
    """A file is itself; a directory is every regular file under it, path order, with caches (`__pycache__`, dot
    directories) left out: they are not text and a re-run would count its own bytecode.  `exclude` paths (a prefix)
    keep a corpus from counting a tree's generated files, such as this tool's own data."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts and not any(x.startswith(".") for x in p.parts)
                      and not any(p.resolve().is_relative_to(e) for e in exclude))
    raise SystemExit("--corpus %s: no such file or directory" % path)


def count_corpus(tok, files: list[pathlib.Path]) -> tuple[collections.Counter, bytes, int]:
    """Token counts over the files' text, a newline after each; the corpus's sha256 and byte count."""
    counts: collections.Counter = collections.Counter()
    digest = hashlib.sha256()
    total = 0
    for path in files:
        raw = path.read_bytes()
        digest.update(raw)
        digest.update(b"\n")
        total += len(raw)
        counts.update(tok.encode(raw.decode("utf-8", "replace")))
    return counts, digest.digest(), total


def select_ids(counts: collections.Counter, vocab: int, n: int, forced: list[int]) -> list[int]:
    """The n ids: the forced ones, then the most frequent others, ties to the lower id; returned ascending."""
    if n <= 0 or n > vocab:
        raise SystemExit("--n %d: 1..%d" % (n, vocab))
    forced_set = set(i for i in forced if 0 <= i < vocab)
    order = sorted(range(vocab), key=lambda i: (-counts[i], i))
    chosen = set(forced_set)
    if len(chosen) >= n:
        return sorted(sorted(forced_set, key=lambda i: (-counts[i], i))[:n])
    for i in order:
        if i not in chosen:
            chosen.add(i)
            if len(chosen) == n:
                break
    return sorted(chosen)


def write_ids(path: pathlib.Path, ids: list[int]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(int(i).to_bytes(4, "little", signed=True) for i in ids))
    return path.stat().st_size


def read_ids(path: pathlib.Path) -> list[int]:
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise SystemExit("%s: not a list of int32 ids" % path)
    ids = [int.from_bytes(raw[i:i + 4], "little", signed=True) for i in range(0, len(raw), 4)]
    if any(ids[i] >= ids[i + 1] for i in range(len(ids) - 1)):
        raise SystemExit("%s: the ids are not strictly ascending" % path)
    return ids


def git_commit(root: pathlib.Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True, help="the .halo pack whose tokenizer ranks the corpus")
    ap.add_argument("--corpus", action="append", required=True,
                    help="a text file, or a directory of them (every regular file, path order); repeatable")
    ap.add_argument("--exclude", action="append", default=[],
                    help="a path under a --corpus directory not to count; repeatable")
    ap.add_argument("--n", type=int, required=True, help="the draft head's rows")
    ap.add_argument("--out", required=True, help="the id file to write")
    ap.add_argument("--provenance", help="the JSON record to write beside it")
    ap.add_argument("--no-specials", action="store_true", help="rank the control tokens with the rest")
    args = ap.parse_args()

    meta, tok = read_pack_tokenizer(args.pack)
    exclude = tuple(pathlib.Path(e).resolve() for e in args.exclude)
    counts: collections.Counter = collections.Counter()
    corpora = []
    for spec in args.corpus:
        files = corpus_files(pathlib.Path(spec), exclude)
        got, digest, total = count_corpus(tok, files)
        counts.update(got)
        entry = {"path": spec, "kind": "dir" if pathlib.Path(spec).is_dir() else "file",
                 "files": len(files), "bytes": total, "sha256": digest.hex()}
        if len(files) == 1:
            entry["file_sha256"] = hashlib.sha256(files[0].read_bytes()).hexdigest()
        corpora.append(entry)
        print("corpus %s: %d files, %d bytes, %d tokens" % (spec, len(files), total, sum(got.values())))
    forced = [] if args.no_specials else sorted(set(tok.special_tokens.values()))
    ids = select_ids(counts, len(tok.tokens), args.n, forced)
    total_counts = sum(counts.values())
    covered = sum(counts[i] for i in ids)
    size = write_ids(pathlib.Path(args.out), ids)
    top = sorted(ids, key=lambda i: (-counts[i], i))[:20]
    print("wrote %s: %d ids, %d bytes, %.2f%% of %d corpus tokens; %d distinct ids seen"
          % (args.out, len(ids), size, 100.0 * covered / max(1, total_counts), total_counts,
             sum(1 for v in counts.values() if v)))
    print("top: " + " ".join("%d:%r(%d)" % (i, tok.decode([i])[:12], counts[i]) for i in top[:10]))
    if args.provenance:
        rec = {
            "format": FORMAT,
            "n": len(ids),
            "file_bytes": size,
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pack": args.pack,
            "pack_name": meta.get("name", ""),
            "pack_sources": meta.get("sources", []),
            "tokenizer": {"vocab": len(tok.tokens), "pre": tok.pre, "special_ids": forced},
            "corpora": corpora,
            "sha256_rule": "sha256 of each file's bytes with one 0x0a after it, in path order (file_sha256: that file's own)",
            "exclude": args.exclude,
            "corpus_tokens": total_counts,
            "covered_tokens": covered,
            "covered_fraction": covered / max(1, total_counts),
            "commit": git_commit(HERE.parents[1]),
            "note": "the sources counted are the working tree the command ran in (this commit, plus whatever was unstaged)",
            "tool": "tools/draft_vocab/make_draft_vocab.py",
            "command": "make_draft_vocab.py " + " ".join(sys.argv[1:]),
        }
        pathlib.Path(args.provenance).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.provenance).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print("wrote %s" % args.provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
