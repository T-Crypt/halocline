"""The draft vocabulary builder's own logic: the ranking, the id file and its reader, the provenance's shape.

No pack and no `regex`: `read_pack_tokenizer` is imported by the tool only, so this runs on a plain python3.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_draft_vocab as dv  # noqa: E402


class SelectTest(unittest.TestCase):
    def test_most_frequent_first_ties_to_the_lower_id(self):
        counts = collections.Counter({5: 10, 1: 10, 3: 7, 2: 0, 0: 0})
        self.assertEqual(dv.select_ids(counts, 6, 2, []), [1, 5])   # the tie at 10: 1 before 5
        # then 3 (count 7), then 0 (count 0: the lowest id left)
        self.assertEqual(dv.select_ids(counts, 6, 4, []), [0, 1, 3, 5])

    def test_forced_ids_are_always_in(self):
        counts = collections.Counter({0: 100, 1: 90, 2: 80, 3: 70})
        self.assertIn(3, dv.select_ids(counts, 4, 2, [3]))
        self.assertEqual(dv.select_ids(counts, 4, 2, [3]), [0, 3])

    def test_forced_ids_alone_can_fill_the_list(self):
        counts = collections.Counter({0: 100})
        self.assertEqual(dv.select_ids(counts, 4, 2, [3, 2]), [2, 3])

    def test_the_list_is_ascending(self):
        counts = collections.Counter({9: 1, 4: 5, 7: 3})
        ids = dv.select_ids(counts, 10, 3, [])
        self.assertEqual(ids, sorted(ids))

    def test_a_bad_n_is_refused(self):
        for n in (0, -1, 5):
            with self.assertRaises(SystemExit):
                dv.select_ids(collections.Counter(), 4, n, [])


class FileTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "ids.bin"
            ids = [0, 1, 42, 248319]
            size = dv.write_ids(p, ids)
            self.assertEqual(size, 4 * len(ids))
            self.assertEqual(dv.read_ids(p), ids)

    def test_the_reader_refuses_what_the_engine_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            odd = pathlib.Path(d) / "odd.bin"
            odd.write_bytes(b"\x01\x00\x00")
            with self.assertRaises(SystemExit):
                dv.read_ids(odd)
            empty = pathlib.Path(d) / "empty.bin"
            empty.write_bytes(b"")
            with self.assertRaises(SystemExit):
                dv.read_ids(empty)
            down = pathlib.Path(d) / "down.bin"
            dv.write_ids(down, [2, 1])
            with self.assertRaises(SystemExit):
                dv.read_ids(down)


class CorpusTest(unittest.TestCase):
    def test_a_dir_is_its_files_in_path_order_and_caches_are_left_out(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "b.txt").write_text("bee")
            (root / "a.txt").write_text("ay")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "x.pyc").write_bytes(b"\x00\x01")
            names = [p.name for p in dv.corpus_files(root)]
            self.assertEqual(names, ["a.txt", "b.txt"])
            self.assertEqual(dv.corpus_files(root / "a.txt")[0].name, "a.txt")

    def test_counting_is_by_token_not_by_character(self):
        # a tokenizer that writes one id per character, then counts: the digest covers the file's bytes and a newline
        class CharTok:
            def encode(self, text):
                return [ord(c) for c in text]
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "x.txt"
            f.write_text("abba")
            counts, digest, total = dv.count_corpus(CharTok(), [f])
            self.assertEqual(counts, collections.Counter({ord("a"): 2, ord("b"): 2}))
            self.assertEqual(total, 4)
            self.assertEqual(len(digest), 32)


class ProvenanceTest(unittest.TestCase):
    def test_the_record_names_what_a_rerun_needs(self):
        rec = json.loads(json.dumps({"format": dv.FORMAT, "n": 3, "corpora": [{"path": "x", "sha256": "00"}]}))
        self.assertIn("int32", rec["format"])
        self.assertEqual(rec["n"], 3)
        self.assertIn("sha256", rec["corpora"][0])


class CommittedDataTest(unittest.TestCase):
    def test_every_committed_file_parses_and_matches_its_name(self):
        data = pathlib.Path(__file__).resolve().parent / "data"
        files = sorted(data.glob("*.bin"))
        self.assertTrue(files, "tools/draft_vocab/data has no id files")
        for f in files:
            ids = dv.read_ids(f)
            n = int(f.stem.rsplit("-n", 1)[1])
            self.assertEqual(len(ids), n, f.name)
            self.assertTrue(all(0 <= i < 248320 for i in ids), f.name)
            self.assertEqual(ids, sorted(set(ids)), f.name)


if __name__ == "__main__":
    unittest.main()
