"""The page contract: what the console's app.js reads of /metrics is what the server sends."""
import re
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "serve" / "chat_template.jinja"
APP_JS = (ROOT / "serve" / "web" / "app.js").read_text(encoding="utf-8")


class PageContract(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "Hi.")
        self.engine.conversation = {"parked": 2, "bytes": 5, "evictions": 1, "restored": 9, "restored_ms": 1.5}
        self.svc = Service(self.engine, tok, ChatTemplate.from_file(TEMPLATE))
        self.m = self.svc.metrics()

    def reads(self, var, text=APP_JS):
        return set(re.findall(rf"\b{var}\.([a-z_]+)", text))

    def test_the_top_level_fields(self):
        top = self.reads("M") - {"slots", "models", "swap"}          # /slots, and the router's
        self.assertTrue({"live", "engine", "history", "requests", "best", "hardware", "totals", "conversation"} <= top)
        self.assertEqual(top - set(self.m), set())

    def test_the_live_fields(self):
        live = self.reads("live")
        self.assertTrue({"state", "phase", "slots"} <= live)
        self.assertEqual(live - set(self.m["live"]), set())

    def test_the_conversation_and_totals_fields(self):
        self.assertEqual(self.reads("conv") - set(self.m["conversation"]), set())
        totals = APP_JS.split("function drawTotals")[1].split("\n}\n")[0]
        self.assertEqual(self.reads("n", totals) - set(self.m["totals"]), set())
        self.assertEqual(self.reads("t", totals) - set(self.m["totals"]), set())

    def test_the_engine_tab_names_no_field_the_engine_cannot_send(self):
        # the union of what the dense and the sparse INFO lines can carry, plus what the server adds
        info = {"model", "path", "max_context", "spill", "expert_total", "kv", "kv_resident", "elastic_kv_ceiling",
                "spec", "mtp_max", "lookup", "drafter", "mtp_kv", "mtp_window", "mtp_ring", "expert_slots",
                "expert_cache_mib", "expert_slots_primary", "expert_cache_primary_mib", "arena_mib", "pcie_frac",
                "conversation_cache_mib", "conversation_cache_slots", "conversation_cache_min_free_mib",
                "conversation_cache_min_tokens", "gpu_weight_bytes", "host_weight_bytes", "host_embedding_bytes",
                "host_vision_bytes", "batch_slots", "vram_elastic", "engine", "version", "quant", "gpus", "context",
                "vision", "video"}
        self.assertEqual(self.reads("eng") - info, set())
