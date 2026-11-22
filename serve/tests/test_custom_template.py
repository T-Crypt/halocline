"""--chat-template: a custom template's own modes and switches reach it; the model's template path is unchanged."""
import unittest

from serve.config import build_parser
from serve.frontend import ChatTemplate, effort_kwargs, openai_to_messages

# echoes what reached it: the effort, a custom switch, and each message's role
ECHO = ("effort={{ reasoning_effort | default('-') }} think={{ enable_thinking | default('-') }} "
        "pre={{ my_preamble | default('-') }} roles={% for m in messages %}{{ m.role }},{% endfor %}")


class Efforts(unittest.TestCase):
    def test_unknown_effort_passes_through_only_with_a_custom_template(self):
        self.assertEqual(effort_kwargs("redteam", passthrough=True), {"reasoning_effort": "redteam"})
        self.assertEqual(effort_kwargs("RedTeam", passthrough=True), {"reasoning_effort": "redteam"})
        self.assertEqual(effort_kwargs("007", passthrough=True), {"reasoning_effort": "007"})     # a mode, not 7
        self.assertEqual(effort_kwargs("off", passthrough=True), {"reasoning_effort": "off"})     # the template's
        with self.assertRaises(ValueError):
            effort_kwargs("redteam")

    def test_model_template_efforts_unchanged(self):
        self.assertEqual(effort_kwargs("xhigh"), {"reasoning_effort": "max"})
        self.assertEqual(effort_kwargs("none"), {"enable_thinking": False})
        self.assertEqual(effort_kwargs("40"), {"reasoning_effort": 40})
        for v in (False, None, ""):   # no effort and thinking off mean the same either way
            self.assertEqual(effort_kwargs(v, passthrough=True), effort_kwargs(v), v)


class Requests(unittest.TestCase):
    REQ = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"}, {"role": "system", "content": "hook"},
                        {"role": "user", "content": "v"}],
           "reasoning_effort": "joker",
           "chat_template_kwargs": {"my_preamble": False, "add_generation_prompt": False, "tools": "x",
                                    "nested": {"no": 1}}}

    def test_custom_template_gets_mode_switches_and_late_system(self):
        messages, _, kw = openai_to_messages(self.REQ, custom_template=True)
        self.assertEqual(kw["reasoning_effort"], "joker")
        self.assertIs(kw["my_preamble"], False)
        self.assertNotIn("add_generation_prompt", kw)            # the server's own keys stay the server's
        self.assertNotIn("tools", kw)
        self.assertNotIn("nested", kw)                           # plain values only
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "system", "user"])
        out = ChatTemplate(ECHO).render(messages, **kw)
        self.assertEqual(out, "effort=joker think=- pre=False roles=system,user,assistant,system,user,")

    def test_model_template_path_is_unchanged(self):
        req = dict(self.REQ, reasoning_effort="high")
        messages, _, kw = openai_to_messages(req)
        self.assertEqual(kw, {"reasoning_effort": "high"})        # custom switches are not forwarded
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user", "user"])
        with self.assertRaises(ValueError):
            openai_to_messages(self.REQ)                          # an unknown effort is still a 400

    def test_thinking_off_still_wins(self):
        req = dict(self.REQ, chat_template_kwargs={"enable_thinking": False, "my_preamble": True})
        _, _, kw = openai_to_messages(req, custom_template=True)
        self.assertIs(kw["enable_thinking"], False)
        self.assertEqual(kw["reasoning_effort"], "joker")   # the mode's instruct form
        self.assertIs(kw["my_preamble"], True)
        _, _, kw = openai_to_messages(dict(req, reasoning_effort="high"))
        self.assertEqual(kw, {"enable_thinking": False})    # the model's template: thinking off alone, as before


class Flag(unittest.TestCase):
    def test_flag_parses(self):
        a = build_parser().parse_args(["--chat-template", "/tmp/t.jinja"])
        self.assertEqual(a.chat_template, "/tmp/t.jinja")
        self.assertIsNone(build_parser().parse_args([]).chat_template)


if __name__ == "__main__":
    unittest.main()
