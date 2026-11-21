"""The Inkling 2 dialect: the model's own Jinja chat template, its block output and its JSON tool calls.

The template (serve/models/inkling_chat_template.jinja) writes a conversation as message blocks, each closed by
`<|end_message|>`: `<|message_user|><|content_text|>...<|end_message|>` and so on. An assistant turn is a list of
such blocks, ended by `<|content_model_end_sampling|>` (the eos token, 200006). A thinking block is
`<|message_model|><|content_thinking|>...<|end_message|>`, a visible answer `<|content_text|>...`, and a tool call
`<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`. The template also
writes the thinking-effort turn itself, so the dialect adds no effort turn of its own.
"""
from __future__ import annotations

import json

from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

from serve import frontend
from serve.models import Dialect


HEADER = "<|message_model|>"
THINKING = "<|content_thinking|>"
TEXT = "<|content_text|>"
TOOL_JSON = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
END_TURN = "<|content_model_end_sampling|>"
BLOCK_TOKENS = (HEADER, THINKING, TEXT, TOOL_JSON, END, END_TURN)
# the template's effort_map names; the frontend's levels reach it as low, medium, high or max
EFFORT_NAMES = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class _Generation(Extension):
    """The `{% generation %}` tag transformers' apply_chat_template defines: the block is rendered only when a
    generation prompt is added, so a conversation's own assistant turns stay out of a history render."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.CallBlock(self.call_method("_render", [nodes.ContextReference()]), [], [], body).set_lineno(lineno)

    def _render(self, context, caller):
        return caller() if context.get("add_generation_prompt") else ""


class InklingTemplate:
    """frontend.ChatTemplate's Jinja settings plus the `{% generation %}` tag this template uses."""

    def __init__(self, source: str):
        def raise_exception(message):
            raise frontend.TemplateRequestError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                             extensions=["jinja2.ext.loopcontrols", _Generation])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        self.source = source
        self.template = env.from_string(source)

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **kwargs)


def _hold(text: str, tags: tuple[str, ...]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
    best = 0
    for tag in tags:
        for n in range(1, len(tag)):
            if text.endswith(tag[:n]):
                best = max(best, n)
    return best


def _json_value_end(text: str, i: int) -> int:
    """Where the JSON value at `i` ends (exclusive), or -1 while it is still arriving. Strings are skipped whole, so
    an `<|end_message|>` inside one does not end the block."""
    n = len(text)
    if i >= n:
        return -1
    c = text[i]
    if c == '"':
        j = i + 1
        while j < n:
            if text[j] == "\\":
                j += 2
            elif text[j] == '"':
                return j + 1
            else:
                j += 1
        return -1
    if c in "{[":
        close = "}" if c == "{" else "]"
        depth = 0
        j = i
        while j < n:
            ch = text[j]
            if ch == '"':
                e = _json_value_end(text, j)
                if e < 0:
                    return -1
                j = e
                continue
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return j + 1 if ch == close else -1
            j += 1
        return -1
    j = i
    while j < n and text[j] not in ",}] \t\r\n":
        j += 1
    if j >= n:
        return -1                  # a number or a literal still arriving
    return j


class InklingParser:
    """Incremental parser of an Inkling turn: thinking, visible text and JSON tool calls, each block closed by
    `<|end_message|>` and the turn by `<|content_model_end_sampling|>`."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.phase = "start"          # start | thinking | text | tool | done
        self.stream_tools = stream_tools
        self.schemas = {t.get("name"): t for t in tools or []}
        self.scall = None             # the ToolCall being written
        self.tname = ""               # the name the header wrote before <|content_invoke_tool_json|>
        self.tbody_name = ""          # the name inside the JSON body
        self.tkey = None              # the JSON key being read
        self.targs_raw = None         # the raw text of the "args" value
        self.targs_sent = 0           # how much of that value has been sent as tool_args
        self.tstate = "obj"           # obj | key | value | comma | after

    @staticmethod
    def _earliest(text: str) -> tuple[int, str | None]:
        """(index, token) of the earliest complete block token in `text`; (-1, None) while none has arrived."""
        best, tok = -1, None
        for t in BLOCK_TOKENS:
            i = text.find(t)
            if i >= 0 and (best < 0 or i < best):
                best, tok = i, t
        return best, tok

    def _open_call(self, out: list[frontend.Event]) -> None:
        self.scall = frontend.ToolCall(name=self.tname, arguments={})
        self.tstate = "obj"
        self.tkey = None
        self.tbody_name = ""
        self.targs_raw = None
        self.targs_sent = 0
        self.phase = "tool"
        self.state = "call"
        if self.stream_tools:
            out.append(frontend.Event("tool_start", call=self.scall))

    def _close_call(self, out: list[frontend.Event]) -> None:
        args = {}
        if self.targs_raw is not None:
            try:
                args = json.loads(self.targs_raw)
            except ValueError:
                args = {}
        if not self.tname and self.tbody_name:
            self.tname = self.tbody_name
        call = frontend.ToolCall(name=self.tname, arguments=args if isinstance(args, dict) else {})
        call.id = self.scall.id
        out.append(frontend.Event("tool_call", call=call))
        self.scall = None
        self.phase = "start"
        self.state = "content"

    def _tool(self, out: list[frontend.Event]) -> bool:
        """Advance the tool-call parse; False: wait for more text."""
        if self.tstate == "obj":
            self.buf = self.buf.lstrip()
            if not self.buf:
                return False
            if self.buf[0] == "{":
                self.buf = self.buf[1:]
                self.tstate = "key"
                return True
            if self.buf[0] == "}":
                self.buf = self.buf[1:]
                self.tstate = "after"
                return True
            return False
        if self.tstate == "key":
            self.buf = self.buf.lstrip()
            if not self.buf:
                return False
            if self.buf[0] == "}":
                self.buf = self.buf[1:]
                self.tstate = "after"
                return True
            if self.buf[0] != '"':
                return False
            e = _json_value_end(self.buf, 0)
            if e < 0:
                return False
            self.tkey = json.loads(self.buf[:e])
            self.buf = self.buf[e:].lstrip()
            self.tstate = "colon"
            return True
        if self.tstate == "colon":
            if not self.buf:
                return False
            if self.buf[0] != ":":
                return False
            self.buf = self.buf[1:]
            self.tstate = "value"
            return True
        if self.tstate == "value":
            self.buf = self.buf.lstrip()
            if not self.buf:
                return False
            e = _json_value_end(self.buf, 0)
            if e < 0:
                if self.tkey == "args" and self.stream_tools:
                    new = self.buf[self.targs_sent:]
                    self.targs_sent = len(self.buf)
                    if new:
                        out.append(frontend.Event("tool_args", new, call=self.scall))
                return False
            raw = self.buf[:e]
            if self.tkey == "args":
                self.targs_raw = raw
                if self.stream_tools:
                    new = raw[self.targs_sent:]
                    self.targs_sent = len(raw)
                    if new:
                        out.append(frontend.Event("tool_args", new, call=self.scall))
            elif self.tkey == "name":
                try:
                    self.tbody_name = json.loads(raw)
                except ValueError:
                    pass
            self.buf = self.buf[e:]
            self.tstate = "comma"
            return True
        if self.tstate == "comma":
            self.buf = self.buf.lstrip()
            if not self.buf:
                return False
            if self.buf[0] == ",":
                self.buf = self.buf[1:]
                self.tstate = "key"
                return True
            if self.buf[0] == "}":
                self.buf = self.buf[1:]
                self.tstate = "after"
                return True
            return False
        # the JSON object is closed: the block ends at <|end_message|>
        self.buf = self.buf.lstrip()
        if not self.buf:
            return False
        if self.buf.startswith(END):
            self.buf = self.buf[len(END):]
            self._close_call(out)
            return True
        return False

    def feed(self, delta: str, *, final: bool = False) -> list[frontend.Event]:
        self.buf += delta
        out: list[frontend.Event] = []
        while True:
            if self.phase == "thinking":
                i = self.buf.find(END)
                if i < 0:
                    keep = 0 if final else _hold(self.buf, (END,))
                    if len(self.buf) > keep:
                        out.append(frontend.Event("reasoning", self.buf[:len(self.buf) - keep]))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                if i:
                    out.append(frontend.Event("reasoning", self.buf[:i]))
                self.buf = self.buf[i + len(END):]
                self.phase = "start"
                self.state = "content"
                continue
            if self.phase == "text":
                i = self.buf.find(END)
                if i < 0:
                    keep = 0 if final else _hold(self.buf, (END,))
                    if len(self.buf) > keep:
                        out.append(frontend.Event("content", self.buf[:len(self.buf) - keep]))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                if i:
                    out.append(frontend.Event("content", self.buf[:i]))
                self.buf = self.buf[i + len(END):]
                self.phase = "start"
                continue
            if self.phase == "tool":
                if not self._tool(out):
                    return out
                continue
            if self.phase == "done":
                return out
            # between blocks: an optional header, then the block's content token
            self.buf = self.buf.lstrip()
            if not self.buf:
                return out
            if self.buf.startswith(HEADER):
                self.buf = self.buf[len(HEADER):]
                continue
            if HEADER.startswith(self.buf):
                return out
            i, tok = self._earliest(self.buf)
            if tok is None:
                return out
            lead = self.buf[:i]
            if tok == TOOL_JSON:
                self.tname = lead.strip()
                self.buf = self.buf[i + len(TOOL_JSON):]
                self._open_call(out)
                continue
            if lead.strip():
                out.append(frontend.Event("content", lead))
            if tok == THINKING:
                self.buf = self.buf[i + len(THINKING):]
                self.phase = "thinking"
                self.state = "reasoning"
                continue
            if tok == TEXT:
                self.buf = self.buf[i + len(TEXT):]
                self.phase = "text"
                self.state = "content"
                continue
            if tok == END:
                self.buf = self.buf[i + len(END):]
                continue
            self.buf = self.buf[i + len(END_TURN):]
            self.phase = "done"
            return out

    def finish(self) -> list[frontend.Event]:
        """End of generation: flush what is held. A call whose JSON closed counts even without its end token; an
        unfinished one is dropped."""
        out = self.feed("", final=True)
        if self.phase == "thinking":
            if self.buf:
                out.append(frontend.Event("reasoning", self.buf))
                self.buf = ""
            self.phase = "start"
            self.state = "content"
        elif self.phase == "text":
            if self.buf:
                out.append(frontend.Event("content", self.buf))
                self.buf = ""
            self.phase = "start"
        elif self.phase == "tool":
            if self.tstate == "after" and self.scall is not None:
                self._close_call(out)
            self.buf = ""
            self.scall = None
            self.phase = "start"
            self.state = "content"
        elif self.phase == "start":
            if self.buf.strip():
                out.append(frontend.Event("content", self.buf))
            self.buf = ""
        return out


class Inkling(Dialect):
    """The Inkling 2 dialect, arch `inkling2`."""

    name = "inkling"
    stop_texts = (END_TURN,)          # the eos token ends the model's message
    reasoning_close = END             # closes a thinking block the model left open
    image_pad = "<|unused_200054|>"   # the template's image placeholder, written after <|content_image|>
    image_start = "<|content_image|>"
    # the template writes its own thinking-effort turn and always keeps earlier turns' thinking, so the dialect
    # adds no effort turn and has no preserve switch
    effort_turn = ""
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True,
            "supports_reasoning_effort": True, "supports_system_role": True,
            "supports_preserve_reasoning": False}

    def template(self, source):
        return InklingTemplate(source)

    def parser(self, thinking, tools, stream_tools=True):
        return InklingParser(thinking=thinking, tools=tools, stream_tools=stream_tools)

    def literal_tags(self, controls):
        return frontend.literal_tags(controls)

    def effort_level(self, level):
        """The frontend's reasoning level as the template's effort name (low, medium, high or max)."""
        if isinstance(level, int) or level not in EFFORT_NAMES:
            raise ValueError(f"unknown reasoning effort {level!r}: use none, low, medium, high or max")
        return level

    def template_kwargs(self, kwargs: dict) -> dict:
        """The frontend's kwargs as this template names them: `enable_thinking: false` is the template's
        `reasoning_effort: "none"` (0.0), its only switch for thinking."""
        out = super().template_kwargs(kwargs)
        if out.get("enable_thinking") is False:
            out = {k: v for k, v in out.items() if k != "enable_thinking"}
            out["reasoning_effort"] = "none"
        return out

    def video_text(self, times, rows):
        raise NotImplementedError("Inkling 2 reads no video")

    def forced_opening(self, name):
        """The text that opens a call to tool `name` (None: any tool), for a forced tool choice."""
        if not name:
            return HEADER
        return f"{HEADER}{name}{TOOL_JSON}" + '{"name":' + json.dumps(name, ensure_ascii=False) + ',"args":'
