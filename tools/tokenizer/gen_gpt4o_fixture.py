#!/usr/bin/env python3
"""Generate the GPT-4o (o200k) pre-tokenizer fixture from the exact llama.cpp regex.

Run with the venv that has the `regex` package:
    ~/scratch/inkling/venv/bin/python tools/tokenizer/gen_gpt4o_fixture.py

The output is tests/program/data/gpt4o_pretok_cases.txt. It is an ASCII-only escaped
format: a CASE line holds the escaped input bytes, the PIECE lines that follow hold
the escaped bytes of each expected piece, and a blank line ends the case. The C++
test decodes both sides and checks the pieces against the byte offsets pretok_gpt4o
returns.
"""

import random
import regex
from pathlib import Path

# The GPT-4o regex llama.cpp maps `tokenizer.ggml.pre = inkling2` to. Kept verbatim:
# alternatives are tried left to right at each position, quantifiers are greedy and
# backtrack, and \s+(?!\S) leaves the last whitespace byte for the next piece.
PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
RX = regex.compile(PATTERN)

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "tests" / "program" / "data" / "gpt4o_pretok_cases.txt"


def escape(raw: str) -> str:
    """Escape the UTF-8 bytes of `raw`: backslash, \\n, \\r, \\t and \\xHH."""
    out = []
    for b in raw.encode("utf-8"):
        if b == 0x5C:
            out.append("\\\\")
        elif b == 0x0A:
            out.append("\\n")
        elif b == 0x0D:
            out.append("\\r")
        elif b == 0x09:
            out.append("\\t")
        elif 0x20 <= b < 0x7F:
            out.append(chr(b))
        else:
            out.append("\\x%02X" % b)
    return "".join(out)


def pieces(text: str):
    """The pieces the regex splits `text` into, as UTF-8 strings."""
    return [m.group() for m in RX.finditer(text)]


def cases():
    """Around 300 varied strings, deterministic so the fixture is reproducible."""
    fixed = [
        # English contractions in mixed case.
        "I'm", "don't", "He'll", "WE'RE", "she's", "it's", "they've", "you'd",
        "I'LL", "can't", "won't", "SHAN'T", "He's not here", "I'M fine",
        "don't stop", "she'll be there", "we've got it", "it'd be nice",
        "rock'n'roll", "'s 't 're", "don't'", "'hello'", "O'Reilly's",
        "y'all'd've", "ain't", "Ma'am", "DON'T", "Wouldn't've", "he'd",
        # Numbers of 1 to 7 digits, and numbers mixed into text.
        "0", "1", "42", "123", "9999", "12345", "654321", "1234567",
        "000", "007", "2024", "12.34", "3.14159", "v2.0", "a1b2c3",
        "2024-01-01", "port8080", "sha256", "x1y2z3", "1,234,567",
        "12 345 6789", "0x1F", "75%", "1+2=3", "ABCD1234efgh5678",
        # Code with symbols, punctuation and newlines.
        "int x = 42;\nprintf(\"hi\");", "a+b*c", "foo_bar", "x->y", "a::b",
        "http://x.com/a?b=1", "#include <stdio.h>", "{}[]();,", "<tag attr='v'>",
        "// comment\nnext line", "x /= 2; y *= 3;", "a.b.c", "a,b,c",
        "foo\nbar\nbaz", "line1\r\nline2", "x\r\ny", "\r\n", "a\r\n\r\nb",
        "  \r\n  ", "if (x) {\n\treturn y;\n}", "a/\nb", "x\r\n/", "main()",
        "C:\\path\\file", "a_b-c.d/e", "1 < 2 && 3 > 2", "x[0] = y[1];",
        "lambda x: x + 1", "SELECT * FROM t;", "a@b.com", "$value ^ 2",
        # Tabs, and runs of spaces before words and at the end.
        "\tx", "a\tb", "\t\tx", "x\t", "  hello", "hello  ", "   a   b   ",
        " leading and trailing  ", "\t leading\ttab \t", "a \t b \n c",
        "no_spaces_here", "   ", "\t ", "\t\t", "a  b  c  d",
        # Chinese, Japanese, Korean.
        "中文", "日本語", "한국어", "汉字ABC", "你好世界", "こんにちは",
        "안녕하세요", "中文English混在", "東京2024", "한글과English",
        "漢字かなカナ", "你好, world!", "日本語のテキストです。",
        # Arabic.
        "مرحبا", "العربية", "123abc", "مرحبا بالعالم", "سماء ٤٢",
        "مرحبا\nالعربية", "aمرحباb", "!؟.",
        # Emoji.
        "😀", "a😀b", "hello😀world", "👍🏽", "🎉🎊", "🙂🙂🙂",
        "emoji😀here", "😀123", "🚀 launch", "café☕time", "x❗y",
        # Combining marks.
        "e\u0301", "cafe\u0301", "a\u0301b\u0302", "\u0301abc", "n\u0303",
        "A\u0300B\u0301", "e\u0301cole", "\u0301\u0302", "z\u0301z\u0301",
        # Accented Latin.
        "café", "École", "naïve", "Zürich", "São Paulo", "Ångström",
        "Øresund", "über", "façade", "señor", "EL NIÑO", "mañana",
        "Æsir", "œuvre", "ÅAåa", "ÁÉÍÓÚ", "áéíóú",
        # Cyrillic and Greek.
        "Привет", "МИР", "Съешь", "Я", "ёлка", "ПриветМир",
        "Ελλάδα", "ΑΒΓαβγ", "Σίσυφος", "Привет, мир!",
        # Whitespace only and the empty string.
        "", " ", "  ", "\t", "\n", "\r\n\r\n", " \t \n \r\n ", "\n\n\n",
        # Mixed scripts and punctuation.
        "Hello, 世界! 123 😀\n\tDone.", "abc中文123😀def", "ÉПривет中文42",
        "'QUOTED' and \"double\"", "a's b't c're d've", "...", "?!?!",
    ]

    rng = random.Random(0x4F323030)  # "O200", fixed for reproducibility
    atoms = [
        "Hello", "world", "HELLO", "camelCase", "UPPER", "lower", "I'm", "don't",
        "123", "4567", "12.3", "中文", "日本語", "Привет", "café", "é", "e\u0301",
        "😀", "a", "B", " ", "  ", "\t", "\n", "\r\n", ".", ",", "!", "?", "-",
        "/", ":", "(", ")", "'", "=", "+", "#", "_", "\u0301", "Ω", "of", "x",
    ]
    generated = []
    for _ in range(len(fixed), 300):
        n = rng.randint(1, 7)
        generated.append("".join(rng.choice(atoms) for _ in range(n)))
    return fixed + generated


def main():
    out = []
    out.append("# GPT-4o pre-tokenizer cases generated by tools/tokenizer/gen_gpt4o_fixture.py.")
    out.append("# regex module %s, Unicode %s. Do not edit by hand." % (
        regex.__version__, __import__("unicodedata").unidata_version))
    for text in cases():
        out.append("CASE " + escape(text))
        for piece in pieces(text):
            out.append("PIECE " + escape(piece))
        out.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="ascii")
    print("wrote %d cases to %s" % (len(cases()), OUT.relative_to(ROOT)))


if __name__ == "__main__":
    main()
