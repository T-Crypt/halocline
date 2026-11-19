#!/usr/bin/env python3
"""Vision check with known answers, through llama-swap (2026-11-19, ticket SRT2J9G).

    python3 vision_check.py MODEL [--url http://127.0.0.1:9090] [--out FILE.json] [--images DIR]

Draws three PNGs whose content is known (so a reply can be scored, unlike lsw_smoke's "said > 10 chars"):
  shapes   3 red circles + 2 blue squares        -> asks the counts      -> must say 3 red/circle and 2 blue/square
  text     the words HALOCLINE 4090 in black      -> asks to read it      -> must contain HALOCLINE and 4090
  bars     bar chart, the tallest bar is green, labelled C -> asks which -> must name C (or green)
Greedy, thinking off, streamed off; one request per image; prints PASS/FAIL, the reply and the wall time.
"""
import argparse, base64, io, json, re, sys, time, urllib.request
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for p in ("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def shapes():
    im = Image.new("RGB", (640, 400), "white"); d = ImageDraw.Draw(im)
    for x in (60, 250, 440):
        d.ellipse((x, 40, x + 120, 160), fill=(220, 20, 20))
    for x in (140, 360):
        d.rectangle((x, 230, x + 120, 350), fill=(20, 40, 220))
    return im


def text():
    im = Image.new("RGB", (800, 240), "white"); d = ImageDraw.Draw(im)
    d.text((40, 70), "HALOCLINE 4090", fill="black", font=font(80))
    return im


def bars():
    im = Image.new("RGB", (640, 420), "white"); d = ImageDraw.Draw(im)
    vals = {"A": (120, (90, 90, 90)), "B": (200, (90, 90, 90)), "C": (330, (30, 170, 60)), "D": (160, (90, 90, 90))}
    f = font(36)
    for i, (k, (h, c)) in enumerate(vals.items()):
        x = 60 + i * 140
        d.rectangle((x, 380 - h, x + 90, 380), fill=c)
        d.text((x + 30, 384), k, fill="black", font=f)
    return im


CASES = [
    ("shapes", shapes, "How many red circles and how many blue squares are in this image? Answer like: N red circles, M blue squares.",
     lambda t: ("3" in t or "three" in t) and ("2" in t or "two" in t) and "red" in t and "blue" in t),
    ("text", text, "Read the text in this image exactly. Reply with only the text.",
     lambda t: "halocline" in t and "4090" in t),
    ("bars", bars, "Which bar in this chart is the tallest? Reply with its letter label and its colour.",
     lambda t: re.search(r"\bc\b", t) is not None and "green" in t),
]


def ask(url, model, im, q):
    buf = io.BytesIO(); im.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {"model": model, "temperature": 0, "max_tokens": 200, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": q},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    return (msg.get("content") or "").strip(), time.time() - t0, out.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--url", default="http://127.0.0.1:9090")
    ap.add_argument("--out"); ap.add_argument("--images", help="also save the PNGs here")
    a = ap.parse_args()
    res, ok_all = [], True
    for name, draw, q, judge in CASES:
        im = draw()
        if a.images:
            im.save(f"{a.images}/{name}.png")
        try:
            text_, wall, usage = ask(a.url, a.model, im, q)
            ok = judge(text_.lower())
        except Exception as e:  # report, keep going
            text_, wall, usage, ok = f"ERROR {e}", 0.0, {}, False
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:7s} {wall:6.1f} s  prompt={usage.get('prompt_tokens')}  {text_[:160]!r}", flush=True)
        res.append({"case": name, "pass": ok, "wall_s": round(wall, 2), "reply": text_, "usage": usage})
    if a.out:
        json.dump({"model": a.model, "utc": time.strftime("%FT%TZ", time.gmtime()), "results": res}, open(a.out, "w"), indent=1)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
