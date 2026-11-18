# The Halocline research site

This directory builds the public research site for the engine. It is not a docs site: the README is the install path,
and this site is the record of what was measured, how, and what is still open.

The built site is `site/_out/`. It is not published from this branch; the workflow exists so the maintainer can publish
after signing off on the screenshots.

## Build and preview

```sh
python3 site/extract.py --write     # regenerate site/data from the eval files
python3 site/build.py               # render site/_out and run every check
python3 -m http.server 18981 --directory site/_out --bind 127.0.0.1
```

Open `http://127.0.0.1:18981/`. The server is local only. Stop it when done.

`python3 site/extract.py --check` recomputes the data and fails if the committed JSON is stale. `site/build.py` also
compares the data files with a fresh extraction, so a stale number cannot reach the page.

## How data flows

```
evals/, logs, SPECS.md, docs/, README.md
        │
        ▼
site/extract.py          one builder per data file; every value carries its source
        │
        ▼
site/data/*.json         values + _sources + _formats
        │
        ▼
site/build.py            verifies the files, binds values, draws SVG charts
        │
        ▼
site/templates/*.html    no numbers in the source; every number is a data key
        │
        ▼
site/_out/               HTML, CSS, JS, the repo assets
```

A source is a JSON pointer (`evals/.../bench.json#quality.kld_mean`), a table row or log line
(`SPECS.md line 66`), a Markdown line (`README.md line 69`), or a computed value whose inputs are named in the string.
Where a number exists only in prose, the value says so and the build still links the file and line.

The data files are:

| File | What it holds |
|---|---|
| `layers.json` | the six stops and their depth in tokens (the plan's UI axis) |
| `machine.json` | the card, host, software, model names, commands, format labels, credits |
| `sparse.json` | the sparse model's facts, pack, placement, cache arms, draft windows, serve runs |
| `dense.json` | the dense model's facts, packs, quality arms, cache and context ceilings, prefill ABBA |
| `compare.json` | the same-card speed series against Strata and NInfer |
| `regress.json` | the sparse non-regression gate, its ABBA rerun, sampling and tool-call results |
| `boundary.json` | one chunk of the rows file, for the probability overlap chart |
| `runs.json` | the runs ledger: one row per arm that defines a published profile |

## Add a finding

1. Extract the number in `site/extract.py` with a source, then run `python3 site/extract.py --write`.
2. Add an entry to `site/content/findings.json`. The claim text carries no digits: every number is a `{placeholder}`
   mapped to a data key in `values`. The build fails on a missing key or a typed digit.
3. Add the eval directory to `evals`, or the docs file to `docs`, and choose a status: `measured`, `open`, or
   `on a branch`.
4. Run `python3 site/build.py` and `python3 -m unittest discover -s site/tests`.

## The checks

`site/build.py` fails when:

- a template types a number in text or in an `alt`, `aria-label` or `title` attribute;
- a claim, rounding or open item types a number outside a placeholder;
- a template or claim asks for a data key that is not in the files;
- a committed data file is stale against a fresh extraction;
- a value names a source file or line that is not there;
- the output reaches any origin except the site, `github.com/T-Crypt/halocline`, and `huggingface.co` (links only);
- a local link or a fragment does not resolve;
- a page is over 400 KB before fonts.

`site/tests/` covers the extractors, the formatters, each check with a planted failure, the escaping, and the charts'
title, description and table fallback.

## Visual and motion rules

The palette, type and motion are the plan's: teal water above the boundary, near-black below, one accent
(`#57ecdc`), one failure colour (`#ff8a5c`). Headings use the system sans stack; every number uses the system
monospace. No fonts are self-hosted and no font licence needs checking.

All motion is CSS and is off under `prefers-reduced-motion`: the hero particles, the caustic bands, the boundary
shimmer, the chart line draw, smooth scrolling and the gauge transition. The page reads fully without JavaScript. The
script only moves the gauge and fills the chart tooltip, and it writes with `textContent`, never `innerHTML`.

## Deploy

`.github/workflows/site.yml` runs `python3 site/build.py` and deploys `site/_out` with GitHub Pages. It uses the four
pinned action commits in the handoff and grants `pages: write` and `id-token: write` only to the deploy job. The
repository's Pages source has to be set to GitHub Actions once; nothing publishes until the maintainer signs off.

The site URL is `https://t-crypt.github.io/halocline/`. Links inside the site are relative; `og:image`,
`twitter:image` and `canonical` are absolute because social previews need them.
