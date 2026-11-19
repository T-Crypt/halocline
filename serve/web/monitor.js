"use strict";
/* The request inspector: every request the server kept, with the figures it recorded for it, from /metrics alone. */
const $ = s => document.querySelector(s);
const num = v => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 0) => num(v) ? v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "-";
const p2 = n => String(n).padStart(2, "0");
const clock = ms => { const d = new Date(ms); return `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`; };
const stamp = s => { const d = new Date(s * 1000); return `${d.toLocaleDateString("en-US", { month: "short", day: "numeric" })} ${clock(s * 1000)}`; };
const kt = n => n >= 100000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ESC[c]);
const html = (el, h) => { if (el.dataset.sig !== h) { el.innerHTML = h; el.dataset.sig = h; } };
const secs = n => num(n) ? `${n.toFixed(2)} s` : "-";
const rate = n => num(n) ? n.toFixed(1) : "-";
const pct = f => num(f) ? `${(f * 100).toFixed(1)}%` : "-";
const ep = p => String(p || "-").replace("/v1/", "").replace("chat/completions", "chat");
const done = f => f === "stop" ? "done" : f === "length" ? "max tokens" : String(f || "-");

let apiKey = sessionStorage.getItem("halo.key") || "";
let M = null, selected = null, all = false;

async function get(path) {
  const r = await fetch(path, { cache: "no-store", headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {} });
  if (r.status === 401) throw Object.assign(new Error("locked"), { locked: true });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/* ---- Now ---- */
function liveItems(live, eng, last) {
  const items = [];
  const st = live.state || "idle";
  const label = st[0].toUpperCase() + st.slice(1);
  const load = live.load || {};
  items.push({ k: "State", v: esc(label), sub: st === "loading" && num(load.percent) ? `${load.percent}% of the weights load` : live.phase ? esc(live.phase) : "" });
  if (st === "generating") {
    items.push({ k: "Decode", v: `${rate(live.tok_s)}<small>tok/s</small>`, sub: `mean ${rate(live.tok_s_mean)} tok/s since the first token` });
  }
  if (st === "reading" || st === "generating") {
    items.push({ k: "Prefill", v: `${num(live.prefill_tok_s_mean) ? fmt(live.prefill_tok_s_mean) : "-"}<small>tok/s</small>`, sub: "mean over newly read tokens" });
  }
  if (num(live.generated) && num(live.max_tokens)) {
    items.push({ k: "Output", v: `${fmt(live.generated)}<small>/ ${fmt(live.max_tokens)} tok</small>`, wide: true,
                 bar: Math.min(1, live.generated / live.max_tokens),
                 sub: `${num(live.prompt_tokens) ? `${fmt(live.prompt_tokens)} prompt tokens` : ""}${num(live.elapsed_s) ? ` · ${live.elapsed_s} s` : ""}` });
  } else if (num(live.prompt_tokens)) {
    items.push({ k: "Prompt", v: `${fmt(live.prompt_tokens)}<small>tokens</small>`, sub: num(live.elapsed_s) ? `${live.elapsed_s} s so far` : "" });
  } else if (last && num(last.decode_tok_s)) {
    items.push({ k: "Last request", v: `${rate(last.decode_tok_s)}<small>tok/s</small>`,
                 sub: `${esc(last.id)} · ${fmt(last.output_tokens)} tokens · ${secs(last.duration_s)}${num(last.ttft_s) ? ` · first token ${last.ttft_s} s` : ""}` });
  }
  items.push({ k: "Context", v: num(eng.max_context) ? `${fmt(eng.max_context)}<small>tokens</small>` : "-", sub: `${fmt((live.slots || {}).total || 1)} slot${((live.slots || {}).total || 1) === 1 ? "" : "s"}` });
  items.push({ k: "Queued", v: fmt(live.queued || 0), sub: `${fmt((live.slots || {}).active || 0)} active` });
  return items;
}
function drawLive(live, eng, last) {
  html($("#live"), liveItems(live, eng, last).map(x => `<div class="lv${x.wide ? " wide" : ""}"><span class="k">${x.k}</span>`
    + `<div class="v">${x.v}</div>${num(x.bar) ? `<div class="bar4"><i style="width:${(x.bar * 100).toFixed(1)}%"></i></div>` : ""}`
    + `${x.sub ? `<div class="sub">${x.sub}</div>` : ""}</div>`).join(""));
}

/* ---- the list ---- */
const active = r => !r.finish || r.finish === "stop" ? "" : r.finish === "error" ? " err" : " warn";
function visible() {
  const q = $("#q").value.trim().toLowerCase(), f = $("#finish").value;
  return (M.requests || []).filter(r => (f === "" || r.finish === f)
    && (`${r.id} ${r.path || ""} ${r.finish || ""}`.toLowerCase().includes(q)));
}
function drawList() {
  const rows = visible(), kept = M.requests_kept ?? (M.requests || []).length;
  text("req-count", `${rows.length} shown · ${fmt(kept)} kept`);
  text("kept", all ? `all ${fmt(kept)} kept` : kept > (M.requests || []).length ? `newest ${(M.requests || []).length} of ${fmt(kept)}: tick every request kept` : `${fmt(kept)} kept`);
  if (!rows.length) { html($("#list"), `<div class="empty">${(M.requests || []).length ? "No request matches the filter." : "No requests yet."}</div>`); return; }
  html($("#list"), rows.map(r => `<button type="button" class="req" role="option" data-id="${esc(r.id)}" aria-selected="${selected === r.id}">`
    + `<div class="r1"><span>${esc(r.id)} · ${clock(r.time * 1000)}</span><span class="bad${active(r)}">${esc(done(r.finish))}</span></div>`
    + `<div class="r2"><span class="ep">${esc(ep(r.path))}${num(r.slot) ? ` · s${r.slot + 1}` : ""}</span>`
    + `<span>${num(r.prompt_tokens) ? `${fmt(r.prompt_tokens)} in` : ""}</span><span>${num(r.output_tokens) ? `· ${fmt(r.output_tokens)} out` : ""}</span>`
    + `<span>${num(r.decode_tok_s) ? `· ${rate(r.decode_tok_s)} t/s` : ""}</span><span>· ${secs(r.duration_s)}</span></div></button>`).join(""));
}
function text(id, v) { const el = document.getElementById(id); if (el.textContent !== v) el.textContent = v; }

/* ---- one request ---- */
function fact(k, v, unit) {
  return v == null || v === "-" ? "" : `<div><dt>${k}</dt><dd>${v}${unit ? `<small>${unit}</small>` : ""}</dd></div>`;
}
function drawDetail() {
  const r = (M.requests || []).find(x => x.id === selected);
  if (!r) {
    html($("#detail"), `<div class="empty">${selected ? "That request is no longer kept." : "Select a request to see the figures the server recorded for it."}</div>`);
    return;
  }
  const t = r.timings || {}, read = num(r.prompt_tokens) && num(r.reused) ? r.prompt_tokens - r.reused : null;
  const pre = num(read) && read > 0 && num(r.prompt_ms) && r.prompt_ms > 0 ? read / (r.prompt_ms / 1000) : null;
  const acc = num(r.drafts_offered) && r.drafts_offered > 0 && num(r.drafts_accepted) ? r.drafts_accepted / r.drafts_offered : null;
  const rows = [
    fact("Wall clock", secs(r.duration_s)),
    fact("Queue wait", secs(r.queue_s)),
    fact("Model load", secs(r.load_s)),
    fact("First token", secs(r.ttft_s)),
    fact("Prompt", num(r.prompt_tokens) ? fmt(r.prompt_tokens) : "-", "tok"),
    fact("Reused", num(r.reused) ? fmt(r.reused) : null, "tok"),
    fact("Read", read != null ? fmt(read) : null, "tok"),
    fact("Prompt time", secs(num(r.prompt_ms) ? r.prompt_ms / 1000 : null)),
    fact("Prefill", num(pre) ? fmt(pre) : null, "tok/s"),
    fact("Output", num(r.output_tokens) ? fmt(r.output_tokens) : null, "tok"),
    fact("Decode", rate(r.decode_tok_s), "tok/s"),
    fact("Per token", num(t.predicted_per_token_ms) ? t.predicted_per_token_ms.toFixed(2) : null, "ms"),
    fact("Drafted", num(r.drafts_offered) ? fmt(r.drafts_offered) : null),
    fact("Accepted", num(r.drafts_accepted) ? fmt(r.drafts_accepted) : null),
    fact("Accept", acc != null ? `${(acc * 100).toFixed(1)}` : null, "%"),
    fact("Tokens / window", num(r.draft_len) ? r.draft_len.toFixed(2) : null),
    fact("Expert hit", num(r.hit_rate) ? (r.hit_rate * 100).toFixed(1) : null, "%"),
    fact("PCIe share", num(r.pcie_share) ? (r.pcie_share * 100).toFixed(1) : null, "%"),
    fact("SSD share", num(r.ssd_share) ? (r.ssd_share * 100).toFixed(1) : null, "%"),
    fact("Slot", num(r.slot) ? String(r.slot + 1) : null),
    fact("Started", stamp(r.time))];
  html($("#detail"), `<div class="dh"><b>${esc(r.id)}</b><span class="acts"><span class="bad${active(r)}">${esc(done(r.finish))}</span>`
    + `<button type="button" class="btn" id="copy">Copy JSON</button></span></div>`
    + `<div class="sub">POST ${esc(r.path || "-")} · ${stamp(r.time)}</div>`
    + `<h4>Figures</h4><dl>${rows.filter(Boolean).join("")}</dl>`
    + `<h4>Content</h4><div class="sub">This server keeps no prompt or answer text; the figures above and the request's end line in the log are all it records. The engine's log carries the same numbers.</div>`
    + `<h4>Raw</h4><pre class="raw">${esc(JSON.stringify(r, null, 2))}</pre>`);
  $("#copy").onclick = async () => {
    try { await navigator.clipboard.writeText(JSON.stringify(r, null, 2)); $("#copy").textContent = "Copied"; }
    catch (e) { $("#copy").textContent = "Select the text to copy it"; }
  };
}
$("#list").addEventListener("click", e => {
  const b = e.target.closest(".req"); if (!b) return;
  selected = b.dataset.id; drawList(); drawDetail();
});
$("#q").addEventListener("input", () => { drawList(); });
$("#finish").addEventListener("change", () => { drawList(); });
$("#all").addEventListener("change", () => { all = $("#all").checked; poll(); });

/* ---- totals ---- */
const dur = ms => !num(ms) || ms <= 0 ? "-" : ms < 60000 ? `${Math.round(ms / 1000)} s` : ms < 3600000 ? `${Math.round(ms / 60000)} min` : `${(ms / 3600000).toFixed(1)} h`;
function drawTotals(t) {
  const n = t && num(t.requests) && t.requests > 0 ? t : null;
  text("totals", n ? `${fmt(n.requests)} requests served · ${fmt(n.prompt)} prompt tokens (${fmt(n.reused)} reused) · ${fmt(n.written)} written` : "No requests served yet.");
  $("#totals").title = n ? `Since ${stamp(n.since)}: ${fmt(n.prompt)} prompt tokens, ${dur(n.prompt_ms)} reading; ${fmt(n.written)} written, ${dur(n.decode_ms)} generating` : "";
}

/* ---- the whole page ---- */
function draw() {
  const live = M.live || {}, eng = M.engine || {}, reqs = M.requests || [];
  const name = (live.state || "idle")[0].toUpperCase() + (live.state || "idle").slice(1);
  $("#state").dataset.s = live.state || "idle";
  text("state", name);
  text("model", eng.model || "");
  drawLive(live, eng, reqs[0]);
  drawList(); drawDetail(); drawTotals(M.totals);
}

let timer = null;
async function poll() {
  clearTimeout(timer);
  try {
    const m = await get(all ? "metrics?requests=all" : "metrics");
    M = m; $("#auth").hidden = true;
    if (selected && !(M.requests || []).some(r => r.id === selected)) selected = null;
    draw();
  } catch (e) {
    $("#auth").hidden = !e.locked;
    M = null;
    $("#state").dataset.s = "offline"; text("state", e.locked ? "Locked" : "Offline");
  }
  timer = setTimeout(poll, 2000);
}
$("#auth").onsubmit = e => {
  e.preventDefault(); apiKey = $("#key").value.trim(); sessionStorage.setItem("halo.key", apiKey); poll();
};
$("#host").textContent = location.host;
fetch("health").then(r => r.json()).then(h => { $("#keyed").hidden = !h.api_key; }).catch(() => {});
poll();
