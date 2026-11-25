// Halocline console: one page for every model in the fleet. Live state arrives over /api/stream once a second;
// history, stored stats and logs are fetched while their view is open.
"use strict";
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => HC.esc(s);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  const S = { state: null, live: [], view: "overview", stats: {}, statsAt: 0, history: null, historyAt: 0, range: "6h",
              hModel: "", logs: { model: null, engine: [], output: [], after: 0, outAfter: 0 }, lsrc: "all", llvl: "all",
              connected: false, es: null };

  // ------------------------------------------------------------------------------------------------ formatting
  const n0 = (v) => v == null ? "-" : Math.round(v).toLocaleString();
  const n1 = (v) => v == null ? "-" : (Math.round(v * 10) / 10).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const pct = (v) => v == null || isNaN(v) ? "-" : Math.round(v * 100) + "%";
  const gib = (b) => b == null ? "-" : (b / 2 ** 30).toFixed(1);
  const mibToGib = (m) => m == null ? "-" : (m / 1024).toFixed(1);
  const tok = (v) => v == null ? "-" : v >= 1e6 ? (v / 1e6).toFixed(2) + "M" : v >= 1e4 ? (v / 1e3).toFixed(1) + "k" : Math.round(v).toLocaleString();
  const tokc = (v) => v == null ? "-" : v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e4 ? Math.round(v / 1e3) + "k" : v >= 1e3 ? (v / 1e3).toFixed(1) + "k" : String(Math.round(v));
  const secs = (v) => v == null ? "-" : v < 1 ? Math.round(v * 1000) + " ms" : v < 120 ? v.toFixed(1) + " s" : Math.round(v / 60) + " min";
  function ago(t) {
    if (!t) return "never";
    const d = Date.now() / 1000 - t;
    if (d < 60) return "just now";
    if (d < 3600) return Math.round(d / 60) + " min ago";
    if (d < 86400) return Math.round(d / 3600) + " h ago";
    return Math.round(d / 86400) + " d ago";
  }
  const hhmmss = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  const STATE_TEXT = { ready: "Ready", loading: "Loading", starting: "Starting", stopping: "Unloading", failed: "Failed",
                       external: "Running outside the console", unloaded: "Not loaded" };

  // a model's color: its place in the fleet, never its rank (the palette holds six; past that, neutral)
  function color(name) {
    const i = (S.state?.models || []).findIndex((m) => m.name === name);
    return i >= 0 && i < 6 ? css(`--s${i + 1}`) : css("--idle");
  }
  const modelOf = (name) => (S.state?.models || []).find((m) => m.name === name);
  const label = (name) => modelOf(name)?.label || name || "-";

  // ------------------------------------------------------------------------------------------------ connection
  function connect() {
    if (S.es) S.es.close();
    const es = new EventSource("api/stream");
    S.es = es;
    es.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      if (!S.connected) { S.live = []; S.state = null; }
      S.connected = true;
      // the stream leaves out what has not changed since the last message: keep the held value
      const prev = S.state || {};
      const st = { ...prev, ...d };
      if ("metrics" in d && d.metrics && prev.metrics && d.metrics_model === undefined) st.metrics = { ...prev.metrics, ...d.metrics };
      else if ("metrics" in d && d.metrics && prev.metrics && d.metrics_model === prev.metrics_model) st.metrics = { ...prev.metrics, ...d.metrics };
      S.live.push(...(d.live || []));
      const cut = st.now - 300;
      S.live = S.live.filter((r) => r.t >= cut);
      S.state = st;
      render();
    };
    es.onerror = async () => {
      S.connected = false;
      setOnAir("offline", "Disconnected", "retrying");
      const r = await fetch("api/state").catch(() => null);
      if (r && r.status === 401) { es.close(); $("auth").hidden = false; }
    };
  }
  // a hidden tab holds no stream; coming back reconnects and refills the last five minutes
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { if (S.es) { S.es.close(); S.es = null; S.connected = false; } }
    else if (!S.es && $("auth").hidden) connect();
  });
  $("auth").addEventListener("submit", (e) => {
    e.preventDefault();
    document.cookie = `hc_key=${encodeURIComponent($("key").value)}; path=/; SameSite=Strict`;
    $("auth").hidden = true;
    connect();
  });

  async function control(name, verb) {
    const r = await fetch(`api/models/${encodeURIComponent(name)}/${verb}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const j = await r.json().catch(() => ({}));
    toast(r.ok ? `${verb === "load" ? "Loading" : "Unloading"} ${label(name)}` : (j.error?.message || `${verb} failed`));
  }
  let toastTimer;
  function toast(text) { const t = $("toast"); t.textContent = text; t.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), 3200); }

  // ------------------------------------------------------------------------------------------------ views
  $("views").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-view]");
    if (!b) return;
    S.view = b.dataset.view;
    for (const x of $("views").children) { x.setAttribute("aria-selected", String(x === b)); x.tabIndex = x === b ? 0 : -1; }
    for (const v of document.querySelectorAll(".view")) v.hidden = v.dataset.view !== S.view;
    history.replaceState(null, "", "#" + S.view);
    HC.hideTip();
    if (S.view === "history") loadHistory(true);
    if (S.view === "logs") { S.logs.model = null; pollLogs(); }
    render();
  });

  $("views").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const tabs = [...$("views").children], i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    next.focus(); next.click(); e.preventDefault();
  });

  function render() {
    const st = S.state;
    if (!st) return;
    renderTop(st);
    if (Date.now() - S.statsAt > 30000) loadStats();
    if (S.view === "overview") renderOverview(st);
    else if (S.view === "fleet") renderFleet(st);
    else if (S.view === "engine") renderEngine(st);
    else if (S.view === "history") { if (Date.now() - S.historyAt > 30000) loadHistory(false); }
    else if (S.view === "hardware") renderHardware(st);
  }

  // ------------------------------------------------------------------------------------------------ top bar
  function setOnAir(s, model, sub) {
    $("onair-dot").dataset.s = s;
    $("onair-model").textContent = model;
    $("onair-state").textContent = sub;
  }
  function renderTop(st) {
    $("clock").textContent = hhmmss(st.now);
    const a = st.active ? modelOf(st.active) : null;
    const live = st.metrics?.live || {};
    const bar = $("swapbar");
    if (st.swap) {
      setOnAir("loading", `${label(st.swap.from) !== "-" ? label(st.swap.from) + " → " : ""}${label(st.swap.to)}`, st.swap.phase === "draining" ? "draining the card" : "starting");
      bar.hidden = false; bar.classList.add("indet"); bar.firstChild.style.width = "";
    } else if (a) {
      const busy = a.state === "ready" && live.state && live.state !== "idle";
      let sub = STATE_TEXT[a.state] || a.state;
      if (busy) sub = live.phase === "prefill" ? "reading the prompt" : `generating · ${n1(live.tok_s)} t/s`;
      if (a.state === "loading" && a.percent != null) sub = `Loading ${Math.round(a.percent)}%`;
      setOnAir(busy ? "busy" : a.state, a.label, sub);
      bar.hidden = a.state !== "loading"; bar.classList.remove("indet");
      if (a.state === "loading") bar.firstChild.style.width = (a.percent || 2) + "%";
    } else {
      setOnAir("unloaded", "Card free", "no Halocline model loaded");
      bar.hidden = true;
    }
    document.title = a ? `${a.label} · Halocline` : "Halocline";
  }

  // ------------------------------------------------------------------------------------------------ overview
  function renderOverview(st) {
    const m = st.metrics, live = m?.live || {}, best = m?.best || {};
    const xs = S.live.map((r) => r.t);
    const dec = live.state && live.state !== "idle" && live.phase !== "prefill" ? live.tok_s : null;
    const lastDec = lastReq(m)?.decode_tok_s;
    $("o-dec").innerHTML = dec != null ? `${n1(dec)}<span class="u">tok/s</span>` : lastDec != null ? `${n1(lastDec)}<span class="u">last</span>` : "-";
    $("o-dec").classList.toggle("dim", dec == null);
    $("o-dec-mean").textContent = n1(live.tok_s_mean ?? lastReq(m)?.decode_tok_s);
    const sb = S.stats[st.active] || {};
    $("o-dec-best").textContent = n1(Math.max(best.decode?.tok_s || 0, sb.decode_best || 0) || null);
    $("o-dec-sub").textContent = st.active ? label(st.active) : "tokens / s";
    HC.line($("o-dec-chart"), { xs, series: [{ name: "Decode", color: css("--accent"), values: S.live.map((r) => r.state ? r.tok_s || 0 : null) }], compact: true, unit: "t/s" });
    const lr = lastReq(m);
    const lastPre = lr?.timings?.prompt_per_second;
    $("o-pre").innerHTML = live.phase === "prefill" ? `${n0(live.prefill_tok_s_mean)}<span class="u">tok/s</span>` : lastPre != null ? `${n0(lastPre)}<span class="u">last</span>` : "-";
    $("o-pre").classList.toggle("dim", live.phase !== "prefill");
    $("o-pre-last").textContent = lr ? `${tok(lr.timings?.prompt_n ?? lr.prompt_tokens)} read in ${secs((lr.prompt_ms || 0) / 1000)}` : "-";
    $("o-pre-best").textContent = n0(Math.max(best.prefill?.tok_s || 0, sb.prefill_best || 0) || null);
    HC.line($("o-pre-chart"), { xs, series: [{ name: "Prefill", color: css("--accent-2"), values: S.live.map((r) => r.state ? r.prefill_tok_s || 0 : null), bars: true }], compact: true, unit: "t/s" });

    // the request: context fill, draft acceptance, prefix-cache hit
    const slot = (st.slots || []).find((s) => s.is_processing) || (st.slots || [])[0];
    $("o-phase").textContent = live.state && live.state !== "idle" ? (live.phase || live.state) : "idle";
    const ctx = slot && slot.n_ctx ? slot.ctx_used / slot.n_ctx : null;
    HC.gauge($("g-ctx"), { value: ctx, label: "Context", text: ctx == null ? "-" : pct(ctx), sub: slot ? `${tokc(slot.ctx_used)} of ${tokc(slot.n_ctx)}` : "", color: css("--accent") });
    const off = slot?.drafts_offered ?? lr?.drafts_offered, acc = slot?.drafts_accepted ?? lr?.drafts_accepted;
    const da = off ? acc / off : null;
    HC.gauge($("g-draft"), { value: da, label: "Drafts kept", text: da == null ? "-" : pct(da), sub: off ? `${tokc(acc)} of ${tokc(off)}` : "no drafter", color: css("--s3") });
    const hit = lr && lr.prompt_tokens ? (lr.reused || 0) / lr.prompt_tokens : null;
    HC.gauge($("g-hit"), { value: hit, label: "Prefix reuse", text: hit == null ? "-" : pct(hit), sub: lr ? `${tokc(lr.reused)} cached` : "", color: css("--s6") });
    $("o-ttft").textContent = secs(slot?.ttft_s ?? lr?.ttft_s);
    $("o-queue").textContent = live.queued != null ? `${live.queued} waiting` : "-";
    $("o-gen").textContent = live.generated != null ? tok(live.generated) : lr ? tok(lr.output_tokens) : "-";

    renderCardMeters(st);
    renderStrip(st, $("o-fleet"), false);
    renderSlots(st);
    renderReqs($("o-reqs"), reqsOf(m).slice(-8).reverse(), false);
  }
  // the engine lists its requests newest first; the page works oldest first
  const reqsOf = (m) => (m?.requests || []).slice().sort((a, b) => a.time - b.time);
  const lastReq = (m) => { const r = reqsOf(m); return r.length ? r[r.length - 1] : null; };

  function renderCardMeters(st) {
    const r = S.live[S.live.length - 1] || {};
    const g = st.static?.gpus?.[0];
    $("o-gpu-name").textContent = g ? g.name.replace("NVIDIA GeForce ", "") : "";
    const eng = st.metrics?.engine || {};
    const total = r.vram_total, used = r.vram_used;
    const exp = eng.expert_cache_mib ? eng.expert_cache_mib * 2 ** 20 : 0;
    const meters = [
      { l: "VRAM", v: `${gib(used)} / ${gib(total)} GiB`, parts: total ? (exp && st.active ? [[exp / total, css("--s4")], [Math.max(0, used - exp) / total, css("--accent")]] : [[used / total, css("--accent")]]) : [] },
      { l: "GPU", v: r.gpu_util != null ? `${r.gpu_util}%` : "-", parts: [[(r.gpu_util || 0) / 100, css("--accent")]] },
      { l: "Power", v: r.gpu_power != null ? `${Math.round(r.gpu_power)} W` : "-", parts: [[(r.gpu_power || 0) / (st.metrics?.hardware?.gpus?.[0]?.power_limit || 450), css("--accent-2")]] },
      { l: "Temperature", v: r.gpu_temp != null ? `${r.gpu_temp} °C` : "-", parts: [[(r.gpu_temp || 0) / 90, r.gpu_temp > 80 ? css("--warn") : css("--s3")]] },
      { l: "Host RAM", v: `${gib(r.ram_used)} / ${gib(r.ram_total)} GiB`, parts: [[r.ram_total ? r.ram_used / r.ram_total : 0, css("--s6")]] },
    ];
    $("o-meters").innerHTML = meters.map((x) => `<div class="meter"><div class="ml"><span>${x.l}</span><b>${esc(x.v)}</b></div><div class="mb">${x.parts.map(([f, c]) => `<i style="width:${Math.max(0, Math.min(100, f * 100)).toFixed(1)}%;background:${c}"></i>`).join("")}</div></div>`).join("")
      + (exp && st.active ? `<div class="legend"><span><i class="swatch" style="--c:${css("--s4")}"></i>expert cache</span><span><i class="swatch" style="--c:${css("--accent")}"></i>weights, KV, workspace</span></div>` : "");
  }

  // ------------------------------------------------------------------------------------------------ fleet cards
  function renderStrip(st, el, full) {
    const html = st.models.map((m) => card(m, st, full)).join("");
    if (el.dataset.sig !== html) { el.innerHTML = html; el.dataset.sig = html; }
    for (const m of st.models) {
      const sp = el.querySelector(`[data-spark="${CSS.escape(m.name)}"]`);
      const rec = S.stats[m.name]?.recent || [];
      if (sp && rec.length > 1) HC.spark(sp, rec, color(m.name));
    }
  }
  function card(m, st, full) {
    const s = S.stats[m.name] || {};
    const on = st.active === m.name;
    const last = on ? st.metrics?.engine : null;
    const tags = [m.release && `release ${m.release}`, m.branch, (last?.max_context || m.max_context) && `${tok(last?.max_context || m.max_context)} ctx`, last?.kv && `KV ${last.kv}`, last?.quant, m.images && "vision"].filter(Boolean);
    const busy = ["loading", "starting", "stopping"].includes(m.state);
    const canLoad = !busy && m.state !== "ready" && m.state !== "external";
    const btn = m.state === "ready" ? `<button class="btn danger" data-act="unload" data-m="${esc(m.name)}">Unload</button>`
      : `<button class="btn primary" data-act="load" data-m="${esc(m.name)}" ${canLoad ? "" : "disabled"}>${st.active && st.active !== m.name && !busy ? "Swap to" : "Load"}</button>`;
    const prog = m.state === "loading" ? `<div class="mprog"><i style="width:${m.percent || 3}%"></i></div>` : "";
    return `<div class="mcard ${on ? "on" : ""}" style="--c:${color(m.name)}" aria-busy="${busy}">
      <div class="mh"><span class="mn">${esc(m.label)}</span><span class="pill"><i class="dot" data-s="${esc(m.state)}"></i>${esc(m.state === "loading" && m.percent != null ? `Loading ${Math.round(m.percent)}%` : m.state === "external" ? "External" : STATE_TEXT[m.state] || m.state)}</span></div>
      <div class="mid">${esc(m.name)}<span class="mport"> :${m.port}${full && m.lease ? " · lease " + esc(m.lease) : ""}</span></div>
      <div class="tags">${tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>
      ${(s.recent || []).length > 1 ? `<div class="spark" data-spark="${esc(m.name)}"></div>` : `<div class="spark-empty">${m.state === "external" ? "history starts now" : m.state === "ready" ? "waiting for its first request" : "no requests yet · load it to start its history"}</div>`}
      <div class="mstats"><span>decode<b>${n1(s.decode_mean)}</b></span><span>best<b>${n1(s.decode_best)}</b></span><span>requests<b>${n0(s.requests)}</b></span>
        ${full ? `<span>mean prefill<b>${n0(s.prefill_mean)}</b></span><span>mean load<b>${s.load_mean_s != null ? secs(s.load_mean_s) : "-"}</b></span><span>last used<b>${esc(ago(s.last))}</b></span>
        <span>tokens out<b>${tok(s.output_tokens)}</b></span><span>drafts kept<b>${s.drafts_offered ? pct(s.drafts_accepted / s.drafts_offered) : "-"}</b></span><span>TTFT<b>${secs(s.ttft_mean)}</b></span>` : ""}</div>
      ${prog}${m.error ? `<div class="merr">${esc(m.error)}</div>` : ""}
      ${full ? `<details class="mcmdd"><summary>Engine command</summary><div class="mcmd">${esc(m.args)}</div></details>${m.notes ? `<div class="hint">${esc(m.notes)}</div>` : ""}` : ""}
      <div class="mact">${m.state === "external" ? '<span class="hint">started outside the console · watched, not controlled</span>' : ""}<span class="grow"></span>${m.state === "external" ? "" : btn}</div>
    </div>`;
  }
  document.addEventListener("click", (e) => {
    const g = e.target.closest("button[data-goto]");
    if (g) { document.querySelector(`.views button[data-view="${g.dataset.goto}"]`).click(); return; }
    const b = e.target.closest("button[data-act]");
    if (!b || b.disabled) return;
    const name = b.dataset.m, verb = b.dataset.act;
    const st = S.state;
    if (verb === "load" && st.active && st.active !== name && modelOf(st.active)?.state === "ready") {
      if (!confirm(`Swap ${label(st.active)} out for ${label(name)}? Running requests finish first.`)) return;
    }
    if (verb === "unload" && !confirm(`Unload ${label(name)}? Running requests finish first.`)) return;
    control(name, verb);
  });

  function renderFleet(st) {
    if (st.mode === "berth") $("f-hint").textContent = "Each runs its own release, started and stopped by Berth under its lease. One on the card at a time.";
    renderStrip(st, $("f-cards"), true);
    const names = st.models.map((m) => m.name);
    HC.bars($("f-compare"), { name: "mean decode", unit: "t/s", labels: names.map(label), values: names.map((n) => S.stats[n]?.decode_mean || 0),
      colors: names.map(color), sub: names.map((n) => S.stats[n]?.requests ? `${n0(S.stats[n].requests)} req` : "no requests") });
    const rows = names.map((n) => { const s = S.stats[n] || {};
      return `<tr><td><i class="swatch" style="--c:${color(n)}"></i>${esc(label(n))}</td><td class="n">${n0(s.requests)}</td><td class="n">${n1(s.decode_mean)}</td><td class="n">${n1(s.decode_best)}</td>
      <td class="n">${n0(s.prefill_mean)}</td><td class="n">${n0(s.prefill_best)}</td><td class="n">${secs(s.ttft_mean)}</td><td class="n">${s.drafts_offered ? pct(s.drafts_accepted / s.drafts_offered) : "-"}</td>
      <td class="n">${s.prompt_tokens ? pct((s.reused || 0) / s.prompt_tokens) : "-"}</td><td class="n">${tok(s.output_tokens)}</td><td class="n">${n0(s.load_count)}</td><td class="n">${s.load_mean_s != null ? secs(s.load_mean_s) : "-"}</td><td>${esc(ago(s.last))}</td></tr>`; }).join("");
    $("f-table").innerHTML = `<div class="tscroll"><table><thead><tr><th>Model</th><th class="n">Requests</th><th class="n">Decode mean</th><th class="n">Best</th><th class="n">Prefill mean</th><th class="n">Best</th><th class="n">TTFT</th><th class="n">Drafts kept</th><th class="n">Prefix reuse</th><th class="n">Tokens out</th><th class="n">Loads</th><th class="n">Load time</th><th>Last used</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  async function loadStats() {
    S.statsAt = Date.now();
    const r = await fetch("api/history?range=30d").catch(() => null);
    if (!r || !r.ok) return;
    const j = await r.json();
    S.stats = j.stats_all || {};
  }

  // ------------------------------------------------------------------------------------------------ slots + requests
  function renderSlots(st) {
    const slots = st.slots || [];
    $("o-slot-note").textContent = slots.length ? `${slots.length} slot${slots.length > 1 ? "s" : ""}` : "";
    if (!slots.length) { $("o-slots").innerHTML = `<div class="empty">${st.active ? "No slots reported" : "No model on the card"}</div>`; return; }
    $("o-slots").innerHTML = `<div class="tscroll"><table><thead><tr><th>Slot</th><th>State</th><th class="n">Context</th><th style="width:28%">Fill</th><th class="n">Prompt</th><th class="n">Out</th><th class="n">t/s</th><th class="n">Drafts kept</th></tr></thead><tbody>${
      slots.map((s) => { const f = s.n_ctx ? s.ctx_used / s.n_ctx : 0;
        return `<tr><td class="n">${s.id}</td><td><span class="pill"><i class="dot" data-s="${s.is_processing ? "busy" : "ready"}"></i>${esc(s.state)}</span></td>
        <td class="n">${tok(s.ctx_used)} / ${tok(s.n_ctx)}</td><td><div class="meter"><div class="mb"><i style="width:${(f * 100).toFixed(1)}%;background:${css("--accent")}"></i></div></div></td>
        <td class="n">${s.is_processing ? `${tok(s.prefill_done)} / ${tok(s.prompt_tokens)}` : tok(s.prompt_tokens)}</td><td class="n">${tok(s.generated)}${s.max_tokens ? " / " + tok(s.max_tokens) : ""}</td>
        <td class="n">${s.is_processing ? n1(s.tok_s) : "-"}</td><td class="n">${s.drafts_offered ? pct(s.drafts_accepted / s.drafts_offered) : "-"}</td></tr>`; }).join("")}</tbody></table></div>`;
  }

  function renderReqs(el, reqs, full) {
    if (!reqs.length) { el.innerHTML = '<div class="empty">No requests yet</div>'; return; }
    const worst = Math.max(...reqs.map((r) => wfTotal(r)), 0.001);
    el.innerHTML = `<div class="tscroll"><table><thead><tr><th>Time</th>${full ? "<th>API</th>" : ""}<th>End</th><th class="n">Prompt</th><th class="n">Reused</th><th class="n">Out</th><th class="n">TTFT</th><th class="n">Prefill t/s</th><th class="n">Decode t/s</th><th class="n">Drafts</th><th>Time split</th></tr></thead><tbody>${
      reqs.map((r) => { const t = r.timings || {}; const dec = r.output_tokens && r.decode_tok_s ? r.output_tokens / r.decode_tok_s : 0;
        const parts = [["q", r.queue_s || 0, "queue"], ["l", r.load_s || 0, "load"], ["p", (r.prompt_ms || 0) / 1000, "prefill"], ["d", dec, "decode"]];
        const w = (v) => (v / worst * 100).toFixed(2);
        return `<tr><td class="n">${hhmmss(r.time)}</td>${full ? `<td>${esc((r.path || "").replace("/v1/", ""))}</td>` : ""}<td class="fin-${esc(r.finish)}">${esc(r.finish || "-")}</td>
        <td class="n">${tok(r.prompt_tokens)}</td><td class="n">${r.prompt_tokens ? pct((r.reused || 0) / r.prompt_tokens) : "-"}</td><td class="n">${tok(r.output_tokens)}</td><td class="n">${secs(r.ttft_s)}</td>
        <td class="n">${n0(t.prompt_per_second)}</td><td class="n">${n1(r.decode_tok_s)}</td><td class="n">${r.drafts_offered ? pct(r.drafts_accepted / r.drafts_offered) : "-"}</td>
        <td><div class="wf" title="${parts.map(([, v, n]) => `${n} ${secs(v)}`).join(" · ")}">${parts.map(([c, v]) => v ? `<i class="${c}" style="width:${w(v)}%"></i>` : "").join("")}</div></td></tr>`; }).join("")}</tbody></table></div>`;
  }
  const wfTotal = (r) => (r.queue_s || 0) + (r.load_s || 0) + (r.prompt_ms || 0) / 1000 + (r.output_tokens && r.decode_tok_s ? r.output_tokens / r.decode_tok_s : 0);

  // ------------------------------------------------------------------------------------------------ engine
  const GROUPS = [
    ["Model", [["model", "Name"], ["quant", "Weights"], ["path", "Pack", (v) => String(v).split("/").pop()], ["engine", "Engine"], ["vision", "Images", yn], ["video", "Video", yn]]],
    ["Context and KV", [["max_context", "Max context", tok], ["context", "Context", tok], ["kv", "KV cache"], ["kv_resident", "KV resident", tok], ["elastic_kv_ceiling", "Elastic ceiling", tok],
      ["conversation_cache_slots", "Conversation cache", (v) => `${v} slots`], ["conversation_cache_mib", "Conversation cache size", (v) => `${mibToGib(v)} GiB`]]],
    ["Speculation", [["spec", "Draft depth"], ["mtp_max", "MTP max"], ["lookup", "Suffix lookup"]]],
    ["Experts", [["expert_total", "Experts", tok], ["expert_slots", "On the card", tok], ["expert_cache_mib", "Cache", (v) => `${mibToGib(v)} GiB`], ["pcie_frac", "PCIe share", pct], ["spill", "Spill"]]],
    ["Runtime", [["gpus", "Cards"], ["pool_workers", "Workers"], ["arena_mib", "Host arena", (v) => `${mibToGib(v)} GiB`], ["vram_free_mib", "VRAM free", (v) => `${n0(v)} MiB`]]],
  ];
  function yn(v) { return v ? "yes" : "no"; }
  function renderEngine(st) {
    const m = st.metrics, e = m?.engine;
    $("e-title").textContent = st.active ? label(st.active) : "Engine";
    if (!e) {
      $("e-sub").textContent = "";
      $("e-shape").innerHTML = '<div class="empty">No engine on the card.<br><button class="btn primary" data-goto="fleet">Choose a model</button></div>';
      for (const id of ["e-mem", "e-totals", "e-reqs"]) $(id).innerHTML = '<div class="empty">Shown once a model is loaded</div>';
      HC.line($("e-req-dec"), { xs: [], series: [], empty: "No engine" }); HC.line($("e-req-draft"), { xs: [], series: [], empty: "No engine" });
      return;
    }
    const mm = modelOf(st.active);
    $("e-sub").textContent = `${e.model} · release ${mm?.release || "?"}${mm?.branch ? " · " + mm.branch : ""} · :${mm?.port}`;
    const shown = new Set();
    $("e-shape").innerHTML = GROUPS.map(([g, keys]) => {
      const rows = keys.filter(([k]) => e[k] != null && e[k] !== "" && !(e[k] === 0 && ["kv_resident", "cvec"].includes(k))).map(([k, l, f]) => { shown.add(k); return `<dt>${l}</dt><dd title="${esc(e[k])}">${esc(f ? f(e[k]) : e[k])}</dd>`; }).join("");
      return rows ? `<div class="kvg"><h4>${g}</h4><dl class="kv">${rows}</dl></div>` : "";
    }).join("");
    // memory: VRAM split, expert residency, host arena
    const r = S.live[S.live.length - 1] || {};
    const total = r.vram_total || 1, exp = (e.expert_cache_mib || 0) * 2 ** 20, free = (e.vram_free_mib || 0) * 2 ** 20;
    const rest = Math.max(0, (r.vram_used || 0) - exp);
    const resid = e.expert_total ? e.expert_slots / e.expert_total : null;
    $("e-mem").innerHTML = `<div class="meters">
      <div class="meter"><div class="ml"><span>VRAM</span><b>${gib(r.vram_used)} / ${gib(r.vram_total)} GiB</b></div><div class="mb">
        ${exp ? `<i style="width:${(exp / total * 100).toFixed(1)}%;background:${css("--s4")}"></i>` : ""}<i style="width:${(rest / total * 100).toFixed(1)}%;background:${css("--accent")}"></i></div></div>
      ${resid != null ? `<div class="meter"><div class="ml"><span>Experts on the card</span><b>${pct(resid)} · ${tok(e.expert_slots)} of ${tok(e.expert_total)}</b></div><div class="mb"><i style="width:${(resid * 100).toFixed(1)}%;background:${css("--s4")}"></i></div></div>` : ""}
      <div class="meter"><div class="ml"><span>Host RAM</span><b>${gib(r.ram_used)} / ${gib(r.ram_total)} GiB</b></div><div class="mb">${e.arena_mib ? `<i style="width:${(e.arena_mib * 2 ** 20 / (r.ram_total || 1) * 100).toFixed(1)}%;background:${css("--s6")}"></i>` : ""}<i style="width:${Math.max(0, ((r.ram_used || 0) - (e.arena_mib || 0) * 2 ** 20) / (r.ram_total || 1) * 100).toFixed(1)}%;background:${css("--idle")}"></i></div></div>
    </div><div class="legend">${exp ? `<span><i class="swatch" style="--c:${css("--s4")}"></i>expert cache</span>` : ""}<span><i class="swatch" style="--c:${css("--accent")}"></i>weights, KV, workspace</span>${e.arena_mib ? `<span><i class="swatch" style="--c:${css("--s6")}"></i>host arena</span>` : ""}<span><i class="swatch" style="--c:${css("--idle")}"></i>everything else</span></div>
    ${(m.requests || []).some((q) => q.hit_rate != null) ? '<div class="panel-h" style="margin:14px 0 0"><h3>Expert cache hits per request</h3><span class="hint">routed experts found on the card</span></div><div class="chart c-small" id="e-hit"></div>' : ""}
    <dl class="kv">${free ? `<dt>VRAM left free</dt><dd>${n0(e.vram_free_mib)} MiB</dd>` : ""}${e.expert_cache_primary_mib ? `<dt>Primary expert cache</dt><dd>${mibToGib(e.expert_cache_primary_mib)} GiB</dd>` : ""}</dl>`;
    if ($("e-hit")) { const rq = reqsOf(m); HC.line($("e-hit"), { xs: rq.map((_, i) => i), xTick: (i) => rq[Math.round(i)]?.id || "", xLabel: (i) => rq[Math.round(i)] ? `${rq[Math.round(i)].id} · ${hhmmss(rq[Math.round(i)].time)}` : "", yMax: 100, fmt: (v) => Math.round(v) + "%", series: [{ name: "Hits", color: css("--s4"), values: rq.map((q) => q.hit_rate != null ? q.hit_rate * 100 : null) }] }); }
    const t = m.totals || {}, best = m.best || {};
    const reuse = t.prompt ? t.reused / t.prompt : null;
    $("e-since").textContent = t.since ? `since ${new Date(t.since * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}` : "";
    $("e-totals").innerHTML = `<dl class="kv"><dt>Requests</dt><dd>${n0(t.requests)}</dd><dt>Prompt tokens</dt><dd>${tok(t.prompt)}</dd><dt>Read from cache</dt><dd>${tok(t.reused)} · ${pct(reuse)}</dd>
      <dt>Tokens written</dt><dd>${tok(t.written)}</dd><dt>Prefill time</dt><dd>${secs((t.prompt_ms || 0) / 1000)}</dd><dt>Decode time</dt><dd>${secs((t.decode_ms || 0) / 1000)}</dd>
      <dt>Best decode</dt><dd>${n1(best.decode?.tok_s)} t/s · ${tok(best.decode?.output_tokens)} out</dd><dt>Best prefill</dt><dd>${n0(best.prefill?.tok_s)} t/s · ${tok(best.prefill?.read_tokens)} read</dd>
      <dt>Requests kept</dt><dd>${n0(m.requests_kept)}</dd></dl>`;
    const reqs = reqsOf(m);
    const idx = reqs.map((_, i) => i);
    const tipFor = (i) => { const q = reqs[i]; return q ? `<div class="tr"><span>prompt</span><b>${tok(q.prompt_tokens)}</b></div><div class="tr"><span>out</span><b>${tok(q.output_tokens)}</b></div>` : ""; };
    HC.line($("e-req-dec"), { xs: idx, xTick: (i) => reqs[Math.round(i)]?.id || "", xLabel: (i) => reqs[Math.round(i)] ? `${reqs[Math.round(i)].id} · ${hhmmss(reqs[Math.round(i)].time)}` : "", unit: "t/s", series: [{ name: "Decode", color: css("--accent"), values: reqs.map((q) => q.decode_tok_s) }], tipExtra: tipFor, empty: "No requests yet" });
    HC.line($("e-req-draft"), { xs: idx, xTick: (i) => reqs[Math.round(i)]?.id || "", xLabel: (i) => reqs[Math.round(i)] ? `${reqs[Math.round(i)].id} · ${hhmmss(reqs[Math.round(i)].time)}` : "", yMax: 100, fmt: (v) => Math.round(v) + "%", series: [{ name: "Kept", color: css("--s3"), values: reqs.map((q) => q.drafts_offered ? q.drafts_accepted / q.drafts_offered * 100 : null) }], tipExtra: tipFor, empty: "No drafts yet" });
    renderReqs($("e-reqs"), reqs.slice().reverse(), true);
  }

  // ------------------------------------------------------------------------------------------------ history
  $("h-range").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-r]");
    if (!b) return;
    S.range = b.dataset.r;
    for (const x of $("h-range").children) x.setAttribute("aria-pressed", String(x === b));
    loadHistory(true);
  });
  $("h-model").addEventListener("change", (e) => { S.hModel = e.target.value; loadHistory(true); });

  async function loadHistory(force) {
    if (!force && Date.now() - S.historyAt < 30000) return;
    S.historyAt = Date.now();
    const q = `range=${S.range}${S.hModel ? "&model=" + encodeURIComponent(S.hModel) : ""}`;
    const [h, rq] = await Promise.all([fetch("api/history?" + q).then((r) => r.json()).catch(() => null),
                                        fetch(`api/requests?range=${S.range}&limit=2000${S.hModel ? "&model=" + encodeURIComponent(S.hModel) : ""}`).then((r) => r.json()).catch(() => [])]);
    if (!h) return;
    S.history = h;
    renderHistory(h, rq);
  }
  function renderHistory(h, reqs) {
    const sel = $("h-model");
    const names = (S.state?.models || []).map((m) => m.name);
    if (sel.options.length !== names.length + 1) sel.innerHTML = `<option value="">Every model</option>` + names.map((n) => `<option value="${esc(n)}">${esc(label(n))}</option>`).join("");
    sel.value = S.hModel;
    const rows = h.series.rows, xs = rows.map((r) => r.t);
    const dom = { xMin: h.series.since, xMax: Date.now() / 1000 };
    // one series per model, so a swap reads as a color change, never two scales on one axis
    const perModel = (key, scale = 1) => {
      const ms = [...new Set(rows.map((r) => r.model).filter(Boolean))];
      const out = ms.map((n) => ({ name: label(n), color: color(n), values: rows.map((r) => r.model === n && r[key] != null ? r[key] * scale : null) }));
      return out.length ? out : [{ name: "-", color: css("--idle"), values: rows.map(() => null) }];
    };
    const host = (key, name, c, scale = 1) => [{ name, color: c, values: rows.map((r) => r[key] != null ? r[key] * scale : null) }];
    HC.line($("h-dec"), { ...dom, xs, series: perModel("tok_s"), unit: "t/s", empty: "No decoding in this range" });
    HC.line($("h-pre"), { ...dom, xs, series: perModel("prefill_tok_s"), unit: "t/s", empty: "No prefill in this range" });
    HC.line($("h-util"), { ...dom, xs, series: host("gpu_util", "GPU", css("--accent")), yMax: 100, unit: "%" });
    HC.line($("h-power"), { ...dom, xs, series: host("gpu_power", "Power", css("--accent-2")), unit: "W" });
    HC.line($("h-vram"), { ...dom, xs, series: host("vram_used", "VRAM", css("--s4"), 1 / 2 ** 30), yMax: rows.length ? Math.ceil(Math.max(...rows.map((r) => r.vram_total || 0)) / 2 ** 30) : null, unit: "GiB" });
    HC.line($("h-ram"), { ...dom, xs, series: host("ram_used", "RAM", css("--s6"), 1 / 2 ** 30), yMax: rows.length ? Math.ceil(Math.max(...rows.map((r) => r.ram_total || 0)) / 2 ** 30) : null, unit: "GiB" });
    // timeline: contiguous buckets of the same model
    const segs = [];
    for (const r of rows) {
      if (!r.model) continue;
      const last = segs[segs.length - 1];
      if (last && last.name === r.model && r.t - last.t1 <= h.series.width * 1.5) last.t1 = r.t + h.series.width;
      else segs.push({ name: r.model, t0: r.t, t1: r.t + h.series.width, color: color(r.model), label: label(r.model) });
    }
    const now = Date.now() / 1000;
    HC.band($("h-band"), { t0: h.series.since, t1: now, segments: segs, marks: (h.events || []).filter((e) => e.kind === "start").map((e) => ({ t: e.t })) });
    HC.scatter($("h-scatter"), { xLog: true, xName: "prompt tokens", yName: "decode t/s", xFmt: tok, yFmt: (v) => Math.round(v),
      points: reqs.filter((r) => r.decode_tok_s).map((r) => ({ x: Math.max(1, r.prompt_tokens || 1), y: r.decode_tok_s, color: color(r.model),
        html: `<div class="tt">${esc(label(r.model))} · ${esc(new Date(r.t * 1000).toLocaleString())}</div><div class="tr"><span>prompt</span><b>${tok(r.prompt_tokens)}</b></div><div class="tr"><span>decode</span><b>${n1(r.decode_tok_s)} t/s</b></div><div class="tr"><span>out</span><b>${tok(r.output_tokens)}</b></div>` })) });
    const evText = { start: "started", ready: "ready", stop: "unloading", stopped: "unloaded", failed: "failed to start", exited: "exited", "swap-out": "swapped out", external: "seen running outside the console", "external-gone": "outside run ended" };
    $("h-events").innerHTML = (h.events || []).length ? h.events.map((e) => `<div class="ev"><span class="et">${esc(new Date(e.t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }))}</span>
      <span><i class="swatch" style="--c:${color(e.model)}"></i>${esc(label(e.model))} ${esc(evText[e.kind] || e.kind)}${e.detail?.load_s ? ` in ${secs(e.detail.load_s)}` : ""}${e.detail?.to ? ` for ${esc(label(e.detail.to))}` : ""}${e.detail?.exit != null && e.kind !== "stopped" ? ` (exit ${e.detail.exit})` : ""}</span></div>`).join("") : '<div class="empty">No events in this range</div>';
    const st = h.stats || {};
    $("h-stats-note").textContent = `${Object.values(st).reduce((a, s) => a + (s.requests || 0), 0).toLocaleString()} requests in range`;
    $("h-stats").innerHTML = Object.keys(st).length ? `<div class="tscroll"><table><thead><tr><th>Model</th><th class="n">Requests</th><th class="n">Decode mean</th><th class="n">Best</th><th class="n">Prefill mean</th><th class="n">TTFT</th><th class="n">Drafts kept</th><th class="n">Prompt</th><th class="n">Out</th><th class="n">Loads</th><th class="n">Load time</th></tr></thead><tbody>${
      Object.values(st).map((s) => `<tr><td><i class="swatch" style="--c:${color(s.model)}"></i>${esc(label(s.model))}</td><td class="n">${n0(s.requests)}</td><td class="n">${n1(s.decode_mean)}</td><td class="n">${n1(s.decode_best)}</td><td class="n">${n0(s.prefill_mean)}</td><td class="n">${secs(s.ttft_mean)}</td><td class="n">${s.drafts_offered ? pct(s.drafts_accepted / s.drafts_offered) : "-"}</td><td class="n">${tok(s.prompt_tokens)}</td><td class="n">${tok(s.output_tokens)}</td><td class="n">${n0(s.load_count)}</td><td class="n">${s.load_mean_s != null ? secs(s.load_mean_s) : "-"}</td></tr>`).join("")}</tbody></table></div>` : '<div class="empty">Nothing stored for this range yet</div>';
  }

  // ------------------------------------------------------------------------------------------------ hardware
  function renderHardware(st) {
    const r = S.live[S.live.length - 1] || {}, hw = st.metrics?.hardware?.gpus?.[0] || {};
    const g = st.static?.gpus?.[0];
    $("hw-gpu-name").textContent = g ? g.name : "GPU";
    $("hw-gpu-sub").textContent = [st.static?.driver && `driver ${st.static.driver}`, st.static?.cuda && `CUDA ${st.static.cuda}`].filter(Boolean).join(" · ");
    const plim = hw.power_limit || 450;
    HC.gauge(hwG(0), { value: (r.gpu_util || 0) / 100, label: "Utilisation", text: r.gpu_util != null ? `${r.gpu_util}%` : "-", color: css("--accent") });
    HC.gauge(hwG(1), { value: r.vram_total ? r.vram_used / r.vram_total : null, label: "VRAM", text: `${gib(r.vram_used)}`, sub: `of ${gib(r.vram_total)} GiB`, color: css("--s4") });
    HC.gauge(hwG(2), { value: (r.gpu_power || 0) / plim, label: "Power", text: r.gpu_power != null ? `${Math.round(r.gpu_power)} W` : "-", sub: `limit ${Math.round(plim)} W`, color: css("--accent-2") });
    HC.gauge(hwG(3), { value: (r.gpu_temp || 0) / 90, label: "Temperature", text: r.gpu_temp != null ? `${r.gpu_temp}°` : "-", color: r.gpu_temp > 80 ? css("--warn") : css("--s3") });
    $("hw-gpu-kv").innerHTML = `<dl class="kv"><dt>SM clock</dt><dd>${n0(r.clock_sm)} MHz</dd><dt>Memory clock</dt><dd>${n0(hw.clock_mem_mhz)} MHz</dd>
      <dt>PCIe link</dt><dd>${hw.pcie_gen ? `gen ${hw.pcie_gen}${hw.pcie_gen_max && hw.pcie_gen_max !== hw.pcie_gen ? ` (max ${hw.pcie_gen_max})` : ""} x${hw.pcie_width}` : "-"}</dd>
      <dt>CPU</dt><dd>${esc(st.static?.cpu_name || "-")}${st.static?.cores ? ` · ${st.static.cores} cores / ${st.static.threads} threads` : ""}</dd></dl>`;
    const hold = st.holders || [];
    $("hw-holders").innerHTML = hold.length ? `<table class="holders"><colgroup><col class="c-pid"><col><col class="c-mib"><col class="c-who"></colgroup><thead><tr><th>PID</th><th>Process</th><th class="n">VRAM</th><th></th></tr></thead><tbody>${hold.map((a) => `<tr><td class="n">${a.pid}</td><td title="${esc(a.cmd)}">${esc(a.cmd || a.name)}</td><td class="n">${n0(a.mib)} MiB</td><td>${a.ours ? '<span class="ok">✓ Halocline</span>' : '<span class="sq">⚠ not ours</span>'}</td></tr>`).join("")}</tbody></table>
      ${hold.some((a) => !a.ours) ? '<div class="hint">Something other than this console\'s engine holds VRAM. If it is not another sanctioned engine under a lease, it is the bug.</div>' : ""}` : '<div class="empty">Nothing holds the card</div>';
    const xs = S.live.map((x) => x.t);
    const one = (k, name, c, scale = 1) => [{ name, color: c, values: S.live.map((x) => x[k] != null ? x[k] * scale : null) }];
    HC.line($("hw-util"), { xs, series: one("gpu_util", "GPU", css("--accent")), yMax: 100, unit: "%" });
    HC.line($("hw-power"), { xs, series: one("gpu_power", "Power", css("--accent-2")), yMax: HC.niceMax(plim), unit: "W" });
    HC.line($("hw-temp"), { xs, series: one("gpu_temp", "Temperature", css("--s3")), yMax: 100, unit: "°C" });
    HC.line($("hw-pcie"), { xs, series: one("pcie_rx", "Host → card", css("--s4")), unit: "MB/s" });
    $("hw-cpu-name").textContent = "%";
    HC.line($("hw-cpu"), { xs, series: one("cpu", "CPU", css("--s5")), yMax: 100, unit: "%" });
    HC.line($("hw-ram"), { xs, series: one("ram_used", "RAM", css("--s6"), 1 / 2 ** 30), yMax: r.ram_total ? Math.ceil(r.ram_total / 2 ** 30) : null, unit: "GiB" });
  }
  function hwG(i) {
    const host = $("hw-gauges");
    while (host.children.length < 4) { const d = document.createElement("div"); d.className = "gauge"; host.appendChild(d); }
    return host.children[i];
  }

  // ------------------------------------------------------------------------------------------------ logs
  for (const [id, key, attr] of [["l-src", "lsrc", "s"], ["l-lvl", "llvl", "l"]]) {
    $(id).addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      S[key] = b.dataset[attr];
      for (const x of $(id).children) x.setAttribute("aria-pressed", String(x === b));
      drawLogs();
    });
  }
  $("l-q").addEventListener("input", drawLogs);
  $("l-model").addEventListener("change", (e) => { S.logs = { model: e.target.value, engine: [], output: [], after: 0, outAfter: 0 }; pollLogs(); });

  async function pollLogs() {
    if (S.view !== "logs") return;
    const sel = $("l-model"), names = (S.state?.models || []).map((m) => m.name);
    if (sel.options.length !== names.length) sel.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(label(n))}</option>`).join("");
    if (!S.logs.model) { S.logs.model = S.state?.active || names[0]; }
    sel.value = S.logs.model || "";
    if (S.logs.model) {
      const r = await fetch(`api/logs?model=${encodeURIComponent(S.logs.model)}&after=${S.logs.after}&out_after=${S.logs.outAfter}`).then((x) => x.json()).catch(() => null);
      if (r && r.model === S.logs.model) {
        S.logs.engine.push(...r.engine); S.logs.output.push(...r.output);
        S.logs.engine = S.logs.engine.slice(-4000); S.logs.output = S.logs.output.slice(-4000);
        if (r.engine.length) S.logs.after = r.engine[r.engine.length - 1].id;
        if (r.output.length) S.logs.outAfter = r.output[r.output.length - 1].id;
        if (r.engine.length || r.output.length) drawLogs();
      }
    }
    setTimeout(pollLogs, 1500);
  }
  function drawLogs() {
    const q = $("l-q").value.trim().toLowerCase();
    const lv = { info: 0, warn: 1, warning: 1, error: 2, err: 2 };
    const min = S.llvl === "error" ? 2 : S.llvl === "warn" ? 1 : 0;
    let rows = [];
    if (S.lsrc !== "out") rows.push(...S.logs.engine.map((r) => ({ ...r, s: r.src || "eng" })));
    if (S.lsrc !== "engine") rows.push(...S.logs.output.map((r) => ({ ...r, s: "out" })));
    rows.sort((a, b) => a.t - b.t);
    rows = rows.filter((r) => (lv[r.lvl] ?? 0) >= min && (!q || r.msg.toLowerCase().includes(q))).slice(-2500);
    const el = $("l-lines");
    const mark = (s) => { const e = esc(s); if (!q) return e; const i = e.toLowerCase().indexOf(esc(q)); return i < 0 ? e : e.slice(0, i) + "<mark>" + e.slice(i, i + esc(q).length) + "</mark>" + e.slice(i + esc(q).length); };
    el.innerHTML = rows.length ? rows.map((r) => `<div class="ll ${lv[r.lvl] === 2 ? "error" : lv[r.lvl] === 1 ? "warn" : ""}"><span class="lt">${hhmmss(r.t)}</span><span class="ls">${esc(r.s)}</span><span class="lm">${mark(r.msg)}</span></div>`).join("") : '<div class="empty">No log lines</div>';
    if ($("l-follow").checked) el.scrollTop = el.scrollHeight;
  }

  // ------------------------------------------------------------------------------------------------ balanced tiling
  // A grid takes the fewest rows its width allows, then the fewest columns that still fill them evenly:
  // 6 cards are 6, 3+3 or 2+2+2, never 5+1; 4 tiles are 4 or 2+2. One observer; it reruns only on resize.
  const GRIDS = [["hero", 280], ["o-fleet", 220], ["f-cards", 330], ["hgrid-h", 420], ["hgrid-hw", 420]];
  const tileRO = new ResizeObserver((entries) => { for (const e of entries) balance(e.target); });
  function balance(el) {
    const min = +el.dataset.min, n = el.children.length;
    if (!n) return;
    const gap = parseFloat(getComputedStyle(el).columnGap) || 16;
    const fit = Math.max(1, Math.floor((el.clientWidth + gap) / (min + gap)));
    const cols = Math.ceil(n / Math.ceil(n / Math.min(fit, n)));
    if (el.dataset.cols !== String(cols)) { el.dataset.cols = cols; el.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`; }
  }
  for (const [id, min] of GRIDS) {
    const el = $(id) || document.querySelector("." + id);
    if (!el) continue;
    el.dataset.min = min;
    tileRO.observe(el);
    new MutationObserver(() => balance(el)).observe(el, { childList: true });
  }

  // ------------------------------------------------------------------------------------------------ start
  const start = (location.hash || "").slice(1);
  if (start && document.querySelector(`.views button[data-view="${CSS.escape(start)}"]`)) document.querySelector(`.views button[data-view="${CSS.escape(start)}"]`).click();
  connect();
  if ("serviceWorker" in navigator && isSecureContext) navigator.serviceWorker.register("sw.js").catch(() => {});
})();
