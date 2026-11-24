// Halocline console charts: canvas line/area, bars, scatter, timeline band, sparkline; SVG gauge. No dependencies.
// Every chart keeps its last spec and redraws on resize; line, bar, scatter and band charts carry a hover tooltip.
"use strict";
(function () {
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const tip = () => document.getElementById("tip");
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const MONO = '11px "Geist Mono", ui-monospace, monospace';

  function showTip(x, y, html) {
    const t = tip();
    t.innerHTML = html;
    t.hidden = false;
    const r = t.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > innerHeight - 8) top = y - r.height - 14;
    t.style.left = left + "px";
    t.style.top = top + "px";
  }
  function hideTip() { tip().hidden = true; }

  // A canvas sized to its box at the device's pixel ratio. draw(ctx, w, h) paints; the spec is kept for resizes.
  const boxes = new WeakMap();
  const ro = new ResizeObserver((entries) => { for (const e of entries) { const b = boxes.get(e.target); if (b) b.paint(); } });
  function box(el) {
    let b = boxes.get(el);
    if (b) return b;
    const c = document.createElement("canvas");
    el.appendChild(c);
    b = { el, c, ctx: c.getContext("2d"), spec: null, draw: null, hover: null };
    b.paint = () => {
      const w = el.clientWidth, h = el.clientHeight;
      if (!w || !h || !b.draw) return;
      const dpr = devicePixelRatio || 1;
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
      b.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      b.ctx.clearRect(0, 0, w, h);
      b.draw(b.ctx, w, h, b.hoverAt);
    };
    c.addEventListener("mousemove", (ev) => { const r = c.getBoundingClientRect(); b.hoverAt = { x: ev.clientX - r.left, y: ev.clientY - r.top, cx: ev.clientX, cy: ev.clientY }; b.paint(); });
    c.addEventListener("mouseleave", () => { b.hoverAt = null; hideTip(); b.paint(); });
    boxes.set(el, b);
    ro.observe(el);
    return b;
  }

  function niceMax(v) {
    if (!(v > 0)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }
  const fmtDefault = (v) => v == null ? "-" : Math.abs(v) >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k" : (Math.round(v * 10) / 10).toString();
  function timeFmt(t, span) {
    const d = new Date(t * 1000);
    if (span > 2 * 86400) return d.toLocaleDateString([], { month: "short", day: "numeric" });
    if (span > 3600) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  }

  // ------------------------------------------------------------------------------------------------ line / area
  // spec: {xs: [t...], series: [{name, color, values, fill}], yMax?, yMin?, fmt?, unit?, compact?, xLabels?, empty?}
  function line(el, spec) {
    const b = box(el);
    b.spec = spec;
    b.draw = (ctx, w, h, hov) => {
      const s = b.spec, xs = s.xs || [];
      const fmt = s.fmt || fmtDefault;
      const pad = s.compact ? { l: 2, r: 2, t: 6, b: 2 } : { l: 44, r: 10, t: 8, b: 20 };
      const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
      let max = s.yMax, min = s.yMin ?? 0;
      if (max == null) {
        let m = 0;
        for (const se of s.series) for (const v of se.values) if (v != null && v > m) m = v;
        max = niceMax(m * 1.08);
      }
      const n = xs.length;
      const t0 = s.xMin ?? xs[0], t1 = s.xMax ?? xs[n - 1];
      const X = (i) => pad.l + (n <= 1 ? pw : ((xs[i] - t0) / ((t1 - t0) || 1)) * pw);
      const Y = (v) => pad.t + ph - ((v - min) / ((max - min) || 1)) * ph;
      // grid
      ctx.font = MONO;
      ctx.lineWidth = 1;
      const rows = s.compact ? 2 : 4;
      for (let i = 0; i <= rows; i++) {
        const v = min + (max - min) * (i / rows), y = Math.round(Y(v)) + 0.5;
        ctx.strokeStyle = css("--grid");
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        if (!s.compact) { ctx.fillStyle = css("--fg-3"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(fmt(v), pad.l - 6, y); }
      }
      if (!s.compact && n > 1) {
        ctx.fillStyle = css("--fg-3"); ctx.textBaseline = "top";
        const ticks = Math.max(2, Math.min(6, Math.floor(pw / 140)));
        for (let i = 0; i <= ticks; i++) {
          const t = t0 + (t1 - t0) * (i / ticks), x = pad.l + pw * (i / ticks);
          ctx.textAlign = i === 0 ? "left" : i === ticks ? "right" : "center";
          ctx.fillText(s.xTick ? s.xTick(t) : s.xLabel ? s.xLabel(t) : timeFmt(t, t1 - t0), x, h - pad.b + 5);
        }
      }
      if (!n || s.series.every((se) => se.values.every((v) => v == null || v === 0)) && s.empty) {
        ctx.fillStyle = css("--fg-3"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.font = '12px "Geist", sans-serif';
        ctx.fillText(s.empty || "No data yet", pad.l + pw / 2, pad.t + ph / 2);
        if (!n) return;
      }
      for (const se of s.series) {
        const col = se.color;
        // runs of non-null values
        let run = [];
        const runs = [];
        se.values.forEach((v, i) => { if (v == null) { if (run.length) runs.push(run); run = []; } else run.push(i); });
        if (run.length) runs.push(run);
        for (const r of runs) {
          if (se.fill !== false) {
            const g = ctx.createLinearGradient(0, pad.t, 0, pad.t + ph);
            g.addColorStop(0, col + "55"); g.addColorStop(1, col + "00");
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.moveTo(X(r[0]), Y(min));
            for (const i of r) ctx.lineTo(X(i), Y(Math.min(se.values[i], max)));
            ctx.lineTo(X(r[r.length - 1]), Y(min));
            ctx.closePath(); ctx.fill();
          }
          ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
          ctx.beginPath();
          r.forEach((i, k) => { const x = X(i), y = Y(Math.min(se.values[i], max)); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
          ctx.stroke();
          if (r.length === 1) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(r[0]), Y(se.values[r[0]]), 3, 0, 7); ctx.fill(); }
        }
      }
      // hover: crosshair at the nearest sample, a ringed dot per series, the tooltip
      if (hov && n && hov.x >= pad.l - 4 && hov.x <= w - pad.r + 4) {
        let best = 0, bd = Infinity;
        for (let i = 0; i < n; i++) { const d = Math.abs(X(i) - hov.x); if (d < bd) { bd = d; best = i; } }
        const x = Math.round(X(best)) + 0.5;
        ctx.strokeStyle = css("--line-2"); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
        let rows2 = "";
        for (const se of s.series) {
          const v = se.values[best];
          if (v == null) continue;
          ctx.fillStyle = css("--surface"); ctx.beginPath(); ctx.arc(x, Y(Math.min(v, max)), 5, 0, 7); ctx.fill();
          ctx.fillStyle = se.color; ctx.beginPath(); ctx.arc(x, Y(Math.min(v, max)), 3.2, 0, 7); ctx.fill();
          rows2 += `<div class="tr"><span><i class="swatch" style="--c:${se.color}"></i>${esc(se.name)}</span><b>${esc(fmt(v))}${s.unit ? " " + esc(s.unit) : ""}</b></div>`;
        }
        const label = s.xLabel ? s.xLabel(xs[best]) : new Date(xs[best] * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
        const extra = s.tipExtra ? s.tipExtra(best) : "";
        showTip(hov.cx, hov.cy, `<div class="tt">${esc(label)}</div>${rows2 || '<div class="tr"><span>no reading</span></div>'}${extra}`);
      }
    };
    b.paint();
  }

  // ------------------------------------------------------------------------------------------------ bars
  // spec: {labels, values, colors, fmt?, unit?, sub?: [text]}
  function bars(el, spec) {
    const b = box(el);
    b.spec = spec;
    b.draw = (ctx, w, h, hov) => {
      const s = b.spec, n = s.values.length, fmt = s.fmt || fmtDefault;
      const pad = { l: 44, r: 10, t: 18, b: 34 };
      const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
      const max = niceMax(Math.max(0, ...s.values.map((v) => v || 0)) * 1.05);
      ctx.font = MONO;
      for (let i = 0; i <= 4; i++) {
        const y = Math.round(pad.t + ph - ph * i / 4) + 0.5;
        ctx.strokeStyle = css("--grid"); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        ctx.fillStyle = css("--fg-3"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(fmt(max * i / 4), pad.l - 6, y);
      }
      if (!n) return;
      const slot = pw / n, bw = Math.min(64, slot * 0.6);
      let hit = -1;
      for (let i = 0; i < n; i++) {
        const v = s.values[i] || 0, bh = (v / max) * ph, x = pad.l + slot * i + (slot - bw) / 2, y = pad.t + ph - bh;
        const on = hov && hov.x >= pad.l + slot * i && hov.x < pad.l + slot * (i + 1);
        if (on) hit = i;
        ctx.fillStyle = s.colors[i] + (on || hov == null ? "" : "b0");
        roundTop(ctx, x, y, bw, bh, 4); ctx.fill();
        ctx.fillStyle = css("--fg"); ctx.textAlign = "center"; ctx.textBaseline = "bottom";
        if (v) ctx.fillText(fmt(v), x + bw / 2, y - 3);
        ctx.fillStyle = css("--fg-2"); ctx.textBaseline = "top";
        ctx.font = '11.5px "Geist", sans-serif';
        ctx.fillText(clip(ctx, s.labels[i], slot - 6), x + bw / 2, pad.t + ph + 6);
        if (s.sub) { ctx.fillStyle = css("--fg-3"); ctx.fillText(clip(ctx, s.sub[i] || "", slot - 6), x + bw / 2, pad.t + ph + 20); }
        ctx.font = MONO;
      }
      if (hit >= 0) showTip(hov.cx, hov.cy, `<div class="tt">${esc(s.labels[hit])}</div><div class="tr"><span>${esc(s.name || "")}</span><b>${esc(fmt(s.values[hit]))}${s.unit ? " " + esc(s.unit) : ""}</b></div>`);
      else if (hov) hideTip();
    };
    b.paint();
  }
  function roundTop(ctx, x, y, w, h, r) {
    r = Math.min(r, h, w / 2);
    ctx.beginPath(); ctx.moveTo(x, y + h); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h); ctx.closePath();
  }
  function clip(ctx, text, width) {
    text = String(text);
    if (ctx.measureText(text).width <= width) return text;
    while (text.length > 2 && ctx.measureText(text + "…").width > width) text = text.slice(0, -1);
    return text + "…";
  }

  // ------------------------------------------------------------------------------------------------ scatter
  // spec: {points: [{x, y, color, label, html}], xLog?, xFmt?, yFmt?, xName, yName}
  function scatter(el, spec) {
    const b = box(el);
    b.spec = spec;
    b.draw = (ctx, w, h, hov) => {
      const s = b.spec, pts = s.points;
      const pad = { l: 44, r: 12, t: 10, b: 34 };
      const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
      const xf = s.xFmt || fmtDefault, yf = s.yFmt || fmtDefault;
      let xmax = 1, ymax = 1, xmin = s.xLog ? 1e9 : 0;
      for (const p of pts) { if (p.x > xmax) xmax = p.x; if (p.y > ymax) ymax = p.y; if (s.xLog && p.x > 0 && p.x < xmin) xmin = p.x; }
      if (s.xLog) { xmin = Math.pow(10, Math.floor(Math.log10(Math.max(1, xmin === 1e9 ? 1 : xmin)))); xmax = Math.pow(10, Math.ceil(Math.log10(xmax))); }
      else xmax = niceMax(xmax);
      ymax = niceMax(ymax * 1.05);
      const X = (v) => pad.l + (s.xLog ? (Math.log10(Math.max(v, xmin)) - Math.log10(xmin)) / ((Math.log10(xmax) - Math.log10(xmin)) || 1) : v / xmax) * pw;
      const Y = (v) => pad.t + ph - (v / ymax) * ph;
      ctx.font = MONO;
      for (let i = 0; i <= 4; i++) {
        const y = Math.round(pad.t + ph - ph * i / 4) + 0.5;
        ctx.strokeStyle = css("--grid"); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        ctx.fillStyle = css("--fg-3"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(yf(ymax * i / 4), pad.l - 6, y);
      }
      ctx.textBaseline = "top";
      if (s.xLog) {
        for (let v = xmin; v <= xmax; v *= 10) { const x = X(v); ctx.textAlign = "center"; ctx.fillText(xf(v), x, pad.t + ph + 5);
          ctx.strokeStyle = css("--grid"); ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, pad.t); ctx.lineTo(Math.round(x) + 0.5, pad.t + ph); ctx.stroke(); }
      } else for (let i = 0; i <= 4; i++) { ctx.textAlign = "center"; ctx.fillText(xf(xmax * i / 4), pad.l + pw * i / 4, pad.t + ph + 5); }
      ctx.fillStyle = css("--fg-3"); ctx.textAlign = "right"; ctx.font = '11px "Geist", sans-serif';
      ctx.fillText(s.xName || "", w - pad.r, h - 12);
      let near = null, nd = 14;
      for (const p of pts) {
        const x = X(p.x), y = Y(p.y);
        if (hov) { const d = Math.hypot(hov.x - x, hov.y - y); if (d < nd) { nd = d; near = p; } }
        ctx.fillStyle = css("--surface"); ctx.beginPath(); ctx.arc(x, y, 5.2, 0, 7); ctx.fill();
        ctx.fillStyle = p.color + "d0"; ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill();
      }
      if (near) {
        ctx.strokeStyle = css("--fg"); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(near.x), Y(near.y), 6.5, 0, 7); ctx.stroke();
        showTip(hov.cx, hov.cy, near.html || `<div class="tr"><span>${esc(s.xName)}</span><b>${esc(xf(near.x))}</b></div><div class="tr"><span>${esc(s.yName)}</span><b>${esc(yf(near.y))}</b></div>`);
      } else if (hov) hideTip();
      if (!pts.length) { ctx.fillStyle = css("--fg-3"); ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.font = '12px "Geist", sans-serif'; ctx.fillText("No requests in this range", pad.l + pw / 2, pad.t + ph / 2); }
    };
    b.paint();
  }

  // ------------------------------------------------------------------------------------------------ timeline band
  // spec: {t0, t1, segments: [{t0, t1, color, label}], marks: [{t, label}]}
  function band(el, spec) {
    const b = box(el);
    b.spec = spec;
    b.draw = (ctx, w, h, hov) => {
      const s = b.spec, pad = { l: 44, r: 10, t: 6, b: 20 };
      const pw = w - pad.l - pad.r, bh = h - pad.t - pad.b;
      const X = (t) => pad.l + ((t - s.t0) / ((s.t1 - s.t0) || 1)) * pw;
      ctx.fillStyle = "rgba(127,247,232,0.04)"; ctx.fillRect(pad.l, pad.t, pw, bh);
      let hit = null;
      for (const g of s.segments) {
        const x0 = Math.max(pad.l, X(g.t0)), x1 = Math.min(w - pad.r, X(g.t1));
        if (x1 <= x0) continue;
        ctx.fillStyle = g.color; ctx.fillRect(x0, pad.t, Math.max(1, x1 - x0 - 2), bh);
        if (hov && hov.x >= x0 && hov.x <= x1) hit = g;
      }
      for (const m of s.marks || []) {
        const x = Math.round(X(m.t)) + 0.5;
        ctx.strokeStyle = css("--warn"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, pad.t - 3); ctx.lineTo(x, pad.t + bh + 3); ctx.stroke();
      }
      ctx.font = MONO; ctx.fillStyle = css("--fg-3"); ctx.textBaseline = "top";
      for (let i = 0; i <= 5; i++) { const t = s.t0 + (s.t1 - s.t0) * i / 5; ctx.textAlign = i === 0 ? "left" : i === 5 ? "right" : "center"; ctx.fillText(timeFmt(t, s.t1 - s.t0), pad.l + pw * i / 5, h - pad.b + 5); }
      if (hit) {
        const dur = hit.t1 - hit.t0;
        showTip(hov.cx, hov.cy, `<div class="tt">${esc(new Date(hit.t0 * 1000).toLocaleString())}</div><div class="tr"><span><i class="swatch" style="--c:${hit.color}"></i>${esc(hit.label)}</span><b>${dur >= 3600 ? (dur / 3600).toFixed(1) + " h" : Math.round(dur / 60) + " min"}</b></div>`);
      } else if (hov) hideTip();
    };
    b.paint();
  }

  // ------------------------------------------------------------------------------------------------ sparkline
  function spark(el, values, color) {
    line(el, { xs: values.map((_, i) => i), series: [{ name: "", color, values }], compact: true, yMin: 0 });
    const b = boxes.get(el);
    b.c.style.pointerEvents = "none";
  }

  // ------------------------------------------------------------------------------------------------ gauge (SVG arc)
  // {value 0..1 or null, label, text, color}
  function gauge(el, g) {
    const r = 40, cx = 50, cy = 50, a0 = Math.PI * 0.8, a1 = Math.PI * 2.2;
    const pt = (a) => [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    const arc = (from, to) => { const [x0, y0] = pt(from), [x1, y1] = pt(to); return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${r} ${r} 0 ${to - from > Math.PI ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`; };
    const v = g.value == null ? null : Math.max(0, Math.min(1, g.value));
    const fill = v == null || v === 0 ? "" : `<path d="${arc(a0, a0 + (a1 - a0) * Math.max(v, 0.005))}" stroke="${g.color}" stroke-width="7" fill="none" stroke-linecap="round"/>`;
    el.innerHTML = `<svg viewBox="0 0 100 84" role="img" aria-label="${esc(g.label)} ${esc(g.text)}">
      <path d="${arc(a0, a1)}" stroke="rgba(127,247,232,0.08)" stroke-width="7" fill="none" stroke-linecap="round"/>${fill}
      <text x="50" y="54" text-anchor="middle" fill="${css("--fg")}" font-family="Geist Mono, monospace" font-size="17" font-weight="500">${esc(g.text)}</text>
      ${g.sub ? `<text x="50" y="68" text-anchor="middle" fill="${css("--fg-3")}" font-family="Geist, sans-serif" font-size="10">${esc(g.sub)}</text>` : ""}
    </svg><div class="gl">${esc(g.label)}</div>`;
  }

  window.HC = { line, bars, scatter, band, spark, gauge, esc, fmt: fmtDefault, niceMax, hideTip };
})();
