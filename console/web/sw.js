// Halocline console service worker: makes the page installable and opens the shell offline.
// Only the page's own files are cached. Live data (/api, /v1) always goes to the network, never to a cache.
"use strict";
const SHELL = "halocline-shell-v1";
const FILES = ["./", "web/app.css", "web/charts.js", "web/app.js", "web/mark.svg", "web/icons/icon-192.png",
               "fonts/geist-latin-wght.woff2", "fonts/geist-mono-latin-wght.woff2", "manifest.webmanifest"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== SHELL).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || /\/(api|v1)\//.test(url.pathname)) return;
  // network first, so a new release shows at once; the cached copy only when the console is unreachable
  e.respondWith(fetch(e.request).then((r) => {
    if (r.ok) { const copy = r.clone(); caches.open(SHELL).then((c) => c.put(e.request, copy)); }
    return r;
  }).catch(() => caches.match(e.request)));
});
