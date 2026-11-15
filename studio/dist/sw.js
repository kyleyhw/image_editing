// Offline support for the hosted editor. Hashed build assets and the model are cached on first use
// (cache-first: their names change when their contents do); the page itself and styles.json are
// network-first, so a new deploy is picked up when online and the cached copy is used offline.
const CACHE = "irodori-v1";
const SHELL = ["./", "./index.html", "./manifest.webmanifest", "./icons/icon-192.png"];

self.addEventListener("install", (e) => { e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  const networkFirst = e.request.mode === "navigate" || url.pathname.endsWith("/styles.json") || url.pathname.endsWith("/index.html");
  e.respondWith(networkFirst ? fromNetwork(e.request) : fromCache(e.request));
});
async function fromNetwork(req) {
  try { const r = await fetch(req); if (r.ok) (await caches.open(CACHE)).put(req, r.clone()); return r; }
  catch { return (await caches.match(req)) || (await caches.match("./index.html")); }
}
async function fromCache(req) {
  const hit = await caches.match(req);
  if (hit) return hit;
  const r = await fetch(req);
  if (r.ok && (r.type === "basic" || r.type === "default")) (await caches.open(CACHE)).put(req, r.clone());
  return r;
}
