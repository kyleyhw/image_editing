// Keeping work between visits (Phase 25), for the browser build: photos and their edits are stored
// in this browser's IndexedDB, on this device only. Presets and the settings clipboard live in
// localStorage. Every call is guarded: private windows or blocked storage just mean nothing is kept.
const DB = "irodori", VERSION = 1;
let dbp = null;

function db() {
  dbp ??= new Promise((res, rej) => {
    const r = indexedDB.open(DB, VERSION);
    r.onupgradeneeded = () => { r.result.createObjectStore("photos", { keyPath: "id" }); r.result.createObjectStore("edits", { keyPath: "id" }); };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
  return dbp;
}
async function tx(store, mode, fn) {
  try {
    const d = await db();
    return await new Promise((res, rej) => {
      const t = d.transaction(store, mode), s = t.objectStore(store), out = fn(s);
      t.oncomplete = () => res(out?.result ?? out);
      t.onerror = () => rej(t.error);
    });
  } catch { return null; }
}

export const savePhoto = (id, name, blob, order) => tx("photos", "readwrite", (s) => s.put({ id, name, blob, order }));
export const deletePhoto = (id) => Promise.all([tx("photos", "readwrite", (s) => s.delete(id)), tx("edits", "readwrite", (s) => s.delete(id))]);
export async function loadPhotos() {
  const all = await tx("photos", "readonly", (s) => s.getAll());
  return (all || []).sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
}
export const saveEdit = (id, edit) => tx("edits", "readwrite", (s) => s.put({ id, ...edit }));
export async function loadEdits() {
  const all = await tx("edits", "readonly", (s) => s.getAll());
  return new Map((all || []).map(({ id, ...e }) => [id, e]));
}
export async function clearAll() {
  await tx("photos", "readwrite", (s) => s.clear());
  await tx("edits", "readwrite", (s) => s.clear());
}

const LS = (k, d) => { try { return JSON.parse(localStorage.getItem(k) || "null") ?? d; } catch { return d; } };
const SS = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } };
export const loadPresets = () => LS("irodori.presets.v1", []);
export const savePresets = (p) => SS("irodori.presets.v1", p);
export const loadClip = () => LS("irodori.clipboard.v1", null);
export const saveClip = (c) => SS("irodori.clipboard.v1", c);
