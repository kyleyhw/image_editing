// Per-browser UI preferences (panel order, collapsed panels, theme). Conveniences only:
// every read/write is guarded so private windows or blocked storage just use defaults.
const KEY = "photostyle.studio.prefs.v1";
export function loadPrefs(defaults) {
  try { return { ...defaults, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch { return { ...defaults }; }
}
export function savePrefs(p) {
  try { localStorage.setItem(KEY, JSON.stringify(p)); } catch { /* storage unavailable */ }
}
