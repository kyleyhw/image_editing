export async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r;
}
export const json = async (path, body) => (await api(path, body)).json();

export async function pollJob(job, onProg) {
  for (;;) {
    const j = await json(`/api/jobs/${job}`);
    onProg?.(j);
    if (j.status !== "running") return j;
    await new Promise((r) => setTimeout(r, 800));
  }
}

export async function downloadBlob(r, fallback) {
  const name = (r.headers.get("content-disposition") || "").match(/filename="?([^";]+)/)?.[1] || fallback;
  const a = document.createElement("a"); a.href = URL.createObjectURL(await r.blob()); a.download = name; a.click();
}

export function loadImage(src) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
}
