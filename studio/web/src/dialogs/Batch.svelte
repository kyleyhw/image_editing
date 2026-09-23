<script>
  import Modal from "./Modal.svelte";
  import { api, downloadBlob, json, loadImage } from "../lib/api.js";
  import { S, renderThumb } from "../lib/studio.svelte.js";
  let { open = $bindable(false) } = $props();
  let consistency = $state(0), grid = $state([]), canvases = $state([]), busy = $state(false);

  async function preview() {
    busy = true;
    const ps = await json("/api/batch_predict", { ids: S.photos.map((p) => p.id), style: S.style, consistency: consistency / 100 });
    grid = S.photos.map((p, i) => ({ id: p.id, params: ps[i] }));
    await Promise.resolve();
    for (const [i, g] of grid.entries()) renderThumb(await loadImage(`/api/photo/${g.id}?edge=400`), g.params, 400, canvases[i]);
    busy = false;
    return ps;
  }
  async function exportAll() {
    const ps = await preview();
    for (const [i, ph] of S.photos.entries()) {
      const r = await api("/api/export", { id: ph.id, params: { ...ps[i], overrides: {} }, format: "jpg" });
      await downloadBlob(r, `${ph.name}.jpg`);
    }
  }
</script>

<Modal bind:open title="Batch">
  <p class="muted">Apply the current style ({S.style}) to every photo in the library.</p>
  <label class="row">Series consistency <input type="range" min="0" max="100" bind:value={consistency} /><output>{consistency}%</output></label>
  <p class="muted small">0% = every photo edited on its own; 100% = one shared edit for the whole set.</p>
  <div id="batchGrid">{#each grid as g, i (g.id)}<canvas bind:this={canvases[i]}></canvas>{/each}</div>
  {#snippet footer()}
    <button class="btn" onclick={preview} disabled={busy}>Preview</button>
    <button class="btn primary" onclick={exportAll} disabled={busy}>Export all JPEGs</button>
  {/snippet}
</Modal>
