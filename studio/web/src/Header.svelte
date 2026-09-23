<script>
  import { S, upload, exportCurrent } from "./lib/studio.svelte.js";
  let { exportFmt = $bindable("jpg"), onCreate, onBatch, onSettings } = $props();
  const name = $derived(S.photos.find((p) => p.id === S.id)?.name || "");
</script>

<header>
  <strong>photostyle Studio</strong>
  <span class="muted">{name}</span>
  <span class="spacer"></span>
  <label class="btn" title="Upload photos">Upload<input id="upload" type="file" accept="image/*,.dng,.cr2,.cr3,.nef,.arw"
    multiple hidden onchange={(e) => upload(e.currentTarget.files)} /></label>
  <button class="btn" onclick={onCreate}>+ Create style</button>
  <button class="btn" onclick={onBatch} disabled={!S.photos.length}>Batch</button>
  <select id="exportFmt" aria-label="Export format" bind:value={exportFmt}>
    <option value="jpg">Export JPEG</option><option value="cube">Export .cube</option>
    <option value="xmp">Export XMP (curves)</option><option value="json">Export JSON</option>
  </select>
  <button id="btnExport" class="btn primary" title="Export (E)" disabled={!S.params} onclick={() => exportCurrent(exportFmt)}>Export</button>
  <button class="btn icon" title="Customise panels" aria-label="Customise panels" onclick={onSettings}>⚙</button>
</header>
