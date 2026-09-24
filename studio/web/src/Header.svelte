<script>
  import { S, upload, exportCurrent } from "./lib/studio.svelte.js";
  let { exportFmt = $bindable("jpg"), design, onCreate, onBatch, onSettings, onLibrary } = $props();
  const name = $derived(S.photos.find((p) => p.id === S.id)?.name || "");
</script>

<header id="topbar">
  {#if design.dock}<button class="btn icon ghost" aria-label="Library" onclick={onLibrary}>☰</button>{/if}
  <span class="brand"><span class="logo" aria-hidden="true"></span>photostyle<span class="brand-sub">Studio</span></span>
  <span class="filename muted">{name}</span>
  <span class="spacer"></span>
  <label class="btn ghost" title="Upload photos">Upload<input id="upload" type="file" accept="image/*,.dng,.cr2,.cr3,.nef,.arw"
    multiple hidden onchange={(e) => upload(e.currentTarget.files)} /></label>
  <button class="btn ghost" onclick={onCreate}>+ Create style</button>
  <button class="btn ghost" onclick={onBatch} disabled={!S.photos.length}>Batch</button>
  <span class="export">
    <select id="exportFmt" aria-label="Export format" bind:value={exportFmt}>
      <option value="jpg">JPEG</option><option value="cube">.cube LUT</option>
      <option value="xmp">Lightroom XMP</option><option value="json">JSON</option>
    </select>
    <button id="btnExport" class="btn primary" title="Export (E)" disabled={!S.params} onclick={() => exportCurrent(exportFmt)}>Export</button>
  </span>
  <button class="btn icon ghost" title="Design and panels" aria-label="Customise panels" onclick={onSettings}>⚙</button>
</header>
