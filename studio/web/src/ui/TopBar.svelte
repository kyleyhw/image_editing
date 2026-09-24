<script>
  import { S, undo, redo, exportCurrent } from "../lib/studio.svelte.js";
  import { I } from "../lib/icons.js";
  let { exportFmt = $bindable("jpg"), onCreate, onBatch } = $props();
  const REPO = "https://github.com/kyleyhw/image_editing";
  const name = $derived(S.photos.find((p) => p.id === S.id)?.name || "");
</script>

<header class="topbar">
  <a class="brand" href="./">photostyle</a>
  {#if S.id}
    <nav class="crumbs" aria-label="Location"><span class="muted">Photos</span><span class="sep">/</span><span class="crumb-file">{name}</span></nav>
  {/if}
  <div class="grow"></div>
  {#if S.id}
    <button class="icon-btn" title="Undo (Ctrl+Z)" aria-label="Undo" disabled={!S.undo.length} onclick={undo}>{@html I.undo}</button>
    <button class="icon-btn" title="Redo (Ctrl+Shift+Z)" aria-label="Redo" disabled={!S.redo.length} onclick={redo}>{@html I.redo}</button>
    <span class="vsep"></span>
    {#if S.caps.create}<button class="btn hide-sm" onclick={onCreate}>New look</button>{/if}
    {#if S.caps.batch}<button class="btn hide-sm" onclick={onBatch} disabled={S.photos.length < 2}>Batch</button>{/if}
    <select id="exportFmt" aria-label="Export format" bind:value={exportFmt}>
      {#each S.formats as [v, label]}<option value={v}>{label}</option>{/each}
    </select>
    <button id="btnExport" class="btn primary" title="Export (E)" disabled={!S.params || S.busy} onclick={() => exportCurrent(exportFmt)}>Export</button>
  {:else}
    <a class="btn" href={REPO} target="_blank" rel="noopener">{@html I.github}<span class="hide-xs">Source</span></a>
  {/if}
</header>
