<script>
  import { S, selectPhoto, photoURL, undo, redo, exportCurrent } from "../lib/studio.svelte.js";
  import { I } from "../lib/icons.js";
  let { exportFmt = $bindable("jpg"), onCreate, onBatch } = $props();
  const REPO = "https://github.com/kyleyhw/image_editing";
</script>

<header class="topbar">
  <a class="brand" href="./" aria-label="photostyle">
    <span class="mark" aria-hidden="true"></span><span class="word">photo<em>style</em></span>
  </a>

  {#if S.id}
    <nav class="film" aria-label="Your photos">
      {#each S.photos as p (p.id)}
        <button class="film-item" class:on={p.id === S.id} title={p.name} aria-label={p.name} onclick={() => selectPhoto(p.id)}>
          <img src={photoURL(p.id, 240)} alt="" />
        </button>
      {/each}
      <label for="upload" class="film-add" title="Add photos" aria-label="Add photos">{@html I.plus}</label>
    </nav>
  {/if}

  <div class="grow"></div>

  {#if S.id}
    <div class="tb-group">
      <button class="icon-btn" title="Undo (Ctrl+Z)" aria-label="Undo" disabled={!S.undo.length} onclick={undo}>{@html I.undo}</button>
      <button class="icon-btn" title="Redo (Ctrl+Shift+Z)" aria-label="Redo" disabled={!S.redo.length} onclick={redo}>{@html I.redo}</button>
    </div>
    {#if S.caps.create}<button class="btn ghost hide-sm" onclick={onCreate}>{@html I.sparkle}New style</button>{/if}
    {#if S.caps.batch}<button class="btn ghost hide-sm" onclick={onBatch} disabled={S.photos.length < 2}>{@html I.layers}Batch</button>{/if}
    <div class="export">
      <select id="exportFmt" aria-label="Export format" bind:value={exportFmt}>
        {#each S.formats as [v, label]}<option value={v}>{label}</option>{/each}
      </select>
      <button id="btnExport" class="btn primary" title="Export (E)" disabled={!S.params || S.busy} onclick={() => exportCurrent(exportFmt)}>
        {@html I.download}<span class="hide-xs">Export</span>
      </button>
    </div>
  {:else}
    <a class="btn ghost" href={REPO} target="_blank" rel="noopener">{@html I.github}<span class="hide-xs">Source</span></a>
  {/if}
</header>
