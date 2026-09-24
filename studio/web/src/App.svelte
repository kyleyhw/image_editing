<script>
  import { onMount } from "svelte";
  import Stage from "./Stage.svelte";
  import TopBar from "./ui/TopBar.svelte";
  import Sidebar from "./ui/Sidebar.svelte";
  import Backdrop from "./ui/Backdrop.svelte";
  import Inspector from "./ui/Inspector.svelte";
  import Welcome from "./ui/Welcome.svelte";
  import CreateStyle from "./dialogs/CreateStyle.svelte";
  import Batch from "./dialogs/Batch.svelte";
  import { S, boot, upload, selectPhoto, setStyle, undo, redo, resetToModel, exportCurrent, commit } from "./lib/studio.svelte.js";
  import { loadPrefs, savePrefs } from "./lib/prefs.js";
  import { I } from "./lib/icons.js";

  let prefs = $state(loadPrefs({ exportFmt: "jpg" }));
  $effect(() => savePrefs($state.snapshot(prefs)));
  $effect(() => { if (!S.formats.some(([v]) => v === prefs.exportFmt)) prefs.exportFmt = S.formats[0][0]; });
  let dialogs = $state({ create: false, batch: false });
  let sheetOpen = $state(false), dragging = $state(0), bootError = $state("");

  onMount(() => { boot().catch((e) => (bootError = String(e))); });

  function onKey(e) {
    if (e.target.matches?.("input, select, textarea") || Object.values(dialogs).some(Boolean) || !S.id) return;
    const k = e.key;
    if (k === "\\") S.holdBefore = true;
    else if (k === "[" || k === "]") { S.strength = Math.max(0, Math.min(1.5, S.strength + (k === "]" ? 0.05 : -0.05))); commit(); }
    else if (k.toLowerCase() === "r" && !e.ctrlKey && !e.metaKey) resetToModel();
    else if (k.toLowerCase() === "e" && !e.ctrlKey && !e.metaKey) exportCurrent(prefs.exportFmt);
    else if (k.toLowerCase() === "y") { const m = ["split", "after", "before"]; S.mode = m[(m.indexOf(S.mode) + 1) % 3]; }
    else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") { e.shiftKey ? redo() : undo(); e.preventDefault(); }
    else if (/^[1-9]$/.test(k) && S.styles[+k - 1]) setStyle(S.styles[+k - 1].name);
    else if (k === "0") S.mode = "before";
    else if (k === "Escape") sheetOpen = false;
    else if (k === "ArrowRight" || k === "ArrowLeft") {
      const i = S.photos.findIndex((p) => p.id === S.id), j = i + (k === "ArrowRight" ? 1 : -1);
      if (S.photos[j]) selectPhoto(S.photos[j].id);
    }
  }
  function onKeyUp(e) { if (e.key === "\\") S.holdBefore = false; }
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
</script>

<svelte:window onkeydown={onKey} onkeyup={onKeyUp}
  ondragenter={(e) => { if (hasFiles(e)) { dragging++; e.preventDefault(); } }}
  ondragleave={() => (dragging = Math.max(0, dragging - 1))}
  ondragover={(e) => { if (hasFiles(e)) e.preventDefault(); }}
  ondrop={(e) => { e.preventDefault(); dragging = 0; upload(e.dataTransfer?.files); }} />

<input id="upload" type="file" accept={S.accept} multiple hidden onchange={(e) => { upload(e.currentTarget.files); e.currentTarget.value = ""; }} />

<Backdrop />
<div class="app" class:empty={!S.id} class:sheet-open={sheetOpen}>
  <TopBar bind:exportFmt={prefs.exportFmt} onCreate={() => (dialogs.create = true)} onBatch={() => (dialogs.batch = true)} />
  {#if S.id}<Sidebar onCreate={() => (dialogs.create = true)} />{/if}
  <main class="center">
    <Stage />
    {#if !S.id && S.ready}<Welcome />{/if}
    {#if bootError}<p class="note boot">{bootError}</p>{/if}
  </main>
  {#if S.id}
    <Inspector bind:open={sheetOpen} />
    <div class="quickbar">
      <label class="qb-strength"><span>Strength</span>
        <input id="strengthQuick" type="range" min="0" max="150" value={Math.round(S.strength * 100)} disabled={!S.params}
               style={`--a:0%;--b:${(S.strength / 1.5) * 100}%`}
               oninput={(e) => (S.strength = Number(e.currentTarget.value) / 100)} onchange={commit} />
        <output>{Math.round(S.strength * 100)}</output></label>
      <button class="btn" onclick={() => (sheetOpen = true)}>{@html I.sliders}Adjust</button>
    </div>
    {#if sheetOpen}<button class="scrim" aria-label="Close adjustments" onclick={() => (sheetOpen = false)}></button>{/if}
  {/if}
</div>

{#if dragging}<div class="dropzone" aria-hidden="true">Drop to open</div>{/if}
{#if S.busy && !S.id}<div class="thinking fixed"><span></span>Opening…</div>{/if}

{#if S.caps.create}<CreateStyle bind:open={dialogs.create} />{/if}
{#if S.caps.batch}<Batch bind:open={dialogs.batch} />{/if}
