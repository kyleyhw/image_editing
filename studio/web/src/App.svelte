<script>
  import { onMount } from "svelte";
  import Header from "./Header.svelte";
  import Library from "./Library.svelte";
  import Stage from "./Stage.svelte";
  import StyleStrip from "./StyleStrip.svelte";
  import Panels from "./Panels.svelte";
  import Dock from "./Dock.svelte";
  import CreateStyle from "./dialogs/CreateStyle.svelte";
  import Batch from "./dialogs/Batch.svelte";
  import Settings from "./dialogs/Settings.svelte";
  import { S, loadPhotos, loadStyles, selectPhoto, setStyle, undo, redo, resetToModel, exportCurrent, commit } from "./lib/studio.svelte.js";
  import { loadPrefs, savePrefs } from "./lib/prefs.js";
  import { PANELS } from "./panels/registry.js";
  import { DESIGNS } from "./lib/designs.js";

  let prefs = $state(loadPrefs({ order: PANELS.map((p) => p.id), collapsed: [], hidden: [], exportFmt: "jpg",
                                 design: "darkroom", tab: "style" }));
  $effect(() => savePrefs($state.snapshot(prefs)));
  // ?design=… in the URL overrides the saved design (handy for comparing them side by side)
  const urlDesign = new URLSearchParams(location.search).get("design");
  if (urlDesign && DESIGNS.some((d) => d.id === urlDesign)) prefs.design = urlDesign;
  const design = $derived(DESIGNS.find((d) => d.id === prefs.design) || DESIGNS[0]);
  let dialogs = $state({ create: false, batch: false, settings: false });
  let sheetOpen = $state(false), drawerOpen = $state(false);

  onMount(async () => {
    await loadStyles(); await loadPhotos();
    if (S.photos.length) await selectPhoto(S.photos[0].id);
  });

  function onKey(e) {
    if (e.target.matches?.("input, select, textarea") || Object.values(dialogs).some(Boolean)) return;
    const k = e.key;
    if (k === "\\") S.holdBefore = true;
    else if (k === "[" || k === "]") { S.strength = Math.max(0, Math.min(1.5, S.strength + (k === "]" ? 0.05 : -0.05))); commit(); }
    else if (k.toLowerCase() === "r" && !e.ctrlKey && !e.metaKey) resetToModel();
    else if (k.toLowerCase() === "e") exportCurrent(prefs.exportFmt);
    else if (k.toLowerCase() === "y") { const m = ["split", "after", "before"]; S.mode = m[(m.indexOf(S.mode) + 1) % 3]; }
    else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") { e.shiftKey ? redo() : undo(); e.preventDefault(); }
    else if (/^[1-9]$/.test(k) && S.styles[+k - 1]) setStyle(S.styles[+k - 1].name);
    else if (k === "Escape") { sheetOpen = false; drawerOpen = false; }
    else if (k === "ArrowRight" || k === "ArrowLeft") {
      const i = S.photos.findIndex((p) => p.id === S.id), j = i + (k === "ArrowRight" ? 1 : -1);
      if (S.photos[j]) selectPhoto(S.photos[j].id);
    }
  }
  function onKeyUp(e) { if (e.key === "\\") S.holdBefore = false; }
</script>

<svelte:window onkeydown={onKey} onkeyup={onKeyUp} />

<div class={`shell design-${design.id}`} class:sheet-open={sheetOpen} class:drawer-open={drawerOpen}>
  <Header bind:exportFmt={prefs.exportFmt} {design} onCreate={() => (dialogs.create = true)} onBatch={() => (dialogs.batch = true)}
          onSettings={() => (dialogs.settings = true)} onLibrary={() => (drawerOpen = !drawerOpen)} />
  <Library onPick={() => (drawerOpen = false)} />
  <section id="stage">
    <Stage />
  </section>
  <StyleStrip layout={design.strip} />
  <Panels {prefs} layout={design.panels} onClose={() => (sheetOpen = false)} />
  {#if design.dock}<Dock onAdjust={() => (sheetOpen = !sheetOpen)} {sheetOpen} />{/if}
  {#if design.dock && (sheetOpen || drawerOpen)}
    <button class="scrim" aria-label="Close" onclick={() => { sheetOpen = false; drawerOpen = false; }}></button>
  {/if}
</div>

<CreateStyle bind:open={dialogs.create} />
<Batch bind:open={dialogs.batch} />
<Settings bind:open={dialogs.settings} {prefs} />
