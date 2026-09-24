<script>
  import { S, view, renderThumb, setStyle } from "./lib/studio.svelte.js";
  import { json } from "./lib/api.js";
  let { layout = "cards" } = $props();

  let canvases = $state({});
  let gen = 0;
  // Rebuild thumbnails when the photo, the style list or the layout changes; a newer run cancels an older one.
  $effect(() => {
    const id = S.id, styles = S.styles.map((s) => s.name);
    S.imgVersion; layout;
    if (!id || !view.img) return;
    const g = ++gen;
    (async () => {
      for (const name of styles) {
        const p = await json("/api/predict", { id, style: name });
        if (g !== gen) return;
        const c = canvases[name];
        if (c) renderThumb(view.img, p, 200, c);
      }
    })();
  });
  const pretty = (n) => n.replace(/_/g, " ");
</script>

<div id="styleStrip" class={`strip-${layout}`} aria-label="Styles">
  <h4 class="strip-title">Styles</h4>
  <button class="thumb" class:sel={S.mode === "before"} onclick={() => (S.mode = "before")} title="The original photo">
    <span class="tframe">{#if S.id}<img src={`/api/photo/${S.id}?edge=240`} alt="original" />{/if}</span>
    <span class="tname">Original</span>
    <small class="tdesc">Your photo, unedited</small>
  </button>
  {#each S.styles as st, i (st.name)}
    <button class="thumb" class:sel={st.name === S.style && S.mode !== "before"} onclick={() => setStyle(st.name)} title={st.description}>
      <span class="tframe"><canvas bind:this={canvases[st.name]}></canvas></span>
      <span class="tname"><kbd class="tkey">{i + 1}</kbd>{pretty(st.name)}</span>
      <small class="tdesc">{st.description || ""}</small>
    </button>
  {/each}
</div>
