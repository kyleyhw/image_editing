<script>
  import { S, view, renderThumb, setStyle } from "./lib/studio.svelte.js";
  import { json } from "./lib/api.js";

  let canvases = $state({});
  let gen = 0;
  // Rebuild thumbnails when the photo or the style list changes; a newer run cancels an older one.
  $effect(() => {
    const id = S.id, styles = S.styles.map((s) => s.name);
    S.imgVersion;
    if (!id || !view.img) return;
    const g = ++gen;
    (async () => {
      for (const name of styles) {
        const p = await json("/api/predict", { id, style: name });
        if (g !== gen) return;
        const c = canvases[name];
        if (c) renderThumb(view.img, p, 140, c);
      }
    })();
  });
</script>

<div id="styleStrip" aria-label="Styles">
  <button class="thumb" onclick={() => (S.mode = "before")}>
    {#if S.id}<img src={`/api/photo/${S.id}?edge=240`} alt="original" />{/if}<span>original</span>
  </button>
  {#each S.styles as st, i (st.name)}
    <button class="thumb" class:sel={st.name === S.style} onclick={() => setStyle(st.name)} title={st.description}>
      <canvas bind:this={canvases[st.name]}></canvas><span>{i + 1} · {st.name}</span>
    </button>
  {/each}
</div>
