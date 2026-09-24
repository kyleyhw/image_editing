<script>
  import { S, view, photoURL, renderThumb, setStyle, setPeek, pretty } from "../lib/studio.svelte.js";
  import { I } from "../lib/icons.js";
  let { onCreate } = $props();
  const REPO = "https://github.com/kyleyhw/image_editing/blob/master/docs/user_guide.md#creating-a-new-style-idea--references--style";
  let canvases = $state({});

  // Previews: each style rendered on this photo, as its params arrive.
  $effect(() => {
    S.imgVersion;
    for (const st of S.styles) {
      const p = S.byStyle[st.name], c = canvases[st.name];
      if (p && c && view.img && c.dataset.for !== `${S.id}/${S.imgVersion}`) {
        renderThumb(view.img, p, 240, c);
        c.dataset.for = `${S.id}/${S.imgVersion}`;
      }
    }
  });
</script>

<aside class="rail" aria-label="Styles">
  <div class="rail-head"><span class="eyebrow">Looks</span><span class="count">{S.styles.length}</span></div>
  <ol class="looks">
    <li>
      <button class="look" class:on={S.mode === "before"} onclick={() => (S.mode = "before")}
              onpointerenter={() => setPeek("__original")} onpointerleave={() => setPeek(null)}>
        <span class="look-thumb"><img src={photoURL(S.id, 240)} alt="" /></span>
        <span class="look-idx">00</span>
        <span class="look-name">Original</span>
      </button>
    </li>
    {#each S.styles as st, i (st.name)}
      <li style={`--i:${i}`}>
        <button class="look" class:on={st.name === S.style && S.mode !== "before"} title={st.description}
                onclick={() => setStyle(st.name)} onpointerenter={() => setPeek(st.name)} onpointerleave={() => setPeek(null)}>
          <span class="look-thumb" class:loading={!S.byStyle[st.name]}><canvas bind:this={canvases[st.name]}></canvas></span>
          <span class="look-idx">{String(i + 1).padStart(2, "0")}</span>
          <span class="look-name">{pretty(st.name)}</span>
        </button>
      </li>
    {/each}
  </ol>
  <div class="rail-foot">
    {#if S.caps.create}
      <button class="btn ghost wide" onclick={onCreate}>{@html I.sparkle}Teach a new look</button>
    {:else}
      <a class="rail-link" href={REPO} target="_blank" rel="noopener">Train your own look <span aria-hidden="true">→</span></a>
    {/if}
  </div>
</aside>
