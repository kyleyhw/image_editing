<script>
  import { S, view, photoURL, renderThumb, selectPhoto, setStyle, setPeek, pretty } from "../lib/studio.svelte.js";
  let { onCreate } = $props();
  const GUIDE = "https://github.com/kyleyhw/image_editing/blob/master/docs/user_guide.md#creating-a-new-style-idea--references--style";
  let canvases = $state({});

  // Look previews: each look rendered on this photo, as its params arrive.
  $effect(() => {
    S.imgVersion;
    for (const st of S.styles) {
      const p = S.byStyle[st.name], c = canvases[st.name];
      if (p && c && view.img && c.dataset.for !== `${S.id}/${S.imgVersion}`) {
        renderThumb(view.img, p, 160, c);
        c.dataset.for = `${S.id}/${S.imgVersion}`;
      }
    }
  });
</script>

<aside class="sidebar panel" aria-label="Photos and looks">
  <section class="side-sec side-photos">
    <div class="side-head"><span>Photos</span><label for="upload" class="link-btn" title="Add photos">Add</label></div>
    <ul class="list">
      {#each S.photos as p (p.id)}
        <li><button class="item" class:on={p.id === S.id} onclick={() => selectPhoto(p.id)} title={p.name}>
          <img class="item-thumb" src={photoURL(p.id, 240)} alt="" /><span class="item-name">{p.name}</span>
        </button></li>
      {/each}
    </ul>
  </section>
  <section class="side-sec side-looks">
    <div class="side-head"><span>Looks</span><span class="muted">{S.styles.length}</span></div>
    <ul class="list looks">
      <li><button class="item" class:on={S.mode === "before"} onclick={() => (S.mode = "before")}
                  onpointerenter={() => setPeek("__original")} onpointerleave={() => setPeek(null)}>
        <img class="item-thumb" src={photoURL(S.id, 240)} alt="" /><span class="item-name">Original</span><kbd>0</kbd>
      </button></li>
      {#each S.styles as st, i (st.name)}
        <li><button class="item look" class:on={st.name === S.style && S.mode !== "before"} title={st.description}
                    onclick={() => setStyle(st.name)} onpointerenter={() => setPeek(st.name)} onpointerleave={() => setPeek(null)}>
          <span class="item-thumb" class:loading={!S.byStyle[st.name]}><canvas bind:this={canvases[st.name]}></canvas></span>
          <span class="item-name">{pretty(st.name)}</span>{#if i < 9}<kbd>{i + 1}</kbd>{/if}
        </button></li>
      {/each}
    </ul>
    {#if S.caps.create}
      <button class="link-btn add-look" onclick={onCreate}>+ New look</button>
    {:else}
      <a class="link-btn add-look" href={GUIDE} target="_blank" rel="noopener">Train your own look</a>
    {/if}
  </section>
</aside>
