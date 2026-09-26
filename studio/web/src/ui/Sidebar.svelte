<script>
  import { S, view, photoURL, renderThumb, selectPhoto, setStyle, setPeek, pretty } from "../lib/studio.svelte.js";
  import LookExample from "./LookExample.svelte";
  let { onCreate } = $props();
  const GUIDE = "https://github.com/kyleyhw/image_editing/blob/master/docs/user_guide.md#creating-a-new-style-idea--references--style";
  let canvases = $state({});
  let openInfo = $state(null);
  const name = (st) => st.title || pretty(st.name);

  // Looks grouped by the kind of photo they suit, in the order the packs give.
  const groups = $derived.by(() => {
    const out = [];
    for (const st of S.styles) {
      const c = st.category || "More looks";
      let g = out.find((x) => x.name === c);
      if (!g) out.push((g = { name: c, looks: [] }));
      g.looks.push(st);
    }
    return out;
  });
  const keyOf = (st) => S.styles.indexOf(st);

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
      {#each groups as g (g.name)}
        <li class="group-head">{g.name}</li>
        {#each g.looks as st (st.name)}
          {@const i = keyOf(st)}
          <li class="look-row">
            <button class="item look" class:on={st.name === S.style && S.mode !== "before"} title={st.description}
                    onclick={() => setStyle(st.name)} onpointerenter={() => setPeek(st.name)} onpointerleave={() => setPeek(null)}>
              <span class="item-thumb" class:loading={!S.byStyle[st.name]}><canvas bind:this={canvases[st.name]}></canvas></span>
              <span class="item-name">{name(st)}</span>{#if i < 9}<kbd>{i + 1}</kbd>{/if}
            </button>
            {#if st.example || st.tutorials?.length}
              <button class="info-btn" class:on={openInfo === st.name} aria-expanded={openInfo === st.name}
                      aria-label={`Example and sources for ${name(st)}`} title="Example and sources"
                      onclick={() => (openInfo = openInfo === st.name ? null : st.name)}>
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>
              </button>
            {/if}
          </li>
          {#if openInfo === st.name}<li class="look-ex-row"><LookExample look={st} /></li>{/if}
        {/each}
      {/each}
    </ul>
    {#if S.caps.create}
      <button class="link-btn add-look" onclick={onCreate}>+ New look</button>
    {:else}
      <a class="link-btn add-look" href={GUIDE} target="_blank" rel="noopener">Train your own look</a>
    {/if}
  </section>
</aside>
