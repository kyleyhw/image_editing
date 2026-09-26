<script>
  // Welcome-screen demo: each look's own example (a photo of the subject its tutorial was written
  // for, before/after the look), one per kind of photo. Move the pointer to compare.
  import { S } from "../lib/studio.svelte.js";
  const base = import.meta.env.BASE_URL;
  let cur = $state(0), x = $state(50);
  const items = $derived.by(() => {
    const seen = new Set(), out = [];
    for (const st of S.styles) {
      if (!st.example || seen.has(st.category)) continue;
      seen.add(st.category);
      out.push(st);
    }
    return out;
  });
  const it = $derived(items[Math.min(cur, items.length - 1)]);
  function move(e) { const r = e.currentTarget.getBoundingClientRect(); x = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100)); }
</script>

{#if it}
  <figure class="showcase">
    <div class="sc-frame" role="img" aria-label={`${it.title}: before and after`} onpointermove={move}>
      <img src={base + it.example.before} alt="" />
      <img class="after" src={base + it.example.after} alt="" style={`clip-path: inset(0 0 0 ${x}%)`} />
      <span class="sc-line" style={`left:${x}%`}></span>
      <span class="tag tag-r">Before</span><span class="tag tag-l">{it.title}</span>
    </div>
    <figcaption>
      {#each items as st, i}<button class="link-btn" class:on={i === cur} onclick={() => (cur = i)}>{st.category}</button>{/each}
    </figcaption>
    <p class="sc-credit">{it.title}, from <a href={it.tutorials?.[0]?.url} target="_blank" rel="noopener">{it.tutorials?.[0]?.title}</a>.
      Photo: <a href={it.example.photo.url} target="_blank" rel="noopener">{it.example.photo.title || "untitled"}</a>
      by {it.example.photo.creator || "unknown"} ({(it.example.photo.license || "").toUpperCase()}).</p>
  </figure>
{/if}
