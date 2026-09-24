<script>
  // Before/after examples (tools/export_web.py hero): public-domain samples graded by the
  // publishable looks. Drag or hover to move the divider; arrows switch example.
  import { onMount } from "svelte";
  import { pretty } from "../lib/studio.svelte.js";
  const base = `${import.meta.env.BASE_URL}samples/`;
  let items = $state([]), cur = $state(0), x = $state(50);
  onMount(() => { fetch(`${base}hero/hero.json`).then((r) => (r.ok ? r.json() : [])).then((v) => (items = v)).catch(() => {}); });
  function move(e) { const r = e.currentTarget.getBoundingClientRect(); x = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100)); }
</script>

{#if items.length}
  <figure class="showcase">
    <div class="sc-frame" role="img" aria-label="Before and after" onpointermove={move}>
      <img src={base + items[cur].before} alt="" />
      <img class="after" src={base + items[cur].after} alt="" style={`clip-path: inset(0 0 0 ${x}%)`} />
      <span class="sc-line" style={`left:${x}%`}></span>
      <span class="tag tag-r">Before</span><span class="tag tag-l">{pretty(items[cur].style)}</span>
    </div>
    <figcaption>
      {#each items as it, i}<button class="link-btn" class:on={i === cur} onclick={() => (cur = i)}>{pretty(it.style)} · {it.before.replace(/\..*/, "").replace(/-/g, " ")}</button>{/each}
    </figcaption>
  </figure>
{/if}
