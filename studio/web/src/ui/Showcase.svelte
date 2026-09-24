<script>
  // Welcome-screen demo: public-domain samples before/after a look (tools/export_web.py hero),
  // with an auto-sweeping divider; cycles through the pairs.
  import { onMount } from "svelte";
  const base = `${import.meta.env.BASE_URL}samples/`;
  let items = $state([]), cur = $state(0);
  onMount(() => {
    fetch(`${base}hero/hero.json`).then((r) => (r.ok ? r.json() : [])).then((v) => (items = v)).catch(() => {});
    const t = setInterval(() => { if (items.length) cur = (cur + 1) % items.length; }, 7000);
    return () => clearInterval(t);
  });
</script>

{#if items.length}
  <figure class="showcase" aria-label="Before and after examples">
    <div class="print back" aria-hidden="true"></div>
    <div class="print">
      {#each items as it, i}
        <div class="pair" class:on={i === cur}>
          <img src={base + it.before} alt="" />
          <img class="after" src={base + it.after} alt="" />
          <span class="sweep" aria-hidden="true"></span>
        </div>
      {/each}
      <span class="tag tag-l sc-before">Before</span>
    </div>
    <div class="dots">{#each items as _, i}<button class:on={i === cur} aria-label={`Example ${i + 1}`} onclick={() => (cur = i)}></button>{/each}</div>
    <figcaption>
      <span class="eyebrow">After</span>
      {#key cur}<em>{items[cur].style.replace(/_/g, " ")}</em>{/key}
    </figcaption>
  </figure>
{/if}
