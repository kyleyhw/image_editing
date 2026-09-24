<script>
  import { S, selectPhoto } from "./lib/studio.svelte.js";
  let { onPick } = $props();
</script>

<aside id="library" aria-label="Library">
  {#each S.photos as p (p.id)}
    <button class="libitem" class:sel={p.id === S.id} onclick={() => { selectPhoto(p.id); onPick?.(); }} aria-label={p.name}>
      <img src={`/api/photo/${p.id}?edge=240`} alt={p.name} loading="lazy" />
    </button>
  {/each}
  {#if !S.photos.length}<p class="muted small libempty">No photos yet</p>{/if}
</aside>
