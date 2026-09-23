<script>
  import { S, selectPhoto } from "./lib/studio.svelte.js";
</script>

<aside id="library" aria-label="Library">
  {#each S.photos as p (p.id)}
    <button class="libitem" class:sel={p.id === S.id} onclick={() => selectPhoto(p.id)} aria-label={p.name}>
      <img src={`/api/photo/${p.id}?edge=240`} alt={p.name} loading="lazy" />
    </button>
  {/each}
</aside>
