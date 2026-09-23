<script>
  import { S } from "./lib/studio.svelte.js";
  import { byId, PANELS } from "./panels/registry.js";
  let { prefs } = $props();
  const order = $derived([...prefs.order.filter((id) => byId[id]), ...PANELS.map((p) => p.id).filter((id) => !prefs.order.includes(id))]
    .filter((id) => !prefs.hidden.includes(id)));
  // provenance: hollow circle = model prediction, filled square = your edit
  const userEdited = $derived({
    curve: !!S.knots, color: !!(S.d.warmth || S.d.tint || S.d.sat), vignette: !!S.d.vig,
    scene: Object.values(S.scene).some(Boolean),
  });
  function toggle(id) {
    prefs.collapsed = prefs.collapsed.includes(id) ? prefs.collapsed.filter((x) => x !== id) : [...prefs.collapsed, id];
  }
</script>

<aside id="panel" aria-label="Adjustments">
  {#each order as id (id)}
    {@const P = byId[id]}
    <section>
      <h3>
        <button class="collapse" aria-expanded={!prefs.collapsed.includes(id)} onclick={() => toggle(id)}>
          {prefs.collapsed.includes(id) ? "▸" : "▾"} {P.title}
        </button>
        {#if P.group}<span class="dot" class:user={userEdited[P.group]} title="hollow = model, filled = your edit"></span>{/if}
      </h3>
      {#if !prefs.collapsed.includes(id)}<P.component />{/if}
    </section>
  {/each}
</aside>
