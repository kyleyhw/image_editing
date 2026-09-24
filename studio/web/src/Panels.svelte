<script>
  import { S } from "./lib/studio.svelte.js";
  import { byId, PANELS } from "./panels/registry.js";
  let { prefs, layout = "stack", onClose } = $props();
  const order = $derived([...prefs.order.filter((id) => byId[id]), ...PANELS.map((p) => p.id).filter((id) => !prefs.order.includes(id))]
    .filter((id) => !prefs.hidden.includes(id)));
  // "tabs": the histogram stays pinned above the tabs
  const tabbed = $derived(order.filter((id) => id !== "histogram"));
  const active = $derived(tabbed.includes(prefs.tab) ? prefs.tab : tabbed[0]);
  // provenance: hollow circle = model prediction, filled square = your edit
  const userEdited = $derived({
    curve: !!S.knots, color: !!(S.d.warmth || S.d.tint || S.d.sat), vignette: !!S.d.vig,
    scene: Object.values(S.scene).some(Boolean),
  });
  function toggle(id) {
    prefs.collapsed = prefs.collapsed.includes(id) ? prefs.collapsed.filter((x) => x !== id) : [...prefs.collapsed, id];
  }
</script>

<aside id="panel" class={`panels-${layout}`} aria-label="Adjustments">
  {#if layout === "sheet"}
    <div class="sheet-head"><span class="grip" aria-hidden="true"></span><strong>Adjustments</strong>
      <button class="btn ghost small" onclick={onClose}>Done</button></div>
  {/if}
  {#if layout === "tabs"}
    {#if order.includes("histogram")}
      {@const H = byId.histogram}<section class="pinned"><H.component /></section>
    {/if}
    <div class="ptabs" role="tablist">
      {#each tabbed as id (id)}
        <button role="tab" class="ptab" class:on={id === active} aria-selected={id === active} onclick={() => (prefs.tab = id)}>
          {byId[id].title}
          {#if byId[id].group && userEdited[byId[id].group]}<span class="dot user"></span>{/if}
        </button>
      {/each}
    </div>
    {@const P = byId[active]}
    <section class="tabbody"><P.component /></section>
  {:else}
    {#each order as id (id)}
      {@const P = byId[id]}
      <section class="card">
        <h3>
          <button class="collapse" aria-expanded={!prefs.collapsed.includes(id)} onclick={() => toggle(id)}>
            <span class="chev">{prefs.collapsed.includes(id) ? "▸" : "▾"}</span> {P.title}
          </button>
          {#if P.group}<span class="dot" class:user={userEdited[P.group]} title="hollow = model, filled = your edit"></span>{/if}
        </h3>
        {#if !prefs.collapsed.includes(id)}<div class="cardbody"><P.component /></div>{/if}
      </section>
    {/each}
  {/if}
</aside>
