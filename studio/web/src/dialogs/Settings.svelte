<script>
  import Modal from "./Modal.svelte";
  import { PANELS, byId } from "../panels/registry.js";
  let { open = $bindable(false), prefs } = $props();
  const order = $derived([...prefs.order.filter((id) => byId[id]), ...PANELS.map((p) => p.id).filter((id) => !prefs.order.includes(id))]);
  function move(id, d) {
    const o = [...order], i = o.indexOf(id), j = i + d;
    if (j < 0 || j >= o.length) return;
    [o[i], o[j]] = [o[j], o[i]]; prefs.order = o;
  }
  function toggleHidden(id) { prefs.hidden = prefs.hidden.includes(id) ? prefs.hidden.filter((x) => x !== id) : [...prefs.hidden, id]; }
  function reset() { prefs.order = PANELS.map((p) => p.id); prefs.hidden = []; prefs.collapsed = []; }
</script>

<Modal bind:open title="Customise panels">
  <p class="muted small">Order and visibility of the adjustment panels. Saved in this browser.</p>
  <ul class="plist">
    {#each order as id, i (id)}
      <li>
        <label><input type="checkbox" checked={!prefs.hidden.includes(id)} onchange={() => toggleHidden(id)} /> {byId[id].title}</label>
        <span class="spacer"></span>
        <button class="btn small" aria-label={`Move ${byId[id].title} up`} onclick={() => move(id, -1)} disabled={i === 0}>↑</button>
        <button class="btn small" aria-label={`Move ${byId[id].title} down`} onclick={() => move(id, 1)} disabled={i === order.length - 1}>↓</button>
      </li>
    {/each}
  </ul>
  {#snippet footer()}<button class="btn" onclick={reset}>Reset layout</button>{/snippet}
</Modal>
