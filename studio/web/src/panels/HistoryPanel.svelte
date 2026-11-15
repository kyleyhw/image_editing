<script>
  // Snapshots (named states of this photo) and the history of edits; click a step to go back to it.
  import { S, historySteps, gotoStep, takeSnapshot, restoreSnapshot, deleteSnapshot } from "../lib/studio.svelte.js";
  const steps = $derived.by(() => { S.undo.length; S.last; return historySteps().slice(-40).reverse(); });
  let name = $state("");
</script>

<div class="sub-head"><span>Snapshots</span></div>
<div class="row-between">
  <input class="text" placeholder="Name (optional)" bind:value={name} aria-label="Snapshot name" />
  <button class="btn small" disabled={!S.id} onclick={() => { takeSnapshot(name); name = ""; }}>Take snapshot</button>
</div>
{#if S.snapshots.length}
  <ul class="mask-list">
    {#each S.snapshots as sn, i}
      <li><button class="mask-chip" onclick={() => restoreSnapshot(sn)}>{sn.name}</button>
        <button class="icon-btn small" aria-label={`Delete ${sn.name}`} onclick={() => deleteSnapshot(i)}>✕</button></li>
    {/each}
  </ul>
{/if}
<div class="sub-head"><span>History</span></div>
<ul class="history">
  {#each steps as st (st.i)}
    <li><button class:on={st.i === S.undo.length} onclick={() => gotoStep(st.i)}>{st.label}</button></li>
  {/each}
</ul>
