<script>
  import { S, setStyle, commit } from "../lib/studio.svelte.js";
  const pct = $derived(Math.round(S.strength * 100));
  const card = $derived(S.styles.find((s) => s.name === S.style));
</script>

<select id="styleSel" aria-label="Style" value={S.style} onchange={(e) => setStyle(e.currentTarget.value)}>
  {#each S.styles as s (s.name)}<option value={s.name}>{s.name}</option>{/each}
</select>
<label class="row">Strength
  <input id="strength" type="range" min="0" max="150" value={pct}
         oninput={(e) => (S.strength = Number(e.currentTarget.value) / 100)} onchange={commit} />
  <output>{pct}%</output>
</label>
{#if S.params?.ood_score > 1.5}
  <div class="warn">This photo is unlike the photos "{S.params.style}" learned from (score {S.params.ood_score.toFixed(2)}).
    Try a lower strength or another style.</div>
{/if}
<p id="explain" class="muted">{S.message || S.params?.explain || ""}</p>
{#if card?.description}<p class="muted small">{card.description}</p>{/if}
