<script>
  import { S, commit, resetToModel, remember, personalise, pretty, setStyle } from "../lib/studio.svelte.js";
  import { describe, effectiveFrom } from "../lib/render.js";
  import Section from "./Section.svelte";
  import Slider from "./Slider.svelte";
  import CurvesPanel from "../panels/CurvesPanel.svelte";
  import ColourPanel from "../panels/ColourPanel.svelte";
  import VignettePanel from "../panels/VignettePanel.svelte";
  import ScenePanel from "../panels/ScenePanel.svelte";
  import HistogramPanel from "../panels/HistogramPanel.svelte";
  import LookExample from "./LookExample.svelte";
  import { I } from "../lib/icons.js";
  let { open = $bindable(false) } = $props();

  const card = $derived(S.styles.find((s) => s.name === S.style));
  const changes = $derived(S.params ? describe(effectiveFrom(S.params, S.strength, S.d, S.knots), S.params.knots) : []);
</script>

<aside class="inspector panel" class:open aria-label="Adjustments">
  <div class="sheet-grip"><span>Adjust</span><button class="icon-btn" aria-label="Close adjustments" onclick={() => (open = false)}>{@html I.close}</button></div>
  <div class="hist-wrap"><HistogramPanel /></div>

  <section class="grp">
    <div class="grp-head">Look</div>
    <select class="wide" aria-label="Look" value={S.style} onchange={(e) => setStyle(e.currentTarget.value)}>
      {#each S.styles as s (s.name)}<option value={s.name}>{s.title || pretty(s.name)}</option>{/each}
    </select>
    {#if card?.description}<p class="desc">{card.description}</p>{/if}
    <Slider id="strength" label="Strength" min={0} max={150} suffix="%" reset={Math.round((S.params?.strength ?? 1) * 100)}
            value={Math.round(S.strength * 100)} disabled={!S.params}
            oninput={(v) => (S.strength = v / 100)} onchange={commit} />
    {#if S.params?.ood_score > 1.5}
      <p class="note">This photo is unlike the ones this look learned from; a lower strength or another look may suit it better.</p>
    {/if}
    {#if changes.length}
      <div class="changes" aria-label="What this edit does">
        {#each changes as c}<span class="chip">{c.label} <b>{c.value}</b></span>{/each}
      </div>
    {/if}
    {#if S.message}<p class="hint">{S.message}</p>{/if}
  </section>

  {#if card?.example || card?.tutorials?.length}
    <Section title="Example & sources" open={false}><LookExample look={card} /></Section>
  {/if}
  <Section title="Curves"><CurvesPanel /></Section>
  <Section title="Colour"><ColourPanel /></Section>
  <Section title="Light"><VignettePanel /></Section>
  {#if S.caps.scene}<Section title="Scene" open={false}><ScenePanel /></Section>{/if}

  <footer class="insp-foot">
    <button id="btnReset" class="btn" onclick={resetToModel} disabled={!S.params}>Reset</button>
    {#if S.caps.remember}
      <button id="btnRemember" class="btn" title="Use this edit to personalise the look" onclick={remember} disabled={!S.params}>Remember</button>
      <button id="btnPersonalise" class="btn" onclick={personalise} disabled={!S.style}>Personalise</button>
    {/if}
  </footer>
</aside>
