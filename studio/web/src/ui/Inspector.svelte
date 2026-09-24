<script>
  import { S, commit, resetToModel, remember, personalise, pretty } from "../lib/studio.svelte.js";
  import { describe, effectiveFrom } from "../lib/render.js";
  import Section from "./Section.svelte";
  import CurvesPanel from "../panels/CurvesPanel.svelte";
  import ColourPanel from "../panels/ColourPanel.svelte";
  import VignettePanel from "../panels/VignettePanel.svelte";
  import ScenePanel from "../panels/ScenePanel.svelte";
  import HistogramPanel from "../panels/HistogramPanel.svelte";
  import { I } from "../lib/icons.js";
  let { open = $bindable(false) } = $props();

  const pct = $derived(Math.round(S.strength * 100));
  const card = $derived(S.styles.find((s) => s.name === S.style));
  const changes = $derived(S.params ? describe(effectiveFrom(S.params, S.strength, S.d, S.knots), S.params.knots) : []);
  const pos = $derived((pct / 150) * 100);
</script>

<aside class="inspector" class:open aria-label="Adjustments">
  <div class="sheet-grip"><button class="icon-btn" aria-label="Close adjustments" onclick={() => (open = false)}>{@html I.close}</button></div>
  <section class="lookcard">
    <span class="eyebrow">Look</span>
    <h2 class="lookcard-name">{S.style ? pretty(S.style) : "—"}</h2>
    {#if card?.description}<p class="lookcard-desc">{card.description}</p>{/if}

    <label class="strength">
      <span class="strength-top"><span class="s-label">Strength</span>
        <output class="bignum">{pct}<small>%</small></output></span>
      <input id="strength" type="range" min="0" max="150" value={pct} disabled={!S.params} style={`--a:0%;--b:${pos}%`}
             oninput={(e) => (S.strength = Number(e.currentTarget.value) / 100)} onchange={commit} />
      <span class="ticks" aria-hidden="true"><span>0</span><span style="left:66.67%">100</span><span>150</span></span>
    </label>

    {#if S.params?.ood_score > 1.5}
      <p class="note">This photo is unlike the ones this look learned from. A lower strength or another look may suit it better.</p>
    {/if}
    {#if changes.length}
      <ul class="changes" aria-label="What this edit does">
        {#each changes as c}<li><span>{c.label}</span><b>{c.value}</b></li>{/each}
      </ul>
    {/if}
    {#if S.message}<p class="hint">{S.message}</p>{/if}
  </section>

  <div class="secs">
    <Section title="Tone"><HistogramPanel /><CurvesPanel /></Section>
    <Section title="Colour"><ColourPanel /></Section>
    <Section title="Light"><VignettePanel /></Section>
    {#if S.caps.scene}<Section title="Scene" open={false}><ScenePanel /></Section>{/if}
  </div>

  <footer class="insp-foot">
    <button id="btnReset" class="btn ghost" onclick={resetToModel} disabled={!S.params}>{@html I.reset}Reset</button>
    {#if S.caps.remember}
      <button id="btnRemember" class="btn ghost" title="Use this edit to personalise the look" onclick={remember} disabled={!S.params}>Remember</button>
      <button id="btnPersonalise" class="btn ghost" onclick={personalise} disabled={!S.style}>Personalise</button>
    {/if}
  </footer>
</aside>
