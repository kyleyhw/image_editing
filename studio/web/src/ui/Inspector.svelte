<script>
  import { S, commit, resetAll, rememberEdit, personalise, pretty, setStyle, resetGroup, copySettings, pasteSettings, syncAll, savePreset, applyPreset, deletePreset, setAuto } from "../lib/studio.svelte.js";
  import HistoryPanel from "../panels/HistoryPanel.svelte";
  import { describe, effectiveFrom } from "../lib/render.js";
  import { SLIDERS } from "../lib/pipeline.js";
  import Section from "./Section.svelte";
  import Slider from "./Slider.svelte";
  import BasicPanel from "../panels/BasicPanel.svelte";
  import CurvesPanel from "../panels/CurvesPanel.svelte";
  import MixerPanel from "../panels/MixerPanel.svelte";
  import GradingPanel from "../panels/GradingPanel.svelte";
  import BWPanel from "../panels/BWPanel.svelte";
  import CalibrationPanel from "../panels/CalibrationPanel.svelte";
  import VignettePanel from "../panels/VignettePanel.svelte";
  import DetailPanel from "../panels/DetailPanel.svelte";
  import EffectsPanel from "../panels/EffectsPanel.svelte";
  import GeometryPanel from "../panels/GeometryPanel.svelte";
  import MasksPanel from "../panels/MasksPanel.svelte";
  import ScenePanel from "../panels/ScenePanel.svelte";
  import HistogramPanel from "../panels/HistogramPanel.svelte";
  import LookExample from "./LookExample.svelte";
  import { I } from "../lib/icons.js";
  let { open = $bindable(false) } = $props();

  const card = $derived(S.styles.find((s) => s.name === S.style));
  const changes = $derived(S.params ? describe(effectiveFrom(S.params, S.strength), S.params.knots) : []);
  const group = (re) => SLIDERS.filter((s) => re.test(s.name)).map((s) => s.name);
  const edited = (names) => names.some((n) => n in S.p);
  const G = {
    basic: group(/^(temp|tint|exposure|contrast|highlights|shadows|whites|blacks|vibrance|saturation)$/),
    mixer: group(/^hsl_/), grading: group(/^grade_/), bw: group(/^bw/), cal: group(/^cal_/), vig: group(/^vig_/),
    detail: group(/^(texture|clarity|dehaze|sharpen|sharpen_masking|nr_luma|nr_colour)$/), fx: group(/^(grain|grain_size|grain_roughness|halation)$/),
  };
</script>

<aside class="inspector panel" class:open aria-label="Adjustments">
  <div class="sheet-grip"><span>Adjust</span><button class="icon-btn" aria-label="Close adjustments" onclick={() => (open = false)}>{@html I.close}</button></div>
  <div class="hist-wrap"><HistogramPanel /></div>

  <section class="grp">
    <div class="grp-head">Look</div>
    <select class="wide" aria-label="Look" value={S.style || ""} onchange={(e) => {
        const v = e.currentTarget.value;
        if (v.startsWith("preset:")) { applyPreset(S.presets.find((p) => p.name === v.slice(7))); e.currentTarget.value = S.style || ""; }
        else setStyle(v); }}>
      <option value="">None</option>
      {#each S.styles as s (s.name)}<option value={s.name}>{s.title || pretty(s.name)}</option>{/each}
      {#if S.presets.length}<optgroup label="My presets">{#each S.presets as pr (pr.name)}<option value={`preset:${pr.name}`}>{pr.name}</option>{/each}</optgroup>{/if}
    </select>
    {#if card?.description}<p class="desc">{card.description}</p>{/if}
    {#if S.caps.auto}
      <label class="toggle auto-toggle" title="A per-photo correction of white balance, exposure and tone, learned from a professional's edits of 5,000 photos (MIT-Adobe FiveK, expert C). The look is applied on top.">
        <input type="checkbox" checked={S.autoOn} onchange={(e) => setAuto(e.currentTarget.checked)} /> Auto-correct first</label>
      {#if S.autoOn && S.auto}
        <div class="changes" aria-label="Auto correction">
          {#each Object.entries(S.auto).filter(([, v]) => Math.abs(v) >= (Math.abs(v) < 5 ? 0.05 : 1)) as [k, v]}
            <span class="chip">{k === "temp" ? "Temp" : k.charAt(0).toUpperCase() + k.slice(1)} <b>{k === "exposure" ? (v > 0 ? "+" : "") + v.toFixed(2) : (v > 0 ? "+" : "") + Math.round(v)}</b></span>
          {/each}
        </div>
      {/if}
    {/if}
    {#if S.style}
      <Slider id="strength" label="Strength" min={0} max={150} suffix="%" reset={Math.round((S.params?.strength ?? 1) * 100)}
              value={Math.round(S.strength * 100)} disabled={!S.params}
              oninput={(v) => (S.strength = v / 100)} onchange={commit} />
    {/if}
    {#if S.params?.ood_score > 1.5}
      <p class="note">This photo is unlike the ones this look learned from; a lower strength or another look may suit it better.</p>
    {/if}
    {#if changes.length}
      <div class="changes" aria-label="What this look does">
        {#each changes as c}<span class="chip">{c.label} <b>{c.value}</b></span>{/each}
      </div>
    {/if}
    {#if S.message}<p class="hint">{S.message}</p>{/if}
  </section>

  {#if card?.example || card?.tutorials?.length}
    <Section title="Example & sources" open={false}><LookExample look={card} /></Section>
  {/if}
  <Section title="Basic" edited={edited(G.basic)} onreset={() => resetGroup(G.basic)}><BasicPanel /></Section>
  <Section title="Curves" edited={!!S.p.curves} onreset={() => resetGroup(["curves"])}><CurvesPanel /></Section>
  <Section title="Masks" open={false} edited={!!S.p.masks?.length} onreset={() => resetGroup(["masks"])}><MasksPanel /></Section>
  <Section title="Colour mixer" open={false} edited={edited(G.mixer)} onreset={() => resetGroup(G.mixer)}><MixerPanel /></Section>
  <Section title="Colour grading" open={false} edited={edited(G.grading)} onreset={() => resetGroup(G.grading)}><GradingPanel /></Section>
  <Section title="Detail" open={false} edited={edited(G.detail)} onreset={() => resetGroup(G.detail)}><DetailPanel /></Section>
  <Section title="Black & white" open={false} edited={edited(G.bw)} onreset={() => resetGroup(G.bw)}><BWPanel /></Section>
  <Section title="Crop & straighten" open={false} edited={!!S.p.crop} onreset={() => resetGroup(["crop"])}><GeometryPanel /></Section>
  <Section title="Vignette" open={false} edited={edited(G.vig)} onreset={() => resetGroup(G.vig)}><VignettePanel /></Section>
  <Section title="Effects" open={false} edited={edited(G.fx)} onreset={() => resetGroup(G.fx)}><EffectsPanel /></Section>
  <Section title="Calibration" open={false} edited={edited(G.cal)} onreset={() => resetGroup(G.cal)}><CalibrationPanel /></Section>
  {#if S.caps.scene}<Section title="Scene" open={false}><ScenePanel /></Section>{/if}
  <Section title="Snapshots & history" open={false}><HistoryPanel /></Section>

  <footer class="insp-foot">
    <button class="btn" title="Copy settings (Ctrl+Shift+C)" onclick={copySettings} disabled={!S.id}>Copy</button>
    <button class="btn" title="Paste settings (Ctrl+Shift+V)" onclick={pasteSettings} disabled={!S.id || !S.clipboard}>Paste</button>
    {#if S.photos.length > 1}<button class="btn" title="Apply the copied settings to every photo" onclick={syncAll} disabled={!S.clipboard}>Sync all</button>{/if}
    <button class="btn" onclick={() => { const n = prompt("Preset name"); if (n) savePreset(n.trim()); }} disabled={!S.id}>Save preset</button>
    {#if S.presets.length}<button class="btn" title="Delete a preset" onclick={() => { const n = prompt(`Delete which preset? (${S.presets.map((p) => p.name).join(", ")})`); if (n) deletePreset(n.trim()); }}>Delete preset</button>{/if}
    <button id="btnReset" class="btn" onclick={resetAll} disabled={!S.id}>Reset all</button>
    {#if S.caps.remember}
      <button id="btnRemember" class="btn" title="Use this edit to personalise the look" onclick={rememberEdit} disabled={!S.params}>Remember</button>
      <button id="btnPersonalise" class="btn" onclick={personalise} disabled={!S.style}>Personalise</button>
    {/if}
  </footer>
</aside>
