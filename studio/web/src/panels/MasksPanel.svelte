<script>
  // Local adjustments (Phase 23): up to four masks, each a shape (linear gradient, radial, brush or
  // the whole photo), optionally limited to a luminance or colour range, with its own sliders.
  import { S, addMask, addLearnedMask, updateMask, setMaskSlider, removeMask, commit } from "../lib/studio.svelte.js";
  import { SLIDERS, SPEC } from "../lib/pipeline.js";
  import Slider from "../ui/Slider.svelte";
  const NAMES = { linear: "Linear gradient", radial: "Radial", brush: "Brush", all: "Range" };
  const sel = $derived(S.p.masks?.[S.maskSel]);
  const spec = (n) => SLIDERS.find((s) => s.name === n);
  const label = (m, i) => `${m.label ?? (m.range?.hue != null && m.type === "all" ? "Colour range" : m.range?.lum && m.type === "all" ? "Luminance range" : NAMES[m.type])} ${i + 1}`;
  const setRange = (patch) => { updateMask(S.maskSel, { range: { ...(sel.range || {}), ...patch } }); };
</script>

<div class="mask-add">
  {#each [["linear", "Linear"], ["radial", "Radial"], ["brush", "Brush"]] as [t, l]}
    <button class="btn small" disabled={!S.id || (S.p.masks?.length ?? 0) >= SPEC.masks.max} onclick={() => addMask(t)}>+ {l}</button>
  {/each}
  <button class="btn small" disabled={!S.id || (S.p.masks?.length ?? 0) >= SPEC.masks.max} onclick={() => addMask("all", { lum: [0.6, 1], feather: 10 })}>+ Luminance</button>
  <button class="btn small" disabled={!S.id || (S.p.masks?.length ?? 0) >= SPEC.masks.max} onclick={() => { addMask("all", { hue: 0, hue_width: 30 }); S.tool = "pickhue"; }}>+ Colour</button>
  {#if S.caps.seg}
    {#each [["sky", "Sky"], ["subject", "Subject"]] as [k, l]}
      <button class="btn small" disabled={!S.id || S.seg?.loading || (S.p.masks?.length ?? 0) >= SPEC.masks.max}
        title={`Select the ${l.toLowerCase()} with a segmentation model (SegFormer-B0, ADE20K; 15 MB, downloaded once, runs on this device)`}
        onclick={() => addLearnedMask(k)}>+ {l}</button>
    {/each}
    {#if S.seg?.loading}<span class="hint">Finding… {S.seg.progress ? Math.round(100 * S.seg.progress) + " %" : ""}</span>{/if}
  {/if}
</div>
{#if S.p.masks?.length}
  <ul class="mask-list">
    {#each S.p.masks as m, i (m.id)}
      <li><button class="mask-chip" class:on={i === S.maskSel} onclick={() => { S.maskSel = i; S.tool = m.type === "all" ? null : "mask"; }}>{label(m, i)}</button>
        <button class="icon-btn small" aria-label={`Delete ${label(m, i)}`} onclick={() => removeMask(i)}>✕</button></li>
    {/each}
  </ul>
{/if}
{#if sel}
  <div class="row-between">
    <label class="toggle"><input type="checkbox" checked={sel.invert} onchange={(e) => { updateMask(S.maskSel, { invert: e.currentTarget.checked }); commit(); }} /> Invert</label>
    <label class="toggle"><input type="checkbox" bind:checked={S.showMask} /> Show mask (O)</label>
  </div>
  {#if sel.type === "radial"}
    <Slider label="Feather" min={0} max={100} value={sel.feather ?? 50} oninput={(v) => updateMask(S.maskSel, { feather: v })} onchange={commit} />
  {/if}
  {#if sel.type === "brush"}
    <div class="sub-head"><span>Brush</span>
      <label class="toggle small"><input type="checkbox" bind:checked={S.brush.erase} /> Erase</label></div>
    <Slider label="Size" min={1} max={30} value={S.brush.size} oninput={(v) => (S.brush.size = v)} />
    <Slider label="Hardness" min={0} max={100} value={S.brush.hardness} oninput={(v) => (S.brush.hardness = v)} />
    <Slider label="Flow" min={5} max={100} value={S.brush.flow} oninput={(v) => (S.brush.flow = v)} />
    <p class="hint">Paint on the photo. Size is a percentage of the long edge.</p>
  {/if}
  {#if sel.range?.lum}
    <div class="sub-head"><span>Luminance range</span></div>
    <Slider label="From" min={0} max={100} value={Math.round(sel.range.lum[0] * 100)} oninput={(v) => setRange({ lum: [v / 100, sel.range.lum[1]] })} onchange={commit} />
    <Slider label="To" min={0} max={100} value={Math.round(sel.range.lum[1] * 100)} oninput={(v) => setRange({ lum: [sel.range.lum[0], v / 100] })} onchange={commit} />
    <Slider label="Smoothness" min={1} max={50} value={sel.range.feather ?? 10} oninput={(v) => setRange({ feather: v })} onchange={commit} />
  {/if}
  {#if sel.range?.hue != null}
    <div class="sub-head"><span>Colour range</span>
      <button class="link-btn" class:on={S.tool === "pickhue"} onclick={() => (S.tool = S.tool === "pickhue" ? null : "pickhue")}>Pick colour</button></div>
    <Slider label="Width" min={5} max={90} value={sel.range.hue_width ?? 30} suffix="°" oninput={(v) => setRange({ hue_width: v })} onchange={commit} />
  {/if}
  {#if sel.type !== "all"}
    <label class="toggle small"><input type="checkbox" checked={!!sel.range?.lum} onchange={(e) => { const { lum: _l, feather: _f, ...r } = sel.range || {}; updateMask(S.maskSel, { range: e.currentTarget.checked ? { ...r, lum: [0.5, 1], feather: 10 } : r }); commit(); }} /> Limit to a luminance range</label>
  {/if}
  <div class="sub-head"><span>Adjust inside the mask</span></div>
  <div class="stack">
    {#each SPEC.masks.local as n (n)}
      {@const s = spec(n)}
      <Slider label={s.label} min={s.min} max={s.max} step={s.step} value={sel.sliders?.[n] ?? 0}
              oninput={(v) => setMaskSlider(S.maskSel, n, v)} onchange={commit} />
    {/each}
  </div>
{:else if !S.p.masks?.length}
  <p class="hint">Add a mask to adjust part of the photo: a gradient for a sky, a radial for a face, a brush for anything else,
    or a luminance or colour range.</p>
{/if}
