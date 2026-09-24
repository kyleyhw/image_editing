<script>
  import { onMount } from "svelte";
  import { S, addSample } from "../lib/studio.svelte.js";
  import { I } from "../lib/icons.js";
  import Showcase from "./Showcase.svelte";
  let samples = $state([]);
  onMount(async () => {
    try { samples = await (await fetch(`${import.meta.env.BASE_URL}samples/samples.json`)).json(); } catch { samples = []; }
  });
  const mb = (b) => (b / 1e6).toFixed(1);
  const credits = $derived(samples.map((s) => `${s.title} by ${s.creator || "unknown"}`).join(" · "));
</script>

<section class="welcome">
  <div class="welcome-top">
  <div class="hero">
    <p class="eyebrow rise" style="--d:0">Colour grading that reads the photo</p>
    <h1 class="rise" style="--d:1">Give every photo<br /><em>the look</em> you meant.</h1>
    <p class="lede rise" style="--d:2">
      photostyle learns a look from a handful of reference photos, then adapts it to each picture:
      tone, colour and light, measured from what is actually in the frame. Everything stays editable.
      {#if S.backend === "browser"}<strong>It runs entirely in your browser; your photos never leave this device.</strong>{/if}
    </p>
    <div class="cta rise" style="--d:3">
      <label for="upload" class="btn primary big">{@html I.image}Open a photo</label>
      <span class="or">or drop one anywhere</span>
    </div>
    {#if S.backend === "browser"}
      <div class="model rise" style="--d:4" class:done={S.model.ready}>
        {#if S.model.error}
          <span class="model-label">Couldn't load the vision model: {S.model.error}</span>
        {:else if S.model.ready}
          <span class="dot"></span><span class="model-label">Vision model ready</span>
        {:else}
          <span class="model-label">Loading the vision model {S.model.total ? `${mb(S.model.loaded)} / ${mb(S.model.total)} MB` : "…"}</span>
          <span class="bar"><span style={`width:${S.model.total ? (100 * S.model.loaded) / S.model.total : 8}%`}></span></span>
        {/if}
      </div>
    {/if}
  </div>
  <Showcase />
  </div>

  {#if samples.length}
    <div class="samples">
      <p class="eyebrow">Or start with one of these</p>
      <div class="sample-grid">
        {#each samples as s, i}
          <button class="sample" style={`--i:${i}`} onclick={() => addSample(s)} disabled={S.busy} aria-label={`Try ${s.title}`}>
            <img src={`${import.meta.env.BASE_URL}samples/${s.thumb}`} alt="" loading="lazy" />
          </button>
        {/each}
      </div>
      <p class="credit" title={credits}>Public-domain photos via Openverse. {credits}.</p>
    </div>
  {/if}

  <ol class="how">
    <li><b>01</b><span>Pick reference photos that have the look you want.</span></li>
    <li><b>02</b><span>photostyle learns how that look responds to light, colour and scene.</span></li>
    <li><b>03</b><span>Apply it to your photo, then refine strength, curves and colour.</span></li>
  </ol>
</section>
