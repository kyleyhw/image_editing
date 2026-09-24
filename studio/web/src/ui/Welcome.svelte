<script>
  import { onMount } from "svelte";
  import { S, addSample } from "../lib/studio.svelte.js";
  import Showcase from "./Showcase.svelte";
  let samples = $state([]);
  onMount(async () => {
    try { samples = await (await fetch(`${import.meta.env.BASE_URL}samples/samples.json`)).json(); } catch { samples = []; }
  });
  const mb = (b) => (b / 1e6).toFixed(1);
</script>

<section class="welcome">
  <div class="welcome-main panel">
    <div class="welcome-text">
      <h1>Open a photo to start</h1>
      <p>Pick a look and it is fitted to your photo: tone, colour and light are set from what is in the frame,
        and every setting stays editable. Export a JPEG or a .cube LUT.</p>
      {#if S.backend === "browser"}<p class="muted">Everything runs in this browser. Photos are not uploaded anywhere.</p>{/if}
      <div class="cta">
        <label for="upload" class="btn primary">Choose a photo…</label>
        <span class="muted">or drop one here</span>
      </div>
      {#if S.backend === "browser"}
        <div class="model">
          {#if S.model.error}<span>Couldn't load the model: {S.model.error}</span>
          {:else if S.model.ready}<span class="ok">Model loaded</span>
          {:else}
            <span>Loading model{S.model.total ? ` · ${mb(S.model.loaded)} of ${mb(S.model.total)} MB` : "…"}</span>
            <span class="bar"><span style={`width:${S.model.total ? (100 * S.model.loaded) / S.model.total : 5}%`}></span></span>
          {/if}
        </div>
      {/if}
    </div>
    <Showcase />
  </div>

  {#if samples.length}
    <div class="samples">
      <div class="side-head"><span>Sample photos</span><span class="muted">public domain, via Openverse</span></div>
      <div class="sample-grid">
        {#each samples as s}
          <button class="sample" onclick={() => addSample(s)} disabled={S.busy} title={`${s.title} by ${s.creator || "unknown"}`}>
            <img src={`${import.meta.env.BASE_URL}samples/${s.thumb}`} alt="" loading="lazy" />
            <span>{s.title}</span>
          </button>
        {/each}
      </div>
    </div>
  {/if}
</section>
