<script>
  // Create a style from an idea: describe -> search openly licensed photos -> pick the ones whose
  // look you like -> review "more like these" -> train. Mirrors `photostyle style ...`.
  import { json, pollJob } from "../lib/api.js";
  import { loadStyles, setStyle } from "../lib/studio.svelte.js";

  let step = $state("describe");          // describe | searching | pick | refs | training | done
  let name = $state(""), describe = $state(""), queries = $state(""), mono = $state(false);
  let project = $state(null), picked = $state(new Set()), dropped = $state(new Set());
  let recipe = $state("gentle"), prog = $state({ value: 0, max: 1, text: "" }), error = $state("");

  const img = (n, edge = 200) => `/api/style/${project.name}/image/${n}?edge=${edge}`;

  async function start() {
    error = "";
    if (!/^[A-Za-z0-9_]{1,64}$/.test(name)) { error = "Name: letters, digits or _."; return; }
    if (!describe.trim()) { error = "Describe the look in a few words."; return; }
    try {
      project = await json("/api/style/new", { name, describe, mono, queries: queries.split("\n") });
      step = "searching";
      const { job } = await json("/api/style/search", { name: project.name, pages: 2 });
      const j = await pollJob(job, (j) => (prog = { value: j.step || 0, max: j.total || 1, text: `${j.loss ?? 0} photos found` }));
      if (j.status !== "done") throw new Error(j.error);
      project = await json(`/api/style/${project.name}`);
      picked = new Set(); step = "pick";
    } catch (e) { error = String(e); step = "describe"; }
  }
  function toggle(set, n) { const s = new Set(set); s.has(n) ? s.delete(n) : s.add(n); return s; }
  async function findMore() {
    project = await json("/api/style/pick", { name: project.name, numbers: [...picked] });
    dropped = new Set(); step = "refs";
  }
  async function train() {
    if (dropped.size) project = await json("/api/style/exclude", { name: project.name, numbers: [...dropped] });
    step = "training"; error = "";
    const { job } = await json("/api/style/train", { name: project.name, recipe });
    const j = await pollJob(job, (j) => (prog = { value: j.step || 0, max: j.total || 1, text: `step ${j.step || 0}/${j.total || "?"}` }));
    if (j.status !== "done") { error = j.error; step = "refs"; return; }
    await loadStyles(); await setStyle(project.name); step = "done";
  }
</script>

{#if step === "describe"}
  <label>Name <input bind:value={name} placeholder="cyberpunk_night" /></label>
  <label>The look, in words <input bind:value={describe} placeholder="neon cyberpunk night city, magenta and cyan" /></label>
  <label>Search queries (optional, one per line)
    <textarea rows="3" bind:value={queries} placeholder="neon city night rain&#10;cyberpunk street"></textarea></label>
  <label><input type="checkbox" bind:checked={mono} /> Black and white</label>
  <p class="muted small">Only openly licensed photos (CC0, public domain, CC BY, CC BY-SA) are searched; credits are kept with the style.
    Searching takes a few minutes.</p>
  <button class="btn primary" onclick={start}>Search for reference photos</button>
{:else if step === "searching" || step === "training"}
  <p>{step === "searching" ? "Searching and filtering photos…" : "Training the style…"} <span class="muted">{prog.text}</span></p>
  <progress value={prog.value} max={prog.max}></progress>
{:else if step === "pick"}
  <p>Click the photos whose <b>look</b> you like (subject doesn't matter). {picked.size} picked.</p>
  <div class="pickgrid">
    {#each project.candidates as c (c.n)}
      <button class="pick" class:on={picked.has(c.n)} onclick={() => (picked = toggle(picked, c.n))}
              title={`${c.title || ""} — ${c.creator || "unknown"} (${c.license})`}>
        <img src={img(c.n)} alt={c.title || `candidate ${c.n}`} loading="lazy" /><span>{c.n}</span>
      </button>
    {/each}
  </div>
  <button class="btn primary" onclick={findMore} disabled={!picked.size}>Find more like these</button>
{:else if step === "refs"}
  <p>{project.refs.length} reference photos: your picks plus the closest matches. Click any you don't want.</p>
  <div class="pickgrid">
    {#each project.refs as n (n)}
      <button class="pick" class:off={dropped.has(n)} onclick={() => (dropped = toggle(dropped, n))}>
        <img src={img(n)} alt={`reference ${n}`} loading="lazy" /><span>{n}</span>
      </button>
    {/each}
  </div>
  <fieldset>
    <legend>Training</legend>
    <label><input type="radio" bind:group={recipe} value="gentle" /> Gentle: looks close to natural photos (~5 min)</label>
    <label><input type="radio" bind:group={recipe} value="strong" /> Strong: looks far from natural, e.g. cyberpunk (~15 min)</label>
    <label><input type="radio" bind:group={recipe} value="instant" /> Instant: no training, via the shared base (needs it built)</label>
  </fieldset>
  <button class="btn primary" onclick={train}>Train</button>
{:else if step === "done"}
  <p>Style <b>{project.name}</b> is ready and selected. Compare it on your photos; set the strength you like.</p>
{/if}
{#if error}<p class="warn">{error}</p>{/if}
