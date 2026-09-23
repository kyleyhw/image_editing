<script>
  import Modal from "./Modal.svelte";
  import IdeaFlow from "./IdeaFlow.svelte";
  import { pollJob } from "../lib/api.js";
  import { S, loadStyles } from "../lib/studio.svelte.js";
  let { open = $bindable(false) } = $props();
  let name = $state(""), mode = $state("paired"), before = $state(null), after = $state(null),
      examples = $state(null), inputs = $state(null), check = $state(""), prog = $state(null);

  const stem = (f) => f.name.replace(/\.[^.]+$/, "");
  const matched = $derived(before && after ? [...after].filter((f) => new Set([...before].map(stem)).has(stem(f))).length : 0);

  async function train() {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) { check = "Name: letters, digits, - or _."; return; }
    const fd = new FormData(); fd.append("name", name); fd.append("mode", mode);
    const add = (field, files) => [...(files || [])].forEach((f) => fd.append(field, f));
    if (mode === "paired") {
      check = `${matched} matched pairs` + (matched < 20 ? " — 20+ recommended." : ".");
      if (matched < 3) return;
      add("before", before); add("after", after);
    } else {
      const n = examples?.length || 0;
      check = `${n} examples` + (n < 20 ? " — 20+ recommended." : ".");
      if (n < 3) return;
      add("examples", examples); add("inputs", inputs);
    }
    prog = { value: 0, max: 1 };
    const { job } = await (await fetch("/api/learn", { method: "POST", body: fd })).json();
    const j = await pollJob(job, (j) => (prog = { value: j.step || 0, max: j.total || 1 }));
    check = j.status === "done" ? `Style "${name}" saved. It is now in the style strip.` : j.error;
    prog = null; await loadStyles();
  }
</script>

<Modal bind:open title="Create a style">
  <fieldset>
    <legend>What do you have?</legend>
    <label><input type="radio" bind:group={mode} value="idea" /> An idea: find openly licensed reference photos for me</label>
    <label><input type="radio" bind:group={mode} value="paired" /> My before/after edits (best)</label>
    <label><input type="radio" bind:group={mode} value="unpaired" /> Photos in a style I like</label>
  </fieldset>
  {#if mode === "idea"}
    <IdeaFlow />
  {:else}
  <label>Name <input bind:value={name} placeholder="my_look" /></label>
  {#if mode === "paired"}
    <label>Before (originals) <input type="file" multiple accept="image/*" onchange={(e) => (before = e.currentTarget.files)} /></label>
    <label>After (your edits, same file names) <input type="file" multiple accept="image/*" onchange={(e) => (after = e.currentTarget.files)} /></label>
    <p class="muted small">{matched} matched pairs. 20+ is enough on the shared base; more is better.</p>
  {:else}
    <label>Example photos in the look <input type="file" multiple accept="image/*" onchange={(e) => (examples = e.currentTarget.files)} /></label>
    <label>Some of your unedited photos (optional) <input type="file" multiple accept="image/*" onchange={(e) => (inputs = e.currentTarget.files)} /></label>
    <p class="muted small">Use photos you own or that are openly licensed. Pairs give clearly better styles.</p>
  {/if}
  <p class="muted small">{check}</p>
  {#if prog}<progress value={prog.value} max={prog.max}></progress>{/if}
  <button class="btn primary" onclick={train} disabled={!!prog}>Train</button>
  {/if}
</Modal>
