<script>
  import { onMount } from "svelte";
  import { S, initGL, render } from "./lib/studio.svelte.js";

  let canvas, wrap, divider;
  onMount(() => initGL(canvas));

  // Re-render on any change that affects the picture.
  $effect(() => {
    S.params; S.strength; S.mode; S.split; S.holdBefore; S.peek;
    JSON.stringify(S.d); JSON.stringify(S.knots);
    render();
    placeDivider();
  });

  function placeDivider() {
    if (!divider || !canvas) return;
    const r = canvas.getBoundingClientRect(), w = wrap.getBoundingClientRect();
    divider.style.left = `${r.left - w.left + S.split * r.width}px`;
    divider.style.top = `${r.top - w.top}px`;
    divider.style.height = `${r.height}px`;
  }
  let dragging = false;
  function move(e) {
    if (!dragging) return;
    const r = canvas.getBoundingClientRect();
    S.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  }
  function key(e) {
    if (e.key === "ArrowLeft") S.split = Math.max(0, S.split - 0.05);
    if (e.key === "ArrowRight") S.split = Math.min(1, S.split + 0.05);
    e.stopPropagation();
  }
  const showing = $derived(S.peek === "__original" || S.holdBefore || S.mode === "before" ? "Original"
    : S.peek ? S.peek.replace(/_/g, " ") : null);
</script>

<svelte:window onresize={placeDivider} />

<div class="stage" class:has-photo={!!S.id}>
  <div id="canvasWrap" bind:this={wrap}>
    <canvas id="view" bind:this={canvas} aria-label="Photo preview" class:ready={!!S.params}></canvas>
    <div id="divider" bind:this={divider} role="slider" aria-label="Before/after divider" aria-valuenow={Math.round(S.split * 100)}
         tabindex="0" hidden={S.mode !== "split" || S.holdBefore || !S.params || !!S.peek}
         onpointerdown={(e) => { dragging = true; e.currentTarget.setPointerCapture(e.pointerId); }}
         onpointermove={move} onpointerup={() => (dragging = false)} onkeydown={key}>
      <span class="tag tag-l">Before</span><span class="tag tag-r">After</span>
    </div>
    {#if showing}<div class="peek-tag">{showing}</div>{/if}
    {#if S.id && !S.params}<div class="thinking"><span></span>Reading the photo…</div>{/if}
    {#if S.sceneBusy}<div class="thinking small"><span></span>Updating scene…</div>{/if}
  </div>
  {#if S.id}
    <div class="compare" role="group" aria-label="Compare">
      {#each [["before", "Before"], ["split", "Split"], ["after", "After"]] as [m, label]}
        <button data-mode={m} class="seg" class:on={S.mode === m} onclick={() => (S.mode = m)}>{label}</button>
      {/each}
      <span class="compare-hint">hold <kbd>\</kbd></span>
    </div>
  {/if}
</div>
