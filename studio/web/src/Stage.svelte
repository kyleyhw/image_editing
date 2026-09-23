<script>
  import { onMount } from "svelte";
  import { S, view, initGL, render } from "./lib/studio.svelte.js";

  let canvas, wrap, divider;
  onMount(() => initGL(canvas));

  // Re-render on any change that affects the picture.
  $effect(() => {
    S.params; S.strength; S.mode; S.split; S.holdBefore;
    JSON.stringify(S.d); JSON.stringify(S.knots);
    render();
    placeDivider();
  });

  function placeDivider() {
    if (!divider || !canvas) return;
    const r = canvas.getBoundingClientRect(), w = wrap.getBoundingClientRect();
    divider.style.left = `${r.left - w.left + S.split * r.width}px`;
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
</script>

<svelte:window onresize={placeDivider} />

<div id="canvasWrap" bind:this={wrap}>
  <canvas id="view" bind:this={canvas} aria-label="Photo preview"></canvas>
  <div id="divider" bind:this={divider} role="slider" aria-label="Before/after divider" aria-valuenow={Math.round(S.split * 100)}
       tabindex="0" hidden={S.mode !== "split" || S.holdBefore || !S.params}
       onpointerdown={(e) => { dragging = true; e.currentTarget.setPointerCapture(e.pointerId); }}
       onpointermove={move} onpointerup={() => (dragging = false)} onkeydown={key}></div>
  {#if !S.photos.length}<div id="empty">Upload photos to start. Everything stays on this computer.</div>{/if}
  {#if S.sceneBusy}<div class="busy">Updating scene…</div>{/if}
</div>
<div id="compareBar">
  <span class="muted">Compare:</span>
  {#each ["split", "after", "before"] as m}
    <button data-mode={m} class="seg" class:on={S.mode === m} onclick={() => (S.mode = m)}>{m[0].toUpperCase() + m.slice(1)}</button>
  {/each}
  <span class="muted small">Hold <kbd>\</kbd> for the original · <kbd>[</kbd>/<kbd>]</kbd> strength · <kbd>R</kbd> reset ·
    <kbd>Ctrl+Z</kbd> undo · <kbd>1</kbd>–<kbd>9</kbd> styles</span>
</div>
