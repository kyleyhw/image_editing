<script>
  import { onMount } from "svelte";
  import { S, view, initGL, render, pretty, drawnP, neutralise, setCrop, commit, effective } from "./lib/studio.svelte.js";
  import { geometryMatrix } from "./lib/pipeline.js";
  import { applyLook } from "./lib/render.js";

  let canvas, wrap, divider, zoomer, cropBox = $state(null);
  onMount(() => initGL(canvas));

  // Re-render on any change that affects the picture.
  $effect(() => {
    S.params; S.strength; S.mode; S.split; S.holdBefore; S.peek; S.clip; S.tool;
    JSON.stringify(S.p);
    render();
    requestAnimationFrame(place);
  });
  $effect(() => { S.zoom; S.pan; requestAnimationFrame(place); });

  /** Fit scale: displayed px per image px when the whole photo fits the stage. */
  function fitScale() {
    const r = wrap.getBoundingClientRect(), pad = 40;
    return Math.min((r.width - pad) / canvas.width, (r.height - pad - 56) / canvas.height, 1);
  }
  function place() {
    if (!canvas || !wrap) return;
    const s = S.zoom ? S.zoom / (window.devicePixelRatio || 1) : fitScale();
    zoomer.style.transform = `translate(${S.pan[0]}px, ${S.pan[1]}px)`;
    canvas.style.width = `${canvas.width * s}px`; canvas.style.height = `${canvas.height * s}px`;
    const r = canvas.getBoundingClientRect(), w = wrap.getBoundingClientRect();
    if (divider) {
      divider.style.left = `${r.left - w.left + S.split * r.width}px`;
      divider.style.top = `${Math.max(0, r.top - w.top)}px`;
      divider.style.height = `${Math.min(r.height, w.height)}px`;
    }
    if (S.tool === "crop" && S.p.crop) {
      const c = S.p.crop;
      cropBox = { left: r.left - w.left + c.x * r.width, top: r.top - w.top + c.y * r.height, width: c.w * r.width, height: c.h * r.height };
    } else cropBox = null;
  }

  // ---------------------------------------------------------------- pointer: split, pan, pick, crop
  let dragging = null;
  const uvOf = (e) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
  function moveSplit(e) {
    if (dragging !== "split") return;
    S.split = Math.min(1, Math.max(0, uvOf(e)[0]));
  }
  function key(e) {
    if (e.key === "ArrowLeft") S.split = Math.max(0, S.split - 0.05);
    if (e.key === "ArrowRight") S.split = Math.min(1, S.split + 0.05);
    e.stopPropagation();
  }

  /** The colour entering the adjustments at a point: the original, through the look if there is one. */
  function sourceColour(u, v) {
    const img = view.img, c = document.createElement("canvas"), W = img.naturalWidth, H = img.naturalHeight;
    const A = geometryMatrix(W, H, drawnP().crop), sx = (A[0][0] * u + A[0][1] * v + A[0][2]) * W, sy = (A[1][0] * u + A[1][1] * v + A[1][2]) * H;
    c.width = 5; c.height = 5;
    const x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(img, sx - 2.5, sy - 2.5, 5, 5, 0, 0, 5, 5);
    const d = x.getImageData(0, 0, 5, 5).data, m = [0, 0, 0];
    for (let i = 0; i < d.length; i += 4) for (let k = 0; k < 3; k++) m[k] += d[i + k] / 255 / 25;
    const e = effective();
    return e.look ? applyLook(e.look, e.K, m) : m;
  }
  function readout(e) {
    const [u, v] = uvOf(e);
    if (u < 0 || v < 0 || u > 1 || v > 1 || !view.main) { S.readout = null; return; }
    const gl = view.main.gl, px = new Uint8Array(4), x = Math.floor(u * canvas.width), y = canvas.height - 1 - Math.floor(v * canvas.height);
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    S.readout = { rgb: [px[0], px[1], px[2]] };
  }

  const pointers = new Map();
  let pinch0 = 0, zoom0 = 1, pan0 = [0, 0], start = [0, 0], crop0 = null, handle = null;
  function down(e) {
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()]; pinch0 = Math.hypot(a[0] - b[0], a[1] - b[1]); zoom0 = currentZoom(); dragging = "pinch"; return;
    }
    const [u, v] = uvOf(e);
    if (S.tool === "wb") { neutralise(sourceColour(u, v)); S.tool = null; return; }
    start = [e.clientX, e.clientY]; pan0 = S.pan.slice();
    dragging = S.zoom ? "pan" : null;
    canvas.setPointerCapture(e.pointerId);
  }
  function move(e) {
    readout(e);
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (dragging === "pinch" && pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      setZoom(Math.max(0.05, Math.min(8, zoom0 * Math.hypot(a[0] - b[0], a[1] - b[1]) / pinch0)));
    } else if (dragging === "pan") S.pan = [pan0[0] + e.clientX - start[0], pan0[1] + e.clientY - start[1]];
  }
  function up(e) { pointers.delete(e.pointerId); if (pointers.size < 2 && dragging === "pinch") dragging = null; if (dragging === "pan") dragging = null; }
  const currentZoom = () => S.zoom || fitScale() * (window.devicePixelRatio || 1);
  function setZoom(z) { S.zoom = Math.abs(z - fitScale() * (window.devicePixelRatio || 1)) < 1e-3 ? 0 : z; if (!S.zoom) S.pan = [0, 0]; }
  function wheel(e) {
    if (!S.id) return;
    e.preventDefault();
    const r = canvas.getBoundingClientRect(), cx = e.clientX - (r.left + r.width / 2), cy = e.clientY - (r.top + r.height / 2);
    const z0 = currentZoom(), z = Math.max(0.05, Math.min(8, z0 * Math.exp(-e.deltaY * 0.0015)));
    S.pan = [S.pan[0] - cx * (z / z0 - 1), S.pan[1] - cy * (z / z0 - 1)];   // keep the point under the cursor fixed
    setZoom(z);
  }
  function dbl() { if (S.tool) return; if (S.zoom) { S.zoom = 0; S.pan = [0, 0]; } else S.zoom = 1; }

  // crop handles: drag the frame (move) or a corner (resize); a fixed aspect is kept
  function cropDown(e, h) {
    e.stopPropagation(); handle = h; start = [e.clientX, e.clientY]; crop0 = { ...S.p.crop };
    e.currentTarget.setPointerCapture(e.pointerId);
  }
  function cropMove(e) {
    if (!handle) return;
    const r = canvas.getBoundingClientRect(), dx = (e.clientX - start[0]) / r.width, dy = (e.clientY - start[1]) / r.height;
    const c = { ...crop0 }, W = canvas.width, H = canvas.height;
    const ratio = c.aspect && c.aspect !== "free" ? (c.w * W) / (c.h * H) : null;
    if (handle === "move") { c.x = Math.min(1 - c.w, Math.max(0, c.x + dx)); c.y = Math.min(1 - c.h, Math.max(0, c.y + dy)); }
    else {
      const left = handle.includes("l"), top = handle.includes("t");
      let x0 = c.x, y0 = c.y, x1 = c.x + c.w, y1 = c.y + c.h;
      if (left) x0 = Math.min(x1 - 0.05, Math.max(0, x0 + dx)); else x1 = Math.max(x0 + 0.05, Math.min(1, x1 + dx));
      if (top) y0 = Math.min(y1 - 0.05, Math.max(0, y0 + dy)); else y1 = Math.max(y0 + 0.05, Math.min(1, y1 + dy));
      if (ratio) {                                   // keep the aspect: height follows width
        const h = ((x1 - x0) * W) / (ratio * H);
        if (top) y0 = Math.max(0, y1 - h); else y1 = Math.min(1, y0 + h);
      }
      Object.assign(c, { x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
    }
    setCrop(c);
  }
  function cropUp() { if (handle) commit(); handle = null; }

  const showing = $derived(S.peek === "__original" || S.holdBefore || S.mode === "before" ? "Original"
    : S.peek ? pretty(S.peek) : null);
</script>

<svelte:window onresize={place} />

<div class="stage" class:has-photo={!!S.id}>
  <div id="canvasWrap" bind:this={wrap} onwheel={wheel} class:picking={S.tool === "wb"} class:zoomed={!!S.zoom}>
    <div class="zoomer" bind:this={zoomer}>
      <canvas id="view" bind:this={canvas} aria-label="Photo preview" class:ready={!!S.id && S.imgVersion > 0}
              onpointerdown={down} onpointermove={move} onpointerup={up} onpointercancel={up}
              onpointerleave={() => (S.readout = null)} ondblclick={dbl}></canvas>
    </div>
    {#if cropBox}
      <div class="cropbox" style={`left:${cropBox.left}px;top:${cropBox.top}px;width:${cropBox.width}px;height:${cropBox.height}px`}
           role="application" aria-label="Crop frame" onpointerdown={(e) => cropDown(e, "move")} onpointermove={cropMove} onpointerup={cropUp}>
        <span class="thirds"></span>
        {#each ["tl", "tr", "bl", "br"] as h}
          <span class={`ch ch-${h}`} role="presentation" onpointerdown={(e) => cropDown(e, h)} onpointermove={cropMove} onpointerup={cropUp}></span>
        {/each}
      </div>
    {/if}
    <div id="divider" bind:this={divider} role="slider" aria-label="Before/after divider" aria-valuenow={Math.round(S.split * 100)}
         tabindex="0" hidden={S.mode !== "split" || S.holdBefore || !S.id || !!S.peek || S.tool === "crop"}
         onpointerdown={(e) => { dragging = "split"; e.currentTarget.setPointerCapture(e.pointerId); }}
         onpointermove={moveSplit} onpointerup={() => (dragging = null)} onkeydown={key}>
      <span class="tag tag-l">Before</span><span class="tag tag-r">After</span>
    </div>
    {#if showing}<div class="peek-tag">{showing}</div>{/if}
    {#if S.tool === "wb"}<div class="peek-tag">Click something that should be neutral grey</div>{/if}
    {#if S.readout}<div class="readout">R {S.readout.rgb[0]} · G {S.readout.rgb[1]} · B {S.readout.rgb[2]}</div>{/if}
    {#if S.id && S.style && !S.params}<div class="thinking"><span></span>Reading the photo…</div>{/if}
    {#if S.sceneBusy}<div class="thinking small"><span></span>Updating scene…</div>{/if}
  </div>
  {#if S.id}
    <div class="compare" role="group" aria-label="Compare">
      {#each [["before", "Before"], ["split", "Split"], ["after", "After"]] as [m, label]}
        <button data-mode={m} class="seg" class:on={S.mode === m} onclick={() => (S.mode = m)}>{label}</button>
      {/each}
      <span class="sep"></span>
      <button class="seg" class:on={!!S.clip} title="Show clipped shadows and highlights (J)" onclick={() => (S.clip = S.clip ? 0 : 1)}>Clipping</button>
      <button class="seg" title="Fit / 100 % (Z, or double-click the photo)" onclick={() => dbl()}>{S.zoom ? `${Math.round(S.zoom * 100)} %` : "Fit"}</button>
    </div>
  {/if}
</div>
