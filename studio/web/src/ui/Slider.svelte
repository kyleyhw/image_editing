<script>
  // Label, range and an editable number box. Bipolar ranges fill from the centre; a `track`
  // gradient (e.g. blue -> yellow for temperature) replaces the fill. Double-click the label or the
  // track to reset to `reset`. Arrow keys move by `step`, Shift + arrow by ten steps.
  /** @type {{ id?: string, label: string, value: number, min?: number, max?: number, step?: number,
   *   reset?: number, suffix?: string, disabled?: boolean, title?: string, track?: string, digits?: number,
   *   oninput: (v: number) => void, onchange?: () => void }} */
  let { id = undefined, label, value, min = -100, max = 100, step = 1, reset = 0, suffix = "",
        disabled = false, title = undefined, track = undefined, digits = undefined, oninput, onchange = undefined } = $props();
  const dp = $derived(digits ?? (step < 1 ? Math.max(0, -Math.floor(Math.log10(step))) : 0));
  const pos = (v) => ((v - min) / (max - min)) * 100;
  const zero = $derived(pos(Math.max(min, Math.min(max, reset))));
  const a = $derived(Math.min(pos(value), zero)), b = $derived(Math.max(pos(value), zero));
  const shown = $derived(`${min < 0 && value > 0 ? "+" : ""}${value.toFixed(dp)}${suffix}`);
  const snap = (v) => Math.max(min, Math.min(max, Math.round(v / step) * step));
  function typed(e) {
    const v = Number(String(e.currentTarget.value).replace(/[^0-9.\-]/g, ""));
    if (Number.isFinite(v)) { oninput(+snap(v).toFixed(dp)); onchange?.(); }
    e.currentTarget.value = shown;
  }
  function key(e) {
    if (!e.shiftKey || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    oninput(+snap(value + (e.key === "ArrowRight" || e.key === "ArrowUp" ? 10 : -10) * step).toFixed(dp)); onchange?.();
  }
</script>

<div class="slider" class:changed={value !== reset} class:custom={!!track} {title}>
  <span class="s-label" role="presentation" ondblclick={() => { oninput(reset); onchange?.(); }}>{label}</span>
  <input {id} type="range" {min} {max} {step} {value} {disabled} aria-label={label}
         style={`--a:${a}%;--b:${b}%${track ? `;--track:${track}` : ""}`}
         ondblclick={() => { oninput(reset); onchange?.(); }} onkeydown={key}
         oninput={(e) => oninput(Number(e.currentTarget.value))} onchange={() => onchange?.()} />
  <input class="num" type="text" inputmode="decimal" value={shown} {disabled} aria-label={`${label} value`}
         onchange={typed} onkeydown={(e) => e.key === "Enter" && e.currentTarget.blur()} />
</div>
