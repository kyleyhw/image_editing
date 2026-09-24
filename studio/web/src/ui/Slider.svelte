<script>
  // Label, range and an editable number box. Bipolar ranges fill from the centre;
  // double-click the label or track to reset to `reset`.
  /** @type {{ id?: string, label: string, value: number, min?: number, max?: number, step?: number,
   *   reset?: number, suffix?: string, disabled?: boolean, title?: string,
   *   oninput: (v: number) => void, onchange?: () => void }} */
  let { id = undefined, label, value, min = -100, max = 100, step = 1, reset = 0, suffix = "",
        disabled = false, title = undefined, oninput, onchange = undefined } = $props();
  const pos = (v) => ((v - min) / (max - min)) * 100;
  const zero = $derived(pos(Math.max(min, Math.min(max, 0))));
  const a = $derived(Math.min(pos(value), zero)), b = $derived(Math.max(pos(value), zero));
  const shown = $derived(`${min < 0 && value > 0 ? "+" : ""}${value}${suffix}`);
  function typed(e) {
    const v = Number(String(e.currentTarget.value).replace(/[^0-9.\-]/g, ""));
    if (Number.isFinite(v)) { oninput(Math.max(min, Math.min(max, Math.round(v)))); onchange?.(); }
    e.currentTarget.value = shown;
  }
</script>

<div class="slider" class:changed={value !== reset} {title}>
  <span class="s-label" role="presentation" ondblclick={() => { oninput(reset); onchange?.(); }}>{label}</span>
  <input {id} type="range" {min} {max} {step} {value} {disabled} aria-label={label} style={`--a:${a}%;--b:${b}%`}
         ondblclick={() => { oninput(reset); onchange?.(); }}
         oninput={(e) => oninput(Number(e.currentTarget.value))} onchange={() => onchange?.()} />
  <input class="num" type="text" inputmode="decimal" value={shown} {disabled} aria-label={`${label} value`}
         onchange={typed} onkeydown={(e) => e.key === "Enter" && e.currentTarget.blur()} />
</div>
