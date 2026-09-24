<script>
  // A range input with label and value readout. Bipolar ranges fill from the centre;
  // double-click resets to `reset`.
  /** @type {{ id?: string, label: string, value: number, min?: number, max?: number, step?: number,
   *   reset?: number, fmt?: (v: number) => string, disabled?: boolean, title?: string,
   *   oninput: (v: number) => void, onchange?: () => void }} */
  let { id = undefined, label, value, min = -100, max = 100, step = 1, reset = 0, fmt = (v) => String(v),
        disabled = false, title = undefined, oninput, onchange = undefined } = $props();
  const pos = (v) => ((v - min) / (max - min)) * 100;
  const a = $derived(Math.min(pos(value), pos(Math.max(min, Math.min(max, 0)))));
  const b = $derived(Math.max(pos(value), pos(Math.max(min, Math.min(max, 0)))));
</script>

<label class="slider" class:changed={value !== reset} {title}
       ondblclick={() => { oninput(reset); onchange?.(); }}>
  <span class="s-label">{label}</span>
  <output class="s-val">{fmt(value)}</output>
  <input {id} type="range" {min} {max} {step} {value} {disabled} style={`--a:${a}%;--b:${b}%`}
         oninput={(e) => oninput(Number(e.currentTarget.value))} onchange={() => onchange?.()} />
</label>
