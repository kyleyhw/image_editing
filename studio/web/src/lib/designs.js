// Studio designs (owner picks one; the others stay available in ⚙ Customise).
// strip: "cards" (horizontal cards) | "list" (vertical list with descriptions) | "dots" (round, in the dock)
// panels: "stack" (collapsible sections) | "tabs" (tabbed inspector, histogram pinned) | "sheet" (slide-up)
export const DESIGNS = [
  { id: "darkroom", name: "Darkroom", blurb: "Refined dark editor: big photo, slim filmstrip, card inspector, large style cards.",
    strip: "cards", panels: "stack" },
  { id: "gallery", name: "Gallery", blurb: "Light and editorial: the photo as a matted print, styles as descriptive cards.",
    strip: "list", panels: "stack" },
  { id: "pro", name: "Pro", blurb: "Dense, Lightroom-like: presets list, bottom filmstrip, tabbed inspector.",
    strip: "list", panels: "tabs" },
  { id: "focus", name: "Focus", blurb: "Minimal and touch-friendly: full-bleed photo, a dock, adjustments in a sheet.",
    strip: "dots", panels: "sheet", dock: true },
];
