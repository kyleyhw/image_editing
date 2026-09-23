// Every adjustment panel, in default order. Add a panel here and it appears in Studio and in
// the Customise dialog (users can reorder, collapse or hide panels; saved per browser).
import StylePanel from "./StylePanel.svelte";
import CurvesPanel from "./CurvesPanel.svelte";
import ColourPanel from "./ColourPanel.svelte";
import ScenePanel from "./ScenePanel.svelte";
import VignettePanel from "./VignettePanel.svelte";
import HistogramPanel from "./HistogramPanel.svelte";
import ActionsPanel from "./ActionsPanel.svelte";

export const PANELS = [
  { id: "style", title: "Style", component: StylePanel },
  { id: "curves", title: "Curves", component: CurvesPanel, group: "curve" },
  { id: "colour", title: "Colour", component: ColourPanel, group: "color" },
  { id: "scene", title: "Scene", component: ScenePanel, group: "scene" },
  { id: "vignette", title: "Vignette", component: VignettePanel, group: "vignette" },
  { id: "histogram", title: "Histogram", component: HistogramPanel },
  { id: "actions", title: "Actions", component: ActionsPanel },
];
export const byId = Object.fromEntries(PANELS.map((p) => [p.id, p]));
