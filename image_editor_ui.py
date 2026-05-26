"""Streamlit UI: pick a trained checkpoint and apply it to an uploaded image.

Auto-discovers every checkpoint under ./checkpoints/, loads it through the
unified loader in `models.checkpoint_io`, and renders side-by-side
comparisons with per-channel histograms and the predicted parameter
vector. Works with all three architectures (fujifilm, generic, tilt_shift)
because the heavy lifting is delegated to the checkpoint loader.

Run with:  streamlit run image_editor_ui.py
"""

from __future__ import annotations

import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.checkpoint_io import build_model_from_checkpoint, load_checkpoint


CHECKPOINT_DIR = "checkpoints"
FUJIFILM_PARAM_NAMES = [
    "Highlight", "Shadow", "Saturation", "WB Red", "WB Blue", "Grain", "Vignette",
]

st.set_page_config(page_title="CDF Image Editor", layout="wide")


def list_checkpoints() -> list[str]:
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    return sorted(
        f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth") or f.endswith(".pt")
    )


@st.cache_resource
def load_pipeline(checkpoint_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    ckpt = load_checkpoint(path, map_location=device)
    model, renderer = build_model_from_checkpoint(ckpt)
    model = model.to(device)
    renderer = renderer.to(device)
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError as exc:
        return None, None, None, f"state_dict mismatch in {checkpoint_name}: {exc}"
    model.eval()
    renderer.eval()
    return model, renderer, ckpt, None


def plot_histogram(image: np.ndarray) -> plt.Figure:
    """Per-channel histogram with overlaid CDFs for an [H, W, 3] image in [0, 1]."""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    cdf_ax = ax.twinx()
    colors = ["red", "green", "blue"]
    for i, color in enumerate(colors):
        hist, bins = np.histogram(image[:, :, i], bins=256, range=(0.0, 1.0))
        ax.plot(bins[:-1], hist, color=color, alpha=0.55, linewidth=1.0)
        cdf = hist.cumsum() / max(hist.sum(), 1)
        cdf_ax.plot(bins[:-1], cdf, color=color, alpha=0.95, linewidth=1.2,
                    linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylabel("count", fontsize=8)
    cdf_ax.set_ylabel("CDF", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    cdf_ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout()
    return fig


def display_params(arch: str, params: torch.Tensor):
    vals = params[0].detach().cpu().numpy()
    if arch == "fujifilm":
        st.subheader("Predicted Parameters (Fujifilm-specific, 7-D)")
        cols = st.columns(len(FUJIFILM_PARAM_NAMES))
        for col, name, v in zip(cols, FUJIFILM_PARAM_NAMES, vals):
            col.metric(name, f"{v:+.3f}")
        return

    n = vals.shape[0]
    if arch == "generic" and n == 21:
        st.subheader("Predicted Parameters (generic, 21-D)")
        st.write("Tone-curve interior deltas (7):")
        st.write(np.round(vals[:7], 4).tolist())
        st.write("Colour matrix offsets dM (9, row-major):")
        st.write(np.round(vals[7:16].reshape(3, 3), 4))
        st.write(f"Colour bias: R={vals[16]:+.4f}  G={vals[17]:+.4f}  B={vals[18]:+.4f}")
        c1, c2 = st.columns(2)
        c1.metric("Grain intensity", f"{vals[19]:+.4f}")
        c2.metric("Vignette strength", f"{vals[20]:+.4f}")
        return

    if arch == "tilt_shift" and n == 24:
        st.subheader("Predicted Parameters (generic + tilt-shift, 24-D)")
        st.write("Generic tone-curve interior deltas (7):")
        st.write(np.round(vals[:7], 4).tolist())
        st.write("Colour matrix offsets (3x3):")
        st.write(np.round(vals[7:16].reshape(3, 3), 4))
        st.write(f"Colour bias: R={vals[16]:+.4f}  G={vals[17]:+.4f}  B={vals[18]:+.4f}")
        c1, c2 = st.columns(2)
        c1.metric("Grain", f"{vals[19]:+.4f}")
        c2.metric("Vignette", f"{vals[20]:+.4f}")
        st.write("Tilt-shift focus band:")
        c3, c4, c5 = st.columns(3)
        c3.metric("center_y", f"{vals[21]:.3f}")
        c4.metric("width", f"{vals[22]:.3f}")
        c5.metric("blur strength", f"{vals[23]:.3f}")
        return

    # Fallback for unknown layouts.
    st.subheader(f"Predicted Parameters ({n}-D)")
    st.write(np.round(vals, 4).tolist())


# ---------- UI ----------

st.title("AI Image Editor: CDF + CNN Stylization")
st.markdown(
    "Pick a trained checkpoint, upload an image, and the network predicts "
    "renderer parameters that imitate the style. Histograms and CDFs are "
    "shown alongside so the global tonal/colour shift is visible."
)

available = list_checkpoints()
if not available:
    st.error(
        f"No checkpoints in ./{CHECKPOINT_DIR}/. Train at least one model with "
        "`python train.py --arch fujifilm` (or `--arch generic`, `--arch tilt_shift`)."
    )
    st.stop()

# Sidebar: pick a checkpoint.
st.sidebar.header("Settings")
selected = st.sidebar.selectbox("Checkpoint", available, index=0)

# Allow forcing a cache refresh in case the user retrains.
if st.sidebar.button("Reload checkpoint"):
    load_pipeline.clear()
    st.rerun()

model, renderer, ckpt, error = load_pipeline(selected)
if error:
    st.error(error)
    st.stop()

st.sidebar.write(f"**arch:** `{ckpt.get('arch')}`")
if ckpt.get("style"):
    st.sidebar.write(f"**style:** `{ckpt.get('style')}`")
if ckpt.get("recipe"):
    st.sidebar.write(f"**recipe:** `{ckpt.get('recipe')}`")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Waiting for an upload.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
image_np = np.array(image)

device = next(model.parameters()).device
transform_input = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()]
)
transform_full = transforms.ToTensor()
input_tensor = transform_input(image).unsqueeze(0).to(device)
full_tensor = transform_full(image).unsqueeze(0).to(device)

with st.spinner("Predicting and rendering..."):
    with torch.no_grad():
        params = model(input_tensor)
        styled_tensor = renderer(full_tensor, params)
    styled = np.clip(styled_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy(), 0, 1)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(image, use_container_width=True)
    st.pyplot(plot_histogram(image_np))
with col2:
    st.subheader("Styled (model output)")
    st.image(styled, use_container_width=True)
    st.pyplot(plot_histogram(styled))

display_params(ckpt.get("arch", "fujifilm"), params)

# Allow download.
buf = BytesIO()
Image.fromarray((styled * 255).astype(np.uint8)).save(buf, format="JPEG", quality=95)
st.download_button(
    "Download styled image",
    data=buf.getvalue(),
    file_name=f"styled_{os.path.splitext(uploaded.name)[0]}.jpg",
    mime="image/jpeg",
)
