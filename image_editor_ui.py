import streamlit as st
import torch
import torchvision.transforms as transforms
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import rawpy
from models.style_net import StyleNet
from models.differentiable_renderer import DifferentiableFujifilm

# Page Config
st.set_page_config(page_title="AI Image Editor", layout="wide")

@st.cache_resource
def load_model(style="fujifilm", recipe="classic_chrome"):
    """
    Loads the model and renderer. Cached for performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = StyleNet().to(device)
    checkpoint_path = f"checkpoints/model_{style}_{recipe}.pth"
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        return None, None, f"Checkpoint not found: {checkpoint_path}"
        
    # Load Renderer
    renderer = DifferentiableFujifilm().to(device)
    renderer.eval()
    
    return model, renderer, None

def plot_histogram(image):
    """
    Plots RGB histogram for an image.
    """
    fig, ax = plt.subplots(figsize=(4, 2))
    
    # Check if float or uint8
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
        
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist, bins = np.histogram(image[:, :, i], bins=256, range=(0, 1))
        ax.plot(bins[:-1], hist, color=color, alpha=0.8)
        
    ax.set_xlim(0, 1)
    ax.axis('off') # Clean look
    return fig

# UI Layout
st.title("AI Image Editor: Fujifilm Simulation")
st.markdown("Upload a photo (JPG, PNG, **DNG**) to apply the **Classic Chrome** film recipe.")

# Sidebar
st.sidebar.header("Settings")
style = st.sidebar.selectbox("Style", ["fujifilm"])
recipe = st.sidebar.selectbox("Recipe", ["classic_chrome"])

# Load Model
model, renderer, error = load_model(style, recipe)

if error:
    st.error(error)
    st.stop()

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "dng"])

if uploaded_file is not None:
    # Load Image
    if uploaded_file.name.lower().endswith(".dng"):
        with rawpy.imread(uploaded_file) as raw:
            rgb = raw.postprocess()
        image = Image.fromarray(rgb)
    else:
        image = Image.open(uploaded_file).convert("RGB")
        
    image_np = np.array(image)
    
    # Process
    with st.spinner("Processing..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare Input
        transform_input = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        transform_full = transforms.ToTensor()
        
        input_tensor = transform_input(image).unsqueeze(0).to(device)
        full_tensor = transform_full(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Predict
            params = model(input_tensor)
            
            # Render
            styled_tensor = renderer(full_tensor, params)
            
        # Convert back to numpy
        styled_image = styled_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        styled_image = np.clip(styled_image, 0, 1)
        
    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)
        st.pyplot(plot_histogram(image_np))
        
    with col2:
        st.subheader("Styled (AI Prediction)")
        st.image(styled_image, use_column_width=True)
        st.pyplot(plot_histogram(styled_image))
        
    # Show Parameters
    st.subheader("Predicted Parameters")
    names = ["Highlight Tone", "Shadow Tone", "Saturation", "WB Red", "WB Blue", "Grain", "Vignette"]
    param_vals = params[0].cpu().numpy()
    
    # Create a nice dataframe or metrics
    cols = st.columns(len(names))
    for i, (name, val) in enumerate(zip(names, param_vals)):
        cols[i].metric(name, f"{val:.3f}")

