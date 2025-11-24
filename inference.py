import argparse
import os
import torch
import torchvision.transforms as transforms
import skimage as ski
import numpy as np
from models.style_net import StyleNet
from models.differentiable_renderer import DifferentiableFujifilm

def load_image(path, max_size=None):
    image = ski.io.imread(path)
    if image.shape[-1] == 4:
        image = ski.color.rgba2rgb(image)
    
    # Resize if needed (for speed/memory, though model is fully convolutional/global)
    # Actually, StyleNet has a ResNet encoder which expects reasonable sizes.
    # But FeatureExtractor uses AdaptiveAvgPool? No, ResNet features are spatial.
    # Wait, FeatureExtractor flattens spatial features: `x.view(x.size(0), -1)`.
    # This implies a FIXED input size if we use a standard Linear layer after flattening!
    
    # Let's check FeatureExtractor.
    # self.features = nn.Sequential(*list(resnet.children())[:-1]) -> Output is (B, 512, 1, 1) usually?
    # ResNet's avgpool layer (the last one before fc) outputs 1x1 spatial.
    # So yes, it is resolution invariant!
    
    # However, for very large images, we might want to resize for speed.
    return image

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")
    
    # 1. Load Model
    model = StyleNet().to(device)
    checkpoint_path = f"checkpoints/model_{args.style}_{args.recipe}.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. Load Renderer
    renderer = DifferentiableFujifilm().to(device)
    renderer.eval() # Disable noise randomness if implemented
    
    # 3. Load Image
    image = load_image(args.image_path)
    
    # Preprocess
    # We need to keep original resolution for output, but maybe downscale for parameter prediction?
    # The model works on any resolution (global pooling), but let's feed it a standard size for consistency.
    
    transform_input = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    transform_full = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    input_tensor = transform_input(image).unsqueeze(0).to(device) # (1, 3, 256, 256)
    full_tensor = transform_full(image).unsqueeze(0).to(device)   # (1, 3, H, W)
    
    with torch.no_grad():
        # Predict parameters using downscaled image
        params = model(input_tensor)
        
        print("Predicted Parameters:")
        names = ["Highlight", "Shadow", "Saturation", "WB Red", "WB Blue", "Grain", "Vignette"]
        for name, val in zip(names, params[0].cpu().numpy()):
            print(f"  {name}: {val:.4f}")
            
        # Apply to FULL resolution image
        # Note: DifferentiableFujifilm needs to handle arbitrary sizes.
        # Our implementation does (it uses element-wise ops and broadcasting).
        styled_tensor = renderer(full_tensor, params)
        
    # Save
    styled_image = styled_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    styled_image = np.clip(styled_image, 0, 1)
    styled_image = ski.util.img_as_ubyte(styled_image)
    
    output_path = args.output_path
    if not output_path:
        base, ext = os.path.splitext(args.image_path)
        output_path = f"{base}_{args.style}_{args.recipe}{ext}"
        
    ski.io.imsave(output_path, styled_image)
    print(f"Saved styled image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--style", type=str, default="fujifilm")
    parser.add_argument("--recipe", type=str, default="classic_chrome")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save output. Defaults to input_style_recipe.jpg")
    
    args = parser.parse_args()
    inference(args)
