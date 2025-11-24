import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import skimage as ski
import numpy as np
from models.style_net import StyleNet
from data_generation.styles.fujifilm import FujifilmGenerator

class ImagePairDataset(Dataset):
    """
    Dataset for loading (Original, Styled) image pairs.
    """
    def __init__(self, root_dir, style="fujifilm", recipe="classic_chrome", transform=None):
        self.root_dir = os.path.join(root_dir, style, recipe)
        self.transform = transform
        self.files = [f for f in os.listdir(self.root_dir) if f.endswith("_original.jpg")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        original_name = self.files[idx]
        styled_name = original_name.replace("_original.jpg", "_styled.jpg")
        
        original_path = os.path.join(self.root_dir, original_name)
        styled_path = os.path.join(self.root_dir, styled_name)
        
        original = ski.io.imread(original_path)
        styled = ski.io.imread(styled_path)
        
        if self.transform:
            original = self.transform(original)
            styled = self.transform(styled)
            
        return original, styled

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # 1. Dataset & DataLoader
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), # Resize for faster training
        transforms.ToTensor()
    ])
    
    dataset = ImagePairDataset(root_dir=args.data_dir, style=args.style, recipe=args.recipe, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 2. Model
    model = StyleNet().to(device)
    
    # 3. Loss & Optimizer
    # Differentiable Renderer to apply predicted parameters
    from models.differentiable_renderer import DifferentiableFujifilm
    renderer = DifferentiableFujifilm().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0
        for i, (original, target) in enumerate(dataloader):
            original = original.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Predict parameters
            params = model(original)
            
            # Apply parameters (Differentiable Simulation)
            rendered = renderer(original, params)
            
            # Calculate Loss: Compare Rendered vs Target
            loss = criterion(rendered, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dataloader)}")
        
    # Save model
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), f"checkpoints/model_{args.style}_{args.recipe}.pth")
    print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="images/styled")
    parser.add_argument("--style", type=str, default="fujifilm")
    parser.add_argument("--recipe", type=str, default="classic_chrome")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)
