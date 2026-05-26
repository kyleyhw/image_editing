import os
import argparse
import numpy as np
import skimage as ski
import requests
from io import BytesIO
from data_generation.styles.film import FilmGenerator
from data_generation.styles.fujifilm import FujifilmGenerator
from data_generation.styles.cyberpunk import CyberpunkGenerator

def download_sample_images(output_dir: str, count: int = 100) -> None:
    """Download diverse sample photographs from picsum.photos.

    Lorem Picsum returns a random image per request. Passing ?random=<i>
    forces a distinct image each iteration (the cache otherwise dedups by URL).

    The previous implementation used source.unsplash.com, which Unsplash
    deprecated and disabled in 2023; that endpoint now returns 503 and never
    populates the input directory.
    """
    print(f"Downloading {count} images from picsum.photos...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(count):
        url = f"https://picsum.photos/1600/1200?random={i}"
        try:
            response = requests.get(url, timeout=30, allow_redirects=True)
        except requests.RequestException as exc:
            print(f"Network error on image {i}: {exc}")
            continue

        if response.status_code == 200 and response.content:
            filepath = os.path.join(output_dir, f"picsum_{i:04d}.jpg")
            with open(filepath, "wb") as f:
                f.write(response.content)
            if (i + 1) % 10 == 0:
                print(f"Downloaded {i + 1}/{count} images...")
        else:
            print(f"Unexpected response for image {i}: HTTP {response.status_code}")

    print(f"Completed downloading {count} images to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training dataset.")
    parser.add_argument("--style", type=str, default="film", choices=["film", "fujifilm", "cyberpunk"], help="Style generator to use.")
    parser.add_argument("--recipe", type=str, default="classic_chrome", help="Recipe name (only for fujifilm style).")
    parser.add_argument("--input_dir", type=str, default="images/original", help="Directory containing original images.")
    parser.add_argument("--output_dir", type=str, default="images/styled", help="Directory to save generated pairs.")
    parser.add_argument("--count", type=int, default=5, help="Number of sample images to download if input_dir is empty.")
    
    args = parser.parse_args()
    
    # 1. Setup Directories
    if not os.path.exists(args.input_dir):
        os.makedirs(args.input_dir)
        
    # Check if we have enough images
    existing_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(existing_files) < 10: # If almost empty, download images
        print("Input directory seems empty. Downloading sample images...")
        download_sample_images(args.input_dir, min(args.count, 100))  # Cap at 100 for now
        
    style_output_dir = os.path.join(args.output_dir, args.style)
    if args.style == "fujifilm":
        style_output_dir = os.path.join(style_output_dir, args.recipe)
        
    if not os.path.exists(style_output_dir):
        os.makedirs(style_output_dir)
        
    # 2. Select Generator
    if args.style == "film":
        generator = FilmGenerator()
    elif args.style == "fujifilm":
        generator = FujifilmGenerator(recipe_name=args.recipe)
    elif args.style == "cyberpunk":
        generator = CyberpunkGenerator()
    else:
        raise ValueError(f"Unknown style: {args.style}")
        
    # 3. Process Images
    print(f"Generating {args.style} dataset ({args.recipe if args.style == 'fujifilm' else 'default'})...")
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(args.input_dir, filename)
            
            try:
                image = ski.io.imread(filepath)
                
                # Handle RGBA
                if image.shape[-1] == 4:
                    image = ski.color.rgba2rgb(image)
                
                # Generate Pair
                original, styled = generator.generate_pair(image)
                
                # Save
                # Convert back to uint8 for saving
                original_uint8 = ski.util.img_as_ubyte(original)
                styled_uint8 = ski.util.img_as_ubyte(styled)
                
                basename = os.path.splitext(filename)[0]
                ski.io.imsave(os.path.join(style_output_dir, f"{basename}_original.jpg"), original_uint8)
                ski.io.imsave(os.path.join(style_output_dir, f"{basename}_styled.jpg"), styled_uint8)
                
                print(f"Processed {filename}")
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    main()
