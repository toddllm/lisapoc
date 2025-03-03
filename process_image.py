from PIL import Image
import numpy as np
from rembg import remove
import argparse
import os

def process_image(input_path, output_path=None, target_size=(240, 160)):
    """
    Process an image by removing background and resizing.
    
    Args:
        input_path: Path to input image
        output_path: Path for output image (if None, will use input filename with '-processed.png')
        target_size: Tuple of (width, height) for output image
    """
    # Generate output path if not provided
    if output_path is None:
        base_path = os.path.splitext(input_path)[0]
        output_path = f"{base_path}-processed.png"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    print(f"Processing {input_path}...")
    
    # Open image and remove background
    input_img = Image.open(input_path)
    output_img = remove(input_img)
    
    # Convert to RGBA to handle transparency
    img = output_img.convert('RGBA')
    
    # Create transparent background
    background = Image.new('RGBA', target_size, (0, 0, 0, 0))
    
    # Calculate resize dimensions while maintaining aspect ratio
    aspect = img.width / img.height
    if aspect > (target_size[0] / target_size[1]):
        new_width = target_size[0]
        new_height = int(new_width / aspect)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect)
    
    # Resize image
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate position to center image
    x = (target_size[0] - new_width) // 2
    y = (target_size[1] - new_height) // 2
    
    # Paste resized image onto transparent background
    background.paste(img, (x, y), img)
    
    # Save with transparency preserved
    background.save(output_path, 'PNG')
    print(f"Saved processed image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images for avatars')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path (optional)')
    parser.add_argument('-w', '--width', type=int, default=240, help='Target width (default: 240)')
    parser.add_argument('--height', type=int, default=160, help='Target height (default: 160)')
    
    args = parser.parse_args()
    
    process_image(
        args.input,
        args.output,
        target_size=(args.width, args.height)
    ) 