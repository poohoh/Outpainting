#!/usr/bin/env python3
"""Create a simple test image for progressive outpainting"""
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

def create_test_image(size=(400, 300), output_path=None, pattern="circles"):
    """Create a simple test image with geometric patterns"""
    if output_path is None:
        # Default to test/data/ directory
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / f"test_center_{size[0]}x{size[1]}_{pattern}.png"
    
    # Create RGB image
    img = Image.new('RGB', size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size[0] // 2, size[1] // 2
    
    if pattern == "circles":
        # Draw concentric circles
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
        radii = [min(size) // 4, min(size) // 6, min(size) // 12]
        
        for radius, color in zip(radii, colors):
            draw.ellipse(
                [center_x - radius, center_y - radius, 
                 center_x + radius, center_y + radius],
                fill=color, outline=(0, 0, 0), width=2
            )
    elif pattern == "squares":
        # Draw nested squares
        colors = [(255, 150, 150), (150, 255, 150), (150, 150, 255)]
        sizes = [min(size) // 3, min(size) // 5, min(size) // 10]
        
        for sq_size, color in zip(sizes, colors):
            draw.rectangle(
                [center_x - sq_size//2, center_y - sq_size//2,
                 center_x + sq_size//2, center_y + sq_size//2],
                fill=color, outline=(0, 0, 0), width=2
            )
    elif pattern == "gradient":
        # Create a simple gradient
        for y in range(size[1]):
            color_val = int(255 * y / size[1])
            draw.line([(0, y), (size[0], y)], fill=(color_val, 100, 255-color_val))
    
    # Draw some text
    draw.text((10, 10), "Test Image", fill=(0, 0, 0))
    draw.text((10, size[1] - 30), f"Size: {size[0]}x{size[1]}", fill=(0, 0, 0))
    draw.text((10, size[1] - 50), f"Pattern: {pattern}", fill=(0, 0, 0))
    
    # Save
    img.save(output_path)
    print(f"Created test image: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    create_test_image()