#!/usr/bin/env python3
"""
Debug script to verify normalization consistency between 
old dataloader and new canvas initialization
"""

import torch
from PIL import Image
from torchvision import transforms

def test_normalization():
    print("=== Normalization Consistency Test ===")
    
    # Transform used in both dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [-1, 1] range
    ])
    
    # Test 1: Original dataloader's gray background
    print("\n1. Original dataloader (humanart_resize.py):")
    gray_img = Image.new('RGB', (512, 512), color=(128, 128, 128))
    gray_tensor = transform(gray_img)
    print(f"   Gray (128,128,128) → Tensor value: {gray_tensor[0,0,0].item():.6f}")
    
    # Test 2: Current canvas initialization
    print("\n2. Current canvas initialization:")
    zero_tensor = torch.zeros(1, 3, 512, 512)
    print(f"   torch.zeros() → Tensor value: {zero_tensor[0,0,0,0].item():.6f}")
    
    # Test 3: What different RGB values become after normalization
    print("\n3. RGB to normalized values mapping:")
    test_values = [0, 64, 128, 192, 255]
    for rgb_val in test_values:
        test_img = Image.new('RGB', (10, 10), color=(rgb_val, rgb_val, rgb_val))
        normalized = transform(test_img)[0,0,0].item()
        print(f"   RGB({rgb_val},{rgb_val},{rgb_val}) → {normalized:.3f}")
    
    # Test 4: Verify consistency
    print("\n4. Consistency Check:")
    is_consistent = abs(gray_tensor[0,0,0].item() - zero_tensor[0,0,0,0].item()) < 1e-6
    print(f"   Gray background == torch.zeros(): {is_consistent}")
    
    if is_consistent:
        print("   ✅ CONSISTENT: Both use normalized value 0 for masked regions")
    else:
        print("   ❌ INCONSISTENT: Different values for masked regions")
    
    # Test 5: SD 1.5 inpainting perspective
    print("\n5. SD 1.5 Inpainting Model Perspective:")
    print("   - Model expects [-1, 1] range")
    print("   - Neutral value (middle) = 0")
    print("   - Masked regions should be neutral → 0 ✅")
    
    return is_consistent

if __name__ == "__main__":
    test_normalization()