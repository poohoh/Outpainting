#!/usr/bin/env python3
"""Test the core progressive logic without SD model"""
import sys
import os
from pathlib import Path
sys.path.append('/app')

from inference_outpainting_progressive_overlap import (
    load_rgb, compute_stride_px, plan_windows_with_center,
    PriorityComposer, build_local_input
)
import numpy as np
from PIL import Image

def test_overlap_calculation():
    """Test overlap calculation logic"""
    print("=== Testing Overlap Calculation ===")
    
    # Test case 1: 25% overlap
    patch = 512
    overlap_ratio = 0.25
    ovx, sx = compute_stride_px(patch, overlap_ratio)
    print(f"Patch={patch}, Overlap ratio={overlap_ratio}")
    print(f"Overlap pixels={ovx}, Stride={sx}")
    assert ovx == 128, f"Expected 128, got {ovx}"
    assert sx == 384, f"Expected 384, got {sx}"
    print("✓ Test passed")
    
    # Test case 2: 50% overlap  
    overlap_ratio = 0.5
    ovx, sx = compute_stride_px(patch, overlap_ratio)
    print(f"Patch={patch}, Overlap ratio={overlap_ratio}")
    print(f"Overlap pixels={ovx}, Stride={sx}")
    assert ovx == 256, f"Expected 256, got {ovx}"
    assert sx == 256, f"Expected 256, got {sx}"
    print("✓ Test passed")


def test_window_planning():
    """Test window planning with arbitrary center size"""
    print("\n=== Testing Window Planning ===")
    
    patch = 512
    overlap_x = 0.25
    overlap_y = 0.25
    center_w = 400
    center_h = 300
    order = ["N", "E", "S", "W"]
    
    windows, canvas_w, canvas_h, cx, cy, sx, sy = plan_windows_with_center(
        patch, overlap_x, overlap_y, center_w, center_h, order
    )
    
    print(f"Center: {center_w}x{center_h}")
    print(f"Canvas: {canvas_w}x{canvas_h}")
    print(f"Center position: ({cx}, {cy})")
    print(f"Stride: ({sx}, {sy})")
    
    # Expected values
    expected_sx = expected_sy = 384  # 512 * (1 - 0.25)
    expected_canvas_w = center_w + 2 * expected_sx  # 400 + 2*384 = 1168
    expected_canvas_h = center_h + 2 * expected_sy  # 300 + 2*384 = 1068
    
    assert canvas_w == expected_canvas_w, f"Expected canvas_w={expected_canvas_w}, got {canvas_w}"
    assert canvas_h == expected_canvas_h, f"Expected canvas_h={expected_canvas_h}, got {canvas_h}"
    assert cx == sx and cy == sy, f"Center position should be ({sx}, {sy}), got ({cx}, {cy})"
    
    print("Windows:")
    for name, win in windows.items():
        if name in ["C"] + order:
            print(f"  {win.name}: ({win.x}, {win.y})")
    
    print("✓ Test passed")


def test_priority_compositing():
    """Test priority-based compositing"""
    print("\n=== Testing Priority Compositing ===")
    
    canvas_h, canvas_w = 100, 100
    composer = PriorityComposer(canvas_h, canvas_w)
    
    # Mark center region
    composer.mark_center(40, 30, 20, 30)  # x, y, h, w
    
    # Check that center is marked with priority 0
    center_region = composer.filled_by[30:50, 40:70]  # y:y+h, x:x+w
    assert np.all(center_region == 0), "Center region should be marked with priority 0"
    
    # Test first-wins compositing
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Create mock patch and mask
    patch_size = 50
    out_patch = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)  # White patch
    mask_fill = np.ones((patch_size, patch_size), dtype=np.uint8)  # Fill entire patch
    
    # Try to composite at position that overlaps with center
    composer.composite_first_wins(canvas, out_patch, mask_fill, 20, 20, pid=1)
    
    # Check that center pixels remain unchanged (priority 0 wins)
    center_pixels = canvas[30:50, 40:70]
    overlapping_pixels = canvas[20:30, 40:70]  # Should be filled
    
    assert np.all(center_pixels == 0), "Center pixels should remain unchanged"
    assert np.any(overlapping_pixels == 255), "Non-center overlapping pixels should be filled"
    
    print("✓ Priority compositing test passed")


def test_with_real_image():
    """Test with the actual test image"""
    print("\n=== Testing with Real Image ===")
    
    # Load test image (adjusted path for new structure)
    test_image_path = Path(__file__).parent.parent / "data" / "test_center.png"
    center_img = load_rgb(str(test_image_path))
    print(f"Loaded test image: {center_img.shape}")
    
    # Test window planning with this image
    patch = 512
    overlap_x = overlap_y = 0.25
    center_h, center_w = center_img.shape[:2]
    order = ["N", "E", "S", "W", "NE", "NW", "SE", "SW"]
    
    windows, canvas_w, canvas_h, cx, cy, sx, sy = plan_windows_with_center(
        patch, overlap_x, overlap_y, center_w, center_h, order
    )
    
    print(f"Input image: {center_w}x{center_h}")
    print(f"Output canvas: {canvas_w}x{canvas_h}")
    print(f"Expansion ratio: {canvas_w/center_w:.2f}x{canvas_h/center_h:.2f}")
    
    # Test local input building
    canvas_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    composer = PriorityComposer(canvas_h, canvas_w)
    
    # Place center image
    canvas_rgb[cy:cy+center_h, cx:cx+center_w] = center_img
    composer.mark_center(cx, cy, center_h, center_w)
    
    # Test building local input for first window
    first_window = windows[order[0]]
    local_rgb, local_mask = build_local_input(
        canvas_rgb, composer, first_window.x, first_window.y, patch
    )
    
    print(f"Local input shape: {local_rgb.shape}")
    print(f"Local mask shape: {local_mask.shape}")
    print(f"Pixels to generate: {np.count_nonzero(local_mask)} / {patch*patch}")
    
    assert local_rgb.shape == (patch, patch, 3), f"Expected {(patch, patch, 3)}, got {local_rgb.shape}"
    assert local_mask.shape == (patch, patch), f"Expected {(patch, patch)}, got {local_mask.shape}"
    assert np.count_nonzero(local_mask) > 0, "Should have pixels to generate"
    
    print("✓ Real image test passed")


if __name__ == "__main__":
    print("Testing Progressive Outpainting Logic...")
    
    try:
        test_overlap_calculation()
        test_window_planning()
        test_priority_compositing()
        test_with_real_image()
        
        print("\n🎉 All tests passed! The progressive outpainting logic is working correctly.")
        print("\nThe code is ready for actual SD inference. To run with a model:")
        print("python inference_outpainting_progressive_overlap.py --center test/data/test_center.png --prompt 'a beautiful landscape'")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()