"""
Inference for SD 1.5 Inpainting using HumanArtDataset for progressive outpainting
- Extended version of inference_resize.py
- Generate whole image with progressive 8-patch manner
- Sequentially generates 8 patches around center region: NW, N, NE, W, E, SW, S, SE
"""

import argparse, sys, time, json, random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# add project root
if '/app' not in sys.path:
    sys.path.append('/app')

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from data.humanart_progressive import create_progressive_dataloader

def get_9_patch_regions(canvas_size: int = 1536, patch_size: int = 512):
    """
    (512*3)x(512*3) -> 9 patches of (512)x(512)
    center for original image, rest for generated
    """
    patches = {
        'NW': (0, 0, patch_size, patch_size),                           # (0:512, 0:512)
        'N':  (0, patch_size, patch_size, patch_size*2),               # (0:512, 512:1024)
        'NE': (0, patch_size*2, patch_size, patch_size*3),             # (0:512, 1024:1536)
        'W':  (patch_size, 0, patch_size*2, patch_size),               # (512:1024, 0:512)
        'CTR': (patch_size, patch_size, patch_size*2, patch_size*2),   # (512:1024, 512:1024) - original input
        'E':  (patch_size, patch_size*2, patch_size*2, patch_size*3),  # (512:1024, 1024:1536)
        'SW': (patch_size*2, 0, patch_size*3, patch_size),             # (1024:1536, 0:512)
        'S':  (patch_size*2, patch_size, patch_size*3, patch_size*2),  # (1024:1536, 512:1024)
        'SE': (patch_size*2, patch_size*2, patch_size*3, patch_size*3) # (1024:1536, 1024:1536)
    }
    return patches

def initialize_canvas_with_center(original_img: torch.Tensor, canvas_size: int = 512*3, patch_size: int = 512):
    """
    Generate (512*3) x (512*3) canvas and put original image to center position
    Initialize other regions to neutral gray (0 in normalized space)
    """
    B, C, H, W = original_img.shape
    assert H == W == patch_size, f"original image must be {patch_size}x{patch_size}"

    # Generate whole canvas with neutral gray (0 in normalized [-1,1] space)
    canvas = torch.zeros(B, C, canvas_size, canvas_size, device=original_img.device, dtype=original_img.dtype)

    # Put original image to center of canvas
    canvas[:, :, patch_size:patch_size*2, patch_size:patch_size*2] = original_img

    return canvas

@torch.no_grad()
def progressive_outpaint_8_patches(
    model,
    sampler,
    original_img: torch.Tensor,  # [1,3,512,512] - center
    prompt: str,
    negative_prompt: str = "",
    steps: int = 30,
    cfg: float = 7.0,
    eta: float = 0.0,
    seed: int = 42,
    canvas_size: int = 512*3,
    patch_size: int = 512,
    conditioning_dir: Path = None,  # directory to save conditioning images
    sample_name: str = "sample",     # sample identifier for file naming
):
    """
    Progressive outpainting: generate sequential patches in (512*3) x (512*3) canvas except for center
    each patch is independent and does not overlap
    """
    device = next(model.parameters()).device
    original_img = original_img.to(device)

    # fix seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # initialize (512*3) x (512*3) canvas and put original img to center
    canvas = initialize_canvas_with_center(original_img=original_img, canvas_size=canvas_size, patch_size=patch_size)

    # define 9 patches region
    patch_regions = get_9_patch_regions(canvas_size=canvas_size, patch_size=patch_size)

    # generate 8 patches (except for center)
    generation_order = ['W', 'E', 'S', 'N', 'SW', 'NW', 'SE', 'NE']

    print(f"[Progressive] Starting 8-patch generation on {canvas_size}×{canvas_size} canvas")
    print(f"[Progressive] Original {patch_size}×{patch_size} image placed at center")
    print(f"[Progressive] Generation order: {generation_order}")

    for idx, patch_name in enumerate(generation_order):
        print(f"\n[Progressive] Step {idx+1}/8: Generating {patch_name} patch")

        # current patch region coordinate
        y0, x0, y1, x1 = patch_regions[patch_name]
        print(f"[Progressive] Patch region: ({y0}:{y1}, {x0}:{x1})")

        # resize entire canvas to 512x512 for model conditioning
        conditioning_img = F.interpolate(canvas, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
        
        # save conditioning image for this step
        if conditioning_dir is not None:
            conditioning_img_uint8 = to_image_uint8(conditioning_img)
            conditioning_pil = Image.fromarray(conditioning_img_uint8[0])
            conditioning_path = conditioning_dir / f"{sample_name}_step{idx+1:02d}_{patch_name}_conditioning.png"
            conditioning_pil.save(conditioning_path)
            print(f"[Progressive] Saved conditioning: {conditioning_path.name}")
        
        # mask is all 1s (generate entire 512x512 region)
        full_mask = torch.ones(1, patch_size, patch_size, device=device, dtype=torch.float32)
            
        # generate condition with current canvas state
        c, uc = prepare_conditioning_correct(
            model=model,
            x_img=conditioning_img,  # entire canvas resized to 512x512
            x_mask=full_mask.unsqueeze(0),  # [1,1,512,512] -> generate all regions
            prompts=[prompt],
        )

        # sampling
        z_shape = (4, patch_size // 8, patch_size // 8)  # 512 -> 64
        z_samples, _ = sampler.sample(
            S=steps,
            conditioning=c,
            batch_size=1,
            shape=z_shape,
            verbose=False,
            unconditional_guidance_scale=cfg,
            unconditional_conditioning=uc,
            eta=eta,
            x_T=None,
        )

        # decoding
        generated_patch = model.decode_first_stage(z_samples)  # [1,3,512,512]

        # put generated patch to canvas
        canvas[:, :, y0:y1, x0:x1] = generated_patch

        print(f"[Progressive] Completed {patch_name} patch generation")

    print(f"\n[Progressive] All 8 patches completed!")
    print(f"[Progressive] Final canvas size: {canvas_size}×{canvas_size}")
    
    return canvas

def prepare_conditioning_correct(
    model,
    x_img: torch.Tensor,  # (B, 3, H, W), [-1, 1]
    x_mask: torch.Tensor,  # (B, 1, H, W), {0, 1}, 1 = to fill
    prompts: List[str],
    scale_factor: float = 0.18215
) -> Tuple[Dict, Dict]:
    """
    Build conditioning following gradio/inpainting.py logic for SD 1.5 inpainting 
    """
    B, _, H, W = x_img.shape

    # create batch dict like gradio
    masked_image = x_img * (1.0 - x_mask)

    batch = {
        "image": x_img,
        "txt": prompts,
        "mask": x_mask,
        "masked_image": masked_image
    }

    # text conditioning - use same method as inpainting.py
    c = model.cond_stage_model.encode(prompts)

    # concatenated conditioning (following gradio logic)
    c_cat = list()
    for ck in model.concat_keys:
        cc = batch[ck].float()
        if ck != model.masked_image_key:
            # downsample mask to latent size
            bchw = [B, 4, H // 8, W // 8]
            cc = F.interpolate(cc, size=bchw[-2:])
        else:
            # encode masked image to latents - scale factor is applied in get_first_stage_encoding
            cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
        c_cat.append(cc)
    c_cat = torch.cat(c_cat, dim=1)

    # unconditional conditioning - try inpainting.py method first
    uc_cross = model.get_unconditional_conditioning(B, "")
    
    # build final conditioning
    cond = {"c_concat": [c_cat], "c_crossattn": [c]}
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

    return cond, uc_full

def to_image_uint8(x: torch.Tensor) -> np.ndarray:
    """
    x: (3, H, W) or (B, 3, H, W), in [-1, 1] -> uint8 HWC
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = (x.clamp(-1, 1) + 1.0) / 2.0  # [0, 1]
    x = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    x = x.permute(0, 2, 3, 1).cpu().numpy()  # BHWC

    return x

def mask_to_image_uint8(x: torch.Tensor) -> np.ndarray:
    """
    x: (1, H, W) or (B, 1, H, W), in [0, 1] -> uint8 HW (grayscale)
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.clamp(0, 1)
    x = (x * 255.0).clamp(0, 255).to(torch.uint8)
    x = x.squeeze(1).cpu().numpy()  # BHW -> remove channel dimension

    return x

def stack_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    h = max(left.height, right.height)
    w = left.width + right.width
    canvas = Image.new('RGB', (w,h), (0,0,0))
    canvas.paste(left.convert('RGB'), (0,0))
    canvas.paste(right.convert('RGB'), (left.width, 0))
    
    return canvas

def load_model_from_config(config_path: str, ckpt_path: str, device: torch.device, precision: str = "fp16"):
    print(f"[Info] Loading config: {config_path}")
    cfg = OmegaConf.load(config_path)

    print(f"[Info] Instantiating model ...")
    model = instantiate_from_config(cfg.model)

    print(f"[Info] Loading checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print(f"[Info] Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    model.eval().to(device)

    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.to(torch.bfloat16)
    
    return model

def create_output_structure(base_dir: str) -> Tuple[Path, Path, Path, Path, Path, Path]:
    """
    Create organized output directory structure:
    base_dir/
    └── YYYY-MM-DD_HH-MM-SS_progressive/
        ├── inputs/       # Input images  
        ├── outputs/      # AI generated results
        ├── comparisons/  # Side-by-side comparisons
        ├── conditioning/ # Conditioning images for each step
        └── masks/        # Mask images (kept for compatibility)

    Returns: (session_dir, inputs_dir, outputs_dir, comparisons_dir, conditioning_dir, masks_dir)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = Path(base_dir) / f"{timestamp}_progressive"

    inputs_dir = session_dir / "inputs"
    outputs_dir = session_dir / "outputs"
    comparisons_dir = session_dir / "comparisons"
    conditioning_dir = session_dir / "conditioning"  # NEW: conditioning images for each step
    masks_dir = session_dir / "masks"  # kept for compatibility

    # create all directories
    for dir_path in [inputs_dir, outputs_dir, comparisons_dir, conditioning_dir, masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[Info] Results will be saved to: {session_dir}")
    print(f"[Info] - Inputs: {inputs_dir}")
    print(f"[Info] - Outputs: {outputs_dir}")
    print(f"[Info] - Comparisons: {comparisons_dir}")
    print(f"[Info] - Conditioning: {conditioning_dir}")

    return session_dir, inputs_dir, outputs_dir, comparisons_dir, conditioning_dir, masks_dir

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/outpainting/outpainting_inference.yaml", help="Config yaml for LatentInpaintDiffusion")
    p.add_argument("--ckpt", type=str, default="checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt", help="Path to sd-v1-5-inpainting checkpoint")
    p.add_argument("--data_root", type=str, default="datasets/humanart", help="Root or annotation source for HumanArtDataset")
    p.add_argument("--outdir", type=str, default="results/inference", help="Base directory for saving outpainting results")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--scale", type=float, default=7.5, help="Classifier-free guidance scale")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--canvas_size", type=int, default=512*3, help="Final canvas size (512*3)")
    p.add_argument("--patch_size", type=int, default=512, help="Each patch size")
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"], help="Weights/activation dtype")
    p.add_argument("--dataset_kwargs", type=str, default="{}", help="JSON string passed to HumanArtDataset(...)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create organized output structure
    session_dir, inputs_dir, outputs_dir, comparisons_dir, conditioning_dir, masks_dir = create_output_structure(args.outdir)
    model = load_model_from_config(args.config, args.ckpt, device, args.precision)
    sampler = DDIMSampler(model)

    # build dataloader using Progressive dataloader
    dataset_kwargs = json.loads(args.dataset_kwargs)
    loader, _ = create_progressive_dataloader(
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        **dataset_kwargs
    )

    # process multiple batches for more diverse results
    num_iterations = 3 
    total_images = 0
    all_batch_times = []
    
    print(f"[Info] Processing {num_iterations} samples with progressive 8-oatch outpainting...")
    print(f"[Info] Canvas: {args.canvas_size}×{args.canvas_size}, Patch: {args.patch_size}×{args.patch_size}")
    print(f"[Info] Steps: {args.steps}, CFG: {args.scale}")
    
    # Create iterator once to get different batches each time
    data_iterator = iter(loader)
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"\n[Info] Processing batch {iteration + 1}/{num_iterations}...")
        batch = next(data_iterator)
        t0 = time.time()

        # Take only first sample since progressive works with batch_size=1
        original_img = batch["original_images"][:1].to(device)  # Full 512x512 original for comparison
        center_crop = batch["center_crops"][:1].to(device)     # Center crop 512x512 for initial condition
        prompt = batch["prompts"][0]
        name = batch["data_keys"][0]

        print(f"[Info] Prompt: '{prompt}'")
        print(f"[Info] Sample name: '{name}'")
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(args.precision in ["fp16", "bf16"])):
            # Progressive 8-patch outpainting
            result_canvas = progressive_outpaint_8_patches(
                model=model,
                sampler=sampler,
                original_img=center_crop,  # 512x512 center crop for initial condition
                prompt=prompt,
                negative_prompt="",
                steps=args.steps,
                cfg=args.scale,
                eta=args.eta,
                seed=args.seed + iteration,
                canvas_size=args.canvas_size,  # 512*3
                patch_size=args.patch_size,   # 512
                conditioning_dir=conditioning_dir,  # save conditioning images
                sample_name=f"{total_images:03d}_{name}",  # unique sample identifier
            )

            # convert tensors to images
            # save to compare (512*3) x (512*3) canvas and original img
            out_imgs = to_image_uint8(result_canvas)  # (512*3) x (512*3)
            orig_imgs = to_image_uint8(original_img)  # (512, 512) full original
            crop_imgs = to_image_uint8(center_crop)   # (512, 512) center crop

        # save results
        final_name = f"{total_images:03d}_{name}_progressive_8patch"
        total_images += 1

        # save original and result images
        orig_pil = Image.fromarray(orig_imgs[0])            # 512×512 full original
        crop_pil = Image.fromarray(crop_imgs[0])            # 512×512 center crop
        out_pil = Image.fromarray(out_imgs[0])              # 1536×1536 result
        
        # generate initial canvas with center crop
        in_canvas = Image.new('RGB', (args.canvas_size, args.canvas_size), (128, 128, 128))
        center_pos = (args.canvas_size - args.patch_size) // 2
        in_canvas.paste(crop_pil, (center_pos, center_pos))
        
        # Compare side-by-side (original canvas vs result)
        side = stack_side_by_side(in_canvas, out_pil)
        
        # save to respective directories
        orig_p = inputs_dir / f"{final_name}_original.png"   # 512×512 full original
        crop_p = inputs_dir / f"{final_name}_center.png"     # 512×512 center crop
        in_canvas_p = inputs_dir / f"{final_name}_canvas.png" # 1536×1536 initial
        out_p = outputs_dir / f"{final_name}.png"            # 1536×1536 result
        side_p = comparisons_dir / f"{final_name}.png"       # Side-by-side

        orig_pil.save(orig_p)
        crop_pil.save(crop_p)
        in_canvas.save(in_canvas_p)
        out_pil.save(out_p)
        side.save(side_p)

        batch_time = time.time() - t0
        all_batch_times.append(batch_time)
        
        print(f"[Success] Sample {iteration + 1} completed in {batch_time:.2f}s")
        print(f"[Success] Saved as: {final_name}")
        print(f"[Success] Final canvas: {args.canvas_size}×{args.canvas_size}")

    # final statistics
    total_time = sum(all_batch_times)
    avg_time_per_sample = total_time / total_images
    
    print(f"\n{'='*60}")
    print(f"[Final Results]")
    print(f"Total samples processed: {total_images}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample: {avg_time_per_sample:.2f}s")
    print(f"Progressive 8-patch outpainting completed!")
    print(f"Final canvas size: {args.canvas_size}×{args.canvas_size}")
    print(f"Results saved to: {session_dir}")

if __name__=="__main__":
    main()