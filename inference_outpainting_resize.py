"""
Inference for SD 1.5 Inpainting using HumanArtDataset for outpainting
- Loads model per your outpainting-inference.yaml (LatentInpaintDiffusion, hybrid conditioning)
- Uses HumanArt DataLoader directly with native batch format
- Applies gradio-compatible conditioning logic for SD 1.5 inpainting
- Saves side-by-side input and generated output
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
from data.humanart import create_dataloader

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

def create_output_structure(base_dir: str) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Create organized output directory structure:
    base_dir/
    └── YYYY-MM-DD_HH-MM-SS/
        ├── inputs/      # Input images  
        ├── outputs/     # AI generated results
        ├── comparisons/ # Side-by-side comparisons
        └── masks/       # Mask images

    Returns: (session_dir, inputs_dir, outputs_dir, comparisons_dir, masks_dir)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = Path(base_dir) / timestamp

    inputs_dir = session_dir / "inputs"
    outputs_dir = session_dir / "outputs"
    comparisons_dir = session_dir / "comparisons"
    masks_dir = session_dir / "masks"

    # create all directories
    for dir_path in [inputs_dir, outputs_dir, comparisons_dir, masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[Info] Results will be saved to: {session_dir}")
    print(f"[Info] - Inputs: {inputs_dir}")
    print(f"[Info] - Outputs: {outputs_dir}")
    print(f"[Info] - Comparisons: {comparisons_dir}")
    print(f"[Info] - Masks: {masks_dir}")

    return session_dir, inputs_dir, outputs_dir, comparisons_dir, masks_dir

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/outpainting/outpainting_inference.yaml", help="Config yaml for LatentInpaintDiffusion")
    p.add_argument("--ckpt", type=str, default="checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt", help="Path to sd-v1-5-inpainting checkpoint")
    p.add_argument("--data_root", type=str, default="datasets/humanart", help="Root or annotation source for HumanArtDataset")
    p.add_argument("--outdir", type=str, default="results/inference", help="Base directory for saving outpainting results")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--scale", type=float, default=7.5, help="Classifier-free guidance scale")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"], help="Weights/activation dtype")
    p.add_argument("--dataset_kwargs", type=str, default="{}", help="JSON string passed to HumanArtDataset(...)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create organized output structure
    session_dir, inputs_dir, outputs_dir, comparisons_dir, masks_dir = create_output_structure(args.outdir)
    model = load_model_from_config(args.config, args.ckpt, device, args.precision)
    sampler = DDIMSampler(model)

    # scale factor per your config (defaults to 0.18215 for SD v1.x)
    scale_factor = getattr(model, "scale_factor", 0.18215)

    # build dataloader using HumanArt's create_dataloader
    dataset_kwargs = json.loads(args.dataset_kwargs)
    loader, _ = create_dataloader(
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        return_ground_truth=True,  # for comparison in inference
        **dataset_kwargs
    )

    # process single batch (for testing/debugging)
    print(f"[Info] Processing one batch with {args.batch_size} samples ...")
    batch = next(iter(loader))
    t0 = time.time()
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(args.precision in ["fp16", "bf16"])):
        # humanart batch already in correct format
        x_img = batch["input_images"].to(device)
        x_mask = batch["masks"].to(device)
        prompts = batch["prompts"]
        names = batch["data_keys"]
        gt_img = batch["ground_truth"].to(device)

        # build conditioning using corrected gradio logic
        c, uc = prepare_conditioning_correct(model, x_img, x_mask, prompts, scale_factor)

        # sample
        B, _, H, W = x_img.shape
        z_shape = (4, H // 8, W // 8)
        samples, _ = sampler.sample(
            S=args.steps,
            conditioning=c,
            batch_size=B,
            shape=z_shape,
            verbose=False,
            unconditional_guidance_scale=args.scale,
            unconditional_conditioning=uc,
            eta=args.eta,
            x_T=None
        )

        # decode
        x_dec = model.decode_first_stage(samples)
        out_imgs = to_image_uint8(x_dec)  # BHWC, uint8
        in_imgs = to_image_uint8(x_img)  # BHWC
        mask_imgs = mask_to_image_uint8(x_mask)  # BHW, uint8 grayscale
        gt_imgs = to_image_uint8(gt_img)
    
    # save to organized structure
    for i in range(out_imgs.shape[0]):
        # set name
        if len(names) == out_imgs.shape[0]:
            name = names[i]
        else:
            name = f"sample_{i:02d}"
            if i == 0:
                print(f"[Warning] Key count mismatch: {len(names)} keys vs {out_imgs.shape[0]} images. Using generic names")

        in_pil = Image.fromarray(in_imgs[i])
        out_pil = Image.fromarray(out_imgs[i])
        side = stack_side_by_side(in_pil, out_pil)
        mask_pil = Image.fromarray(mask_imgs[i], mode="L")  # grayscale mode
        gt_pil = Image.fromarray(gt_imgs[i])
        
        # save to respective directories
        in_p = inputs_dir / f"{name}.png"
        out_p = outputs_dir / f"{name}.png"
        side_p = comparisons_dir / f"{name}.png"
        mask_p = masks_dir / f"{name}.png"
        gt_p = inputs_dir / f"{name}_gt.png"

        in_pil.save(in_p)
        out_pil.save(out_p)
        side.save(side_p)
        mask_pil.save(mask_p)
        gt_pil.save(gt_p)

    dt = time.time() - t0
    print(f"[Success] Generated and saved {out_imgs.shape[0]} images in {dt:.2f}s")

    print(f"Done. Results saved to: {session_dir}")

if __name__=="__main__":
    main()