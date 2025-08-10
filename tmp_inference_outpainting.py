import torch
import argparse, os, sys, datetime, glob
sys.path.append('/app/stable-diffusion')
sys.path.append('/app')

import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from data.humanart import create_dataloader

class OutpaintingInference:
    def __init__(self, config_path="/app/configs/unet/outpainting.yaml", device="cuda"):
        self.device = device
        self.config_path = config_path

        # load config
        self.config = OmegaConf.load(config_path)

        # load model
        self.model = instantiate_from_config(self.config.model)

        # load checkpoint
        self.load_checkpoint("/app/checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt")

        self.model = self.model.to(device)
        self.model.eval()

        # initialize DDIM Sampler
        self.sampler = DDIMSampler(self.model)

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            print(f"checkpoint loading: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if "state_dict" in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # if need to change key name
            self.model.load_state_dict(state_dict, strict=False)
            print("successfully loaded checkpoint")
        else:
            print("no checkpoint")
    

    def prepare_outpainting_input(self, batch):
        # extract datas from batch
        input_images = batch['input_images']  # B, 3, H, W
        masks = batch['masks']  # B, 1, H, W

        # use vae for latent space
        with torch.no_grad():
            posterior = self.model.encode_first_stage(input_images)  # B, 4, H/8, W/8

            input_latents = posterior.mode()

            latent_h, latent_w = input_latents.shape[-2:]
            masks_latent = F.interpolate(masks, size=(latent_h, latent_w), mode='nearest')  # B, 1, H/8, W/8
        
        return input_latents, masks_latent

    def run_inference(self, batch, num_steps=50, guidance_scale=7.0, eta=0.0):
        batch_size = batch['input_images'].shape[0]

        # prepare input
        input_latents, masks_latent = self.prepare_outpainting_input(batch)

        masked_input_latents = input_latents * (1 - masks_latent)

        # unet conditional input
        c_concat = torch.cat([
            masked_input_latents,  # B, 4, H/8, W/8
            masks_latent,  # B, 1, H/8, W/8
        ], dim=1)  # B, 5, H/8, W/8
        
        # text conditioning
        prompts = batch.get('prompts', [""] * batch_size)
        c_crossattn = None
        unconditional_conditioning = None

        # prompt embedding (if model has text encoder)
        if hasattr(self.model, 'get_learned_conditioning'):
            c_crossattn = self.model.get_learned_conditioning(prompts)

            # hybrid conditioning
            conditioning = {
                'c_concat': [c_concat],
                'c_crossattn': [c_crossattn]
            }

            if guidance_scale > 1.0:
                unconditional_crossattn = self.model.get_learned_conditioning([""] * batch_size)
                unconditional_conditioning = {
                    'c_concat': [c_concat],
                    'c_crossattn': [unconditional_crossattn]
                }
        else:
            # fallback
            conditioning = c_concat
        
        # sample shape
        latent_shape = [4, input_latents.shape[-2], input_latents.shape[-1]]  # 4, H/8, W/8

        print(f"outpainting sampling start - steps: {num_steps}, guidance: {guidance_scale}")
        print(f"input latents: {input_latents.shape}")
        print(f"outpaint masks: {masks_latent.shape}")
        print(f"condition input: {c_concat.shape}")

        # DDIM sample start
        outpainted_samples, _ = self.sampler.sample(
            S=num_steps,
            conditioning=conditioning,
            batch_size=batch_size,
            shape=latent_shape,
            verbose=False,
            unconditional_conditioning=unconditional_conditioning,
            eta=eta,
            x_T=None,  # from random noise
            mask=masks_latent,
            x0=input_latents,  # from center image
        )

        # latent to image
        with torch.no_grad():
            outpainted_images = self.model.decode_first_stage(outpainted_samples)
        
        # [-1,1] -> [0,1]
        outpainted_images = torch.clamp((outpainted_images + 1.0) / 2.0, min=0.0, max=1.0)

        return outpainted_images
            

if __name__ == "__main__":
    config_path = "/app/configs/unet/outpainting.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # sample image
    input_images = torch.randn(1, 3, 512, 512).to(device)
    masks = torch.ones(1, 1, 512, 512).to(device)

    batch = {
        'input_images': input_images,
        'masks': masks,
        'prompts': ["a beautiful landscape with mountains and trees"]
    }

    # outpainting pipeline
    outpainting = OutpaintingInference(config_path=config_path, device=device)

    # prepare input
    input_latents, masks_latent = outpainting.prepare_outpainting_input(batch)

    print(f"Input shape: {input_images.shape}")
    print(f"Input type: {type(input_images)}")
    print(f"input latent shape: {input_latents.shape}")
    print(f"input latent type: {type(input_latents)}")

    # run inference
    num_steps=20
    guidance_scale=7.0
    outpainted_images = outpainting.run_inference(batch, num_steps=num_steps, guidance_scale=guidance_scale)

    print(f"outpainted images shape: {outpainted_images.shape}")
    print(f"outpainted images type: {type(outpainted_images)}")