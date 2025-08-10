import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse, os, sys, datetime, glob
sys.path.append('/app/stable-diffusion')
sys.path.append('/app')

import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion
from data.humanart import create_dataloader
import wandb
from torch.cuda.amp import autocast
from torch.amp import GradScaler

class OutpaintingTrainer:
    def __init__(self, config_path="/app/configs/unet/outpainting.yaml", device="cuda"):
        self.device = device
        self.config_path = config_path

        # load config
        self.config = OmegaConf.load(config_path)

        # initialize model
        self.model = instantiate_from_config(self.config.model)

        # load pretrained checkpoint
        pretrained_path = self.config.training.get('pretrained_checkpoint', "/app/checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt")
        self.load_pretrained(pretrained_path)

        self.model = self.model.to(device)
        self.model.train()

        # training hyperparameters from config
        self.training_config = self.config.training
        self.num_epochs = self.training_config.num_epochs
        self.batch_size = self.training_config.batch_size
        self.base_learning_rate = self.training_config.learning_rate
        self.save_interval = self.training_config.save_interval
        self.log_interval = self.training_config.log_interval
        self.warmup_steps = self.training_config.get('warmup_steps', 1000)
        self.use_ema = self.training_config.get('use_ema', True)

        # early stopping
        self.early_stopping_patience = self.training_config.get('early_stopping_patience', 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # optimizer and scheduler
        self.setup_optimizer()

        # mixed precision
        self.scaler = GradScaler()

        # EMA for model weights
        if self.use_ema:
            self.setup_ema()
        
        # checkpoint directory
        self.checkpoint_dir = self.training_config.get('checkpoint_dir', "/app/results/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f"Fine-tuning configuration loaded:")
        print(f"  - Epochs: {self.num_epochs}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Base Learning rate: {self.base_learning_rate}")
        print(f"  - Warmup steps: {self.warmup_steps}")
        print(f"  - Use EMA: {self.use_ema}")
        print(f"  - Device: {self.device}")
    
    def load_pretrained(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if "state_dict" in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # load with strict=False to allow partial loading
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            print(f"Missing keys (exptected for fine-tuning): {len(missing_keys)}")
            for key in missing_keys[:5]:
                print(f"  - {key}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:
                print(f"  - {key}")
            
            print("Pretrained checkpoint loaded successfully for fine-tuning")
        else:
            raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
    
    def setup_optimizer(self):
        optimizer_config = self.training_config.optimizer

        # TODO: separate parameters for different learning rates
        # if self.training_config.get('use_differential_lr', True):

        # only unet parameters
        learnable_params = []
        for name, param in self.model.model.named_parameters():
            learnable_params.append(param)
        
        param_groups = [{'params': learnable_params, 'lr': self.training_config.lr}]

        # optimizer
        if optimizer_config.type == "AdamW":
            self.optimizer = optim.AdamW(
                param_groups,
                betas=tuple(optimizer_config.betas),
                weight_decay=optimizer_config.weight_decay,
                eps=optimizer_config.get('eps', 1e-8)
            )
        elif optimizer_config.type == "Adam":
            self.optimizer = optim.Adam(
                param_groups,
                betas=tuple(optimizer_config.betas),
                weight_decay=optimizer_config.get('weight_decay',0),
                eps=optimizer_config.get('eps', 1e-8)
            )
        
        # learning rate scheduler
        scheduler_config = self.training_config.scheduler
        if scheduler_config.type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=scheduler_config.get('eta_min', 1e-7)
            )
        elif scheduler_config.type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma
            )
        
        # warmup scheduler wrapper
        if self.warmup_steps > 0:
            self.warmup_scheduler = self.get_warmup_scheduler()
    
    def get_warmup_scheduler(self):
        def warmup_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.0
        return optim.lr_scheduler.LambdaLR(self.optimizer, warmup_lambda)

    def setup_ema(self):
        from copy import deepcopy
        self.ema_model = deepcopy(self.model)
        self.ema_decay = self.training_config.get('ema_decay', 0.9999)

        # freeze ema model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update_ema(self):
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def prepare_training_batch(self, batch):
        input_images = batch['input_images'].to(self.device)  # B, 3, H, W
        target_images = batch['target_images'].to(self.device)  # B, 3, H, W
        masks = batch['masks'].to(self.device)  # B, 1, H, W
        prompts = batch.get('prompts', [""] * input_images.shape[0])

        # encode to latent space using VAE
        with torch.no_grad():
            # input latents (masked region)
            input_posterior = self.model.encode_first_stage(input_images)
            input_latents = input_posterior.mode()

            # target latents (full outpainted image)
            target_posterior = self.model.encode_first_stage(target_images)
            target_latents = target_posterior.mode()

            # resize masks to latent dimensions
            latent_h, latent_w = input_latents.shape[-2:]
            masks_latent = F.interpolate(masks, size=(latent_h, latent_w), mode='nearest')
        
        return input_latents, target_latents, masks_latent, prompts

    def compute_loss(self, input_latents, target_latents, masks_latent, prompts, timesteps=None):
        batch_size = input_latents.shape[0]

        # sample random timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device).long()
        
        # add noise to target latents
        noise = torch.randn_like(target_latents)
        noisy_latents = self.model.q_sample(x_start=target_latents, t=timesteps, noise=noise)

        # masked input latents for conditioning
        masked_input_latents = input_latents * (1 - masks_latent)

        # conditioning input: masked latents + mask
        c_concat = torch.cat([
            masked_input_latents,  # B, 4, H/8, W/8
            masks_latent  # B, 1, H/8, W/8
        ], dim=1)  # B, 5, H/8, W/8

        # text conditioning
        c_crossattn = None
        if hasattr(self.model, 'get_learned_conditioning') and prompts:
            c_crossattn = self.model.get_learned_conditioning(prompts)
        
        # predict noise
        if c_crossattn is not None:
            # hybrid conditioning
            conditioning = {
                'c_concat': [c_concat],
                'c_crossattn': [c_crossattn]
            }
            noise_pred = self.model.apply_model(noisy_latents, timesteps, conditioning)
        else:
            # concat conditioning only
            x_in = torch.cat([noisy_latents, c_concat], dim=1)
            noise_pred = self.model.apply_model(x_in, timesteps)

        # main diffusion loss
        main_loss = F.mse_loss(noise_pred, noise, reduction='mean')

        # additional losses for fine-tuning stability
        total_loss = main_loss

        # add consistency loss in masked regions
        if self.training_config.get('use_consistency_loss', True):
            consistency_weight = self.training_config.get('consistency_loss_weight', 0.1)
            # ensure predicted outpainting preserves original content in unmasked regions
            with torch.no_grad():
                clean_pred = self.model.predict_start_from_noise(noisy_latents, timesteps, noise_pred)

            consistency_loss = F.mse_loss(
                clean_pred * (1 - masks_latent),
                target_latents * (1 - masks_latent),
                reduction='mean'
            )
            total_loss += consistency_weight * consistency_loss

        return total_loss, noise_pred, noise

    def train_step(self, batch, global_step):
        pass

    def validate_step(self, batch):
        pass

    def save_checkpoint(self, epoch, loss, global_step, is_best=False):
        pass

    def load_checkpoint(self, checkpoint_path):
        # load training checkpoint
        pass

    def train(self, train_dataloader, val_dataloader=None, resume_from=None):
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Outpainting fine-tuning')
    parser.add_argument('--config', type=str, default='/app/configs/unet/outpainting.yaml')
    # parser.add_argument('--data_path', type=str, required=True)
    # parser.add_argument('--val_data_path', type=str)
    parser.add_argument('--resume', type=str)

    return parser.parse_args()


if __name__=="__main__":
    args = parse_arguments()

    # create trainer
    trainer = OutpaintingTrainer(args.config)

    # create dataloaders


    # start training