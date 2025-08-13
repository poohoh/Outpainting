# Simple HumanArt dataset for progressive outpainting
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class HumanArtProgressiveDataset(Dataset):
    def __init__(
        self,
        dataset_root="/app/datasets/humanart",
        mapping_file="mapping_file_training.json",
        image_size=512,
        transform=None,
        use_2d_only=True,
        excluded_categories=['cartoon', 'digital_art', 'kids_drawing', 'shadow_play', 'sketch'],
    ):
        """
        Simple dataset for progressive outpainting
        
        Returns:
        - original_images: Full original image resized to image_size×image_size
        - center_crops: Center crop from original (approximately half dimensions each)
                       then resized to image_size×image_size for initial condition
                       
        CENTER CROP GUARANTEE: 
        For original image WxH, center crop will be approximately (W/2)×(H/2)
        Crop bounds: [W/4 : 3W/4] × [H/4 : 3H/4] (integer division)
        
        Args:
            dataset_root: Root path of HumanArt dataset
            mapping_file: JSON file containing image-prompt mappings
            image_size: Target image size (square)
            transform: Custom transform (if None, default transform is used)
            use_2d_only: Use only 2D virtual human images
            excluded_categories: Categories to exclude from 2D virtual human
        """
        self.dataset_root = dataset_root
        self.image_size = image_size

        # load mapping file
        mapping_path = os.path.join(dataset_root, mapping_file)
        with open(mapping_path, 'r') as f:
            self.data_mapping = json.load(f)
        
        # filter dataset
        self.data_keys = list(self.data_mapping.keys())
        if use_2d_only or excluded_categories:
            self.data_keys = self._filter_dataset(use_2d_only, excluded_categories)
        
        print(f"HumanArt Progressive Dataset loaded: {len(self.data_keys)} images")

        # default transform - simple resize and normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # [-1, 1] range
            ])
        else:
            self.transform = transform

    def _filter_dataset(self, use_2d_only, excluded_categories):
        # filter dataset based on categories
        filtered_keys = []

        for key in self.data_keys:
            img_path = self.data_mapping[key]['img_path']

            # filter 2d virtual human only
            if use_2d_only and not img_path.startswith('images/2D_virtual_human/'):
                continue

            # filter excluded categories
            if excluded_categories and img_path.startswith('images/2D_virtual_human/'):
                path_parts = img_path.split('/')
                if len(path_parts) >= 3:
                    category = path_parts[2]  # category
                    if category in excluded_categories:
                        continue
            
            filtered_keys.append(key)
        
        return filtered_keys
    
    def load_image(self, img_path):
        # load and preprocess image
        full_path = os.path.join(self.dataset_root, img_path)

        try:
            image = Image.open(full_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            
            # return dummy image
            return Image.new('RGB', (512, 512), color=(128, 128, 128))
        
    def __len__(self):
        return len(self.data_keys)
    
    def __getitem__(self, idx):
        # get data info
        data_key = self.data_keys[idx]
        data_info = self.data_mapping[data_key]
        
        img_path = data_info['img_path']
        prompt = data_info['prompt']
        
        # load original image (keep original resolution)
        image = self.load_image(img_path)
        orig_w, orig_h = image.size
        
        # center crop: take middle half of each dimension
        # Mathematical guarantee: crop size = original_size // 2 (approximately)
        left = orig_w // 4
        top = orig_h // 4
        right = orig_w * 3 // 4
        bottom = orig_h * 3 // 4
        
        # verify that crop is approximately half the original size
        crop_w = right - left
        crop_h = bottom - top
        expected_w = orig_w // 2
        expected_h = orig_h // 2
        
        # Assert that crop is indeed approximately half (allowing for integer division)
        assert abs(crop_w - expected_w) <= 1, f"Crop width {crop_w} not half of original {orig_w}"
        assert abs(crop_h - expected_h) <= 1, f"Crop height {crop_h} not half of original {orig_h}"
        
        # create center crop from original resolution
        center_crop = image.crop((left, top, right, bottom))
        
        # resize both to target size
        original_resized = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        center_crop_resized = center_crop.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # transform to tensors
        original_tensor = self.transform(original_resized)
        center_crop_tensor = self.transform(center_crop_resized)

        return {
            'original_images': original_tensor,     # 3, H, W - full original for comparison
            'center_crops': center_crop_tensor,     # 3, H, W - center crop for initial condition
            'prompts': prompt,                      # str
            'data_keys': data_key                  # str
        }
    
def collate_fn_progressive(batch):
    # collate function for progressive dataloader
    original_images = torch.stack([item['original_images'] for item in batch])
    center_crops = torch.stack([item['center_crops'] for item in batch])
    prompts = [item['prompts'] for item in batch]
    data_keys = [item['data_keys'] for item in batch]

    return {
        'original_images': original_images,  # B, 3, H, W - full original images
        'center_crops': center_crops,        # B, 3, H, W - center crops for initial condition
        'prompts': prompts,                  # list[str]
        'data_keys': data_keys              # list[str]
    }

def create_progressive_dataloader(
    dataset_root="/app/datasets/humanart",
    batch_size=4,
    num_workers=4,
    shuffle=True,
    image_size=512,
    use_2d_only=True,
    excluded_categories=['cartoon', 'digital_art', 'kids_drawing', 'shadow_play', 'sketch'],
):
    """
    Create DataLoader for Progressive Outpainting
    
    Args:
        dataset_root: Root path of HumanArt dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        image_size: Target image size
        use_2d_only: Use only 2D virtual human images
        excluded_categories: Categories to exclude
    
    Returns:
        dataloader: PyTorch DataLoader
        dataset: HumanArtProgressiveDataset instance
    """
    dataset = HumanArtProgressiveDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        use_2d_only=use_2d_only,
        excluded_categories=excluded_categories,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_progressive,
        pin_memory=True
    )

    return dataloader, dataset

if __name__=="__main__":
    # test progressive dataset
    print("\n=== Progressive Dataset Test ===")
    dataset = HumanArtProgressiveDataset(
        image_size=512
    )

    sample = dataset[0]
    print(f"Original images shape: {sample['original_images'].shape}")
    print(f"Center crops shape: {sample['center_crops'].shape}")
    print(f"prompt: {sample['prompts'][:50]}...")

    # test dataloader
    print("\n=== Progressive DataLoader Test ===")
    dataloader, _ = create_progressive_dataloader(
        batch_size=2,
        num_workers=0
    )

    batch = next(iter(dataloader))
    print(f"Batch original images shape: {batch['original_images'].shape}")
    print(f"Batch center crops shape: {batch['center_crops'].shape}")
    print(f"Batch prompts count: {len(batch['prompts'])}")

    print("\nProgressive Dataset test completed!")