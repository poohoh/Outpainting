# human art data loader with circular mask
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import math

class HumanArtCircleDataset(Dataset):
    def __init__(
        self,
        dataset_root="/app/datasets/humanart",
        mapping_file="mapping_file_training.json",
        image_size=512,
        center_size=256,
        transform=None,
        use_2d_only=True,
        excluded_categories=['cartoon', 'digital_art', 'kids_drawing', 'shadow_play', 'sketch'],
        return_ground_truth=False  # for training
    ):
        """
        Args:
            dataset_root: Root path of HumanArt dataset
            mapping_file: JSON file containing image-prompt mappings
            image_size: Final image size (square)
            center_size: Size of center crop region
            transform: Custom transform (if None, default transform is used)
            use_2d_only: Use only 2D virtual human images
            excluded_categories: Categories to exclude from 2D virtual human
            return_ground_truth: Whether to return GT image (needed for training)
        """
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.center_size = center_size
        self.return_ground_truth = return_ground_truth

        # load mapping file
        mapping_path = os.path.join(dataset_root, mapping_file)
        with open(mapping_path, 'r') as f:
            self.data_mapping = json.load(f)
        
        # filter dataset
        self.data_keys = list(self.data_mapping.keys())
        if use_2d_only or excluded_categories:
            self.data_keys = self._filter_dataset(use_2d_only, excluded_categories)
        
        print(f"HumanArt Circle Dataset loaded: {len(self.data_keys)} images")

        # default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # [-1, 1] range
            ])
        else:
            self.transform = transform
        
        # calculate crop offset
        self.crop_offset = (image_size - center_size) // 2

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
    
    def create_circular_mask(self, size, center_size):
        """
        Create circular mask for outpainting
        Returns:
            mask: torch.Tensor of shape (1, size, size)
                  0 for center circle (keep), 1 for outer area (generate)
        """
        mask = torch.ones(1, size, size)
        
        # Create coordinate grids
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        
        # Center coordinates
        center = size // 2
        
        # Calculate distance from center
        distance = torch.sqrt((x - center).float() ** 2 + (y - center).float() ** 2)
        
        # Circular mask radius
        radius = center_size // 2
        
        # Set center circle to 0 (keep original), outer to 1 (generate)
        mask[0][distance <= radius] = 0
        
        return mask
    
    def create_outpainting_setup(self, image):
        """
        Create outpainting setup from full image with circular mask
        Returns:
            input_image: Image with center crop, rest filled with gray
            mask: 0 for center circle (keep), 1 for outer area (generate)
            ground_truth: Original full image (if return_ground_truth=True)
        """
        # resize to target size
        full_image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # extract center crop
        center_crop = full_image.crop((
            self.crop_offset,
            self.crop_offset,
            self.crop_offset + self.center_size,
            self.crop_offset + self.center_size
        ))

        # create circular mask (0 for center circle, 1 for outpainting region)
        mask = self.create_circular_mask(self.image_size, self.center_size)
        
        # Apply circular mask to create smooth blended input image
        # Convert mask to PIL for image processing
        mask_pil = Image.fromarray((mask[0].numpy() * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Create input image with circular masking directly from full image
        full_image_array = np.array(full_image)
        mask_array = np.array(mask_pil) / 255.0  # normalize to [0,1]
        
        # Start with gray background
        input_image_array = np.full((self.image_size, self.image_size, 3), 128, dtype=np.uint8)
        
        # Apply circular masking with smooth transition
        for c in range(3):  # RGB channels
            input_image_array[:, :, c] = (
                full_image_array[:, :, c] * (1 - mask_array) +  # keep center circle from original
                input_image_array[:, :, c] * mask_array          # gray background for outer area
            )
        
        input_image = Image.fromarray(input_image_array.astype(np.uint8))
        
        # transform images
        input_tensor = self.transform(input_image)

        if self.return_ground_truth:
            gt_tensor = self.transform(full_image)
            return input_tensor, mask, gt_tensor
        else:
            return input_tensor, mask
        
    def __len__(self):
        return len(self.data_keys)
    
    def __getitem__(self, idx):
        # get data info
        data_key = self.data_keys[idx]
        data_info = self.data_mapping[data_key]
        
        img_path = data_info['img_path']
        prompt = data_info['prompt']
        
        # load image
        image = self.load_image(img_path)

        # create outpainting setup
        if self.return_ground_truth:
            input_image, mask, ground_truth = self.create_outpainting_setup(image)
            return {
                'input_images': input_image,    # 3, H, W
                'masks': mask,                  # 1, H, W
                'prompts': prompt,              # str
                'ground_truth': ground_truth,   # 3, H, W
                'data_key': data_key
            }
        else:
            input_image, mask = self.create_outpainting_setup(image)
            return {
                'input_images': input_image,    # 3, H, W
                'masks': mask,                  # 1, H, W
                'prompts': prompt,              # str
                'data_key': data_key
            }
    
def collate_fn(batch):
    # collate function for dataloader
    input_images = torch.stack([item['input_images'] for item in batch])
    masks = torch.stack([item['masks'] for item in batch])
    prompts = [item['prompts'] for item in batch]
    data_keys = [item['data_key'] for item in batch]

    result = {
        'input_images': input_images,   # B, 3, H, W
        'masks': masks,                 # B, 1, H, W
        'prompts': prompts,             # list[str]
        'data_keys': data_keys          # list[str]
    }

    # add ground truth if available
    if 'ground_truth' in batch[0]:
        ground_truths = torch.stack([item['ground_truth'] for item in batch])
        result['ground_truth'] = ground_truths  # B, 3, H, W
    
    return result

def create_dataloader(
    dataset_root="/app/datasets/humanart",
    batch_size=4,
    num_workers=4,
    shuffle=True,
    image_size=512,
    center_size=256,
    use_2d_only=True,
    excluded_categories=['cartoon', 'digital_art', 'kids_drawing', 'shadow_play', 'sketch'],
    return_ground_truth=False
):
    """
    Create DataLoader for HumanArt dataset with circular masking
    
    Args:
        dataset_root: Root path of HumanArt dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        image_size: Final image size
        center_size: Size of center crop region
        use_2d_only: Use only 2D virtual human images
        excluded_categories: Categories to exclude
        return_ground_truth: Return ground truth images (for training)
    
    Returns:
        dataloader: PyTorch DataLoader
        dataset: HumanArtCircleDataset instance
    """
    dataset = HumanArtCircleDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        center_size=center_size,
        use_2d_only=use_2d_only,
        excluded_categories=excluded_categories,
        return_ground_truth=return_ground_truth
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader, dataset

if __name__=="__main__":
    # test inference mode
    print("\n=== Circle Inference Mode ===")
    dataset_inference = HumanArtCircleDataset(
        image_size=512,
        center_size=256,
        return_ground_truth=False
    )

    sample = dataset_inference[0]
    print(f"Input images shape: {sample['input_images'].shape}")
    print(f"Masks shape: {sample['masks'].shape}")
    print(f"Prompt: {sample['prompts'][:50]}...")

    # test training mode
    print("\n=== Circle Training Mode ===")
    dataset_training = HumanArtCircleDataset(
        image_size=512,
        center_size=256,
        return_ground_truth=True
    )

    sample = dataset_training[0]
    print(f"Input images shape: {sample['input_images'].shape}")
    print(f"Masks shape: {sample['masks'].shape}")
    print(f"Ground truth shape: {sample['ground_truth'].shape}")
    print(f"Prompt: {sample['prompts'][:50]}...")

    # test dataloader
    print("\n=== Circle DataLoader Test ===")
    dataloader, _ = create_dataloader(
        batch_size=2,
        num_workers=0,
        return_ground_truth=True
    )

    batch = next(iter(dataloader))
    print(f"Batch input images shape: {batch['input_images'].shape}")
    print(f"Batch masks shape: {batch['masks'].shape}")
    print(f"Batch ground truth shape: {batch['ground_truth'].shape}")
    print(f"Batch prompts count: {len(batch['prompts'])}")

    print("\nCircle Dataset test completed!")