"""
HumanArt Dataset for Single-shot Center-to-Full Outpainting
중앙 이미지를 입력으로 받아서 전체 캔버스로 확장하는 outpainting 학습용 데이터셋
"""
import os
import json
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import DDPMScheduler
import random

class HumanArtOutpaintingDataset(Dataset):
    def __init__(
        self,
        dataset_root="/app/datasets/humanart",
        mapping_file="mapping_file_training.json",
        target_size=512,      # 최종 이미지 크기
        center_size=256,      # 중앙 영역 크기 (논문의 r=0.5: 50%×50% = 25% 면적)
        noise_scheduler=None,
        transform=None,
        use_2d_artwork_only=True,  # 2D artwork만 사용
        excluded_categories=['cartoon', 'digital_art', 'kids_drawing', 'shadow_play', 'sketch']  # 제외할 카테고리 리스트 (기본값)
    ):
        self.dataset_root = dataset_root
        self.target_size = target_size
        self.center_size = center_size
        self.use_2d_artwork_only = use_2d_artwork_only
        self.excluded_categories = excluded_categories
        
        # DDPM 노이즈 스케줄러 (SD 1.5와 동일한 설정)
        if noise_scheduler is None:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                steps_offset=1,
                prediction_type="epsilon"
            )
        else:
            self.noise_scheduler = noise_scheduler
            
        # 데이터셋 로드
        mapping_path = os.path.join(dataset_root, mapping_file)
        with open(mapping_path, 'r') as f:
            self.data_mapping = json.load(f)
        
        self.data_keys = list(self.data_mapping.keys())
        original_count = len(self.data_keys)
        print(f"✓ HumanArt 데이터셋 로드 완료: {original_count}개 이미지")
        
        # 카테고리별 필터링
        if self.use_2d_artwork_only or self.excluded_categories:
            filtered_keys = []
            category_counts = {}  # 카테고리별 개수 추적
            excluded_count = 0
            
            for key in self.data_keys:
                img_path = self.data_mapping[key]['img_path']
                
                # 2D artwork만 사용하는 경우
                if self.use_2d_artwork_only and not img_path.startswith('images/2D_virtual_human/'):
                    excluded_count += 1
                    continue
                    
                # 제외 카테고리 체크
                should_exclude = False
                if self.excluded_categories and img_path.startswith('images/2D_virtual_human/'):
                    # img_path에서 카테고리 추출: images/2D_virtual_human/cartoon/xxx.jpg -> cartoon
                    path_parts = img_path.split('/')
                    if len(path_parts) >= 3:
                        category = path_parts[2]
                        if category in self.excluded_categories:
                            should_exclude = True
                            excluded_count += 1
                        else:
                            # 포함된 카테고리 카운트
                            category_counts[category] = category_counts.get(category, 0) + 1
                
                if not should_exclude:
                    filtered_keys.append(key)
            
            self.data_keys = filtered_keys
            print(f"✓ 카테고리 필터링 완료: {original_count}개 → {len(self.data_keys)}개 이미지 ({excluded_count}개 제외)")
            
            if category_counts:
                print("✓ 사용된 2D virtual human 카테고리:")
                for category, count in sorted(category_counts.items()):
                    print(f"  - {category}: {count:,}개")
        
        # 기본 전처리 파이프라인
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # [-1, 1] 범위로 정규화
            ])
        else:
            self.transform = transform
            
        # 중앙 크롭을 위한 계산
        self.crop_offset = (target_size - center_size) // 2
        
    def __len__(self):
        return len(self.data_keys)
    
    def load_image(self, img_path):
        """이미지 로드 및 전처리"""
        full_path = os.path.join(self.dataset_root, img_path)
        
        try:
            # 이미지 로드
            image = Image.open(full_path).convert('RGB')
            
            # 강제 리사이징 (aspect ratio 무시하고 정사각형으로)
            max_size = max(image.size)
            image = image.resize((max_size, max_size), Image.LANCZOS)
            
            return image
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패 {img_path}: {e}")
            # 더미 이미지 생성
            return Image.new('RGB', (512, 512), color=(128, 128, 128))
    
    def create_center_crop_setup(self, original_image):
        """
        중앙 크롭 설정 생성
        - 원본 이미지를 target_size로 리사이즈 (Ground Truth)
        - 중앙 center_size 영역만 유지하고 나머지는 마스킹
        """
        # 1. 원본 이미지를 target_size로 리사이즈 (Ground Truth)
        gt_image = original_image.resize((self.target_size, self.target_size), Image.LANCZOS)
        gt_tensor = self.transform(gt_image)  # [3, H, W], [-1, 1] 범위
        
        # 2. 중앙 영역 추출
        center_crop = gt_image.crop((
            self.crop_offset, 
            self.crop_offset,
            self.crop_offset + self.center_size,
            self.crop_offset + self.center_size
        ))
        
        # 3. 입력 이미지 생성 (중앙에 center_crop 배치, 나머지는 회색)
        input_image = Image.new('RGB', (self.target_size, self.target_size), color=(128, 128, 128))
        input_image.paste(center_crop, (self.crop_offset, self.crop_offset))
        input_tensor = self.transform(input_image)  # [3, H, W]
        
        # 4. 마스크 생성 (중앙 영역은 0, 나머지는 1)
        mask = torch.ones(1, self.target_size, self.target_size)  # [1, H, W]
        mask[:, self.crop_offset:self.crop_offset+self.center_size, 
             self.crop_offset:self.crop_offset+self.center_size] = 0
        
        return gt_tensor, input_tensor, mask
    
    def add_diffusion_noise(self, clean_image):
        """Diffusion 노이즈 추가"""
        # 랜덤 타임스텝 선택
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,))
        
        # 노이즈 생성
        noise = torch.randn_like(clean_image)
        
        # 노이즈가 추가된 이미지 생성
        noisy_image = self.noise_scheduler.add_noise(clean_image, noise, timesteps)
        
        return noisy_image, noise, timesteps
    
    def __getitem__(self, idx):
        """데이터셋 아이템 반환"""
        # 데이터 정보 가져오기
        data_key = self.data_keys[idx]
        data_info = self.data_mapping[data_key]
        
        img_path = data_info['img_path']
        prompt = data_info['prompt']
        
        # 이미지 로드
        original_image = self.load_image(img_path)
        
        # 중앙 크롭 설정 생성
        gt_image, input_image, mask = self.create_center_crop_setup(original_image)
        
        # Diffusion 노이즈 추가 (Ground Truth에)
        noisy_gt, noise_target, timesteps = self.add_diffusion_noise(gt_image)
        
        # SD 1.5 Inpainting 형식으로 입력 구성 (9채널)
        # [noisy_latent:4 + mask:1 + masked_image:4] = 9채널
        # 여기서는 RGB 이미지이므로 [noisy_gt:3 + mask:1 + input_image:3] = 7채널
        # 실제 학습에서는 VAE latent space에서 4채널로 변환됨
        
        model_input = torch.cat([
            noisy_gt,     # [3, H, W] - 노이즈가 추가된 GT 이미지
            mask,         # [1, H, W] - 마스크 (outpainting 영역 = 1)
            input_image   # [3, H, W] - 중앙 이미지가 있는 입력
        ], dim=0)  # [7, H, W]
        
        return {
            'model_input': model_input,     # [7, H, W] - 모델 입력
            'noise_target': noise_target,   # [3, H, W] - 노이즈 예측 타겟
            'timesteps': timesteps,         # [1] - 타임스텝
            'prompt': prompt,               # str - 텍스트 프롬프트
            'gt_image': gt_image,          # [3, H, W] - Ground Truth 이미지
            'input_image': input_image,     # [3, H, W] - 입력 이미지 (중앙만)
            'mask': mask,                  # [1, H, W] - 마스크
            'data_key': data_key           # str - 데이터 키
        }

def collate_fn(batch):
    """배치 데이터 처리"""
    model_inputs = torch.stack([item['model_input'] for item in batch])
    noise_targets = torch.stack([item['noise_target'] for item in batch])
    timesteps = torch.stack([item['timesteps'] for item in batch]).squeeze()
    prompts = [item['prompt'] for item in batch]
    gt_images = torch.stack([item['gt_image'] for item in batch])
    input_images = torch.stack([item['input_image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    data_keys = [item['data_key'] for item in batch]
    
    return {
        'model_inputs': model_inputs,      # [B, 7, H, W]
        'noise_targets': noise_targets,    # [B, 3, H, W]
        'timesteps': timesteps,            # [B]
        'prompts': prompts,                # List[str]
        'gt_images': gt_images,           # [B, 3, H, W]
        'input_images': input_images,      # [B, 3, H, W]
        'masks': masks,                   # [B, 1, H, W]
        'data_keys': data_keys            # List[str]
    }

def create_dataloader(
    dataset_root="/app/datasets/humanart",
    batch_size=4,
    num_workers=4,
    shuffle=True,
    target_size=512,
    center_size=256,  # 논문의 r=0.5 설정
    use_2d_artwork_only=True,  # 2D artwork만 사용
    excluded_categories=['cartoon', 'digital_art', 'kids_drawing', 'shadow_play', 'sketch']  # 제외할 카테고리 리스트 (기본값)
):
    """데이터로더 생성 헬퍼 함수"""
    
    dataset = HumanArtOutpaintingDataset(
        dataset_root=dataset_root,
        target_size=target_size,
        center_size=center_size,
        use_2d_artwork_only=use_2d_artwork_only,
        excluded_categories=excluded_categories
    )
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader, dataset

if __name__ == "__main__":
    # 테스트 코드
    print("🧪 HumanArt Outpainting Dataset 테스트")
    
    # 데이터셋 생성
    dataset = HumanArtOutpaintingDataset(
        target_size=512,
        center_size=320
    )
    
    # 샘플 확인
    sample = dataset[0]
    print(f"✓ 모델 입력 shape: {sample['model_input'].shape}")
    print(f"✓ 노이즈 타겟 shape: {sample['noise_target'].shape}")
    print(f"✓ 타임스텝: {sample['timesteps']}")
    print(f"✓ 프롬프트: {sample['prompt'][:50]}...")
    print(f"✓ GT 이미지 shape: {sample['gt_image'].shape}")
    print(f"✓ 입력 이미지 shape: {sample['input_image'].shape}")
    print(f"✓ 마스크 shape: {sample['mask'].shape}")
    
    # 데이터로더 테스트
    dataloader, _ = create_dataloader(batch_size=2, num_workers=0)
    batch = next(iter(dataloader))
    
    print(f"\n📦 배치 테스트:")
    print(f"✓ 배치 모델 입력 shape: {batch['model_inputs'].shape}")
    print(f"✓ 배치 노이즈 타겟 shape: {batch['noise_targets'].shape}")
    print(f"✓ 배치 타임스텝 shape: {batch['timesteps'].shape}")
    print(f"✓ 배치 프롬프트 개수: {len(batch['prompts'])}")
    
    print("🎉 데이터셋 테스트 완료!")