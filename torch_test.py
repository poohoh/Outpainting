import torch
import os

print("=== PyTorch CUDA 환경 테스트 ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '설정되지 않음')}")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"GPU 개수: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print("\n=== GPU 정보 ===")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  메모리: {props.total_memory / 1024**3:.1f} GB")
    
    print("\n=== 간단한 GPU 테스트 ===")
    device = torch.device('cuda')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.mm(x, y)
    print(f"GPU 행렬 곱셈 성공: {z.device}")
    print("🎉 PyTorch GPU 환경이 정상적으로 작동합니다!")
else:
    print("❌ CUDA를 사용할 수 없습니다.")