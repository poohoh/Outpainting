import torch
import sys
sys.path.append("/app")

ckpt = torch.load("checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt", map_location="cpu")
sd = ckpt.get("state_dict", ckpt)

print("Total keys in checkpoint:", len(sd))
print("\nEMA related keys:")
ema_keys = [k for k in sd.keys() if "model_ema" in k]
print("EMA keys count:", len(ema_keys))
if ema_keys:
    print("Sample EMA keys:")
    for i, key in enumerate(ema_keys[:10]):
        print(f"  {key}")
    if len(ema_keys) > 10:
        print(f"  ... and {len(ema_keys) - 10} more")

print("\nNon-EMA model keys:")
model_keys = [k for k in sd.keys() if k.startswith("model.") and "model_ema" not in k]
print("Model keys count:", len(model_keys))
if model_keys:
    print("Sample model keys:")
    for i, key in enumerate(model_keys[:10]):
        print(f"  {key}")
    if len(model_keys) > 10:
        print(f"  ... and {len(model_keys) - 10} more")