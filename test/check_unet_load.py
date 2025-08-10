import argparse, torch
from omegaconf import OmegaConf
import sys
sys.path.append("/app")
from ldm.util import instantiate_from_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default="configs/outpainting/outpainting_inference.yaml")
    ap.add_argument('--ckpt', default="checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt")
    ap.add_argument('--device', default="cuda" if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    # model instantiate
    cfg = OmegaConf.load(args.config)
    model = instantiate_from_config(cfg.model)

    # checkpoint load
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    try:
        missing, unexpected = model.load_state_dict(sd, strict=True)
        print("[strict=True] state_dict load success")
    except RuntimeError as e:
        print("[strict=True] load error:", e)
        model.load_state_dict(sd, strict=False)
    
    # unet handle
    unet = model.model.diffusion_model

    # check architecture
    print("Unet in_channels:", getattr(unet, "in_channels", None))
    print("Unet out_channels:", getattr(unet, "out_channels", None))
    print("Unet model_channels:", getattr(unet, "model_channels", None))
    print("context_dim (if any):", getattr(unet, "context_dim", None))
    print("scale_factor on model", getattr(model, "scale_factor", None))

    # necessary condition
    assert unet.in_channels == 9, "in_channels is not 9"
    assert unet.out_channels == 4, "out_channels is not 4"
    assert getattr(unet, "context_dim", 768) in (768,), "context_dim has to be 768"
    sf = getattr(model, "scale_factor", 0.18215)
    assert abs(float(sf) - 0.18215) < 1e-4, "scale_factor has to be 0.18215"

    # dummy forward
    device = torch.device(args.device)
    model.to(device).eval()
    with torch.no_grad():
        x = torch.randn(1, 9, 64, 64, device=device)
        t = torch.tensor([0], device=device, dtype=torch.long)
        ctx = torch.randn(1, 77, 768, device=device)
        y = unet(x, t, context=ctx)

    print("Forward OK. Output shape:", tuple(y.shape))
    assert y.shape==(1, 4, 64, 64), "output shape is not (1,4,64,64)"

    print("test pass")

if __name__=="__main__":
    main()