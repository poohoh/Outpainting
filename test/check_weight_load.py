# check_weight_load.py
# Usage:
#   python check_weight_load.py \
#       --config configs/outpainting/outpainting_inference.yaml \
#       --ckpt   checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt \
#       --device cuda
import argparse, sys, torch
from typing import Dict, Tuple
from omegaconf import OmegaConf

# your repo layout assumption
sys.path.append("/app")
from ldm.util import instantiate_from_config

# -----------------------------
# Helpers
# -----------------------------
def shape_map(sd: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in sd.items() if isinstance(v, torch.Tensor)}

def subdict(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in sd.items() if k.startswith(prefix)}

def compare_state_dicts(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor], tag_a="A", tag_b="B"):
    sa, sb = shape_map(sd_a), shape_map(sd_b)
    missing = sorted(set(sb) - set(sa))        # present in model but not in ckpt
    unexpected = sorted(set(sa) - set(sb))     # present in ckpt but not in model
    diffshape = sorted([k for k in sb if (k in sa and sa[k] != sb[k])])

    print(f"[cmp] {tag_a} vs {tag_b} :: missing={len(missing)}, unexpected={len(unexpected)}, diffshape={len(diffshape)}")
    if missing[:5]:    print("  - missing(sample):", missing[:5])
    if unexpected[:5]: print("  - unexpected(sample):", unexpected[:5])
    if diffshape[:5]:  print("  - diffshape(sample):", [(k, sa[k], sb[k]) for k in diffshape[:2]])

    assert not missing and not unexpected and not diffshape, f"state_dict mismatch between {tag_a} and {tag_b}"

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/outpainting/outpainting_inference.yaml")
    ap.add_argument("--ckpt",   default="checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) Instantiate model from Stability/CompVis config
    cfg = OmegaConf.load(args.config)
    model = instantiate_from_config(cfg.model)

    # 2) Load checkpoint (handle EMA meta keys cleanly)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd_full = ckpt.get("state_dict", ckpt)

    def _try_load(sd, strict=True):
        try:
            model.load_state_dict(sd, strict=strict)
            print(f"[load] strict={strict} : success")
            return True
        except RuntimeError as e:
            print(f"[load] strict={strict} : {e}")
            return False

    ok = _try_load(sd_full, strict=True)
    if not ok:
        # strip EMA keys then retry strict
        sd_noema = {k: v for k, v in sd_full.items() if not k.startswith("model_ema.")}
        print(f"[load] stripping EMA keys: removed {len(sd_full) - len(sd_noema)} entries")
        ok = _try_load(sd_noema, strict=True)
        if not ok:
            print("[load] fallback to strict=False")
            _ = _try_load(sd_noema, strict=False)  # last resort

    # 3) Fingerprint & state_dict match — UNet / VAE
    unet = model.model.diffusion_model
    vae  = model.first_stage_model
    cond = model.cond_stage_model  # FrozenCLIPEmbedder 래퍼(일반적)

    print("\n--- UNet fingerprint ---")
    print("in_channels     :", getattr(unet, "in_channels", None))
    print("out_channels    :", getattr(unet, "out_channels", None))
    print("model_channels  :", getattr(unet, "model_channels", None))
    print("context_dim     :", getattr(unet, "context_dim", None))
    sf = float(getattr(model, "scale_factor", 0.18215))
    print("scale_factor    :", sf)

    assert unet.in_channels == 9,  "UNet.in_channels must be 9 for inpainting (4+4+1)"
    assert unet.out_channels == 4, "UNet.out_channels must be 4 (latent)"
    assert float(abs(sf - 0.18215)) < 1e-6, "scale_factor must be 0.18215 (v1 VAE)"

    # Compare UNet weights (ckpt vs instantiated)
    sd_unet_ckpt  = subdict(sd_full, "model.diffusion_model.")
    sd_unet_model = unet.state_dict()
    compare_state_dicts(sd_unet_ckpt, sd_unet_model, tag_a="ckpt.UNet", tag_b="model.UNet")

    # Compare VAE weights (ckpt vs instantiated)
    sd_vae_ckpt  = subdict(sd_full, "first_stage_model.")
    sd_vae_model = vae.state_dict()
    compare_state_dicts(sd_vae_ckpt, sd_vae_model, tag_a="ckpt.VAE", tag_b="model.VAE")

    # 4) CLIP Text encoder structural checks (HF model via FrozenCLIPEmbedder)
    print("\n--- CLIP Text fingerprint ---")
    # 일반적으로 cond has: .tokenizer (HF) and .transformer (HF CLIPTextModel)
    text_model = getattr(cond, "transformer", cond)  # 안전장치
    cfg_t = getattr(text_model, "config", None)
    print("class           :", text_model.__class__.__name__)
    if cfg_t is not None:
        print("hidden_size     :", getattr(cfg_t, "hidden_size", None))
        print("num_layers      :", getattr(cfg_t, "num_hidden_layers", None))
        print("max_pos_embed   :", getattr(cfg_t, "max_position_embeddings", None))
        print("vocab_size      :", getattr(cfg_t, "vocab_size", None))
        print("name_or_path    :", getattr(cfg_t, "_name_or_path", None))
        assert cfg_t.hidden_size == 768
        assert cfg_t.num_hidden_layers == 12
        assert cfg_t.max_position_embeddings == 77
        # OpenAI CLIP 기본 토크나이저 vocab
        assert getattr(cfg_t, "vocab_size", 49408) in (49408,)

    # Ensure it's text‑only (no vision weights mixed in)
    tm_keys = list(text_model.state_dict().keys())
    assert not any(k.startswith("vision_model.") for k in tm_keys), "Unexpected vision weights inside text model"

    # 5) Smoke forward tests (UNet / VAE / CLIP)
    model.to(device).eval()
    with torch.no_grad():
        # (A) CLIP Text: tokenize & forward → (B,77,768)
        tokenizer = getattr(cond, "tokenizer", None)
        assert tokenizer is not None, "cond_stage_model must have a tokenizer"
        toks = tokenizer(["a photo of a cat"], max_length=77, padding="max_length",
                         truncation=True, return_tensors="pt")
        hs = text_model(
            input_ids=toks["input_ids"].to(device),
            attention_mask=toks["attention_mask"].to(device)
        ).last_hidden_state
        print("[CLIP] hidden_state shape:", tuple(hs.shape))
        assert hs.shape == (1, 77, 768)

        # (B) VAE: [-1,1] 입력 → encode/decode → shape 검사
        x_img = torch.rand(1, 3, 512, 512, device=device) * 2 - 1  # [-1,1]
        posterior = vae.encode(x_img)  # DiagonalGaussianDistribution in ldm
        z = posterior.sample() * sf
        print("[VAE] latent shape:", tuple(z.shape))
        assert z.shape == (1, 4, 64, 64)
        x_rec = vae.decode(z / sf)
        print("[VAE] recon shape :", tuple(x_rec.shape))
        assert x_rec.shape == (1, 3, 512, 512)

        # (C) UNet: (z, z_masked, mask) 형태의 9채널 더미 입력으로 포워드
        B, H, W = 1, 64, 64
        z_noisy   = torch.randn(B, 4, H, W, device=device)
        z_masked  = torch.randn(B, 4, H, W, device=device)
        mask      = torch.rand(B, 1, H, W, device=device)  # [0,1]
        unet_in   = torch.cat([z_noisy, z_masked, mask], dim=1)  # (B,9,H,W)

        t = torch.tensor([0], dtype=torch.long, device=device)
        y = unet(unet_in, t, context=hs)  # context: (B,77,768)
        print("[UNet] out shape  :", tuple(y.shape))
        assert y.shape == (B, 4, H, W)

    print("\nALL CHECKS PASSED ✅")

if __name__ == "__main__":
    main()
