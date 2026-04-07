"""
PersonViT Inference Script
--------------------------
Class-based wrapper to load any PersonViT backbone and extract CLS + patch tokens.
"""
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

# ── resolve imports from the PersonViT finetuning tree ──────────────────
_FINETUNING_DIR = (
    Path(__file__).resolve().parent.parent
    / "training-pipeline"
    / "Architecture"
    / "models"
    / "personvit"
    / "finetuning"
)
sys.path.insert(0, str(_FINETUNING_DIR))

from model.backbones.vit_pytorch import (
    vit_base_patch16_224_TransReID,
    vit_small_patch16_224_TransReID,
    vit_tiny_patch16_224_TransReID,
)


PERSONVIT_BACKBONES = {
    "personvit-vit_tiny":  {"factory": vit_tiny_patch16_224_TransReID,  "embed_dim": 192},
    "personvit-vit_small": {"factory": vit_small_patch16_224_TransReID, "embed_dim": 384},
    "personvit-vit_base":  {"factory": vit_base_patch16_224_TransReID,  "embed_dim": 768},
}


class PersonViTExtractor:
    """
    Load a PersonViT backbone once, then extract tokens from any image.

    Usage:
        extractor = PersonViTExtractor("personvit-vit_small")
        extractor.load("path/to/checkpoint.pth")
        tokens = extractor.extract_token("photo.jpg")
    """

    def __init__(self, model_key: str = "personvit-vit_small", device: str = "cuda",
                 img_size: tuple = (256, 128), stride_size: int = 16):
        if model_key not in PERSONVIT_BACKBONES:
            raise ValueError(f"Unknown model key '{model_key}'. "
                             f"Choose from: {list(PERSONVIT_BACKBONES)}")
        self.model_key = model_key
        self.device = device
        self.img_size = img_size
        self.stride_size = stride_size
        self.model = None
        self._transform = None

    def load(self, checkpoint_path, hw_ratio=2):
        """Load the model and weights onto the target device."""
        cfg = PERSONVIT_BACKBONES[self.model_key]
        self.model = cfg["factory"](
            img_size=self.img_size,
            stride_size=self.stride_size,
            drop_path_rate=0.0,
            camera=0,
            view=0,
            local_feature=False,
        )
        self.model.load_param(checkpoint_path, hw_ratio=hw_ratio)
        self.model = self.model.to(self.device).eval()
        self._transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def extract_token(self, image, target_size=None):
        """
        Run an image through the model and return CLS + patch tokens.

        Args:
            image: PIL Image or path string.
            target_size: Optional (H, W) tuple. If provided, resize the image
                         to this size and skip the default resize.

        Returns:
            dict with:
                "cls_token"    – (1, 1, D) tensor
                "patch_tokens" – (1, N, D) tensor
        """
        if self.model is None:
            raise RuntimeError("Call .load() before .extract_token()")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if target_size is not None:
            image = image.resize((target_size[1], target_size[0]), Image.BICUBIC)
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            pixel_values = transform(image).unsqueeze(0).to(self.device)
        else:
            pixel_values = self._transform(image).unsqueeze(0).to(self.device)

        # Forward all tokens (CLS + patches) instead of just CLS
        m = self.model
        B = pixel_values.shape[0]
        x = m.patch_embed(pixel_values)
        cls_tokens = m.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + m.pos_embed
        x = m.pos_drop(x)
        for blk in m.blocks:
            x = blk(x)
        x = m.norm(x)                                                # (B, 1+N, D)

        cls_token = x[:, 0:1, :]                                     # (1, 1, D)
        patch_tokens = x[:, 1:, :]                                   # (1, N, D)

        return {"cls_token": cls_token, "patch_tokens": patch_tokens}


# ── quick test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PersonViT token extraction")
    parser.add_argument("--model", default="personvit-vit_small",
                        help="Key from PERSONVIT_BACKBONES")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    extractor = PersonViTExtractor(args.model, args.device)
    extractor.load(args.checkpoint)
    tokens = extractor.extract_token(args.image)

    print(f"CLS token   : {tokens['cls_token'].shape}")
    print(f"Patch tokens: {tokens['patch_tokens'].shape}")
