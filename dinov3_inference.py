"""
DINOv3 Inference Script
-----------------------
Class-based wrapper to load any DINOv3 backbone and extract CLS + patch tokens.
"""
import torch
from transformers import DINOv3ViTModel, DINOv3ViTImageProcessorFast
from PIL import Image


DINOV3_BACKBONES = {
    # ViT (Vision Transformer)   
    "dinov3-vits16-pretrain": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3-vitsplus-pretrain": "facebook/dinov3-vitsplus-pretrain-lvd1689m",
    "dinov3-vitb16-pretrain": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-vitl16-pretrain": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3-vit7b16-pretrain": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}


class DINOv3Extractor:
    """
    Load a DINOv3 backbone once, then extract tokens from any image.

    Usage:
        extractor = DINOv3Extractor("dinov3-vits16-pretrain")
        extractor.load()
        tokens = extractor.extract_token("photo.jpg")
    """

    def __init__(self, model_key: str = "dinov3-vits16-pretrain", device: str = "cuda"):
        self.model_id = DINOV3_BACKBONES.get(model_key, model_key)
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        """Load the model and processor onto the target device."""
        self.model = DINOv3ViTModel.from_pretrained(self.model_id)
        self.model = self.model.to(self.device).eval()
        self.processor = DINOv3ViTImageProcessorFast.from_pretrained(self.model_id)

    @torch.no_grad()
    def extract_token(self, image):
        """
        Run an image through the model and return CLS + patch tokens.

        Args:
            image: PIL Image or path string.

        Returns:
            dict with:
                "cls_token"    – (1, 1, D) tensor
                "patch_tokens" – (1, N, D) tensor
        """
        if self.model is None:
            raise RuntimeError("Call .load() before .extract_token()")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0:1, :]                           # (1, 1, D)
        num_reg = self.model.config.num_register_tokens              # typically 4
        patch_tokens = last_hidden[:, 1 + num_reg:, :]               # (1, N, D) — skip CLS + registers

        return {"cls_token": cls_token, "patch_tokens": patch_tokens}


# ── quick test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DINOv3 token extraction")
    parser.add_argument("--model", default="dinov3-vits16-pretrain",
                        help="Key from DINOV3_BACKBONES or HF model id")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    extractor = DINOv3Extractor(args.model, args.device)
    extractor.load()
    tokens = extractor.extract_token(args.image)

    print(f"CLS token   : {tokens['cls_token'].shape}")
    print(f"Patch tokens: {tokens['patch_tokens'].shape}")
