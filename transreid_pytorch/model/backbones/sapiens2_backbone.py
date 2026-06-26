import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sapiens.backbones.standalone.sapiens2 import Sapiens2


class Sapiens2Backbone(nn.Module):
    """Thin ReID-compatible wrapper around the standalone Sapiens2 ViT.

    Forward returns the CLS token as a (B, embed_dim) global descriptor.
    Weights are loaded from a local .safetensors path (cfg.MODEL.PRETRAIN_PATH).
    """

    def __init__(self, arch: str = "sapiens2_0.4b", img_size=(256, 128), ckpt_path: str = ""):
        super().__init__()
        self.model = Sapiens2(arch=arch, img_size=img_size, patch_size=16, out_type="raw")
        self.in_planes: int = self.model.embed_dims

        assert ckpt_path and os.path.isfile(ckpt_path), \
            f"Sapiens2 checkpoint not found: {ckpt_path!r}. Set MODEL.PRETRAIN_PATH in your config."

        state_dict = load_file(ckpt_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"[Sapiens2Backbone] Loaded {arch} from {ckpt_path}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model(x)[0]   # (B, num_tokens, embed_dim)
        return tokens[:, 0]         # CLS token → (B, embed_dim)
