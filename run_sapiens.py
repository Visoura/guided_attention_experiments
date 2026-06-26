import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sapiens.backbones.standalone.sapiens2 import Sapiens2

# Build the model and load the pretrained checkpoint
model = Sapiens2(arch="sapiens2_0.4b", img_size=(1024, 768), patch_size=16).eval().cuda()  # img_size is (H, W)
ckpt_path = "/home/wahdan/Desktop/fun/sapiens2/sapiens2_host/seg/sapiens2_0.4b_pretrain.safetensors"
model.load_state_dict(load_file(ckpt_path))

# Forward pass on a single image (RGB; ImageNet normalization recommended)
x = torch.randn(1, 3, 1024, 768).cuda()

with torch.no_grad():
    features = model(x)[0]  # dense backbone features: (B, num_tokens, embed_dim)


print(f"model output: {x}")