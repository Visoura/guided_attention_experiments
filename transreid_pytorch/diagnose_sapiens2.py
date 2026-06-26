"""
Diagnostic script: verifies Sapiens2 weights load correctly and checks
feature discriminativeness on real Market-1501 query images.

Run from the transreid_pytorch directory:
    python3 diagnose_sapiens2.py
"""

import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file
from sapiens.backbones.standalone.sapiens2 import Sapiens2

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH  = "/home/wahdan/Desktop/fun/sapiens2/sapiens2_host/seg/sapiens2_0.4b_pretrain.safetensors"
QUERY_DIR  = "/home/wahdan/Desktop/GP/guided_attention_experiments/Data/Market-1501/query"
ARCH       = "sapiens2_0.4b"
IMG_SIZE   = (256, 128)   # (H, W) — same as test config
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_IDS      = 10           # how many identities to sample
N_PER_ID   = 3            # images per identity (for same-ID pairs)
# ──────────────────────────────────────────────────────────────────────────────

def pid_from_filename(fname):
    return int(os.path.basename(fname).split("_")[0])

def load_query_images(query_dir, n_ids, n_per_id):
    """Return {pid: [path, ...]} for the first n_ids that have >= n_per_id images."""
    by_pid = defaultdict(list)
    for f in sorted(os.listdir(query_dir)):
        if not f.endswith(".jpg"):
            continue
        pid = pid_from_filename(f)
        by_pid[pid].append(os.path.join(query_dir, f))

    selected = {}
    for pid, paths in by_pid.items():
        if len(paths) >= n_per_id:
            selected[pid] = paths[:n_per_id]
        if len(selected) == n_ids:
            break
    return selected

def build_transform(img_size):
    H, W = img_size
    # resample=2 in preprocessor_config.json → PIL BILINEAR (not BICUBIC)
    return transforms.Compose([
        transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_model(arch, img_size, ckpt_path, device):
    print(f"\n{'='*60}")
    print(f"  Loading Sapiens2 ({arch})")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Image size: {img_size}  Device: {device}")
    print(f"{'='*60}")

    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    model = Sapiens2(arch=arch, img_size=img_size, patch_size=16, out_type="raw")
    state_dict = load_file(ckpt_path)

    # ── Weight load check ────────────────────────────────────────────────────
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    total_params  = sum(p.numel() for p in model.parameters())
    loaded_params = sum(state_dict[k].numel() for k in state_dict if k in dict(model.named_parameters()))

    print(f"\n[Weight check]")
    print(f"  Model params  : {total_params:,}")
    print(f"  Loaded params : {loaded_params:,}")
    print(f"  Missing keys  : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"    First 5 missing  : {missing[:5]}")
    if unexpected:
        print(f"    First 5 unexpected: {unexpected[:5]}")

    # ── Sanity: check a weight isn't all zeros ────────────────────────────────
    sample_key = next(iter(state_dict))
    sample_w   = state_dict[sample_key]
    print(f"\n  Sample weight '{sample_key}': shape={tuple(sample_w.shape)}, "
          f"mean={sample_w.float().mean().item():.6f}, std={sample_w.float().std().item():.6f}")

    model = model.eval().to(device)
    print(f"\n  Model moved to {device}. Ready.\n")
    return model

@torch.no_grad()
def extract_features(model, paths, transform, device, num_extra_tokens):
    feats = {}
    for path in paths:
        img = Image.open(path).convert("RGB")
        x   = transform(img).unsqueeze(0).to(device)
        tokens = model(x)[0]                      # (1, N, D)
        cls_feat  = tokens[:, 0]                  # (1, D)
        patch_feat = tokens[:, num_extra_tokens:].mean(dim=1)  # (1, D)
        feats[path] = {
            "cls":   F.normalize(cls_feat,   dim=-1).cpu(),
            "patch": F.normalize(patch_feat, dim=-1).cpu(),
        }
    return feats

def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1).item()

def run_similarity_analysis(model, pid_to_paths, transform, device, num_extra_tokens):
    print(f"{'='*60}")
    print(f"  Similarity analysis on {len(pid_to_paths)} identities")
    print(f"{'='*60}\n")

    # Extract features for all selected images
    all_paths = [p for paths in pid_to_paths.values() for p in paths]
    feats = extract_features(model, all_paths, transform, device, num_extra_tokens)

    pids  = list(pid_to_paths.keys())
    same_cls,  diff_cls  = [], []
    same_patch, diff_patch = [], []

    # Same-ID pairs (different cameras of same person)
    for pid, paths in pid_to_paths.items():
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                same_cls.append(cosine(feats[paths[i]]["cls"],   feats[paths[j]]["cls"]))
                same_patch.append(cosine(feats[paths[i]]["patch"], feats[paths[j]]["patch"]))

    # Diff-ID pairs (different persons)
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a = pid_to_paths[pids[i]][0]
            b = pid_to_paths[pids[j]][0]
            diff_cls.append(cosine(feats[a]["cls"],   feats[b]["cls"]))
            diff_patch.append(cosine(feats[a]["patch"], feats[b]["patch"]))

    def stats(vals):
        t = torch.tensor(vals)
        return t.mean().item(), t.std().item(), t.min().item(), t.max().item()

    def row(label, same, diff):
        sm, ss, smin, smax = stats(same)
        dm, ds, dmin, dmax = stats(diff)
        gap = sm - dm
        print(f"  {label}")
        print(f"    same-ID  ({len(same):3d} pairs): mean={sm:.4f}  std={ss:.4f}  range=[{smin:.4f}, {smax:.4f}]")
        print(f"    diff-ID  ({len(diff):3d} pairs): mean={dm:.4f}  std={ds:.4f}  range=[{dmin:.4f}, {dmax:.4f}]")
        print(f"    gap (same - diff): {gap:.4f}  {'✓ positive' if gap > 0 else '✗ NEGATIVE — features not discriminative'}\n")

    row("CLS token   (index 0):", same_cls,   diff_cls)
    row("Patch tokens (mean-pool):", same_patch, diff_patch)

    # ── Per-identity consistency check ────────────────────────────────────────
    print(f"  Per-identity within-camera similarity (should be high if same person):")
    for pid, paths in list(pid_to_paths.items())[:5]:
        if len(paths) >= 2:
            s = cosine(feats[paths[0]]["cls"], feats[paths[1]]["cls"])
            print(f"    PID={pid:04d}: {os.path.basename(paths[0])} ↔ {os.path.basename(paths[1])}  cls_sim={s:.4f}")
    print()

def main():
    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(ARCH, IMG_SIZE, CKPT_PATH, DEVICE)
    num_extra_tokens = model.num_extra_tokens  # 9 = 1 CLS + 8 storage

    print(f"[Architecture]")
    print(f"  embed_dims      : {model.embed_dims}")
    print(f"  num_layers      : {model.num_layers}")
    print(f"  num_extra_tokens: {num_extra_tokens}  (1 CLS + {model.n_storage_tokens} storage)")
    print(f"  patch_size      : {model.patch_size}")
    H, W = IMG_SIZE
    n_patches = (H // model.patch_size) * (W // model.patch_size)
    print(f"  patch grid      : {H//model.patch_size}×{W//model.patch_size} = {n_patches} tokens at {IMG_SIZE}")
    print(f"  total tokens    : {num_extra_tokens + n_patches}\n")

    # ── Load query images ─────────────────────────────────────────────────────
    pid_to_paths = load_query_images(QUERY_DIR, N_IDS, N_PER_ID)
    print(f"[Dataset] Loaded {len(pid_to_paths)} identities × {N_PER_ID} images each from:")
    print(f"  {QUERY_DIR}\n")
    for pid, paths in pid_to_paths.items():
        print(f"  PID={pid:04d}: {[os.path.basename(p) for p in paths]}")
    print()

    transform = build_transform(IMG_SIZE)

    # ── Run analysis ──────────────────────────────────────────────────────────
    run_similarity_analysis(model, pid_to_paths, transform, DEVICE, num_extra_tokens)

    print(f"{'='*60}")
    print("  Interpretation:")
    print("  • gap > 0.05 → features are meaningfully discriminative (zero-shot)")
    print("  • gap < 0.01 → features collapse, fine-tuning required")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
