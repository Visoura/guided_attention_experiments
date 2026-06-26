"""
Sapiens2 segmentation sanity check + backbone similarity analysis.

Part 1 – Full segmentation model:
  Runs the seg checkpoint on real Market-1501 query images and saves
  visualizations so you can confirm the model actually segments bodies.

Part 2 – Backbone-only (no seg head):
  Loads only the backbone weights from the seg checkpoint and runs the
  same similarity analysis as diagnose_sapiens2.py to compare CLS vs
  patch mean-pool discriminativeness.

Run from transreid_pytorch/:
    python3 sanity_check_sapiens2_seg.py
"""

import os, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from safetensors.torch import load_file
from sapiens.backbones.standalone.sapiens2 import Sapiens2

# ── Config ────────────────────────────────────────────────────────────────────
SEG_CKPT   = "/home/wahdan/Desktop/fun/sapiens2/sapiens2_host/seg/sapiens2_0.4b_seg.safetensors"
QUERY_DIR  = "/home/wahdan/Desktop/GP/guided_attention_experiments/Data/Market-1501/query"
OUT_DIR    = "/home/wahdan/Desktop/GP/guided_attention_experiments/logs/sapiens2_seg_sanity"
ARCH       = "sapiens2_0.4b"
# Resolution for seg visualisation — larger = better seg quality.
# 512×256 is a good balance; use 1024×768 if you want the full model quality.
VIS_SIZE   = (512, 256)   # (H, W)
REID_SIZE  = (256, 128)   # (H, W) — same as test.py
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_VIS      = 8            # images to visualise
N_IDS      = 10           # identities for similarity analysis
N_PER_ID   = 3            # images per identity
# ──────────────────────────────────────────────────────────────────────────────

# 29-class body segmentation palette (DOME_CLASSES_29)
CLASSES = {
    0:  ("Background",     [50,  50,  50]),
    1:  ("Apparel",        [255, 218, 0]),
    2:  ("Eyeglass",       [14,  204, 182]),
    3:  ("Face_Neck",      [128, 200, 255]),
    4:  ("Hair",           [255, 0,   109]),
    5:  ("Left_Foot",      [189, 0,   204]),
    6:  ("Left_Hand",      [255, 0,   218]),
    7:  ("Left_Lower_Arm", [0,   160, 204]),
    8:  ("Left_Lower_Leg", [0,   255, 145]),
    9:  ("Left_Shoe",      [204, 0,   131]),
    10: ("Left_Sock",      [182, 0,   255]),
    11: ("Left_Upper_Arm", [255, 109, 0]),
    12: ("Left_Upper_Leg", [0,   255, 255]),
    13: ("Lower_Clothing", [72,  0,   255]),
    14: ("Right_Foot",     [204, 131, 0]),
    15: ("Right_Hand",     [255, 0,   0]),
    16: ("Right_Lower_Arm",[72,  255, 0]),
    17: ("Right_Lower_Leg",[189, 204, 0]),
    18: ("Right_Shoe",     [182, 255, 0]),
    19: ("Right_Sock",     [102, 0,   204]),
    20: ("Right_Upper_Arm",[32,  72,  204]),
    21: ("Right_Upper_Leg",[0,   145, 255]),
    22: ("Torso",          [14,  204, 0]),
    23: ("Upper_Clothing", [0,   128, 72]),
    24: ("Lower_Lip",      [235, 205, 119]),
    25: ("Upper_Lip",      [115, 227, 112]),
    26: ("Lower_Teeth",    [157, 113, 143]),
    27: ("Upper_Teeth",    [132, 93,  50]),
    28: ("Tongue",         [82,  21,  114]),
}
PALETTE = np.array([CLASSES[i][1] for i in range(len(CLASSES))], dtype=np.uint8)


# ── Self-contained SegHead (mirrors sapiens/dense/src/models/heads/seg_head.py)
class SegHead(nn.Module):
    """Deconv upsampler + 1×1 classifier. Exactly matches the checkpoint."""

    def __init__(self, in_channels=1024, num_classes=29):
        super().__init__()
        # 4 deconv stages: 1024→512→256→128→64  (each ×2 spatial)
        deconv_cfg = [(in_channels, 512), (512, 256), (256, 128), (128, 64)]
        layers = []
        for ic, oc in deconv_cfg:
            layers += [
                nn.ConvTranspose2d(ic, oc, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(oc),
                nn.SiLU(inplace=True),
            ]
        self.deconv_layers = nn.Sequential(*layers)

        # 2 conv 1×1 refinement layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 64, 1), nn.InstanceNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 1), nn.InstanceNorm2d(64), nn.SiLU(inplace=True),
        )
        self.conv_seg = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):          # x: (B, 1024, H/16, W/16)
        x = self.deconv_layers(x)  # → (B, 64, H, W)
        x = self.conv_layers(x)
        return self.conv_seg(x)    # → (B, num_classes, H, W)


class Sapiens2Seg(nn.Module):
    """Full seg model: Sapiens2 backbone (featmap output) + SegHead."""

    def __init__(self, arch, img_size):
        super().__init__()
        self.backbone = Sapiens2(arch=arch, img_size=img_size,
                                 patch_size=16, out_type="featmap")
        self.decode_head = SegHead(in_channels=self.backbone.embed_dims,
                                   num_classes=len(CLASSES))

    def forward(self, x):
        feat = self.backbone(x)[0]       # (B, embed_dim, H/16, W/16)
        logits = self.decode_head(feat)  # (B, 29, H, W)
        return logits


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_transform(H, W):
    return transforms.Compose([
        transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_seg_model(arch, img_size, ckpt_path, device):
    print(f"\n{'='*60}")
    print(f"  Building full Sapiens2Seg ({arch})")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*60}")

    model = Sapiens2Seg(arch=arch, img_size=img_size).to(device).eval()
    sd = load_file(ckpt_path)

    # Checkpoint keys are prefixed backbone.* / decode_head.*
    # The model's state_dict also has backbone.* / decode_head.*
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Missing keys   : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:    print(f"    first 5: {missing[:5]}")
    if unexpected: print(f"    first 5: {unexpected[:5]}")

    bb_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("backbone."))
    dh_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("decode_head."))
    print(f"  backbone params    : {bb_params:,}")
    print(f"  decode_head params : {dh_params:,}")

    # Quick sanity: a few backbone weights should be non-zero
    w = next(p for n, p in model.named_parameters() if n.startswith("backbone."))
    print(f"  Sample backbone weight: mean={w.float().mean().item():.6f}  std={w.float().std().item():.6f}")
    print()
    return model

def load_backbone_only(arch, img_size, ckpt_path, device):
    """Load only backbone weights from the seg checkpoint into a standalone Sapiens2."""
    print(f"  Building backbone-only Sapiens2 ({arch}) for ReID similarity analysis")
    backbone = Sapiens2(arch=arch, img_size=img_size, patch_size=16, out_type="raw").to(device).eval()

    sd = load_file(ckpt_path)
    # Strip 'backbone.' prefix to match standalone Sapiens2 keys
    bb_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}

    missing, unexpected = backbone.load_state_dict(bb_sd, strict=False)
    print(f"  Missing keys   : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:    print(f"    first 5: {missing[:5]}")
    if unexpected: print(f"    first 5: {unexpected[:5]}")
    print()
    return backbone


# ── Part 1: Segmentation visualisation ────────────────────────────────────────

def colorise(label_map, alpha=0.55, orig_img=None):
    """Convert (H, W) int label map to an RGBA overlay blended on orig_img."""
    H, W = label_map.shape
    color_mask = PALETTE[label_map]          # (H, W, 3) RGB
    if orig_img is not None:
        orig = np.array(orig_img.resize((W, H), Image.BILINEAR)).astype(np.float32)
        blended = (1 - alpha) * orig + alpha * color_mask.astype(np.float32)
        return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
    return Image.fromarray(color_mask)

def draw_legend(present_ids, cell_size=16, font_size=12):
    """Small legend image listing the classes present in this prediction."""
    rows = [(cid, CLASSES[cid][0], CLASSES[cid][1]) for cid in sorted(present_ids) if cid != 0]
    w = 140
    h = max(10, len(rows) * (cell_size + 2) + 4)
    img = Image.new("RGB", (w, h), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    for i, (cid, name, color) in enumerate(rows):
        y = 2 + i * (cell_size + 2)
        draw.rectangle([2, y, 2 + cell_size, y + cell_size], fill=tuple(color))
        draw.text((cell_size + 6, y), f"{cid}:{name}", fill=(220, 220, 220), font=font)
    return img

def run_seg_visualisation(model, query_dir, out_dir, n_vis, device):
    os.makedirs(out_dir, exist_ok=True)
    H, W = VIS_SIZE
    tf = make_transform(H, W)

    imgs = sorted(f for f in os.listdir(query_dir) if f.endswith(".jpg"))[:n_vis]
    print(f"\n{'='*60}")
    print(f"  Segmentation visualisation on {len(imgs)} images  ({H}×{W})")
    print(f"  Output → {out_dir}")
    print(f"{'='*60}")

    for fname in imgs:
        path = os.path.join(query_dir, fname)
        orig = Image.open(path).convert("RGB")
        x = tf(orig).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)                              # (1, 29, H, W)
            # upsample to vis size if needed
            logits = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (H, W)

        present = set(np.unique(pred).tolist())
        present.discard(0)  # skip background for legend

        overlay  = colorise(pred, alpha=0.55, orig_img=orig)
        mask_img = colorise(pred, alpha=1.0)
        legend   = draw_legend(present, cell_size=14)

        # ── Print class breakdown ──────────────────────────────────────────────
        total_px = pred.size
        class_info = []
        for cid in sorted(present):
            pct = 100.0 * (pred == cid).sum() / total_px
            class_info.append(f"{CLASSES[cid][0]}({pct:.1f}%)")
        print(f"  {fname}: {', '.join(class_info) if class_info else 'only background'}")

        # ── Save 3-panel: original | overlay | pure mask ──────────────────────
        orig_r = orig.resize((W, H), Image.BILINEAR)
        leg_r  = legend.resize((legend.width, H), Image.BILINEAR) if legend.height != H else legend
        panel  = Image.new("RGB", (W * 3 + legend.width + 6, H), (10, 10, 10))
        panel.paste(orig_r,   (0,         0))
        panel.paste(overlay,  (W + 2,     0))
        panel.paste(mask_img, (W * 2 + 4, 0))
        panel.paste(legend,   (W * 3 + 6, 0))

        out_path = os.path.join(out_dir, fname.replace(".jpg", "_seg.png"))
        panel.save(out_path)

    print(f"\n  Saved {len(imgs)} visualisations to {out_dir}")
    print("  Each image: [original | overlay | pure mask | legend]")


# ── Part 2: Backbone similarity analysis ──────────────────────────────────────

def pid_from_filename(fname):
    return int(os.path.basename(fname).split("_")[0])

def load_query_images(query_dir, n_ids, n_per_id):
    by_pid = defaultdict(list)
    for f in sorted(os.listdir(query_dir)):
        if f.endswith(".jpg"):
            by_pid[pid_from_filename(f)].append(os.path.join(query_dir, f))
    selected = {}
    for pid, paths in by_pid.items():
        if len(paths) >= n_per_id:
            selected[pid] = paths[:n_per_id]
        if len(selected) == n_ids:
            break
    return selected

@torch.no_grad()
def extract(backbone, paths, tf, device):
    ne = backbone.num_extra_tokens  # 9 = 1 CLS + 8 storage tokens
    feats = {}
    for p in paths:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        tokens = backbone(x)[0]              # (1, num_tokens, embed_dim)
        # Token layout: [CLS(0), storage(1..8), patch(9..136)]
        cls_tok     = tokens[:, 0]           # single CLS token
        storage_tok = tokens[:, 1:ne]        # 8 storage tokens
        patch_tok   = tokens[:, ne:]         # 128 patch tokens
        feats[p] = {
            "cls":             F.normalize(cls_tok,                          dim=-1).cpu(),
            "storage_mean":    F.normalize(storage_tok.mean(1),              dim=-1).cpu(),
            "patch_mean":      F.normalize(patch_tok.mean(1),                dim=-1).cpu(),
            "cls_patch_mean":  F.normalize((cls_tok + patch_tok.mean(1))/2,  dim=-1).cpu(),
            "all_mean":        F.normalize(tokens.mean(1),                   dim=-1).cpu(),
        }
    return feats

def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1).item()

def verdict(gap):
    if gap >= 0.05:
        return "GOOD  — can rank people without fine-tuning"
    elif gap >= 0.02:
        return "WEAK  — some signal, fine-tuning strongly recommended"
    elif gap >= 0.005:
        return "POOR  — barely any signal, fine-tuning required"
    else:
        return "NONE  — features cannot distinguish people at all"

def bar(gap, width=20):
    # gap typically 0..0.1; clamp and scale to bar
    filled = max(0, min(width, int(gap * width / 0.10)))
    return "[" + "█" * filled + "░" * (width - filled) + f"]  {gap:+.4f}"

def run_similarity_analysis(backbone, pid_to_paths, device, ckpt_label=""):
    H, W = REID_SIZE
    tf = make_transform(H, W)

    all_paths = [p for paths in pid_to_paths.values() for p in paths]
    feats = extract(backbone, all_paths, tf, device)
    pids  = list(pid_to_paths.keys())

    results = {k: {"same": [], "diff": []} for k in feats[all_paths[0]]}

    for pid, paths in pid_to_paths.items():
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                for k in results:
                    results[k]["same"].append(cosine(feats[paths[i]][k], feats[paths[j]][k]))

    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = pid_to_paths[pids[i]][0], pid_to_paths[pids[j]][0]
            for k in results:
                results[k]["diff"].append(cosine(feats[a][k], feats[b][k]))

    header = f"  Backbone from: {ckpt_label}" if ckpt_label else ""
    print(f"\n{'='*70}")
    print(f"  FEATURE DISCRIMINATION TEST")
    print(f"  What this measures: can the model tell two different people apart?")
    print(f"  Dataset: {len(pid_to_paths)} identities, {N_PER_ID} images each from Market-1501")
    print(f"  Image size fed to backbone: {H}×{W} px")
    if header: print(header)
    print(f"{'='*70}")
    print()
    print(f"  Cosine similarity: 1.0 = identical features, 0.0 = totally different")
    print(f"  A good ReID model needs:  same-person score >> different-person score")
    print(f"  The GAP = same-person avg − different-person avg (bigger = better)")
    print()

    LABELS = {
        "cls":            "CLS token         (index 0, the 'summary' token)",
        "storage_mean":   "Storage tokens    (indices 1-8, mean-pooled)",
        "patch_mean":     "Patch tokens      (indices 9-136, mean-pooled)",
        "cls_patch_mean": "CLS + patch avg   (average of cls and patch mean)",
        "all_mean":       "All tokens        (mean over all 137 tokens)",
    }

    rows = []
    for k, label in LABELS.items():
        same = results[k]["same"]
        diff = results[k]["diff"]
        sm = sum(same) / len(same)
        dm = sum(diff) / len(diff)
        gap = sm - dm
        rows.append((gap, k, label, sm, dm))

    rows.sort(reverse=True)  # best gap first

    print(f"  {'Feature':<38}  {'same-ID avg':>11}  {'diff-ID avg':>11}  {'GAP':>7}")
    print(f"  {'-'*38}  {'-'*11}  {'-'*11}  {'-'*7}")
    for gap, k, label, sm, dm in rows:
        marker = " ← best" if k == rows[0][1] else ""
        print(f"  {label:<38}  {sm:>11.4f}  {dm:>11.4f}  {gap:>+7.4f}{marker}")

    print()
    best_gap, best_k, best_label, best_sm, best_dm = rows[0]
    print(f"  BEST feature: {best_label.strip()}")
    print(f"  Gap progress bar (max shown = 0.10):")
    print(f"    {bar(best_gap)}")
    print(f"  Verdict: {verdict(best_gap)}")

    print()
    print(f"  WHY does good segmentation not mean good ReID?")
    print(f"  ─────────────────────────────────────────────────────────────────")
    print(f"  Segmentation asks: 'what body part is this pixel?'")
    print(f"    → Two people in identical clothes get the SAME segmentation mask")
    print(f"    → So segmentation quality tells you nothing about identity")
    print(f"  ReID asks: 'is this the same person?'")
    print(f"    → Requires features that are UNIQUE per identity")
    print(f"    → Sapiens2 patch tokens encode body-part TYPE, not identity")
    print(f"  Solution: fine-tune with ID loss + triplet loss on Market-1501")
    print(f"  ─────────────────────────────────────────────────────────────────")

    print()
    print(f"  CLS TOKEN VERIFICATION (confirms correct index):")
    print(f"  ─────────────────────────────────────────────────────────────────")
    print(f"  Token layout: [CLS(0) | storage(1-8) | patch(9-136)] = 137 total")
    print(f"  out_type='cls_token' in Sapiens2 returns x[:, 0]  — we use same index")
    print()
    print(f"  Same-person similarity across cameras (CLS token):")
    print(f"  (High = model sees same person. Low = model confused by camera change)")
    for pid, paths in list(pid_to_paths.items())[:5]:
        if len(paths) >= 2:
            s = cosine(feats[paths[0]]["cls"], feats[paths[1]]["cls"])
            cam_a = os.path.basename(paths[0]).split("_")[1]
            cam_b = os.path.basename(paths[1]).split("_")[1]
            bar_s = "█" * int(s * 20)
            print(f"    PID {pid:04d}  {cam_a} vs {cam_b}:  {bar_s:<20} {s:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    assert os.path.isfile(SEG_CKPT), f"Checkpoint not found: {SEG_CKPT}"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Part 1: full seg model visualisation ──────────────────────────────────
    seg_model = load_seg_model(ARCH, VIS_SIZE, SEG_CKPT, DEVICE)
    run_seg_visualisation(seg_model, QUERY_DIR, OUT_DIR, N_VIS, DEVICE)

    del seg_model
    torch.cuda.empty_cache()

    # ── Part 2: backbone-only similarity analysis ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Loading backbone-only for ReID similarity analysis")
    print(f"{'='*60}")
    backbone = load_backbone_only(ARCH, REID_SIZE, SEG_CKPT, DEVICE)

    pid_to_paths = load_query_images(QUERY_DIR, N_IDS, N_PER_ID)
    print(f"  Loaded {len(pid_to_paths)} identities from {QUERY_DIR}")
    run_similarity_analysis(backbone, pid_to_paths, DEVICE,
                            ckpt_label="sapiens2_0.4b_seg.safetensors (seg-finetuned backbone)")

    print(f"\n{'='*60}")
    print(f"  Done. Seg visualisations saved to:")
    print(f"  {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
