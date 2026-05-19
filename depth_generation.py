"""
generate_depth_dataset.py
───────────────────────────────────────────────────────────────────────────────
Creates a COMPLETE depth-training dataset using:

1. glass_dataset.py
   → unified loading of Trans10K / GDD / RealFrames

2. inpaint_glass.py
   → Costanzino et al. glass in-painting augmentation

Pipeline
────────
For every sample:
    RGB image + binary glass mask
        ↓
    Generate N in-painted variants
        ↓
    Run frozen depth model on all variants
        ↓
    Median aggregation
        ↓
    Save:
        image/
        mask/
        depth/

Final folder structure:
───────────────────────
output_root/
    train/
        images/
        masks/
        depth/

    val/
        images/
        masks/
        depth/

    test/
        images/
        masks/
        depth/

Depth maps are saved as:
    .npy  → raw float32 depth
    .png  → visualisation

Usage
─────
python generate_depth_dataset.py ^
    --trans10k D:\datasets\Trans10k ^
    --gdd D:\datasets\GDD ^
    --real D:\datasets\RealFrames ^
    --output D:\datasets\glass_depth_dataset ^
    --encoder vitl ^
    --n 5
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm

# Your shared files
from datasets.glass_dataset import GlassDataset
from paint_glass import inpaint_glass, aggregate_virtual_depths


import sys
sys.path.append("D:/ToM_Depth_Estimation/Depth-Anything-V2")

# Depth Anything V2
from depth_anything_v2.dpt import DepthAnythingV2


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_depth_visual(depth, out_path):
    """
    Save normalized depth visualization PNG.
    """
    d = depth.copy()

    d = d - d.min()

    if d.max() > 0:
        d = d / d.max()

    d = (d * 255).astype(np.uint8)

    Image.fromarray(d).save(out_path)


def load_original_sample(dataset, idx):
    """
    Load original RGB + binary mask WITHOUT transforms.
    """

    item = dataset._dataset[idx]

    img_path, mask_path = item.samples[0]

    image = np.array(Image.open(img_path).convert("RGB"))

    mask_raw = np.array(Image.open(mask_path).convert("L"))
    mask = (mask_raw > 0).astype(np.uint8)

    return image, mask


# ─────────────────────────────────────────────────────────────────────────────
# Depth model
# ─────────────────────────────────────────────────────────────────────────────

def build_depth_model(encoder):

    cfg = ENCODER_CONFIGS[encoder]

    model = DepthAnythingV2(**cfg)

    ckpt = f"checkpoints/depth_anything_v2_{encoder}.pth"

    model.load_state_dict(
        torch.load(ckpt, map_location="cpu")
    )

    model = model.to(DEVICE).eval()

    return model


@torch.no_grad()
def predict_depth(model, image_rgb):

    depth = model.infer_image(image_rgb)

    return depth.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Split helper
# ─────────────────────────────────────────────────────────────────────────────

def get_split(idx, total):

    ratio = idx / total

    if ratio < 0.8:
        return "train"

    elif ratio < 0.9:
        return "val"

    return "test"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):

    # Dataset
    dataset = GlassDataset(
        trans10k_root=args.trans10k,
        gdd_root=args.gdd,
        real_root=args.real,
        trans10k_splits=["train", "validation", "test"],
    )

    total = len(dataset)

    print(f"\nTotal samples: {total}")

    # Depth model
    model = build_depth_model(args.encoder)

    # Create output folders
    for split in ["train", "val", "test"]:

        ensure_dir(Path(args.output) / split / "images")
        ensure_dir(Path(args.output) / split / "masks")
        ensure_dir(Path(args.output) / split / "depth")
        ensure_dir(Path(args.output) / split / "depth_vis")

    # Process
    for idx in tqdm(range(total)):

        sample = dataset[idx]

        stem = f"{sample['source']}_{sample['stem']}"

        split = get_split(idx, total)

        # ------------------------------------------------------------
        # Original image + mask
        # ------------------------------------------------------------

        image_t = sample["image"]
        mask_t = sample["mask"]

        image = image_t.permute(1, 2, 0).cpu().numpy()

        # de-normalize
        image = (
            image * np.array([0.229, 0.224, 0.225])
            + np.array([0.485, 0.456, 0.406])
        )

        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        mask = mask_t[0].cpu().numpy().astype(np.uint8)

        # ------------------------------------------------------------
        # In-painting
        # ------------------------------------------------------------

        variants = inpaint_glass(
            image=image,
            mask=mask,
            n=args.n,
            strategy=args.strategy,
        )

        # ------------------------------------------------------------
        # Predict depth for each variant
        # ------------------------------------------------------------

        depth_maps = []

        for variant in variants:

            depth = predict_depth(model, variant)

            depth_maps.append(depth)

        # ------------------------------------------------------------
        # Median aggregation
        # ------------------------------------------------------------

        virtual_depth = aggregate_virtual_depths(depth_maps)

        # ------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------

        image_path = (
            Path(args.output)
            / split
            / "images"
            / f"{stem}.png"
        )

        mask_path = (
            Path(args.output)
            / split
            / "masks"
            / f"{stem}.png"
        )

        depth_path = (
            Path(args.output)
            / split
            / "depth"
            / f"{stem}.npy"
        )

        depth_vis_path = (
            Path(args.output)
            / split
            / "depth_vis"
            / f"{stem}.png"
        )

        Image.fromarray(image).save(image_path)

        Image.fromarray(mask * 255).save(mask_path)

        np.save(depth_path, virtual_depth)

        save_depth_visual(virtual_depth, depth_vis_path)

    print("\nDONE")
    print(f"Dataset saved to: {args.output}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--trans10k", type=str, default=None)
    parser.add_argument("--gdd", type=str, default=None)
    parser.add_argument("--real", type=str, default=None)

    parser.add_argument("--output", type=str, required=True)

    parser.add_argument(
        "--encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl"]
    )

    parser.add_argument("--n", type=int, default=5)

    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "contrast", "palette"]
    )

    args = parser.parse_args()

    main(args)