"""
generate_complete_depth_dataset.py
───────────────────────────────────────────────────────────────────────────────
Creates FULL dataset with:
    train/
        images/
        masks/
        depths/
        depth_vis/

    validation/
        images/
        masks/
        depths/
        depth_vis/

    test/
        images/
        masks/
        depths/
        depth_vis/

Uses:
    • glass_dataset.py
    • inpaint_glass.py
    • Depth Anything V2

This version follows your WORKING generate_pseudo_labels.py logic.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

# =============================================================================
# ADD DEPTH ANYTHING V2 TO PATH
# =============================================================================

import sys
sys.path.append("D:/ihub-data/ToM_Depth_Estimation/Depth-Anything-V2")

# -----------------------------------------------------------------------------
# Depth Anything V2
# -----------------------------------------------------------------------------

from depth_anything_v2.dpt import DepthAnythingV2

# =============================================================================
# IMPORT SHARED CODE
# =============================================================================

from datasets.glass_dataset import (
    GlassDataset,
    _load_rgb
)

from paint_glass import (
    inpaint_glass,
    aggregate_virtual_depths
)

# =============================================================================
# CONFIG
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_TYPE = "vits"

CHECKPOINT_PATH = (
    r"D:/ihub-data/ToM_Depth_Estimation/Depth-Anything-V2/checkpoints/"
    r"depth_anything_v2_vits.pth"
)

# -----------------------------------------------------------------------------
# DATASET ROOTS
# -----------------------------------------------------------------------------

TRANS10K_ROOT = r"D:/ihub-data/ToM_Depth_Estimation/datasets/Trans10k"

GDD_ROOT = r"D:/ihub-data/ToM_Depth_Estimation/datasets/GDD"

REAL_ROOT = None

# -----------------------------------------------------------------------------
# OUTPUT ROOT
# -----------------------------------------------------------------------------

OUTPUT_ROOT = Path(
    r"D:/ihub-data/ToM_Depth_Estimation/pseudo_generation"
)

# -----------------------------------------------------------------------------
# INPAINTING
# -----------------------------------------------------------------------------

N_VARIANTS = 5

COLOR_STRATEGY = "random"

# =============================================================================
# MODEL CONFIGS
# =============================================================================

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384]
    },

    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768]
    },

    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024]
    }
}

# =============================================================================
# CREATE OUTPUT DIRS
# =============================================================================
for folder in [
    "images",
    "masks",
    "depths",
    "depth_vis"
]:
    (OUTPUT_ROOT / folder).mkdir(
    parents=True,
    exist_ok=True
)




# =============================================================================
# LOAD MODEL
# =============================================================================

print("\nLoading Depth Anything V2...\n")

model = DepthAnythingV2(
    **model_configs[MODEL_TYPE]
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE).eval()

print(f"Loaded Depth Anything V2 ({MODEL_TYPE})")
print(f"Using device: {DEVICE}\n")

# =============================================================================
# HELPERS
# =============================================================================

def normalize_depth_for_viz(depth):

    d_min = depth.min()
    d_max = depth.max()

    norm = (depth - d_min) / (d_max - d_min + 1e-8)

    return norm


def save_depth_visualization(
    image,
    mask,
    depth,
    save_path
):

    viz_depth = normalize_depth_for_viz(depth)

    plt.figure(figsize=(12, 4))

    # RGB
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("RGB")
    plt.axis("off")

    # MASK
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Glass Mask")
    plt.axis("off")

    # DEPTH
    plt.subplot(1, 3, 3)
    plt.imshow(viz_depth, cmap="inferno")
    plt.title("Pseudo Depth")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def infer_depth(image_rgb):

    with torch.no_grad():

        depth = model.infer_image(image_rgb)

    depth = depth.astype(np.float32)

    return depth

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_dataset():

    from datasets.glass_dataset import (
        GlassDataset,
        Trans10KDataset,
        GDDDataset,
        RealFramesDataset,
        _val_transforms,
        _load_rgb
    )

    datasets = []

    # ------------------------------------------------------------------
    # ALL TRANS10K
    # ------------------------------------------------------------------

    datasets.append(

        Trans10KDataset(
            root=TRANS10K_ROOT,
            splits=["train", "validation", "test"],
            transform=_val_transforms()
        )
    )

    # ------------------------------------------------------------------
    # ALL GDD
    # ------------------------------------------------------------------

    datasets.append(

        GDDDataset(
            root=GDD_ROOT,
            split="train",
            transform=_val_transforms()
        )
    )

    datasets.append(

        GDDDataset(
            root=GDD_ROOT,
            split="test",
            transform=_val_transforms()
        )
    )

    # ------------------------------------------------------------------
    # OPTIONAL REAL DATA
    # ------------------------------------------------------------------

    if REAL_ROOT is not None:

        datasets.append(

            RealFramesDataset(
                root=REAL_ROOT,
                transform=_val_transforms()
            )
        )

    dataset = GlassDataset(datasets)

    print(f"Samples found: {len(dataset)}\n")

    concat_dataset = dataset.dataset

    for idx in tqdm(range(len(dataset))):

        sample = dataset[idx]

        # ---------------------------------------------------------------------
        # Recover ORIGINAL PATHS from ConcatDataset
        # ---------------------------------------------------------------------

        cumulative_sizes = concat_dataset.cumulative_sizes

        dataset_idx = 0

        for c in cumulative_sizes:

            if idx < c:
                break

            dataset_idx += 1

        prev_cum = (
            0 if dataset_idx == 0
            else cumulative_sizes[dataset_idx - 1]
        )

        # local_idx = idx - prev_cum

        # subdataset = concat_dataset.datasets[dataset_idx]

        # img_path, mask_path = subdataset.samples[local_idx]

        local_idx = idx - prev_cum

        subdataset = concat_dataset.datasets[dataset_idx]

        dataset_name = type(subdataset).__name__
        stem = f"{dataset_name}_{sample['stem']}"

        img_path, mask_path = subdataset.samples[local_idx]

        # ---------------------------------------------------------------------
        # LOAD IMAGE + MASK
        # ---------------------------------------------------------------------

        image = _load_rgb(img_path)

        mask_raw = np.array(
        Image.open(mask_path).convert("L"),
        dtype=np.uint8
    )

        # Fix mismatched shapes
        if image.shape[:2] != mask_raw.shape[:2]:

            print("\nFIXING MASK")
            print("Image:", img_path)
            print("Mask :", mask_path)

            mask_raw = cv2.resize(
                mask_raw,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        mask = (mask_raw > 0).astype(np.uint8)

        # ---------------------------------------------------------------------
        # INPAINTING
        # ---------------------------------------------------------------------

        variants = inpaint_glass(
            image=image,
            mask=mask,
            n=N_VARIANTS,
            strategy=COLOR_STRATEGY,
            seed=idx
        )

        # ---------------------------------------------------------------------
        # DEPTH PREDICTIONS
        # ---------------------------------------------------------------------

        depth_predictions = []

        for variant in variants:

            depth = infer_depth(variant)

            depth_predictions.append(depth)

        # ---------------------------------------------------------------------
        # MEDIAN AGGREGATION
        # ---------------------------------------------------------------------

        pseudo_depth = aggregate_virtual_depths(
            depth_predictions
        )

        # ---------------------------------------------------------------------
        # SAVE RGB
        # ---------------------------------------------------------------------

        image_save_path = (
            OUTPUT_ROOT /
            "images" /
            f"{stem}.png"
        )

        Image.fromarray(image).save(
            image_save_path
        )

        # ---------------------------------------------------------------------
        # SAVE MASK
        # ---------------------------------------------------------------------

        mask_save_path = (
            OUTPUT_ROOT /
            "masks" /
            f"{stem}.png"
        )

        Image.fromarray(
            (mask * 255).astype(np.uint8)
        ).save(mask_save_path)

        # ---------------------------------------------------------------------
        # SAVE DEPTH (.npy)
        # ---------------------------------------------------------------------

        depth_save_path = (
            OUTPUT_ROOT /
            "depths" /
            f"{stem}.npy"
        )

        np.save(
            depth_save_path,
            pseudo_depth
        )

        # ---------------------------------------------------------------------
        # SAVE VISUALIZATION
        # ---------------------------------------------------------------------

        viz_save_path = (
            OUTPUT_ROOT /
            "depth_vis" /
            f"{stem}.png"
        )

        # save_depth_visualization(
        #     image=image,
        #     mask=mask,
        #     depth=pseudo_depth,
        #     save_path=viz_save_path
        # )



# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    process_dataset()

    print("\nPseudo depth dataset generation completed.\n")