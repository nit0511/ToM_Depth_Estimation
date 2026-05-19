<<<<<<< HEAD
# generate_pseudo_labels.py

import os
=======
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
>>>>>>> e58d14c60f531b7117191abfbfb030ec6b6a5d50
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

<<<<<<< HEAD
import sys
sys.path.append("D:/ToM_Depth_Estimation/Depth-Anything-V2")
=======
# =============================================================================
# ADD DEPTH ANYTHING V2 TO PATH
# =============================================================================

import sys
sys.path.append("D:/ihub-data/ToM_Depth_Estimation/Depth-Anything-V2")
>>>>>>> e58d14c60f531b7117191abfbfb030ec6b6a5d50

# -----------------------------------------------------------------------------
# Depth Anything V2
# -----------------------------------------------------------------------------

from depth_anything_v2.dpt import DepthAnythingV2

<<<<<<< HEAD

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_TYPE = "vits"     # small model

CHECKPOINT_PATH = r"D:/ToM_Depth_Estimation/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"

IMAGE_DIR = r"D:/ToM_Depth_Estimation/datasets/Trans10k/train/images"
MASK_DIR  = r"D:/ToM_Depth_Estimation/datasets/Trans10k/train/masks"

OUTPUT_DEPTH_DIR = r"pseudo_generation/outputs/pseudo_depth"
OUTPUT_VIZ_DIR   = r"pseudo_generation/outputs/visualizations"

os.makedirs(OUTPUT_DEPTH_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)

N_VARIANTS = 5

IMG_EXTS = [".jpg", ".jpeg", ".png"]


# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------
=======
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
>>>>>>> e58d14c60f531b7117191abfbfb030ec6b6a5d50

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384]
<<<<<<< HEAD
    }
}

model = DepthAnythingV2(**model_configs[MODEL_TYPE])
=======
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
>>>>>>> e58d14c60f531b7117191abfbfb030ec6b6a5d50

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE).eval()

<<<<<<< HEAD
print(f"\nLoaded Depth Anything V2 ({MODEL_TYPE})")
print(f"Using device: {DEVICE}\n")


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def load_rgb(path):

    img = np.array(
        Image.open(path).convert("RGB"),
        dtype=np.uint8
    )

    return img


def load_mask(path, target_shape):

    mask = np.array(
        Image.open(path).convert("L"),
        dtype=np.uint8
    )

    # resize mask to image size
    mask = cv2.resize(
        mask,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # any non-zero -> glass
    mask = (mask > 0).astype(np.uint8)

    return mask


def random_recolor_glass(image, mask):

    recolored = image.copy()

    random_color = np.random.randint(
        40,
        215,
        size=3,
        dtype=np.uint8
    )

    recolored[mask == 1] = random_color

    return recolored


def infer_depth(image_rgb):

    depth = model.infer_image(image_rgb)

    depth = depth.astype(np.float32)

    return depth

=======
print(f"Loaded Depth Anything V2 ({MODEL_TYPE})")
print(f"Using device: {DEVICE}\n")

# =============================================================================
# HELPERS
# =============================================================================
>>>>>>> e58d14c60f531b7117191abfbfb030ec6b6a5d50

def normalize_depth_for_viz(depth):

    d_min = depth.min()
    d_max = depth.max()

    norm = (depth - d_min) / (d_max - d_min + 1e-8)

    return norm


<<<<<<< HEAD
# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

image_paths = []

for ext in IMG_EXTS:
    image_paths.extend(
        sorted(Path(IMAGE_DIR).glob(f"*{ext}"))
    )

print(f"Found {len(image_paths)} images\n")

for img_path in tqdm(image_paths):

    stem = img_path.stem

    
    mask_path = Path(MASK_DIR) / f"{stem}_mask.png"

    if not mask_path.exists():
        print(f"Mask missing for {stem}")
        continue

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    image = load_rgb(img_path)

    mask = load_mask(
    mask_path,
    image.shape[:2]
)

    # -------------------------------------------------------------------------
    # MULTI-COLOR IN-PAINTING + DEPTH INFERENCE
    # -------------------------------------------------------------------------

    depth_predictions = []

    for n in range(N_VARIANTS):

        recolored = random_recolor_glass(
            image,
            mask
        )

        with torch.no_grad():

            depth = infer_depth(recolored)

        depth_predictions.append(depth)

    # -------------------------------------------------------------------------
    # MEDIAN AGGREGATION
    # -------------------------------------------------------------------------

    depth_stack = np.stack(
        depth_predictions,
        axis=0
    )

    pseudo_depth = np.median(
        depth_stack,
        axis=0
    ).astype(np.float32)

    # -------------------------------------------------------------------------
    # SAVE DEPTH
    # -------------------------------------------------------------------------

    save_path = Path(OUTPUT_DEPTH_DIR) / f"{stem}.npy"

    np.save(save_path, pseudo_depth)

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------

    viz_depth = normalize_depth_for_viz(pseudo_depth)
=======
def save_depth_visualization(
    image,
    mask,
    depth,
    save_path
):

    viz_depth = normalize_depth_for_viz(depth)
>>>>>>> e58d14c60f531b7117191abfbfb030ec6b6a5d50

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

<<<<<<< HEAD
    viz_save_path = Path(OUTPUT_VIZ_DIR) / f"{stem}.png"

    plt.savefig(viz_save_path)

    plt.close()

print("\nPseudo-label generation completed.\n")
=======
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
>>>>>>> e58d14c60f531b7117191abfbfb030ec6b6a5d50
