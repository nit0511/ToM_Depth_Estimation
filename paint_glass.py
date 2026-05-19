"""
inpaint_glass.py
────────────────────────────────────────────────────────────────────────────────
In-painting function from Costanzino et al. ICCV 2023
"Learning Depth Estimation for Transparent and Mirror Surfaces"

Core idea (Eq. 2–3 from paper):
  1. For each image + binary glass mask, sample N random RGB colors
  2. Replace glass pixels with each color → N augmented images
  3. Run frozen depth model on all N variants
  4. Take per-pixel median → virtual depth pseudo-label

This file implements steps 1–2 (the in-painting itself).
See generate_pseudo_labels.py for steps 3–4.

Usage
─────
    from inpaint_glass import inpaint_glass, batch_inpaint

    # Single image
    inpainted_list = inpaint_glass(image, mask, n=5)

    # Batch (for pseudo-label generation loop)
    results = batch_inpaint(image_paths, mask_paths, n=5, out_dir="pseudo/inpainted")
"""

import os
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Colour sampling
# ─────────────────────────────────────────────────────────────────────────────

# Fixed seed used by paper for reproducibility during pseudo-label generation.
# Change only if you want different colors across runs.
PAPER_SEED = 0

def _sample_colors(
    n: int,
    image: np.ndarray,
    mask: np.ndarray,
    strategy: str = "random",
    seed: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """
    Sample N RGB colors for in-painting.

    Args:
        n        : Number of colors to sample.
        image    : RGB image [H, W, 3] uint8 — used for contrast-aware sampling.
        mask     : Binary mask [H, W] uint8 {0, 1}.
        strategy : Sampling strategy:
                     "random"    — uniform random RGB (paper default)
                     "contrast"  — avoids colors similar to glass-region border
                     "palette"   — evenly spaced hues for maximal diversity
        seed     : Random seed for reproducibility. None = non-deterministic.

    Returns:
        List of N (R, G, B) tuples, each in range [0, 255].
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if strategy == "random":
        # Paper Eq.2: uniformly sample RGB color c_i for each variant
        colors = [
            (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            for _ in range(n)
        ]

    elif strategy == "contrast":
        # Avoid colors too similar to the mean of the glass-border region.
        # Helps with the edge case described in paper Sec 3: "certain colors
        # might result ineffective" (e.g. white on white wall).
        border_kernel = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15))) - mask
        border_pixels = image[border_kernel > 0]
        if len(border_pixels) > 0:
            border_mean = border_pixels.mean(axis=0)   # [R, G, B]
        else:
            border_mean = np.array([128, 128, 128])

        colors = []
        attempts = 0
        while len(colors) < n and attempts < n * 20:
            c = np_rng.integers(0, 256, size=3)
            # Reject if perceptual distance to border is too small
            dist = np.linalg.norm(c.astype(float) - border_mean)
            if dist > 60:   # threshold in RGB space
                colors.append(tuple(int(v) for v in c))
            attempts += 1
        # Fill remaining slots with random if contrast check keeps failing
        while len(colors) < n:
            c = np_rng.integers(0, 256, size=3)
            colors.append(tuple(int(v) for v in c))

    elif strategy == "palette":
        # Evenly-spaced hues in HSV → convert to RGB
        # Ensures maximal color diversity across N variants
        colors = []
        for i in range(n):
            hue = int((i / n) * 180)           # OpenCV hue: 0-179
            sat = rng.randint(150, 255)
            val = rng.randint(100, 255)
            hsv_pixel = np.array([[[hue, sat, val]]], dtype=np.uint8)
            rgb_pixel  = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)
            r, g, b = int(rgb_pixel[0, 0, 0]), int(rgb_pixel[0, 0, 1]), int(rgb_pixel[0, 0, 2])
            colors.append((r, g, b))
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: 'random', 'contrast', 'palette'")

    return colors


# ─────────────────────────────────────────────────────────────────────────────
# Core in-painting function
# ─────────────────────────────────────────────────────────────────────────────

def inpaint_glass(
    image: np.ndarray,
    mask: np.ndarray,
    n: int = 5,
    strategy: str = "random",
    seed: Optional[int] = PAPER_SEED,
    return_colors: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[Tuple[int, int, int]]]]:
    """
    In-paint glass regions with N different solid colors.

    Implements Eq. 2 from Costanzino et al. ICCV 2023:
        Ĩ_k(p) = c       if M_k(p) = 1   (glass pixel)
        Ĩ_k(p) = I_k(p)  otherwise        (keep original)

    Args:
        image         : RGB image as numpy array [H, W, 3] uint8.
        mask          : Binary glass mask [H, W] uint8 with values {0, 1}.
                        1 = glass pixel, 0 = background.
        n             : Number of in-painted variants to generate. Paper uses 5.
        strategy      : Color sampling strategy ("random", "contrast", "palette").
                        "random" matches the paper exactly.
        seed          : Random seed. Paper fixes seed=0 for reproducibility.
        return_colors : If True, also return the sampled colors.

    Returns:
        List of N in-painted images, each [H, W, 3] uint8.
        If return_colors=True: (list_of_images, list_of_colors).

    Raises:
        ValueError : If image/mask shapes are incompatible.
    """
    # ── Input validation ──────────────────────────────────────────────────
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must be [H, W, 3] RGB array, got shape {image.shape}"
        )
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be [H, W] array, got shape {mask.shape}"
        )
    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"image spatial dims {image.shape[:2]} != mask dims {mask.shape}"
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    # Ensure mask is binary {0, 1}
    binary_mask = (mask > 0).astype(np.uint8)   # handles 0/255 or 0/1 input

    if binary_mask.sum() == 0:
        # No glass pixels — return N copies of original image unchanged
        copies = [image.copy() for _ in range(n)]
        if return_colors:
            return copies, [(0, 0, 0)] * n
        return copies

    # ── Sample N colors ───────────────────────────────────────────────────
    colors = _sample_colors(n, image, binary_mask, strategy=strategy, seed=seed)

    # ── Generate N in-painted images ──────────────────────────────────────
    # Build boolean mask for indexing: [H, W] bool
    glass_pixels = binary_mask.astype(bool)   # True wherever glass

    inpainted_images = []
    for color in colors:
        # Copy original image; replace glass pixels with solid color
        variant = image.copy()
        variant[glass_pixels, 0] = color[0]   # R channel
        variant[glass_pixels, 1] = color[1]   # G channel
        variant[glass_pixels, 2] = color[2]   # B channel
        inpainted_images.append(variant)

    if return_colors:
        return inpainted_images, colors
    return inpainted_images


# ─────────────────────────────────────────────────────────────────────────────
# Median depth aggregation  (Eq. 3 from paper)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_virtual_depths(depth_maps: List[np.ndarray]) -> np.ndarray:
    """
    Compute per-pixel median across N depth maps.

    Implements Eq. 3 from Costanzino et al.:
        D̃*_k = median{ Ψ(Ĩ^i_k), i ∈ [0, N-1] }

    Args:
        depth_maps : List of N depth arrays, each [H, W] float32.

    Returns:
        Median depth map [H, W] float32.
    """
    if not depth_maps:
        raise ValueError("depth_maps list is empty")
    stacked = np.stack(depth_maps, axis=0)   # [N, H, W]
    return np.median(stacked, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Batch utility — for pseudo-label generation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def batch_inpaint(
    image_paths: List[Path],
    mask_paths: List[Path],
    n: int = 5,
    strategy: str = "random",
    out_dir: Optional[str] = None,
    seed: Optional[int] = PAPER_SEED,
) -> List[Tuple[List[np.ndarray], List[Tuple[int, int, int]]]]:
    """
    Run in-painting over a list of (image, mask) path pairs.

    Used in the pseudo-label generation loop (generate_pseudo_labels.py).

    Args:
        image_paths : List of paths to RGB images.
        mask_paths  : List of paths to corresponding binary masks.
        n           : Number of in-painted variants per image.
        strategy    : Color sampling strategy.
        out_dir     : If provided, save in-painted images to this directory.
                      Files saved as: {stem}_variant_{i}.jpg
        seed        : Random seed (incremented per image for diversity).

    Returns:
        List of (variants, colors) tuples, one per input image.
    """
    assert len(image_paths) == len(mask_paths), \
        "image_paths and mask_paths must have the same length"

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for idx, (img_path, msk_path) in enumerate(zip(image_paths, mask_paths)):
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask_raw = np.array(Image.open(msk_path).convert("L"), dtype=np.uint8)
        mask = (mask_raw > 0).astype(np.uint8)

        # Increment seed per image so colors differ across images
        img_seed = seed + idx if seed is not None else None

        variants, colors = inpaint_glass(
            image, mask, n=n, strategy=strategy,
            seed=img_seed, return_colors=True
        )

        if out_dir:
            stem = Path(img_path).stem
            for i, variant in enumerate(variants):
                out_path = Path(out_dir) / f"{stem}_variant_{i}.jpg"
                Image.fromarray(variant).save(out_path, quality=95)

        results.append((variants, colors))

        if (idx + 1) % 500 == 0:
            print(f"  [batch_inpaint] {idx + 1}/{len(image_paths)} done")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper  (sanity check / debugging)
# ─────────────────────────────────────────────────────────────────────────────

def visualise_inpainting(
    image: np.ndarray,
    mask: np.ndarray,
    variants: List[np.ndarray],
    colors: List[Tuple[int, int, int]],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Build a side-by-side comparison grid:
    [original | mask overlay | variant_0 | variant_1 | ... | variant_N-1]

    Args:
        image     : Original RGB image [H, W, 3].
        mask      : Binary glass mask [H, W].
        variants  : List of N in-painted images.
        colors    : List of N (R, G, B) colors used.
        save_path : If provided, save the grid as a PNG.

    Returns:
        Grid image [H, W_total, 3] uint8.
    """
    H, W = image.shape[:2]

    # Mask overlay: tint glass region red on original
    overlay = image.copy()
    glass_px = mask.astype(bool)
    overlay[glass_px] = (
        overlay[glass_px] * 0.4 + np.array([255, 60, 60]) * 0.6
    ).astype(np.uint8)

    panels = [image, overlay] + variants
    labels = ["Original", "Glass mask"] + [
        f"Variant {i}\nRGB{colors[i]}" for i in range(len(variants))
    ]

    # Draw label text on each panel
    labelled = []
    for panel, label in zip(panels, labels):
        p = panel.copy()
        for j, line in enumerate(label.split("\n")):
            cv2.putText(
                p, line, (6, 18 + j * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                p, line, (6, 18 + j * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (20, 20, 20), 1, cv2.LINE_AA
            )
        labelled.append(p)

    grid = np.concatenate(labelled, axis=1)   # [H, W*(N+2), 3]

    if save_path:
        Image.fromarray(grid).save(save_path)
        print(f"[visualise] Saved grid to {save_path}")

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# CLI — run:  python inpaint_glass.py --image img.jpg --mask mask.png
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Costanzino et al. glass in-painting — sanity check"
    )
    parser.add_argument("--image",    required=True, help="Path to RGB image")
    parser.add_argument("--mask",     required=True, help="Path to binary glass mask")
    parser.add_argument("--n",        type=int, default=5, help="Number of variants")
    parser.add_argument("--strategy", default="random",
                        choices=["random", "contrast", "palette"],
                        help="Color sampling strategy")
    parser.add_argument("--out",      default="inpaint_check.png",
                        help="Output visualisation path")
    parser.add_argument("--seed",     type=int, default=0, help="Random seed")
    args = parser.parse_args()

    print(f"\n── Costanzino et al. in-painting check ──")
    print(f"  image    : {args.image}")
    print(f"  mask     : {args.mask}")
    print(f"  N        : {args.n}")
    print(f"  strategy : {args.strategy}")
    print(f"  seed     : {args.seed}")

    image = np.array(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    mask_raw = np.array(Image.open(args.mask).convert("L"), dtype=np.uint8)
    mask = (mask_raw > 0).astype(np.uint8)

    print(f"\n  Image shape : {image.shape}")
    print(f"  Glass pixels: {mask.sum()} / {mask.size} "
          f"({100 * mask.mean():.1f}%)")

    variants, colors = inpaint_glass(
        image, mask,
        n=args.n,
        strategy=args.strategy,
        seed=args.seed,
        return_colors=True,
    )

    print(f"\n  Sampled colors:")
    for i, c in enumerate(colors):
        print(f"    Variant {i}: RGB{c}")

    assert len(variants) == args.n, "Wrong number of variants returned"
    for i, v in enumerate(variants):
        assert v.shape == image.shape, f"Variant {i} shape mismatch"
        # Verify glass pixels were replaced
        changed = (v[mask.astype(bool)] != image[mask.astype(bool)]).any(axis=1)
        assert changed.all(), f"Variant {i}: some glass pixels were NOT replaced"

    print(f"\n  ✅ All {args.n} variants correct — glass pixels replaced")

    grid = visualise_inpainting(image, mask, variants, colors, save_path=args.out)
    print(f"  Grid shape : {grid.shape}")
    print(f"\n  Saved visualisation → {args.out}\n")