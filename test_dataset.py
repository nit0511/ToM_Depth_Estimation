# test_glass_dataset.py
# ============================================================
# Test script for unified glass dataset loader
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.data_unified import build_dataloader


# ============================================================
# Helper Function
# ============================================================

def denormalize(image_tensor):

    """
    Convert normalized tensor back to RGB image
    """

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    image = image_tensor.permute(1, 2, 0).cpu().numpy()

    image = (image * std) + mean

    image = np.clip(image, 0, 1)

    return image


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    loader = build_dataloader(

        trans10k_root="D:/ToM_Depth_Estimation/datasets/Trans10k",

        gdd_root="D:/ToM_Depth_Estimation/datasets/GDD",

        real_root="datasets/real_frames",

        batch_size=4,

        num_workers=0,

        train=True
    )

    # --------------------------------------------------------
    # Get one batch
    # --------------------------------------------------------

    batch = next(iter(loader))

    images  = batch["image"]
    masks   = batch["mask"]
    sources = batch["source"]
    stems   = batch["stem"]

    # --------------------------------------------------------
    # Print info
    # --------------------------------------------------------

    print("\n========== DATASET CHECK ==========\n")

    print("Images Shape :", images.shape)
    print("Masks Shape  :", masks.shape)

    print("\nSources:")
    print(sources)

    print("\nStems:")
    print(stems)

    print("\nImage Range:")
    print(images.min().item(), images.max().item())

    print("\nMask Unique Values:")
    print(torch.unique(masks))

    # --------------------------------------------------------
    # Visualize batch
    # --------------------------------------------------------

    batch_size = images.shape[0]

    plt.figure(figsize=(12, 6))

    for i in range(batch_size):

        # ----------------------------------------------------
        # RGB Image
        # ----------------------------------------------------

        image = denormalize(images[i])

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        mask = masks[i][0].cpu().numpy()

        # ----------------------------------------------------
        # Show image
        # ----------------------------------------------------

        plt.subplot(2, batch_size, i + 1)

        plt.imshow(image)

        plt.title(f"{sources[i]}")

        plt.axis("off")

        # ----------------------------------------------------
        # Show mask
        # ----------------------------------------------------

        plt.subplot(2, batch_size, batch_size + i + 1)

        plt.imshow(mask, cmap="gray")

        plt.title(stems[i])

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()