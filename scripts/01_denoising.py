"""Reproduce Notebook 1: noise synthesis + classical denoising.

Synthesizes Gaussian and salt-and-pepper noise on a test image, then removes
it with hand-coded median / averaging / Gaussian / bilateral filters and
reports the PSNR of each result.

Usage
-----
    python scripts/01_denoising.py --image data/lena.bmp
    python scripts/01_denoising.py --image data/lena.bmp --save out.png --no-show
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Make the `restoration` package importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restoration import (
    psnr,
    g_noise,
    i_noise,
    median_filter,
    averaging_filter,
    gaussian_filter,
    bilateral_filter,
)

# Default image path (drop your own image into data/).
DEFAULT_IMAGE = os.path.join("data", "lena.bmp")


def parse_args():
    parser = argparse.ArgumentParser(description="Classical image denoising demo.")
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Path to the input image (default: %(default)s).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the result figure (PNG).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive Matplotlib window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        sys.exit(
            f"Image not found: {args.image}\n"
            "Drop your own image into data/ or pass --image <path>."
        )

    # Load and convert to RGB.
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"Failed to read image: {args.image}")
    image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Synthesize noisy versions -------------------------------------
    g10 = g_noise(image_original, 10)
    g30 = g_noise(image_original, 30)
    g50 = g_noise(image_original, 50)
    i01 = i_noise(image_original, 0.1)
    i005 = i_noise(image_original, 0.05)

    # --- Apply filters and score with PSNR -----------------------------
    median_filtered = median_filter(g50, 5)
    psnr_median = psnr(image_original, median_filtered)

    averaging_filtered = averaging_filter(i01, 3)
    psnr_averaging = psnr(image_original, averaging_filtered)

    gaussian_filtered = gaussian_filter(i01, 3, 15)
    psnr_gaussian = psnr(image_original, gaussian_filtered)

    bilateral_filtered = bilateral_filter(g50, 5, 30, 15)
    psnr_bilateral = psnr(image_original, bilateral_filtered)

    print("PSNR (dB):")
    print(f"  Median    (on Gaussian sigma=50) : {psnr_median:.2f}")
    print(f"  Averaging (on impulse p=0.10)    : {psnr_averaging:.2f}")
    print(f"  Gaussian  (on impulse p=0.10)    : {psnr_gaussian:.2f}")
    print(f"  Bilateral (on Gaussian sigma=50) : {psnr_bilateral:.2f}")

    # --- Visualize noisy images ----------------------------------------
    plt.figure(figsize=(12, 4))
    for idx, (img, name) in enumerate(
        [
            (g10, "Gaussian sigma=10"),
            (g30, "Gaussian sigma=30"),
            (g50, "Gaussian sigma=50"),
            (i01, "Impulse p=0.10"),
            (i005, "Impulse p=0.05"),
        ],
        start=1,
    ):
        plt.subplot(1, 5, idx)
        plt.imshow(img)
        plt.title(name, fontsize=9)
        plt.axis("off")
    plt.tight_layout()

    # --- Visualize denoised results ------------------------------------
    plt.figure(figsize=(12, 4))
    for idx, (img, title) in enumerate(
        [
            (median_filtered, f"Median\nPSNR: {psnr_median:.2f} dB"),
            (averaging_filtered, f"Averaging\nPSNR: {psnr_averaging:.2f} dB"),
            (gaussian_filtered, f"Gaussian\nPSNR: {psnr_gaussian:.2f} dB"),
            (bilateral_filtered, f"Bilateral\nPSNR: {psnr_bilateral:.2f} dB"),
        ],
        start=1,
    ):
        plt.subplot(1, 4, idx)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
