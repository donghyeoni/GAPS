"""Reproduce Notebook 2: bandwidth-limited image reconstruction.

Builds an image pyramid (512 -> 256 -> 128), transmits only a budgeted subset
of patches at each level, and reconstructs the full-resolution image.

Two pipelines are compared and scored with PSNR:

- Baseline : random patch selection + bilinear interpolation.
- Ours     : gradient-based patch selection + custom forward interpolation
             (sparse expand/assemble followed by neighborhood-weighted fill).

Other patch selectors (frequency / Laplacian) are available in
``restoration.patch_select`` for experimentation.

Usage
-----
    python scripts/02_reconstruction.py --image data/4611.png
"""

import argparse
import os
import sys

import cv2

# Make the `restoration` package importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restoration import (
    psnr,
    downsampling,
    bilinear,
    expand,
    assemble,
    restoration1,
    restoration3,
    random_choose,
    gradient_choose,
)

# Default image path (drop your own image into data/).
DEFAULT_IMAGE = os.path.join("data", "4611.png")

# Patch (grid) size shared by both pipelines.
K_BASE = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bandwidth-limited image reconstruction demo."
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Path to the input image (default: %(default)s).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=K_BASE,
        help="Patch (grid) side length (default: %(default)s).",
    )
    return parser.parse_args()


def run_baseline(image_512, image_256, image_128, k):
    """Random patch selection + bilinear interpolation."""
    image_high = image_128
    image_mid = random_choose(image_256, k)
    image_low = random_choose(image_512, k)

    image_re_256 = restoration1(image_high, image_mid)
    image_re_512 = restoration1(image_re_256, image_low)

    p1 = psnr(image_256, image_re_256)
    p2 = psnr(image_512, image_re_512)
    return p1, p2


def run_ours(image_512, image_256, image_128, image_high_up, k):
    """Gradient-based selection + custom forward interpolation."""
    image_low = random_choose(image_512, k)

    # Level 128 -> 256: gradient-selected detail patches over a sparse base.
    g1 = gradient_choose(image_high_up, image_256, k)
    img_sparse = expand(image_128)
    img_combined = assemble(g1, img_sparse)
    image_re_256_m2 = restoration3(img_combined)

    # Level 256 -> 512.
    image_re_256_m2_up = bilinear(image_re_256_m2)
    img_sparse2 = expand(image_re_256_m2)
    # (Detail scoring at the higher level; kept as in the original notebook.)
    _g2 = gradient_choose(image_re_256_m2_up, image_512, k)
    img_combined2 = assemble(image_low, img_sparse2)
    image_re_512_m2 = restoration3(img_combined2)

    p1 = psnr(image_256, image_re_256_m2)
    p2 = psnr(image_512, image_re_512_m2)
    return p1, p2


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        sys.exit(
            f"Image not found: {args.image}\n"
            "Drop your own image into data/ or pass --image <path>."
        )

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"Failed to read image: {args.image}")
    image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Build the pyramid: full (512), 256, 128, and a bilinear upsample of 128.
    image_512 = image_original
    image_256 = downsampling(image_original)
    image_128 = downsampling(image_256)
    image_high_up = bilinear(image_128)

    print(f"Pyramid sizes: {image_512.shape} -> {image_256.shape} -> {image_128.shape}")
    print(f"Patch size: {args.patch_size}\n")

    b1, b2 = run_baseline(image_512, image_256, image_128, args.patch_size)
    o1, o2 = run_ours(image_512, image_256, image_128, image_high_up, args.patch_size)

    print("PSNR (dB) -- higher is better:")
    print(f"  {'':<10}{'256 level':>12}{'512 level':>12}")
    print(f"  {'Baseline':<10}{b1:>12.2f}{b2:>12.2f}")
    print(f"  {'Ours':<10}{o1:>12.2f}{o2:>12.2f}")


if __name__ == "__main__":
    main()
