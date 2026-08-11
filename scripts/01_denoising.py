"""Classical denoising benchmark.

Synthesizes Gaussian (sigma=50) and salt-and-pepper (p=0.10) noise on a test
image, then applies every hand-coded filter (median / averaging / Gaussian /
bilateral) to every noise type and reports the full PSNR matrix, so each
filter's strength and weakness per noise type is visible.

Usage
-----
    python scripts/01_denoising.py --seed 0 --save results/denoising_comparison.png --no-show
    python scripts/01_denoising.py --image data/your_uav_frame.png
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

# Default: the committed 512x512 test image.
DEFAULT_IMAGE = os.path.join("results", "input_dog.png")


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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed numpy RNG for reproducible noise synthesis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

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
    noise_cases = [
        ("Gaussian s=50", g_noise(image_original, 50)),
        ("Impulse p=0.10", i_noise(image_original, 0.1)),
    ]

    # --- Apply every filter to every noise type and score with PSNR ----
    filters = [
        ("Median 5x5", lambda im: median_filter(im, 5)),
        ("Averaging 3x3", lambda im: averaging_filter(im, 3)),
        ("Gaussian 3x3 s=1", lambda im: gaussian_filter(im, 3, 1.0)),
        ("Bilateral 5x5", lambda im: bilateral_filter(im, 5, 30, 15)),
    ]

    results = {}  # {noise_name: [("Noisy", img, psnr), (filter_name, img, psnr), ...]}
    for noise_name, noisy in noise_cases:
        rows = [("Noisy", noisy, psnr(image_original, noisy))]
        for filter_name, fn in filters:
            out = fn(noisy)
            rows.append((filter_name, out, psnr(image_original, out)))
        results[noise_name] = rows

    print("PSNR (dB) -- higher is better:")
    header = f"  {'':<18}" + "".join(f"{n:>18}" for n, _ in noise_cases)
    print(header)
    labels = [r[0] for r in results[noise_cases[0][0]]]
    for i, label in enumerate(labels):
        line = f"  {label:<18}"
        for noise_name, _ in noise_cases:
            line += f"{results[noise_name][i][2]:>18.2f}"
        print(line)

    # --- Visualize: one row per noise type ------------------------------
    n_rows = len(noise_cases)
    n_cols = len(labels)
    plt.figure(figsize=(3 * n_cols, 3.2 * n_rows))
    for r, (noise_name, _) in enumerate(noise_cases):
        for c, (name, img, val) in enumerate(results[noise_name]):
            plt.subplot(n_rows, n_cols, r * n_cols + c + 1)
            plt.imshow(img)
            plt.title(f"{noise_name}\n{name}: {val:.2f} dB", fontsize=9)
            plt.axis("off")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
