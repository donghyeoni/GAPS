"""Shared helpers for the per-filter denoising demos.

Each demo synthesizes the two benchmark noise types (Gaussian sigma=50 and
impulse p=0.10) on the test image, applies one filter, and reports PSNR.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Make the `restoration` package importable when run from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from restoration import psnr, g_noise, i_noise  # noqa: E402

DEFAULT_IMAGE = os.path.join("results", "input_dog.png")


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Path to the input image (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed numpy RNG for reproducible noise synthesis.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the comparison figure (PNG).",
    )
    return parser.parse_args()


def load_rgb(path):
    if not os.path.isfile(path):
        sys.exit(
            f"Image not found: {path}\n"
            "Drop your own image into data/ or pass --image <path>."
        )
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_demo(filter_name, filter_fn):
    """Standard demo body: noise synthesis, filtering, PSNR, optional figure."""
    args = parse_args(f"{filter_name} denoising demo.")
    if args.seed is not None:
        np.random.seed(args.seed)

    image_original = load_rgb(args.image)
    noise_cases = [
        ("Gaussian s=50", g_noise(image_original, 50)),
        ("Impulse p=0.10", i_noise(image_original, 0.1)),
    ]

    print(f"[{filter_name}] PSNR (dB) -- higher is better:")
    panels = []
    for noise_name, noisy in noise_cases:
        filtered = filter_fn(noisy)
        p_noisy = psnr(image_original, noisy)
        p_filtered = psnr(image_original, filtered)
        print(f"  {noise_name:<15} noisy {p_noisy:6.2f}  ->  filtered {p_filtered:6.2f}")
        panels.append((noisy, f"{noise_name}\nnoisy: {p_noisy:.2f} dB"))
        panels.append((filtered, f"{noise_name}\n{filter_name}: {p_filtered:.2f} dB"))

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(3 * len(panels), 3.4))
        for idx, (img, title) in enumerate(panels, start=1):
            plt.subplot(1, len(panels), idx)
            plt.imshow(img)
            plt.title(title, fontsize=9)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
