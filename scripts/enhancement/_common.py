"""Shared pipeline for the bandwidth-limited enhancement experiments.

Pyramid 512 -> 256 -> 128 with a 2^14-pixel transmission budget per level.
A pipeline is a (selection, fill) pair:

- selection : random / frequency (FFT) / laplacian / gradient
- fill      : bilinear / bicubic / forward (sparse expand + weighted fill)

``random + bilinear`` is the baseline. Selection is always scored on the
receiver-side upsample, so it costs no side-channel bandwidth.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Make the `restoration` package importable when run from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from restoration import (  # noqa: E402
    psnr,
    downsampling,
    bilinear,
    bicubic_upsampling,
    expand,
    assemble,
    restoration1,
    restoration2,
    restoration3,
    random_choose,
    frequency_choose,
    laplacian_choose,
    gradient_choose,
)

DEFAULT_IMAGE = os.path.join("results", "input_dog.png")
K_BASE = 4

SELECTIONS = ["random", "frequency", "laplacian", "gradient"]
FILLS = ["bilinear", "bicubic", "forward"]
BASELINE = ("random", "bilinear")


def parse_args(description, with_figures=False):
    parser = argparse.ArgumentParser(description=description)
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed numpy RNG for reproducible random patch selection.",
    )
    if with_figures:
        parser.add_argument(
            "--save",
            default=None,
            help="Optional path to save the pipeline progression figure (PNG).",
        )
        parser.add_argument(
            "--save-patches",
            default=None,
            help="Optional path to save the selected-patches figure (PNG).",
        )
        parser.add_argument(
            "--combos",
            default=None,
            help="Comma-separated 'selection+fill' subset to run "
                 "(e.g. 'random+bilinear,gradient+bicubic'; default: all).",
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


def build_pyramid(image_original):
    """Full (512), 256, 128, and a bilinear upsample of the 128 level."""
    image_512 = image_original
    image_256 = downsampling(image_original)
    image_128 = downsampling(image_256)
    image_high_up = bilinear(image_128)
    return image_512, image_256, image_128, image_high_up


def select(name, reference, target, patch_size):
    """Pick the transmitted patches of ``target`` (reference guides scoring)."""
    if name == "random":
        return random_choose(target, patch_size)
    if name == "frequency":
        return frequency_choose(reference, target, patch_size)
    if name == "laplacian":
        return laplacian_choose(reference, target, patch_size)
    return gradient_choose(reference, target, patch_size)


def fill(mode, lower, selected):
    """Reconstruct a level from the lower-resolution image + selected patches."""
    if mode == "bilinear":
        return restoration1(lower, selected)
    if mode == "bicubic":
        return restoration2(lower, selected)
    combined = assemble(selected, expand(lower))
    return restoration3(combined)


def upsample(mode, lower):
    """The pure 2x interpolation each fill mode is built on (for figures)."""
    if mode == "bilinear":
        return bilinear(lower)
    if mode == "bicubic":
        return bicubic_upsampling(lower)
    return restoration3(expand(lower))


def run_pipeline(sel_name, fill_mode, image_512, image_256, image_128,
                 image_high_up, patch_size, seed, stages=False):
    if seed is not None:
        np.random.seed(seed)  # identical random selections across pipelines

    sel_256 = select(sel_name, image_high_up, image_256, patch_size)
    re_256 = fill(fill_mode, image_128, sel_256)

    ref_512 = bilinear(re_256)  # receiver-side scoring reference
    sel_512 = select(sel_name, ref_512, image_512, patch_size)
    re_512 = fill(fill_mode, re_256, sel_512)

    out = {
        "p256": psnr(image_256, re_256),
        "p512": psnr(image_512, re_512),
        "re_256": re_256,
        "re_512": re_512,
        "sel_256": sel_256,
        "sel_512": sel_512,
    }
    if stages:  # intermediate interpolation panels for the progression figure
        out["up_256"] = upsample(fill_mode, image_128)
        out["up_512"] = ref_512 if fill_mode == "bilinear" else upsample(fill_mode, re_256)
    return out


def run_single(name, sel_name, fill_mode):
    """Standard body for the one-method demo scripts."""
    args = parse_args(f"{name} reconstruction demo.")
    image_original = load_rgb(args.image)
    image_512, image_256, image_128, image_high_up = build_pyramid(image_original)

    print(f"Pyramid sizes: {image_512.shape} -> {image_256.shape} -> {image_128.shape}")
    print(f"Patch size: {args.patch_size}\n")

    r = run_pipeline(sel_name, fill_mode, image_512, image_256, image_128,
                     image_high_up, args.patch_size, args.seed)
    print(f"[{name}]  ({sel_name} selection + {fill_mode} fill)")
    print(f"  PSNR (dB):  256 level {r['p256']:6.2f}   512 level {r['p512']:6.2f}")
