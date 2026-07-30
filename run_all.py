"""Regenerate every committed artifact under ``results/`` in one command.

The original notebooks operated on aerial frames that are not redistributed.
To make the pipeline reproducible with **no external data**, this script
synthesizes a deterministic 512x512 test image (fixed seed) and runs both
scripts on it:

* ``results/input_synthetic.png``   -- the generated 512x512 test image
* ``results/01_denoising.png``      -- noisy + denoised comparison figure
* ``results/01_denoising.log``      -- PSNR of each classical filter
* ``results/02_reconstruction.log`` -- baseline vs. "ours" PSNR (256 / 512)

The original notebook figures are preserved under
``results/notebook_reference/``.

Usage
-----
    python run_all.py
"""

import os
import subprocess
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, "results")


def make_synthetic_image(size=512, seed=0):
    """A deterministic 512x512 BGR image with smooth gradients, discs and edges
    (structure the gradient-based patch selector can act on) plus fine texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    r = (xx / size * 255).astype(np.float32)
    g = (yy / size * 255).astype(np.float32)
    b = ((np.sin(xx / 24.0) + np.cos(yy / 24.0)) * 60 + 128).astype(np.float32)
    img = np.stack([b, g, r], axis=2)
    for _ in range(8):
        cx, cy = rng.integers(0, size, 2)
        rad = int(rng.integers(25, 80))
        color = rng.integers(0, 256, 3).astype(np.float32)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
        img[mask] = color
    img += rng.normal(0, 8, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def run(name, args):
    log_path = os.path.join(OUT_DIR, f"{name}.log")
    print(f"  {name} ...")
    proc = subprocess.run([sys.executable] + args, cwd=REPO_ROOT,
                          capture_output=True, text=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n" + proc.stderr)
    if proc.returncode != 0:
        print(f"    WARNING: {name} exited with {proc.returncode} (see log)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"

    img_path = os.path.join(OUT_DIR, "input_synthetic.png")
    cv2.imwrite(img_path, make_synthetic_image())
    print(f"Synthetic input written to {os.path.relpath(img_path, REPO_ROOT)}")

    run("01_denoising",
        ["scripts/01_denoising.py", "--image", img_path,
         "--save", os.path.join(OUT_DIR, "01_denoising.png"),
         "--no-show", "--seed", "0"])
    run("02_reconstruction",
        ["scripts/02_reconstruction.py", "--image", img_path, "--seed", "0"])

    print("Done. Artifacts under results/.")


if __name__ == "__main__":
    main()
