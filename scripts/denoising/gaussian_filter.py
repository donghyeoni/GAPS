"""Gaussian smoothing filter demo.

Usage
-----
    python scripts/denoising/gaussian_filter.py --seed 0
"""

from _common import run_demo
from restoration import gaussian_filter

if __name__ == "__main__":
    run_demo("Gaussian 3x3 s=1", lambda im: gaussian_filter(im, 3, 1.0))
