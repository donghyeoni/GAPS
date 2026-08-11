"""Bilateral (edge-preserving) filter demo.

Usage
-----
    python scripts/denoising/bilateral_filter.py --seed 0
"""

from _common import run_demo
from restoration import bilateral_filter

if __name__ == "__main__":
    run_demo("Bilateral 5x5", lambda im: bilateral_filter(im, 5, 30, 15))
