"""Averaging (box) filter demo.

Usage
-----
    python scripts/denoising/averaging_filter.py --seed 0
"""

from _common import run_demo
from restoration import averaging_filter

if __name__ == "__main__":
    run_demo("Averaging 3x3", lambda im: averaging_filter(im, 3))
