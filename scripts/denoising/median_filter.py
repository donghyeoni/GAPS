"""Median filter demo (impulse-detection variant: only 0/255 pixels replaced).

Usage
-----
    python scripts/denoising/median_filter.py --seed 0
"""

from _common import run_demo
from restoration import median_filter

if __name__ == "__main__":
    run_demo("Median 5x5", lambda im: median_filter(im, 5))
