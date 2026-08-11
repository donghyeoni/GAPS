"""Baseline: random patch selection + bilinear interpolation.

Usage
-----
    python scripts/enhancement/baseline.py --seed 0
"""

from _common import run_single

if __name__ == "__main__":
    run_single("Baseline", "random", "bilinear")
