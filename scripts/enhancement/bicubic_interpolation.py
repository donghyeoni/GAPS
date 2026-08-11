"""Ours (interpolation 1): random selection + bicubic interpolation.

Usage
-----
    python scripts/enhancement/bicubic_interpolation.py --seed 0
"""

from _common import run_single

if __name__ == "__main__":
    run_single("Bicubic interpolation", "random", "bicubic")
