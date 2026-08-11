"""Ours (interpolation 2): random selection + forward interpolation
(sparse expand + neighborhood-weighted fill).

Usage
-----
    python scripts/enhancement/forward_interpolation.py --seed 0
"""

from _common import run_single

if __name__ == "__main__":
    run_single("Forward interpolation", "random", "forward")
