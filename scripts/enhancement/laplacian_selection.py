"""Ours (selection 2): Laplacian first-residual selection + bilinear fill.

Usage
-----
    python scripts/enhancement/laplacian_selection.py --seed 0
"""

from _common import run_single

if __name__ == "__main__":
    run_single("Laplacian selection", "laplacian", "bilinear")
