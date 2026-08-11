"""Ours (selection 3): gradient-magnitude selection + bilinear fill.

Usage
-----
    python scripts/enhancement/gradient_selection.py --seed 0
"""

from _common import run_single

if __name__ == "__main__":
    run_single("Gradient selection", "gradient", "bilinear")
