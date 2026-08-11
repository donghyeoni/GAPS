"""Ours (selection 1): FFT high-frequency energy selection + bilinear fill.

Usage
-----
    python scripts/enhancement/frequency_selection.py --seed 0
"""

from _common import run_single

if __name__ == "__main__":
    run_single("Frequency selection", "frequency", "bilinear")
