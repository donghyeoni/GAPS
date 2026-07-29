"""Classical (non-deep-learning) image restoration, implemented from scratch.

This package collects hand-written NumPy implementations of the building
blocks used in the accompanying report *Classical Image Restoration for UAV
Imaging*:

- ``metrics``      : image quality metric (PSNR).
- ``noise``        : synthetic noise generators (Gaussian, salt-and-pepper).
- ``denoise``      : spatial denoising filters (median, averaging,
                     Gaussian, bilateral).
- ``pyramid``      : image-pyramid helpers (downsampling, expand, assemble).
- ``interpolate``  : upsampling / interpolation and patch-merge routines.
- ``patch_select`` : patch (region) selection strategies for a limited
                     transmission budget.

Only NumPy is used for the numerical work. OpenCV is used solely for image
I/O and color conversion, and Matplotlib for visualization in the scripts.
"""

from .metrics import psnr
from .noise import g_noise, i_noise
from .denoise import (
    median_filter,
    averaging_filter,
    gaussian_filter,
    bilateral_filter,
)
from .pyramid import downsampling, expand, assemble
from .interpolate import (
    bilinear,
    cubic_weight,
    bicubic_upsampling,
    restoration1,
    restoration2,
    restoration3,
)
from .patch_select import (
    random_choose,
    gradient_choose,
    frequency_choose,
    laplacian_choose,
)

__all__ = [
    "psnr",
    "g_noise",
    "i_noise",
    "median_filter",
    "averaging_filter",
    "gaussian_filter",
    "bilateral_filter",
    "downsampling",
    "expand",
    "assemble",
    "bilinear",
    "cubic_weight",
    "bicubic_upsampling",
    "restoration1",
    "restoration2",
    "restoration3",
    "random_choose",
    "gradient_choose",
    "frequency_choose",
    "laplacian_choose",
]
