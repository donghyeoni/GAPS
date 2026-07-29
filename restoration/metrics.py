"""Image quality metrics."""

import numpy as np


def psnr(img1, img2):
    """Peak Signal-to-Noise Ratio between two images.

    The maximum pixel value is inferred from the dtype: 1.0 for floating
    point images (assumed to be in the [0, 1] range) and 255.0 otherwise.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Images of identical shape.

    Returns
    -------
    float
        PSNR in decibels. ``inf`` when the images are identical.
    """
    assert img1.shape == img2.shape, "Image shapes differ!"

    # Float images are assumed to live in [0, 1]; integer images in [0, 255].
    if img1.dtype == np.float32 or img1.dtype == np.float64:
        max_pixel = 1.0
    else:
        max_pixel = 255.0

    # Mean Squared Error.
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")  # identical images -> infinite PSNR

    return 20 * np.log10(max_pixel / np.sqrt(mse))
