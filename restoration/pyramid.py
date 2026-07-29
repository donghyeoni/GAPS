"""Image-pyramid helpers: downsampling and sparse expand/assemble.

These functions support the bandwidth-limited reconstruction pipeline, where
an image pyramid (512 -> 256 -> 128) is built and only a budgeted subset of
patches is transmitted at each level.
"""

import numpy as np


def downsampling(image):
    """Gaussian-blur then 2x2-average downsample by a factor of two.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C).

    Returns
    -------
    np.ndarray
        Image of size (H//2, W//2, C), uint8.
    """
    image = image.astype(np.float32)
    H, W, C = image.shape
    sigma, kernel_size = 1, 3

    # Build the Gaussian kernel.
    ax = np.arange(-kernel_size // 2, kernel_size // 2 + 1)  # kernel extent
    xx, yy = np.meshgrid(ax, ax)                             # coordinate grid
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / np.sum(kernel)                         # normalize

    # Blur.
    k = kernel.shape[0]
    pad = k // 2
    padded_img = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    blurred = np.zeros_like(image, dtype=np.float32)

    for y in range(H):
        for x in range(W):
            for c in range(C):
                region = padded_img[y : y + k, x : x + k, c]
                blurred[y, x, c] = np.sum(region * kernel)

    # 2x2 average downsampling.
    H, W, C = blurred.shape
    blurred = blurred[: H - H % 2, : W - W % 2, :]
    downsampled = blurred.reshape(H // 2, 2, W // 2, 2, C).mean(axis=(1, 3))
    return downsampled.astype(np.uint8)


def expand(image):
    """Insert the image into a sparse 2x-larger grid.

    Each source pixel is placed on an even (row, col) position of the output;
    the remaining positions stay zero.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C).

    Returns
    -------
    np.ndarray
        Sparse image of size (2H, 2W, C).
    """
    H, W, C = image.shape
    img_sparse = np.zeros((H * 2, W * 2, C), dtype=image.dtype)
    img_sparse[::2, ::2] = image  # place a pixel every 2 positions
    return img_sparse


def assemble(image1, img_sparse):
    """Fill empty pixels of ``image1`` with values from ``img_sparse``.

    A pixel of ``image1`` is considered empty when all its channels are zero.

    Parameters
    ----------
    image1 : np.ndarray
        Partially filled image (H, W, C).
    img_sparse : np.ndarray
        Sparse image supplying values for the empty pixels.

    Returns
    -------
    np.ndarray
        Merged image.
    """
    # Empty-pixel mask: True where all RGB channels are zero.
    mask = np.all(image1 == 0, axis=2)
    result = image1.copy()
    # Copy sparse values only at empty positions.
    result[mask] = img_sparse[mask]
    return result
