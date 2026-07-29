"""Hand-written spatial denoising filters (NumPy only).

All filters operate on (H, W, C) uint8 images and pad the input with a
reflected border so that the output keeps the original spatial size.
"""

import numpy as np


def median_filter(image, size):
    """Median filter with impulse-noise detection.

    Only pixels that look like impulse noise (value 0 or 255) are replaced
    by the median of their neighborhood; all other pixels are left intact.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C).
    size : int
        Square window side length (odd).

    Returns
    -------
    np.ndarray
        Filtered image, uint8.
    """
    H, W, C = image.shape
    pad = size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    filtered = np.copy(image)  # keep the original, modify only what is needed

    for y in range(H):
        for x in range(W):
            for c in range(C):
                # Apply the filter only on suspected impulse noise (0 or 255).
                pixel_val = image[y, x, c]
                if pixel_val == 0 or pixel_val == 255:
                    window = padded[y : y + size, x : x + size, c]
                    filtered[y, x, c] = np.median(window)

    return filtered.astype(np.uint8)


def averaging_filter(image, size):
    """Box (averaging) filter.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C).
    size : int
        Square kernel side length.

    Returns
    -------
    np.ndarray
        Filtered image (same dtype as input).
    """
    # Uniform averaging kernel.
    kernel = np.ones((size, size)) / (size * size)
    pad_size = size // 2
    padded = np.pad(
        image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="reflect"
    )

    output = np.zeros_like(image)

    for c in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i : i + size, j : j + size, c]
                output[i, j, c] = np.sum(region * kernel)

    return output


def gaussian_filter(image, size, std):
    """Gaussian smoothing filter.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C).
    size : int
        Square kernel side length (odd).
    std : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Filtered image clipped to [0, 255], uint8.
    """
    pad = size // 2
    H, W, C = image.shape

    # Build the (normalized) Gaussian kernel.
    k_x = np.arange(-pad, pad + 1)
    xx, yy = np.meshgrid(k_x, k_x)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * std ** 2))
    kernel = kernel / np.sum(kernel)

    # Pad the image.
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    # Convolve.
    filtered = np.zeros_like(image, dtype=np.float64)
    for y in range(H):
        for x in range(W):
            for c in range(C):
                region = padded[y : y + size, x : x + size, c]
                filtered[y, x, c] = np.sum(region * kernel)

    return np.clip(filtered, 0, 255).astype(np.uint8)


def bilateral_filter(image, size, sigma_color, sigma_space):
    """Edge-preserving bilateral filter.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C).
    size : int
        Filter window side length (odd).
    sigma_color : float
        Standard deviation of the range (color) Gaussian.
    sigma_space : float
        Standard deviation of the spatial (distance) Gaussian.

    Returns
    -------
    np.ndarray
        Filtered image clipped to [0, 255], uint8.
    """
    img = image.astype(np.float32)
    H, W, C = img.shape
    pad = size // 2
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    output = np.zeros_like(img)

    # Spatial (distance) Gaussian kernel.
    ax = np.arange(-pad, pad + 1)
    xx, yy = np.meshgrid(ax, ax)
    spatial_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_space ** 2))

    for y in range(H):
        for x in range(W):
            for c in range(C):
                center_val = padded[y + pad, x + pad, c]
                region = padded[y : y + size, x : x + size, c]

                # Range Gaussian: high weight for similar colors, low otherwise.
                color_diff = region - center_val
                color_kernel = np.exp(-(color_diff ** 2) / (2 * sigma_color ** 2))

                # Final weight is the product of spatial and range kernels.
                bilateral_weights = spatial_kernel * color_kernel
                bilateral_weights /= np.sum(bilateral_weights)

                output[y, x, c] = np.sum(region * bilateral_weights)

    return np.clip(output, 0, 255).astype(np.uint8)
