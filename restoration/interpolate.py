"""Upsampling / interpolation and patch-merge routines.

- ``bilinear``           : 2x bilinear upsampling (baseline).
- ``bicubic_upsampling`` : 2x bicubic upsampling (backward mapping).
- ``restoration3``       : neighborhood-weighted forward interpolation used
                           to fill sparse images.
- ``restoration1/2/3``   : merge a low-resolution upsample into a partially
                           transmitted higher-resolution image.
"""

import numpy as np


def bilinear(image):
    """2x bilinear upsampling (baseline interpolation).

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C).

    Returns
    -------
    np.ndarray
        Upsampled image of size (2H, 2W, C), uint8.
    """
    H, W, C = image.shape
    new_H, new_W = H * 2, W * 2
    upsampled = np.zeros((new_H, new_W, C), dtype=image.dtype)

    # Iterate over the pixels of the upsampled image.
    for y in range(new_H):
        for x in range(new_W):
            # Corresponding location in the source image.
            src_y = y / 2.0
            src_x = x / 2.0
            # The four surrounding source pixels.
            y0 = int(np.floor(src_y))
            x0 = int(np.floor(src_x))
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)

            dy = src_y - y0
            dx = src_x - x0

            for c in range(C):
                top = (1 - dx) * image[y0, x0, c] + dx * image[y0, x1, c]
                bottom = (1 - dx) * image[y1, x0, c] + dx * image[y1, x1, c]
                value = (1 - dy) * top + dy * bottom
                upsampled[y, x, c] = np.clip(value, 0, 255)

    return upsampled.astype(np.uint8)


def restoration1(image1, image2):
    """Merge a bilinear upsample of ``image1`` into ``image2``.

    Empty pixels of ``image2`` (all channels zero) are filled from the
    upsampled ``image1``.
    """
    image1_up = bilinear(image1).astype(np.uint8)
    mask = np.all(image2 == 0, axis=-1)
    output = image2.copy()
    output[mask] = image1_up[mask]
    return output


def cubic_weight(t):
    """Cubic convolution weight (a = -0.5) used by bicubic interpolation."""
    t = abs(t)
    if t <= 1:
        return (1.5 * t - 2.5) * t * t + 1
    elif t < 2:
        return ((-0.5 * t + 2.5) * t - 4) * t + 2
    else:
        return 0


def bicubic_upsampling(image):
    """2x bicubic upsampling using backward mapping.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C).

    Returns
    -------
    np.ndarray
        Upsampled image of size (2H, 2W, C), uint8.
    """
    H, W, C = image.shape
    new_H, new_W = H * 2, W * 2
    upsampled = np.zeros((new_H, new_W, C), dtype=np.float32)

    for y in range(new_H):
        for x in range(new_W):
            src_y = y / 2.0
            src_x = x / 2.0

            y_int = int(np.floor(src_y))
            x_int = int(np.floor(src_x))

            for c in range(C):
                value = 0.0
                for m in range(-1, 3):
                    for n in range(-1, 3):
                        yy = np.clip(y_int + m, 0, H - 1)
                        xx = np.clip(x_int + n, 0, W - 1)

                        wy = cubic_weight(src_y - (y_int + m))
                        wx = cubic_weight(src_x - (x_int + n))

                        value += image[yy, xx, c] * wy * wx

                upsampled[y, x, c] = np.clip(value, 0, 255)

    return upsampled.astype(np.uint8)


def restoration2(image1, image2):
    """Merge a bicubic upsample of ``image1`` into ``image2``.

    Empty pixels of ``image2`` (all channels zero) are filled from the
    bicubic upsampling of ``image1``.
    """
    image1_up = bicubic_upsampling(image1).astype(np.uint8)
    mask = np.all(image2 == 0, axis=-1)
    output = image2.copy()
    output[mask] = image1_up[mask]
    return output


def restoration3(image):
    """Fill empty pixels via distance-weighted 3x3 neighborhood averaging.

    A pixel is empty when all its channels are zero. Each empty pixel is
    filled with a weighted average of its valid 8-neighbors, weighted by
    inverse Chebyshev distance (closer neighbors count more).

    Parameters
    ----------
    image : np.ndarray
        Sparse image (H, W, C).

    Returns
    -------
    np.ndarray
        Filled image clipped to [0, 255], uint8.
    """
    img = image.copy().astype(np.float32)
    H, W, C = img.shape
    filled = img.copy()
    mask = np.any(img != 0, axis=2)  # True where the pixel has any nonzero channel

    for y in range(H):
        for x in range(W):
            if not mask[y, x]:  # only fill empty pixels
                vals = []
                weights = []

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < H) and (0 <= nx < W) and mask[ny, nx]:
                            dist = max(abs(dy), abs(dx))       # 0 (self) or 1
                            weight = 1.0 / (dist + 1e-5)       # closer -> larger weight
                            vals.append(img[ny, nx])
                            weights.append(weight)

                if vals:
                    vals = np.array(vals)
                    weights = np.array(weights).reshape(-1, 1)
                    weighted_avg = np.sum(vals * weights, axis=0) / np.sum(weights)
                    filled[y, x] = weighted_avg

    return np.clip(filled, 0, 255).astype(np.uint8)
