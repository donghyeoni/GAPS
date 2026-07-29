"""Patch (region) selection strategies for a limited transmission budget.

Given a per-level budget, each strategy chooses which fixed-size patches of a
higher-resolution image to keep; the rest are set to zero and later filled by
interpolation. ``random_choose`` is the baseline; the gradient-, frequency-,
and Laplacian-based strategies form the proposed ("ours") selection.

The budget ``k = 2**14 // (patch_size ** 2)`` keeps the total number of
transmitted pixels constant (2**14) across patch sizes.
"""

import numpy as np
import cv2

from .pyramid import downsampling
from .interpolate import bilinear


def random_choose(image, patch_size):
    """Baseline: keep a random subset of patches.

    Parameters
    ----------
    image : np.ndarray
        Source image (H, W, C).
    patch_size : int
        Square patch side length.

    Returns
    -------
    np.ndarray
        Image with only the selected patches copied; the rest are zero.
    """
    H, W, C = image.shape
    n_h = H // patch_size
    n_w = W // patch_size
    total_patches = n_h * n_w
    k = 2 ** 14 // (patch_size * patch_size)

    # All patch indices.
    patch_indices = [(i, j) for i in range(n_h) for j in range(n_w)]
    selected = np.random.choice(len(patch_indices), k, replace=False)
    selected_patches = [patch_indices[i] for i in selected]

    # Output image (all pixels start at zero).
    output = np.zeros_like(image)
    # Copy only the selected patches.
    for i, j in selected_patches:
        y, x = i * patch_size, j * patch_size
        output[y : y + patch_size, x : x + patch_size, :] = image[
            y : y + patch_size, x : x + patch_size, :
        ]
    return output


def _fft_spectrum_energy(patch, ratio=0.25):
    """High-frequency energy of a grayscale patch.

    The low-frequency center of the shifted spectrum is masked out and the
    standard deviation of the remaining (high-frequency) magnitudes is
    returned as an energy score.
    """
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    H, W = magnitude.shape
    cy, cx = H // 2, W // 2

    # Exclude the low-frequency center.
    low = int(min(H, W) * ratio / 2)
    mask = np.ones_like(magnitude, dtype=bool)
    mask[cy - low : cy + low, cx - low : cx + low] = False

    high_freq_energy = np.std(magnitude[mask])
    return high_freq_energy


def frequency_choose(image, target, grid_size):
    """Select the patches with the highest high-frequency energy.

    Parameters
    ----------
    image : np.ndarray
        Reference image used to score patches (BGR, as read by OpenCV).
    target : np.ndarray
        Image the selected patches are copied from.
    grid_size : int
        Square patch side length.

    Returns
    -------
    np.ndarray
        Image with only the top-scoring patches copied from ``target``.
    """
    H, W, _ = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    energies = []
    positions = []
    k = 2 ** 14 // (grid_size * grid_size)
    for y in range(0, H, grid_size):
        for x in range(0, W, grid_size):
            patch = gray_image[y : y + grid_size, x : x + grid_size]
            energy = _fft_spectrum_energy(patch)
            energies.append(energy)
            positions.append((y, x))

    energies = np.array(energies)
    positions = np.array(positions)

    # Keep the k patches with the largest energy.
    top_idx = np.argsort(energies)[-k:]
    masked_image = np.zeros_like(image)

    for idx in top_idx:
        y, x = positions[idx]
        masked_image[y : y + grid_size, x : x + grid_size] = target[
            y : y + grid_size, x : x + grid_size
        ]

    return masked_image


def gradient_choose(image, target, grid_size):
    """Select the patches with the largest total gradient magnitude.

    Parameters
    ----------
    image : np.ndarray
        Reference image used to score patches (BGR, as read by OpenCV).
    target : np.ndarray
        Image the selected patches are copied from.
    grid_size : int
        Square patch side length.

    Returns
    -------
    np.ndarray
        Image with only the top-scoring patches copied from ``target``.
    """
    H, W, C = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gradient magnitude over the whole image (reflect-padded finite diff).
    gray_padded = np.pad(gray, pad_width=1, mode="reflect")  # (H+2, W+2)

    gx = np.abs(gray_padded[1:-1, 2:] - gray_padded[1:-1, 1:-1])  # (H, W)
    gy = np.abs(gray_padded[2:, 1:-1] - gray_padded[1:-1, 1:-1])  # (H, W)

    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # (H, W)

    gradients = []
    positions = []
    k = 2 ** 14 // (grid_size * grid_size)

    for y in range(0, H, grid_size):
        for x in range(0, W, grid_size):
            patch_grad = grad_mag[y : y + grid_size, x : x + grid_size]
            mean_grad = np.sum(patch_grad)
            gradients.append(mean_grad)
            positions.append((y, x))

    gradients = np.array(gradients)
    positions = np.array(positions)

    top_idx = np.argsort(gradients)[-k:]
    masked_image = np.zeros_like(image)

    for idx in top_idx:
        y, x = positions[idx]
        masked_image[y : y + grid_size, x : x + grid_size] = target[
            y : y + grid_size, x : x + grid_size
        ]

    return masked_image


def laplacian_choose(image, target, grid_size):
    """Select patches by Laplacian-pyramid detail (residual std-dev).

    A one-level Laplacian is formed as the absolute difference between the
    image and a bilinear upsample of its downsample. Patches with the highest
    detail (standard deviation of the residual) are kept.

    Parameters
    ----------
    image : np.ndarray
        Reference image used to score patches (BGR, as read by OpenCV).
    target : np.ndarray
        Image the selected patches are copied from.
    grid_size : int
        Square patch side length.

    Returns
    -------
    np.ndarray
        Image with only the top-scoring patches copied from ``target``.
    """
    H, W, C = image.shape

    image_down = downsampling(image)
    image_up = bilinear(image_down)

    L1 = np.abs(image - image_up)
    L1 = cv2.cvtColor(L1, cv2.COLOR_BGR2GRAY)

    scores = []
    positions = []
    k = 2 ** 14 // (grid_size * grid_size)
    for y in range(0, H, grid_size):
        for x in range(0, W, grid_size):
            patch = L1[y : y + grid_size, x : x + grid_size]
            score = np.std(patch)
            scores.append(score)
            positions.append((y, x))

    scores = np.array(scores)
    positions = np.array(positions)

    top_idx = np.argsort(scores)[-k:]
    masked_image = np.zeros_like(image)

    for idx in top_idx:
        y, x = positions[idx]
        masked_image[y : y + grid_size, x : x + grid_size] = target[
            y : y + grid_size, x : x + grid_size
        ]

    return masked_image
