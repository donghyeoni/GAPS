"""Synthetic noise generators used to build denoising test cases."""

import numpy as np


def g_noise(image, std):
    """Add zero-mean additive Gaussian noise.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C), uint8.
    std : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 255], uint8.
    """
    H, W, C = image.shape
    gauss = np.random.normal(0, std, image.shape)
    noisy = image + gauss
    noisy_image = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy_image


def i_noise(image, prob):
    """Add salt-and-pepper (impulse) noise.

    A fraction ``prob`` of the pixels is corrupted; half are set to 255
    (salt) and half to 0 (pepper).

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C), uint8.
    prob : float
        Fraction of pixels to corrupt, in [0, 1].

    Returns
    -------
    np.ndarray
        Noisy image, same dtype as input.
    """
    noisy = image.copy()
    H, W, _ = image.shape
    num_pixels = H * W
    num_noisy = int(prob * num_pixels)

    # All pixel coordinates as flat (1D) indices.
    all_coords = np.arange(num_pixels)
    np.random.shuffle(all_coords)

    # Split the corrupted set into salt and pepper coordinates.
    salt_coords = all_coords[: num_noisy // 2]
    pepper_coords = all_coords[num_noisy // 2 : num_noisy]

    # Convert flat indices back to 2D coordinates.
    salt_y, salt_x = np.unravel_index(salt_coords, (H, W))
    pepper_y, pepper_x = np.unravel_index(pepper_coords, (H, W))

    # Apply salt (255) and pepper (0).
    noisy[salt_y, salt_x, :] = 255
    noisy[pepper_y, pepper_x, :] = 0

    return noisy
