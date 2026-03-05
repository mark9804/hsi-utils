"""Reconstruct an RGB image from a multi-band HSI cube.

Uses CIE 1964 color matching functions for spectral integration,
with an option to load a pre-existing RGB image from a .mat file.
"""

import numpy as np
from ._wavelength_data import (
    WAVELENGTHS_28,
    wavelength_to_xyz,
    _XYZ_TO_SRGB,
    _srgb_gamma,
)


def hsi_to_rgb(
    cube: np.ndarray,
    wavelengths: np.ndarray | None = None,
    gamma: float = 2.2,
) -> np.ndarray:
    """Reconstruct an RGB image from a multi-band HSI cube.

    Uses CIE 1964 color matching functions to integrate spectral bands
    into XYZ tristimulus values, then converts to sRGB.

    Args:
        cube: (H, W, C) float array in [0, 1].
        wavelengths: (C,) array of band wavelengths in nm.
            Defaults to WAVELENGTHS_28.
        gamma: gamma correction exponent (default 2.2).
            Set to 0 or negative to use standard sRGB gamma curve.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    if wavelengths is None:
        wavelengths = WAVELENGTHS_28
    cube = np.clip(cube, 0.0, 1.0)

    # Get XYZ color matching weights for each band: (C, 3)
    xyz_weights = wavelength_to_xyz(wavelengths)

    # Spectral integration: (H, W, C) @ (C, 3) -> (H, W, 3)
    xyz_image = cube @ xyz_weights

    # XYZ -> linear sRGB: (H, W, 3) @ (3, 3) -> (H, W, 3)
    linear_rgb = xyz_image @ _XYZ_TO_SRGB.T
    linear_rgb = np.clip(linear_rgb, 0.0, None)

    # Normalize to [0, 1]
    max_val = linear_rgb.max()
    if max_val > 0:
        linear_rgb = linear_rgb / max_val

    # Gamma correction
    if gamma > 0:
        rgb = np.power(linear_rgb, 1.0 / gamma)
    else:
        rgb = _srgb_gamma(linear_rgb)

    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def load_or_reconstruct_rgb(
    mat_data: dict,
    cube_key: str = "truth",
    rgb_key: str = "rgb",
    wavelengths: np.ndarray | None = None,
) -> np.ndarray:
    """Load a pre-existing RGB image from .mat data, or reconstruct from cube.

    Args:
        mat_data: dictionary from scipy.io.loadmat().
        cube_key: key for the HSI cube in mat_data.
        rgb_key: key for a pre-stored RGB image in mat_data.
            If present and valid, this is returned directly.
        wavelengths: passed to hsi_to_rgb if reconstruction is needed.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    # Try loading pre-stored RGB
    if rgb_key in mat_data:
        rgb = np.asarray(mat_data[rgb_key])
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            return rgb

    # Fallback to reconstruction
    cube = np.asarray(mat_data[cube_key], dtype=np.float64)
    return hsi_to_rgb(cube, wavelengths)
