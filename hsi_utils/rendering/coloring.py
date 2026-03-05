"""Wavelength-dependent pseudo-coloring for HSI spectral bands.

Converts single-channel grayscale images (one spectral band) into
false-color RGB images using CIE 1964 color matching functions.
"""

import numpy as np
from ._wavelength_data import WAVELENGTHS_28, wavelength_to_srgb


def colorize_channel(
    gray_image: np.ndarray,
    wavelength_nm: float,
    brightness: float = 5.0,
) -> np.ndarray:
    """Apply wavelength-dependent pseudo-coloring to a grayscale band.

    Matches the MATLAB dispCubeAshwin.m algorithm:
    1. Normalize by top-50 pixel mean, scale by brightness
    2. Build a colormap from gray ramp * wavelength RGB color
    3. Index into colormap

    Args:
        gray_image: (H, W) float array, single spectral band.
        wavelength_nm: center wavelength for this band (nm).
        brightness: intensity scaling factor (default 5.0).

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    img = np.asarray(gray_image, dtype=np.float64).copy()

    # Brightness normalization (matches MATLAB dispCubeAshwin.m):
    # Scale so that top-50 pixel mean maps to 50, then multiply by brightness.
    # The resulting values are used directly as colormap indices [0, 255].
    flat = np.sort(img.ravel())
    ref = flat[-50:].mean()
    if ref > 0:
        img = img * (50.0 / ref)
    if brightness > 1:
        img = img * brightness

    # Wavelength -> base sRGB color (3,)
    base_rgb = wavelength_to_srgb(wavelength_nm)

    # Build 256-entry colormap: gray_ramp * base_color, then normalize peak to 1
    gray_ramp = np.linspace(0, 1, 256).reshape(-1, 1)
    colormap = gray_ramp * base_rgb.reshape(1, 3)
    cmap_max = colormap.max()
    if cmap_max > 0:
        colormap = colormap / cmap_max

    # Scaled pixel values are colormap indices (MATLAB: ind2rgb8 convention)
    indices = np.round(img).astype(np.intp)
    indices = np.clip(indices, 0, 255)
    colored = colormap[indices]  # (H, W, 3), float [0, 1]

    return (colored * 255).astype(np.uint8)


def colorize_cube(
    cube: np.ndarray,
    wavelengths: np.ndarray | None = None,
    brightness: float = 5.0,
    channels: list[int] | None = None,
) -> list[tuple[float, np.ndarray]]:
    """Colorize multiple channels from an HSI cube.

    Args:
        cube: (H, W, C) float array, HSI data cube.
        wavelengths: (C,) array of wavelengths in nm. Defaults to WAVELENGTHS_28.
        brightness: intensity scaling factor.
        channels: subset of channel indices to colorize. None = all.

    Returns:
        List of (wavelength_nm, colored_image) tuples, where each
        colored_image is (H, W, 3) uint8.
    """
    if wavelengths is None:
        wavelengths = WAVELENGTHS_28
    n_channels = cube.shape[2]

    if channels is None:
        channels = list(range(n_channels))

    # Clip to [0, 1] before coloring (matches MATLAB: recon(find(recon>1))=1)
    cube_clipped = np.clip(cube, 0.0, 1.0)

    results = []
    for ch in channels:
        wl = float(wavelengths[ch])
        colored = colorize_channel(cube_clipped[:, :, ch], wl, brightness)
        results.append((wl, colored))

    return results
