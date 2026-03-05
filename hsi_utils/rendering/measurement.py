"""Render raw CASSI measurement as a grayscale image."""

import numpy as np


def render_measurement(measurement: np.ndarray) -> np.ndarray:
    """Normalize and render a CASSI measurement as a uint8 grayscale image.

    Args:
        measurement: (H, W) float array (raw compressed measurement).

    Returns:
        (H, W) uint8 grayscale image, min-max normalized.
    """
    m = np.asarray(measurement, dtype=np.float64)
    lo, hi = m.min(), m.max()
    if hi > lo:
        m = (m - lo) / (hi - lo)
    else:
        m = np.zeros_like(m)
    return (m * 255).astype(np.uint8)
