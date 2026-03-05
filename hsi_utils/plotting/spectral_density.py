"""Spectral density curve plotting for HSI reconstruction comparison.

Reproduces the spectral density plot from show_line.m + createfigure.m:
extract mean spectral density over an ROI, normalize to max=1, compute
Pearson correlation against ground truth, and draw comparison curves.
"""

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..rendering._wavelength_data import WAVELENGTHS_28


@dataclass
class SpectralInput:
    """One HSI cube + metadata for spectral density comparison."""

    cube: np.ndarray  # (H, W, C), float [0,1]
    label: str  # legend label, e.g. "MST-L"
    color: str | None = None  # line color (matplotlib color string)
    is_ground_truth: bool = False  # if True, treated as the reference


def compute_spectral_density(
    cube: np.ndarray,
    roi: tuple[int, int, int, int],
    clip_max: float | None = None,
) -> np.ndarray:
    """Extract mean spectral density from a ROI, normalized to max=1.

    Args:
        cube: (H, W, C) float array.
        roi: (x, y, w, h) pixel coordinates.
        clip_max: if set, clip cube values to this max before extraction.

    Returns:
        (C,) float64 array with values in [0, 1].
    """
    x, y, w, h = roi
    data = cube.copy()
    if clip_max is not None:
        data = np.clip(data, None, clip_max)
    patch = data[y : y + h, x : x + w, :]  # (h, w, C)
    spectrum = patch.mean(axis=(0, 1))  # (C,)
    peak = spectrum.max()
    if peak > 0:
        spectrum = spectrum / peak
    return spectrum.astype(np.float64)


def draw_spectral_density(
    inputs: list[SpectralInput],
    roi: tuple[int, int, int, int],
    wavelengths: np.ndarray | None = None,
    clip_max: float | None = None,
    output_path: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Image.Image:
    """Draw spectral density curves for multiple reconstructions.

    Computes Pearson correlation of each non-GT input against the GT input,
    and displays it in the legend.

    Args:
        inputs: list of SpectralInput. Exactly one should have is_ground_truth=True.
        roi: (x, y, w, h) region over which to compute the spectral density.
        wavelengths: (C,) array of wavelengths. Defaults to WAVELENGTHS_28.
        clip_max: optional clip threshold applied before extraction (e.g. 0.7).
        output_path: if provided, save the plot as a PNG file.
        figsize: matplotlib figure size.

    Returns:
        PIL Image of the spectral density plot.
    """
    if wavelengths is None:
        wavelengths = WAVELENGTHS_28

    # Separate GT from predictions
    gt_input = None
    pred_inputs = []
    for inp in inputs:
        if inp.is_ground_truth:
            gt_input = inp
        else:
            pred_inputs.append(inp)

    gt_spectrum = None
    if gt_input is not None:
        gt_spectrum = compute_spectral_density(gt_input.cube, roi, clip_max)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot GT first
    if gt_input is not None and gt_spectrum is not None:
        ax.plot(
            wavelengths,
            gt_spectrum,
            marker=".",
            markersize=16,
            linewidth=2.5,
            color=gt_input.color or "red",
            label=f" {gt_input.label}",
        )

    # Plot predictions with correlation
    for inp in pred_inputs:
        spectrum = compute_spectral_density(inp.cube, roi, clip_max)
        if gt_spectrum is not None:
            corr = float(np.corrcoef(gt_spectrum, spectrum)[0, 1])
            label = f" {inp.label}, corr: {corr:.4f}"
        else:
            label = f" {inp.label}"
        ax.plot(
            wavelengths,
            spectrum,
            marker=".",
            # markersize=16,
            linewidth=2.5,
            color=inp.color,
            label=label,
        )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Wavelength (nm)", fontsize=28, fontfamily="sans-serif")
    ax.set_ylabel("Density", fontsize=28, fontfamily="sans-serif")
    ax.tick_params(labelsize=22)
    for spine in ax.spines.values():
        spine.set_linewidth(3.5)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    ax.legend(fontsize=22, edgecolor="white", loc="lower right")

    fig.tight_layout()

    # Render to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()

    if output_path:
        img.save(output_path)

    plt.close(fig)
    return img
