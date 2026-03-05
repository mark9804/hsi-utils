"""ROI zoom inset overlay for HSI visualization images."""

from enum import Enum
import numpy as np


class InsetPosition(Enum):
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


def draw_magnified_inset(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    inset_position: InsetPosition = InsetPosition.TOP_LEFT,
    inset_scale: float = 3.0,
    border_color: tuple[int, int, int] = (255, 255, 0),
    border_width: int = 2,
    margin: int = 4,
) -> np.ndarray:
    """Draw a zoom inset on the image.

    Draws a rectangle around the ROI, extracts and scales the region,
    then pastes it at the specified corner with a colored border.

    Args:
        image: (H, W, 3) uint8 RGB image.
        roi: (x, y, w, h) pixel coordinates of the region to magnify.
        inset_position: corner placement for the zoomed inset.
        inset_scale: zoom factor for the inset.
        border_color: RGB color for the ROI rectangle and inset border.
        border_width: width of the border in pixels.
        margin: pixel margin from image edge for inset placement.

    Returns:
        (H, W, 3) uint8 image with inset overlaid (copy of input).
    """
    out = image.copy()
    h, w = out.shape[:2]
    x, y, rw, rh = roi
    color = np.array(border_color, dtype=np.uint8)

    # Draw ROI rectangle on the image
    _draw_rect(out, x, y, rw, rh, color, border_width)

    # Extract and scale the ROI region
    patch = image[y : y + rh, x : x + rw]
    new_h = int(rh * inset_scale)
    new_w = int(rw * inset_scale)

    # Simple nearest-neighbor upscale (no extra dependency)
    row_idx = (np.arange(new_h) * rh / new_h).astype(int)
    col_idx = (np.arange(new_w) * rw / new_w).astype(int)
    row_idx = np.clip(row_idx, 0, rh - 1)
    col_idx = np.clip(col_idx, 0, rw - 1)
    scaled = patch[np.ix_(row_idx, col_idx)]

    # Determine inset placement
    if inset_position == InsetPosition.TOP_LEFT:
        ix, iy = margin, margin
    elif inset_position == InsetPosition.TOP_RIGHT:
        ix, iy = w - new_w - margin, margin
    elif inset_position == InsetPosition.BOTTOM_LEFT:
        ix, iy = margin, h - new_h - margin
    else:  # BOTTOM_RIGHT
        ix, iy = w - new_w - margin, h - new_h - margin

    # Clamp to image bounds
    ix = max(0, min(ix, w - new_w))
    iy = max(0, min(iy, h - new_h))

    # Paste scaled patch
    out[iy : iy + new_h, ix : ix + new_w] = scaled

    # Draw border around the inset
    _draw_rect(out, ix, iy, new_w, new_h, color, border_width)

    return out


def _draw_rect(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: np.ndarray,
    thickness: int,
) -> None:
    """Draw a rectangle border on the image (in-place)."""
    ih, iw = img.shape[:2]
    # Top edge
    y0, y1 = max(0, y), min(ih, y + thickness)
    x0, x1 = max(0, x), min(iw, x + w)
    img[y0:y1, x0:x1] = color
    # Bottom edge
    y0, y1 = max(0, y + h - thickness), min(ih, y + h)
    img[y0:y1, x0:x1] = color
    # Left edge
    y0, y1 = max(0, y), min(ih, y + h)
    x0, x1 = max(0, x), min(iw, x + thickness)
    img[y0:y1, x0:x1] = color
    # Right edge
    x0, x1 = max(0, x + w - thickness), min(iw, x + w)
    img[y0:y1, x0:x1] = color
