"""
Unified I/O handler for .mat (v5/v7/v7.3+) and .exr files.
"""

from pathlib import Path
from typing import Any, Union

import numpy as np
import scipy.io as sio

import OpenEXR
import Imath


def loadmat(
    path: Union[str, Path], variable_names: list[str] | None = None
) -> dict[str, Any]:
    """Load a .mat file, trying scipy first, then mat73 for v7.3+.

    Args:
        path: Path to the .mat file.
        variable_names: Optional list of keys to selectively load.

    Returns:
        Dict mapping variable names to their values.
    """
    try:
        kw = {"variable_names": variable_names} if variable_names else {}
        return sio.loadmat(str(path), **kw)
    except NotImplementedError:
        import mat73

        kw = {"only_include": variable_names} if variable_names else {}
        return mat73.loadmat(str(path), use_attrdict=True, **kw)


def whosmat(path: Union[str, Path]) -> list[tuple[str, tuple[int, ...], str]]:
    """Read .mat file metadata (name, shape, dtype) without loading data.

    For v5/v7 files, uses scipy.io.whosmat.
    For v7.3+ files, falls back to mat73.

    Returns:
        List of (name, shape, dtype_str) tuples.
    """
    try:
        return sio.whosmat(str(path))
    except NotImplementedError:
        import mat73

        mat = mat73.loadmat(str(path))
        result = []
        for name, value in mat.items():
            arr = np.asarray(value)
            result.append((name, arr.shape, str(arr.dtype)))
        return result


# ---------------------------------------------------------------------------
# EXR support
# ---------------------------------------------------------------------------

# Standard spectral channel names used by the KAIST dataset
_KAIST_WAVELENGTHS = list(range(420, 730, 10))  # 420nm .. 720nm, 31 bands
_KAIST_CHANNEL_NAMES = [f"w{w}nm" for w in _KAIST_WAVELENGTHS]


def loadexr(
    path: Union[str, Path],
    channel_names: list[str] | None = None,
) -> dict[str, Any]:
    """Load an EXR file, returning a dict mapping channel names to 2D arrays.

    Follows the same convention as ``loadmat``: each key maps to a numpy
    array.  In addition, two convenience keys are synthesised when
    ``channel_names`` is ``None``:

    - ``"rgb"``: (H, W, 3) float32 array of the R, G, B channels.
    - ``"cube"``: (H, W, 31) float32 array of the 31 spectral bands
      (w420nm … w720nm), if all 31 channels are present.

    Args:
        path: Path to the .exr file.
        channel_names: Optional list of channel names to selectively load.
            If ``None``, all channels in the file are loaded.

    Returns:
        Dict mapping channel names (and ``"rgb"``/``"cube"``) to numpy arrays.
    """

    exr = OpenEXR.InputFile(str(path))
    header = exr.header()

    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    available = list(header["channels"].keys())
    to_load = channel_names if channel_names is not None else available

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    result: dict[str, Any] = {}
    for name in to_load:
        if name not in available:
            continue
        raw = exr.channel(name, pt)
        result[name] = np.frombuffer(raw, dtype=np.float32).reshape(height, width)

    # Synthesise convenience keys when loading all channels
    if channel_names is None:
        # RGB
        if all(c in result for c in ("R", "G", "B")):
            result["rgb"] = np.stack([result["R"], result["G"], result["B"]], axis=-1)

        # Spectral cube
        spectral = [result[c] for c in _KAIST_CHANNEL_NAMES if c in result]
        if len(spectral) == len(_KAIST_CHANNEL_NAMES):
            result["cube"] = np.stack(spectral, axis=-1)

    exr.close()
    return result


def whosexr(path: Union[str, Path]) -> list[tuple[str, tuple[int, ...], str]]:
    """Read .exr file metadata (channel name, shape, pixel type) without
    loading pixel data.

    Returns:
        List of (name, (H, W), pixel_type_str) tuples.
    """
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()

    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    result = []
    for name, chan in header["channels"].items():
        result.append((name, (height, width), str(chan.type)))

    exr.close()
    return result

