"""
Unified I/O handler for .mat (v5/v7/v7.3+) files.
"""

from pathlib import Path
from typing import Any, Union

import numpy as np
import scipy.io as sio


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
