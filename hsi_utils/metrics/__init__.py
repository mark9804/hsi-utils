import torch
import numpy as np
from hsi_utils.metrics.metrics_utils import torch_ssim, torch_psnr

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    return torch_ssim(img1, img2, window_size, size_average)

def psnr(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return torch_psnr(img, ref)


_lpips_net = None
_hsi_to_rgb_weights = None  # (28, 3) CIE spectral -> XYZ -> sRGB


def _get_lpips_net():
    global _lpips_net # Init singleton LPIPS network instance
    if _lpips_net is None:
        import lpips
        _lpips_net = lpips.LPIPS(net="vgg", verbose=False).cuda().eval()
        for p in _lpips_net.parameters():
            p.requires_grad = False
    return _lpips_net


def _get_hsi_to_rgb_weights(device: torch.device) -> torch.Tensor:
    """Precompute (28, 3) matrix: HSI bands -> linear sRGB via CIE 1964."""
    global _hsi_to_rgb_weights
    if _hsi_to_rgb_weights is not None and _hsi_to_rgb_weights.device == device:
        return _hsi_to_rgb_weights

    from hsi_utils.rendering._wavelength_data import (
        WAVELENGTHS_28,
        wavelength_to_xyz,
        _XYZ_TO_SRGB,
    )
    # (28, 3) XYZ weights, then XYZ -> linear sRGB
    xyz_w = wavelength_to_xyz(WAVELENGTHS_28)           # (28, 3)
    rgb_w = xyz_w @ _XYZ_TO_SRGB.T                     # (28, 3)
    rgb_w = np.clip(rgb_w, 0.0, None)
    _hsi_to_rgb_weights = torch.from_numpy(rgb_w).float().to(device)
    return _hsi_to_rgb_weights


def _hsi_to_rgb_tensor(hsi: torch.Tensor) -> torch.Tensor:
    """Convert (C, H, W) HSI tensor to (3, H, W) sRGB tensor in [0, 1].

    Uses simple linear projection + per-image normalization + gamma=2.2.
    """
    W = _get_hsi_to_rgb_weights(hsi.device)  # (28, 3)
    # (C, H, W) -> (H, W, C) @ (C, 3) -> (H, W, 3) -> (3, H, W)
    rgb = torch.einsum("chw,cd->hwd", hsi.clamp(0, 1), W)
    max_val = rgb.max()
    if max_val > 0:
        rgb = rgb / max_val
    rgb = rgb.pow(1.0 / 2.2).clamp(0, 1)
    return rgb.permute(2, 0, 1)  # (3, H, W)


def lpips_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute LPIPS between two HSI tensors.

    Args:
        pred:   (C, H, W) predicted HSI cube, float in [0, 1].
        target: (C, H, W) ground-truth HSI cube, float in [0, 1].

    Returns:
        Scalar tensor (lower = more similar).
    """
    net = _get_lpips_net()
    # LPIPS expects (B, 3, H, W) in [-1, 1]
    rgb_pred = _hsi_to_rgb_tensor(pred).unsqueeze(0) * 2 - 1
    rgb_target = _hsi_to_rgb_tensor(target).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return net(rgb_pred, rgb_target).squeeze()