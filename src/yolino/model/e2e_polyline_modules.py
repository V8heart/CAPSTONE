# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# Differentiable E2E polyline head skeleton (soft local pooling, optional
# feature-aware attention, Bézier tensor ops). Original scipy/BFS postproc
# remains separate in yolino.postprocessing.line_fit.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolino.utils.enums import LINE, Variables
from yolino.utils.logger import Log


def bernstein_matrix(degree: int, num_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Matrix T of shape [num_samples, degree + 1] with rows B(t_i) for uniform t in [0, 1].
    curve_points = T @ P  (P: [degree+1, 2] control points) → [num_samples, 2].
    """
    n = int(degree)
    if n < 1:
        raise ValueError("Bézier degree must be >= 1 (at least two control points).")
    t = torch.linspace(0.0, 1.0, int(num_samples), device=device, dtype=dtype)
    row_list = []
    for i in range(n + 1):
        c = math.comb(n, i)
        row = c * ((1.0 - t) ** (n - i)) * (t**i)
        row_list.append(row)
    return torch.stack(row_list, dim=1)


def bezier_sample_points(control_points: torch.Tensor, bernstein_t: torch.Tensor) -> torch.Tensor:
    """
    Args:
        control_points: [B, N, K, 2] (K = degree+1)
        bernstein_t: [T, K] precomputed Bernstein basis rows (same K)
    Returns:
        [B, N, T, 2]
    """
    # [B, N, T, 2] = einsum('tk,bnkd->bntd', bernstein_t, control_points)
    return torch.einsum("tk,bnkd->bntd", bernstein_t, control_points)


class SoftArgmaxSegmentTokens(nn.Module):
    """
    Phase 1: replace hard threshold / local winner with temperature-scaled softmax
    over a (2k+1)^2 neighborhood, then weighted sum of neighbor activations.
    Operates on the head grid (same H x W as geometry logits after reshape).
    """

    def __init__(self, window_size: int = 3, temperature: float = 0.5):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd (e.g. 3 or 5).")
        self.window_size = int(window_size)
        self.temperature = float(temperature)

    def forward(self, geom_act: torch.Tensor, conf_index: int) -> torch.Tensor:
        """
        Args:
            geom_act: [B, H, W, P, V] activated training variables (sigmoid/linear per config).
            conf_index: channel index of confidence within V.
        Returns:
            soft_geom: [B, H, W, P, V] same layout, conf unchanged (copied through softmax weights).
        """
        b, h, w, p, v = geom_act.shape
        k = self.window_size
        pad = k // 2
        # [B*P, V, H, W]
        x = geom_act.permute(0, 3, 4, 1, 2).reshape(b * p, v, h, w)
        conf = x[:, conf_index : conf_index + 1]  # [B*P,1,H,W]
        conf_unf = F.unfold(conf, kernel_size=k, padding=pad)  # [B*P, k*k, H*W]
        conf_unf = conf_unf / max(self.temperature, 1e-6)
        wts = F.softmax(conf_unf, dim=1)  # softmax over neighborhood

        soft_ch = []
        for c in range(v):
            ch = x[:, c : c + 1]
            ch_unf = F.unfold(ch, kernel_size=k, padding=pad)
            mixed = (wts * ch_unf).sum(dim=1, keepdim=True)  # [B*P,1,H*W]
            mixed = mixed.view(b * p, 1, h, w)
            soft_ch.append(mixed)
        out = torch.cat(soft_ch, dim=1)  # [B*P, V, H, W]
        out = out.view(b, p, v, h, w).permute(0, 3, 4, 1, 2).contiguous()
        return out


def mid_dir_geom_to_midpoints_pixels(
    geom_act: torch.Tensor,
    stride: float,
    img_h: int,
    img_w: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert MID_DIR activated geometry to midpoint and endpoints in **pixel** coords.

    Channel semantics match the grid / ``get_image_lines`` UV_SPLIT convention used by
    ``plot.draw_cell`` (see comment there: row-like, col-like in the training tensor).
    For MID_DIR from ``to_md_cell_space``, the first four slots are::

        ch0 = midpoint fraction along the **row** axis within the cell (vertical / V),
        ch1 = midpoint fraction along the **col** axis within the cell (horizontal / H),
        ch2 = segment delta along **row** (dV, same units as ch0),
        ch3 = segment delta along **col** (dH, same units as ch1).

    Pixel positions follow ``Grid.get_image_lines``: ``V = row * stride + ...``,
    ``H = col * stride + ...``.  For ``F.grid_sample`` / OpenCV we return **image (x, y)**
    with ``x = H`` (column / width) and ``y = V`` (row / height), i.e. the same ordering
    as ``pixels_to_grid_sample_grid`` expects.

    geom_act: [B, H, W, P, 4+] — uses first four channels as above.
    Returns:
        mid_px: [B, H, W, P, 2]  (x_px, y_px) = (H, V)
        end_a:  [B, H, W, P, 2]
        end_b:  [B, H, W, P, 2]
    """
    b, hh, ww, p, _ = geom_act.shape
    device, dtype = geom_act.device, geom_act.dtype
    col = torch.arange(ww, device=device, dtype=dtype).view(1, 1, ww, 1, 1).expand(b, hh, ww, p, 1)
    row = torch.arange(hh, device=device, dtype=dtype).view(1, hh, 1, 1, 1).expand(b, hh, ww, p, 1)
    s = torch.tensor(float(stride), device=device, dtype=dtype)
    mid_v = geom_act[..., 0:1]
    mid_h = geom_act[..., 1:2]
    d_v = geom_act[..., 2:3]
    d_h = geom_act[..., 3:4]
    # V / H in pixels (row-major image indices).
    v_mid = (row + mid_v) * s
    h_mid = (col + mid_h) * s
    half_v = d_v * (0.5 * s)
    half_h = d_h * (0.5 * s)
    v_a = v_mid - half_v
    h_a = h_mid - half_h
    v_b = v_mid + half_v
    h_b = h_mid + half_h
    # (x, y) for grid_sample / cv2.line without UV_SPLIT swap = (H, V).
    mid_px = torch.cat([h_mid, v_mid], dim=-1)
    end_a = torch.cat([h_a, v_a], dim=-1)
    end_b = torch.cat([h_b, v_b], dim=-1)
    return mid_px, end_a, end_b


def pixels_to_grid_sample_grid(xy_px: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """
    xy_px: [B, N, 2] with (x=width dim, y=height dim) in pixel coords.
    Returns grid for F.grid_sample: [B, N, 1, 2] with (x_norm, y_norm) in [-1, 1].
    """
    x = xy_px[..., 0:1]
    y = xy_px[..., 1:2]
    gx = 2.0 * x / max(float(img_w - 1), 1.0) - 1.0
    gy = 2.0 * y / max(float(img_h - 1), 1.0) - 1.0
    g = torch.cat([gx, gy], dim=-1).view(xy_px.shape[0], xy_px.shape[1], 1, 2)
    return g


