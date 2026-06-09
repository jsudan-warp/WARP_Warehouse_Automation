"""Lens undistortion (Phase 3 Part 1) — apply a known lens model to frames.

Given a camera matrix K and distortion coefficients `dist`, undistort frames.
For video we precompute the remap once and reuse it (much faster than calling
cv2.undistort per frame).

Resolution gotcha (from the brief): `dist` is resolution-independent but K scales
with pixel size. If the lens was calibrated at 4K and we process the 848x478
preview, K must be scaled to the processing resolution first — `scale_K` does that.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_intrinsics(path: str | Path) -> dict | None:
    """Load intrinsics.npz -> {K, dist, image_size, ...} or None if absent."""
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    out = {
        "K": data["K"].astype(np.float64),
        "dist": data["dist"].astype(np.float64).reshape(1, -1),
        "image_size": tuple(int(v) for v in data["image_size"]),  # (w, h) at calibration
    }
    for k in ("rms", "model", "n_images", "square_size_m"):
        if k in data:
            out[k] = data[k].item() if data[k].shape == () else data[k]
    return out


def scale_K(K: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    """Scale a camera matrix from one resolution to another. Sizes are (w, h)."""
    (fw, fh), (tw, th) = from_size, to_size
    sx, sy = tw / fw, th / fh
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx   # fx
    K2[0, 2] *= sx   # cx
    K2[1, 1] *= sy   # fy
    K2[1, 2] *= sy   # cy
    return K2


class Undistorter:
    """Cached remap for a fixed processing size. alpha=0 crops to valid pixels;
    alpha=1 keeps the whole FOV (black wedges)."""

    def __init__(self, intr: dict, proc_size: tuple[int, int], alpha: float = 0.0):
        w, h = proc_size
        K = scale_K(intr["K"], intr["image_size"], proc_size)
        dist = intr["dist"]
        newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            K, dist, None, newK, (w, h), cv2.CV_16SC2)
        self.K, self.dist, self.newK = K, dist, newK
        self.roi = tuple(int(v) for v in roi)
        self.proc_size = proc_size
        self.alpha = alpha

    def __call__(self, img: np.ndarray, crop: bool = False) -> np.ndarray:
        und = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
        if crop and self.roi[2] > 0 and self.roi[3] > 0:
            x, y, ww, hh = self.roi
            und = und[y:y + hh, x:x + ww]
        return und


def undistort_image(img: np.ndarray, intr: dict, alpha: float = 0.0,
                    crop: bool = False) -> np.ndarray:
    """One-shot undistort of a single image at its own resolution."""
    h, w = img.shape[:2]
    return Undistorter(intr, (w, h), alpha)(img, crop=crop)
