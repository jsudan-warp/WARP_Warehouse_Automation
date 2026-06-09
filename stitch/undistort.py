"""No-calibration barrel undistortion for the 2.8 mm ceiling lens.

We have no checkerboard. But we can pin a sensible camera matrix from the known
field of view and then estimate the radial distortion empirically. Two helpers:

  - focal_from_hfov / build_K : K from HFOV (107 deg) at the processing resolution.
  - undistort_maps / apply    : precomputed remap for a given (k1, k2) Brown model.

The distortion coefficient itself is estimated elsewhere (calib_by_matching.py)
by the value that makes the real overlapping seams match best.
"""
from __future__ import annotations

import numpy as np
import cv2


def focal_from_hfov(width_px: int, hfov_deg: float) -> float:
    """Pinhole focal length in pixels from horizontal FOV."""
    return (width_px / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)


def build_K(width: int, height: int, hfov_deg: float = 107.0) -> np.ndarray:
    f = focal_from_hfov(width, hfov_deg)
    return np.array([[f, 0, width / 2.0],
                     [0, f, height / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def undistort_maps(width, height, K, k1, k2=0.0, k3=0.0, alpha=1.0, new_camera="same"):
    """Precompute (mapx, mapy, newK, valid_mask) for a Brown radial model.

    new_camera='same' keeps newK=K so the undistorted image stays in the SAME pixel
    frame (stable, comparable across k1 — best for estimating k1). 'optimal' uses
    getOptimalNewCameraMatrix (alpha=1 keeps FOV with black wedges; 0 crops)."""
    dist = np.array([[k1, k2, 0.0, 0.0, k3]], dtype=np.float64)
    if new_camera == "same":
        newK = K.copy()
    else:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), alpha)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, (width, height), cv2.CV_16SC2)
    # validity: undistort a white image; nonzero => real (non-black) pixel
    white = np.full((height, width), 255, np.uint8)
    valid = cv2.remap(white, mapx, mapy, cv2.INTER_NEAREST) > 0
    return mapx, mapy, newK, valid.astype(np.uint8) * 255


def apply(img, mapx, mapy):
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


def undistort_maps_full(width, height, K, k1, k2=0.0, k3=0.0, samples=60):
    """Undistort into an EXPANDED canvas so NO field of view is cropped.

    `newK=K` (the calibration-sweep setting) keeps the frame fixed, which clips
    the corrected edges off-frame — and those edges are where tiles overlap. Here
    we undistort the image border, take its bounding box, shift the principal point
    so everything fits, and size the output to hold the whole corrected FOV.
    Returns (mapx, mapy, newK, (Wn, Hn), valid_mask)."""
    dist = np.array([[k1, k2, 0.0, 0.0, k3]], dtype=np.float64)
    xs = np.linspace(0, width - 1, samples)
    ys = np.linspace(0, height - 1, samples)
    border = np.array([[x, 0] for x in xs] + [[x, height - 1] for x in xs] +
                      [[0, y] for y in ys] + [[width - 1, y] for y in ys],
                      np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(border, K, dist, P=K).reshape(-1, 2)
    # guard: the Brown inversion can diverge for a few extreme corner points at
    # strong k1, exploding the canvas. Clamp to a sane multiple of the frame.
    und = np.clip(und, [-1.0 * width, -1.0 * height], [2.0 * width, 2.0 * height])
    minx, miny = und.min(0)
    maxx, maxy = und.max(0)
    newK = K.copy().astype(np.float64)
    newK[0, 2] -= minx
    newK[1, 2] -= miny
    Wn, Hn = int(np.ceil(maxx - minx)), int(np.ceil(maxy - miny))
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, (Wn, Hn), cv2.CV_16SC2)
    valid = cv2.remap(np.full((height, width), 255, np.uint8), mapx, mapy, cv2.INTER_NEAREST)
    return mapx, mapy, newK, (Wn, Hn), valid
