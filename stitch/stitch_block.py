"""Stitch the connected {07,08,12,13} block into one mosaic (undistorted tiles).

Connectivity (post-undistortion): 07-08 (strong), 12-13 (strong), 08-13 (link).
We chain SIMILARITY transforms (rotation + uniform scale + translation) — the right,
stable model for same-height near-nadir floor tiles with thin overlaps (a full
homography is ill-conditioned when matches sit in a narrow band).

Spanning tree rooted at 07:   07 -> 08 -> 13 -> 12
Run:  ./venv/bin/python stitch/stitch_block.py
"""
from __future__ import annotations

import numpy as np
import cv2

from edge_match import OUT  # type: ignore

STILLS_U = OUT / "stills_undist"
ROOT = "07"
EDGES = [("08", "07"), ("13", "08"), ("12", "13")]   # (child, parent)

_SIFT = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.006,
                        edgeThreshold=16, sigma=1.6)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def detect(img):
    g = _CLAHE.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return _SIFT.detectAndCompute(g, None)


def similarity_child_to_parent(child, parent):
    """Estimate a 3x3 similarity mapping child-image pixels into parent-image frame."""
    kpc, desc = detect(child)
    kpp, desp = detect(parent)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc, desp, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    pc = np.float32([kpc[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pp = np.float32([kpp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(pc, pp, method=cv2.RANSAC,
                                         ransacReprojThreshold=4.0)
    if M is None:
        raise RuntimeError(f"failed to estimate transform {parent}<-child")
    H = np.vstack([M, [0, 0, 1]])
    scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    return H, int(inl.sum()), scale


def corners(w, h):
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def main():
    tiles = [ROOT] + [c for c, _ in EDGES]
    imgs = {t: cv2.imread(str(STILLS_U / f"{t}.png")) for t in tiles}
    if any(v is None for v in imgs.values()):
        raise SystemExit("missing undistorted stills — run edge_match_undist.py first")
    h, w = imgs[ROOT].shape[:2]

    T = {ROOT: np.eye(3)}
    print(f"\nStitching block {tiles} (root {ROOT}); similarity chain:")
    for child, parent in EDGES:
        H, inl, scale = similarity_child_to_parent(imgs[child], imgs[parent])
        T[child] = T[parent] @ H
        print(f"  {child} <- {parent}: {inl} inliers, scale={scale:.3f}")

    # global canvas bounds
    allc = []
    for t in tiles:
        allc.append(cv2.perspectiveTransform(corners(w, h), T[t]).reshape(-1, 2))
    allc = np.vstack(allc)
    minx, miny = allc.min(0)
    maxx, maxy = allc.max(0)
    off = np.array([[1, 0, -minx], [0, 1, -miny], [0, 0, 1]], np.float64)
    W, Hc = int(np.ceil(maxx - minx)), int(np.ceil(maxy - miny))
    print(f"  canvas: {W}x{Hc}")

    # feather-blend
    acc = np.zeros((Hc, W, 3), np.float64)
    wsum = np.zeros((Hc, W), np.float64)
    for t in tiles:
        M = off @ T[t]
        warp = cv2.warpPerspective(imgs[t], M, (W, Hc))
        cover = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, Hc),
                                    flags=cv2.INTER_NEAREST)
        weight = cv2.distanceTransform((cover > 0).astype(np.uint8), cv2.DIST_L2, 3)
        weight = weight + 1e-3
        acc += warp.astype(np.float64) * weight[..., None]
        wsum += weight
    mosaic = (acc / np.maximum(wsum[..., None], 1e-6)).clip(0, 255).astype(np.uint8)

    out = OUT / "mosaic_block.png"
    cv2.imwrite(str(out), mosaic)
    print(f"\n  saved mosaic -> {out}")
    return str(out)


if __name__ == "__main__":
    main()
