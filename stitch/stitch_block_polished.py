"""Polished stitch of the {07,08,12,13} block.

Improvements over stitch_block.py:
  1) GLOBAL similarity bundle adjustment. Every tile gets a similarity transform
     [[a,-b,tx],[b,a,ty]]. The constraint "matched points land in the same global
     spot" is LINEAR in (a,b,tx,ty), so we pool ALL pairwise inlier correspondences
     and solve one linear least-squares (07 fixed as reference). The strong
     horizontal seams (07-08, 12-13) then help constrain the weak vertical one (08-13).
  2) Exposure/gain compensation (log-domain least-squares over the overlap means).
  3) Multi-band blending (falls back to feather if cv2.detail is unavailable).

Run:  ./venv/bin/python stitch/stitch_block_polished.py
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import cv2

from edge_match import OUT  # type: ignore

STILLS_U = OUT / "stills_undist"
TILES = ["07", "08", "12", "13"]
ROOT = "07"
MIN_INLIERS = 8        # include a pair in the bundle only above this

_SIFT = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.006,
                        edgeThreshold=16, sigma=1.6)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def detect(img):
    g = _CLAHE.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return _SIFT.detectAndCompute(g, None)


def inlier_correspondences(kpi, desi, kpj, desj):
    """Return matched inlier point arrays (Ni x 2 in image i, same in image j)."""
    if desi is None or desj is None:
        return np.empty((0, 2)), np.empty((0, 2))
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desi, desj, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 4:
        return np.empty((0, 2)), np.empty((0, 2))
    pi = np.float32([kpi[m.queryIdx].pt for m in good])
    pj = np.float32([kpj[m.trainIdx].pt for m in good])
    _, inl = cv2.estimateAffinePartial2D(pi.reshape(-1, 1, 2), pj.reshape(-1, 1, 2),
                                         method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if inl is None:
        return np.empty((0, 2)), np.empty((0, 2))
    keep = inl.ravel().astype(bool)
    return pi[keep], pj[keep]


def global_similarity_ba(corr, tiles, root):
    """Linear least-squares for per-tile similarity params [a,b,tx,ty]; root fixed."""
    nonroot = [t for t in tiles if t != root]
    col = {t: 4 * k for k, t in enumerate(nonroot)}
    root_u = np.array([1.0, 0.0, 0.0, 0.0])

    rows, rhs = [], []

    def terms(t, x, y, sign):
        # contributions of tile t's [a,b,tx,ty] to (x_eq, y_eq) for point (x,y)
        ax = {"a": x, "b": -y, "tx": 1.0, "ty": 0.0}
        ay = {"a": y, "b": x, "tx": 0.0, "ty": 1.0}
        return ax, ay

    for (i, j), (pi, pj) in corr.items():
        for (xi, yi), (xj, yj) in zip(pi, pj):
            for eq_sel in (0, 1):       # x-equation then y-equation
                rowv = np.zeros(4 * len(nonroot))
                r = 0.0
                for (t, x, y, s) in ((i, xi, yi, +1.0), (j, xj, yj, -1.0)):
                    ax, ay = terms(t, x, y, s)
                    sel = ax if eq_sel == 0 else ay
                    coeffs = np.array([sel["a"], sel["b"], sel["tx"], sel["ty"]]) * s
                    if t == root:
                        r -= coeffs @ root_u    # move known root to RHS
                    else:
                        rowv[col[t]:col[t] + 4] += coeffs
                rows.append(rowv)
                rhs.append(r)

    A = np.array(rows)
    c = np.array(rhs)
    u, *_ = np.linalg.lstsq(A, c, rcond=None)

    T = {root: np.eye(3)}
    for t in nonroot:
        a, b, tx, ty = u[col[t]:col[t] + 4]
        T[t] = np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])
    return T


def exposure_gains(warps, masks, tiles, root):
    """Per-tile multiplicative gain to equalise brightness across overlaps (log-LSQ)."""
    grays = {t: cv2.cvtColor(warps[t], cv2.COLOR_BGR2GRAY).astype(np.float64) for t in tiles}
    nonroot = [t for t in tiles if t != root]
    idx = {t: k for k, t in enumerate(nonroot)}
    rows, rhs = [], []
    for i, j in combinations(tiles, 2):
        ov = (masks[i] > 0) & (masks[j] > 0)
        if ov.sum() < 50:
            continue
        mi, mj = grays[i][ov].mean(), grays[j][ov].mean()
        if mi < 1 or mj < 1:
            continue
        row = np.zeros(len(nonroot))
        if i != root:
            row[idx[i]] += 1
        if j != root:
            row[idx[j]] -= 1
        rows.append(row)
        rhs.append(np.log(mj) - np.log(mi))
    # weak prior: gains near 1
    for t in nonroot:
        row = np.zeros(len(nonroot)); row[idx[t]] = 0.05
        rows.append(row); rhs.append(0.0)
    if not rows:
        return {t: 1.0 for t in tiles}
    g, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    gains = {root: 1.0}
    for t in nonroot:
        gains[t] = float(np.clip(np.exp(g[idx[t]]), 0.6, 1.7))
    return gains


def blend(warps, masks, tiles, W, H):
    """Multi-band blend; fall back to distance-weighted feather."""
    try:
        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(5)
        blender.prepare((0, 0, W, H))
        for t in tiles:
            blender.feed(warps[t].astype(np.int16), masks[t], (0, 0))
        res, _ = blender.blend(None, None)
        return np.clip(res, 0, 255).astype(np.uint8), "multiband"
    except Exception as e:  # noqa: BLE001
        acc = np.zeros((H, W, 3), np.float64)
        wsum = np.zeros((H, W), np.float64)
        for t in tiles:
            wt = cv2.distanceTransform((masks[t] > 0).astype(np.uint8), cv2.DIST_L2, 3) + 1e-3
            acc += warps[t].astype(np.float64) * wt[..., None]
            wsum += wt
        return (acc / np.maximum(wsum[..., None], 1e-6)).clip(0, 255).astype(np.uint8), f"feather ({e})"


def main():
    imgs = {t: cv2.imread(str(STILLS_U / f"{t}.png")) for t in TILES}
    if any(v is None for v in imgs.values()):
        raise SystemExit("missing undistorted stills — run edge_match_undist.py first")
    h, w = imgs[ROOT].shape[:2]
    feats = {t: detect(imgs[t]) for t in TILES}

    corr = {}
    print("\nPolished block stitch — pairwise inliers used in the global bundle:")
    for i, j in combinations(TILES, 2):
        pi, pj = inlier_correspondences(*feats[i], *feats[j])
        if len(pi) >= MIN_INLIERS:
            corr[(i, j)] = (pi, pj)
            print(f"  {i}-{j}: {len(pi)} inliers  (used)")
        else:
            print(f"  {i}-{j}: {len(pi)} inliers  (skipped, < {MIN_INLIERS})")

    T = global_similarity_ba(corr, TILES, ROOT)

    # residual check: mean reprojection error over all used correspondences (px)
    errs = []
    for (i, j), (pi, pj) in corr.items():
        gi = cv2.perspectiveTransform(pi.reshape(-1, 1, 2), T[i]).reshape(-1, 2)
        gj = cv2.perspectiveTransform(pj.reshape(-1, 1, 2), T[j]).reshape(-1, 2)
        errs.append(np.linalg.norm(gi - gj, axis=1))
    err = np.concatenate(errs)
    print(f"  global BA reprojection error: mean {err.mean():.2f}px, median "
          f"{np.median(err):.2f}px, 90th {np.percentile(err,90):.2f}px")

    # canvas
    allc = np.vstack([cv2.perspectiveTransform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), T[t]).reshape(-1, 2)
        for t in TILES])
    minx, miny = allc.min(0); maxx, maxy = allc.max(0)
    off = np.array([[1, 0, -minx], [0, 1, -miny], [0, 0, 1]], np.float64)
    W, H = int(np.ceil(maxx - minx)), int(np.ceil(maxy - miny))

    warps, masks = {}, {}
    for t in TILES:
        M = off @ T[t]
        warps[t] = cv2.warpPerspective(imgs[t], M, (W, H))
        masks[t] = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H),
                                       flags=cv2.INTER_NEAREST)

    gains = exposure_gains(warps, masks, TILES, ROOT)
    print(f"  exposure gains: " + ", ".join(f"{t}={gains[t]:.2f}" for t in TILES))
    for t in TILES:
        warps[t] = np.clip(warps[t].astype(np.float64) * gains[t], 0, 255).astype(np.uint8)

    mosaic, how = blend(warps, masks, TILES, W, H)
    out = OUT / "mosaic_block_polished.png"
    cv2.imwrite(str(out), mosaic)
    print(f"  blend: {how};  canvas {W}x{H}")
    print(f"  saved -> {out}")
    return str(out)


if __name__ == "__main__":
    main()
