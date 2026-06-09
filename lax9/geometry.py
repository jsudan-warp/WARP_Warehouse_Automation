"""Pure geometry helpers — homography, footprint projection, similarity fit, A*.

Kept free of OpenCV where practical so the unit tests run fast and deterministic.
Homography utilities accept plain numpy 3x3 matrices.
"""
from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------
def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map Nx2 points through a 3x3 homography. Returns Nx2.

    pts may be (2,) for a single point or (N,2).
    """
    H = np.asarray(H, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    single = pts.ndim == 1
    p = np.atleast_2d(pts)
    ones = np.ones((p.shape[0], 1))
    hom = np.hstack([p, ones])          # Nx3
    out = (H @ hom.T).T                  # Nx3
    w = out[:, 2:3]
    # guard against division by ~0 (points at the horizon)
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    res = out[:, :2] / w
    return res[0] if single else res


def homography_roundtrip_error(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Forward then inverse a homography; return per-point residual (pixels)."""
    Hinv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    fwd = apply_homography(H, pts)
    back = apply_homography(Hinv, fwd)
    pts2 = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    return np.linalg.norm(back - pts2, axis=1)


# ---------------------------------------------------------------------------
# Footprint / base projection
# ---------------------------------------------------------------------------
def mask_base_point(mask: np.ndarray) -> tuple[float, float]:
    """Floor-contact point of a binary mask: centroid-x at the bottom-most row.

    The base is the only point that obeys the floor-plane assumption (a tall
    stack seen at an angle is offset on the floor by parallax). We take the
    lowest occupied row of the mask (largest image y) and the horizontal
    centroid of the mask within a small band at that bottom edge.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("empty mask")
    y_bottom = ys.max()
    band = ys >= (y_bottom - 2)          # bottom 3 rows
    x_base = float(xs[band].mean())
    return (x_base, float(y_bottom))


def box_base_point(box_xyxy: Iterable[float]) -> tuple[float, float]:
    """Floor-contact point of an axis-aligned box: bottom-edge midpoint."""
    x1, y1, x2, y2 = box_xyxy
    return ((x1 + x2) / 2.0, max(y1, y2))


# ---------------------------------------------------------------------------
# Similarity fit (map pixels -> metres): rotation + uniform scale + translation
# ---------------------------------------------------------------------------
def fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares similarity (Umeyama) mapping src (Nx2) -> dst (Nx2).

    Returns (T, scale, rms) where T is a 3x3 homogeneous matrix such that
    dst ≈ (T @ [src,1])[:2]. Requires >= 2 non-degenerate correspondences.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 2:
        raise ValueError("need matching Nx2 arrays with N>=2")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    var_s = (sc ** 2).sum() / src.shape[0]
    cov = (dc.T @ sc) / src.shape[0]     # 2x2
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    scale = (D * np.diag(S)).sum() / var_s if var_s > 1e-12 else 1.0
    t = mu_d - scale * (R @ mu_s)

    T = np.eye(3)
    T[:2, :2] = scale * R
    T[:2, 2] = t

    proj = apply_homography(T, src)
    rms = float(np.sqrt(((proj - dst) ** 2).sum(axis=1).mean()))
    return T, float(scale), rms


# ---------------------------------------------------------------------------
# A* on an occupancy grid (True = blocked)
# ---------------------------------------------------------------------------
def astar(occ: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
          allow_diagonal: bool = True) -> list[tuple[int, int]] | None:
    """Shortest path on a boolean occupancy grid. (row, col) coords.

    occ[r, c] True means blocked. Returns a list of (row, col) cells from start
    to goal inclusive, or None if no path / endpoints blocked / out of bounds.
    """
    occ = np.asarray(occ, dtype=bool)
    h, w = occ.shape

    def in_bounds(rc):
        return 0 <= rc[0] < h and 0 <= rc[1] < w

    if not (in_bounds(start) and in_bounds(goal)):
        return None
    if occ[start] or occ[goal]:
        return None
    if start == goal:
        return [start]

    if allow_diagonal:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heur(a, b):
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        # octile distance
        return (dr + dc) + (np.sqrt(2) - 2) * min(dr, dc) if allow_diagonal else (dr + dc)

    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    g = {start: 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    closed = set()

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        for dr, dc in steps:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not in_bounds(nxt) or occ[nxt] or nxt in closed:
                continue
            # forbid cutting diagonally between two blocked orthogonal cells
            if dr != 0 and dc != 0:
                if occ[cur[0] + dr, cur[1]] and occ[cur[0], cur[1] + dc]:
                    continue
            step_cost = np.sqrt(2) if (dr != 0 and dc != 0) else 1.0
            ng = g[cur] + step_cost
            if ng < g.get(nxt, np.inf):
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng + heur(nxt, goal), nxt))
    return None
