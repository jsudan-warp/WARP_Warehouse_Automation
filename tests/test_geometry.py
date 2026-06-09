"""Pure-geometry unit tests (no OpenCV, no data needed)."""
import numpy as np
import pytest

from lax9.geometry import (
    apply_homography, homography_roundtrip_error,
    mask_base_point, box_base_point, fit_similarity, astar,
)


def test_homography_roundtrip():
    # a non-trivial projective transform
    H = np.array([[1.2, 0.1, 30.0],
                  [0.05, 0.9, -20.0],
                  [1e-4, 2e-4, 1.0]])
    pts = np.array([[0, 0], [100, 0], [100, 200], [0, 200], [37, 91]], float)
    err = homography_roundtrip_error(H, pts)
    assert np.all(err < 1e-6)


def test_apply_homography_single_point():
    H = np.eye(3)
    H[0, 2] = 5.0
    out = apply_homography(H, np.array([1.0, 2.0]))
    assert out.shape == (2,)
    assert np.allclose(out, [6.0, 2.0])


def test_box_base_point():
    # bottom-edge midpoint is the floor-contact point
    assert box_base_point([10, 20, 30, 80]) == (20.0, 80.0)


def test_mask_base_point():
    # a tall rectangle: base must be at the BOTTOM, centred in x
    mask = np.zeros((100, 100), bool)
    mask[10:90, 40:60] = True       # rows 10..89, cols 40..59
    x, y = mask_base_point(mask)
    assert y == 89
    assert abs(x - 49.5) < 1.0


def test_fit_similarity_recovers_known_transform():
    # build a known similarity: rotate 30deg, scale 0.05 (px->m), translate
    theta = np.deg2rad(30)
    s = 0.05
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    t = np.array([3.0, -7.0])
    src = np.array([[0, 0], [800, 0], [800, 600], [0, 600], [123, 456]], float)
    dst = (s * (R @ src.T).T) + t

    T, scale, rms = fit_similarity(src, dst)
    assert abs(scale - s) < 1e-9
    assert rms < 1e-9
    assert np.allclose(apply_homography(T, src), dst, atol=1e-7)


def test_fit_similarity_noise_is_small():
    rng = np.random.default_rng(0)
    s, theta = 0.04, np.deg2rad(-12)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([1.0, 2.0])
    src = rng.uniform(0, 1000, size=(20, 2))
    dst = (s * (R @ src.T).T) + t + rng.normal(0, 0.002, size=(20, 2))
    T, scale, rms = fit_similarity(src, dst)
    assert abs(scale - s) < 1e-3
    assert rms < 0.01


def test_astar_simple_path():
    occ = np.zeros((10, 10), bool)
    path = astar(occ, (0, 0), (9, 9))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (9, 9)


def test_astar_respects_walls():
    occ = np.zeros((10, 10), bool)
    occ[:, 5] = True            # full vertical wall
    assert astar(occ, (0, 0), (0, 9)) is None  # wall has no gap -> blocked

    occ[0, 5] = False           # open a gap
    path = astar(occ, (0, 0), (0, 9))
    assert path is not None
    assert all(not occ[r, c] for r, c in path)


def test_astar_blocked_endpoints():
    occ = np.zeros((5, 5), bool)
    occ[4, 4] = True
    assert astar(occ, (0, 0), (4, 4)) is None
    assert astar(occ, (0, 0), (0, 0)) == [(0, 0)]


def test_astar_no_diagonal_cut_through_corner():
    # A diagonal pinch in the middle of an open grid: the planner must route
    # around it, never squeezing diagonally between two blocked orthogonal cells.
    occ = np.zeros((4, 4), bool)
    occ[1, 2] = True
    occ[2, 1] = True
    path = astar(occ, (0, 0), (3, 3))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (3, 3)
    # invariant: no diagonal step cuts between two blocked orthogonal neighbours
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dr, dc = r1 - r0, c1 - c0
        if dr != 0 and dc != 0:
            assert not (occ[r0 + dr, c0] and occ[r0, c0 + dc])
