"""Tests for the distortion-tolerant image->floor warp (lens-model-free path).

Synthetic camera: world floor points -> ideal pixels (homography) -> barrel
distortion. We then fit pixel->world and check the warp recovers metric floor
coordinates, including on held-out points.
"""
import numpy as np
import pytest

from lax9.warp import FloorWarp, _poly_terms
from lax9.geometry import apply_homography


def _world_grid(nx=11, ny=9):
    xs = np.linspace(-5, 5, nx)
    ys = np.linspace(-3, 3, ny)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _world_to_ideal_pixel(W, H):
    return apply_homography(H, W)


def _barrel(pix, k1=-0.18, c=(424.0, 239.0), R=500.0):
    c = np.array(c)
    d = pix - c
    r2 = (d ** 2).sum(axis=1, keepdims=True) / (R ** 2)
    return c + d * (1.0 + k1 * r2)


H_PERSP = np.array([[70.0, 5.0, 424.0],
                    [-4.0, 72.0, 239.0],
                    [1e-4, 2e-4, 1.0]])
H_AFFINE = np.array([[70.0, 0.0, 424.0],
                     [0.0, 72.0, 239.0],
                     [0.0, 0.0, 1.0]])


def test_poly_term_counts():
    assert len(_poly_terms(1)) == 3
    assert len(_poly_terms(2)) == 6
    assert len(_poly_terms(3)) == 10


def test_poly_recovers_affine_exactly():
    W = _world_grid()
    pix = _world_to_ideal_pixel(W, H_AFFINE)         # no distortion, pure affine
    w = FloorWarp(model="poly", poly_degree=1).fit(pix, W)
    assert w.fit_rms < 1e-6
    rng = np.random.default_rng(0)
    Wt = np.stack([rng.uniform(-5, 5, 50), rng.uniform(-3, 3, 50)], axis=1)
    pixt = _world_to_ideal_pixel(Wt, H_AFFINE)
    assert w.residuals(pixt, Wt)["rms"] < 1e-6


def test_tps_interpolates_control_points():
    W = _world_grid()
    pix = _barrel(_world_to_ideal_pixel(W, H_PERSP))
    w = FloorWarp(model="tps").fit(pix, W)
    assert w.fit_rms < 1e-6                            # exact interpolation at controls


def test_nonlinear_beats_affine_under_barrel():
    rng = np.random.default_rng(1)
    W = _world_grid(13, 11)
    pix = _barrel(_world_to_ideal_pixel(W, H_PERSP))
    Wt = np.stack([rng.uniform(-4.5, 4.5, 80), rng.uniform(-2.7, 2.7, 80)], axis=1)
    pixt = _barrel(_world_to_ideal_pixel(Wt, H_PERSP))

    affine = FloorWarp("poly", poly_degree=1).fit(pix, W).residuals(pixt, Wt)["rms"]
    poly3 = FloorWarp("poly", poly_degree=3).fit(pix, W).residuals(pixt, Wt)["rms"]
    tps = FloorWarp("tps").fit(pix, W).residuals(pixt, Wt)["rms"]

    # the distortion-tolerant fits must dramatically beat a plain affine map
    assert tps < 0.3 * affine
    assert poly3 < 0.6 * affine
    assert tps < 0.05            # < 5 cm-equivalent on held-out floor points


def test_single_point_call_shape():
    W = _world_grid()
    pix = _barrel(_world_to_ideal_pixel(W, H_PERSP))
    w = FloorWarp(model="tps").fit(pix, W)
    out = w(np.array([420.0, 240.0]))
    assert out.shape == (2,)


def test_serialization_roundtrip():
    W = _world_grid()
    pix = _barrel(_world_to_ideal_pixel(W, H_PERSP))
    q = np.array([[400.0, 250.0], [600.0, 300.0], [424.0, 239.0]])
    for model, deg in (("tps", 2), ("poly", 3)):
        w = FloorWarp(model=model, poly_degree=deg).fit(pix, W)
        w2 = FloorWarp.from_dict(w.to_dict())
        assert np.allclose(w(q), w2(q), atol=1e-9)


def test_poly_needs_enough_points():
    W = _world_grid(2, 2)                              # 4 points
    pix = _world_to_ideal_pixel(W, H_AFFINE)
    with pytest.raises(ValueError):
        FloorWarp("poly", poly_degree=3).fit(pix, W)   # cubic needs >= 10
