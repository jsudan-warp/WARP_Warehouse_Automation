"""Distortion-tolerant image->metric-floor mapping (Phase 4/5, lens-model-free).

Decision (Phase 3 gate): skip lens undistortion. Instead, fit ONE warp per camera
straight from measured floor correspondences (image pixel -> world X,Y in metres).
The warp bends with the 2.8 mm barrel lens, lands pixels on metric floor
coordinates in a single step, and anchors each camera independently — so
non-overlapping cameras need no shared features.

Two models (brief: "a thin-plate spline or a 2nd/3rd-order polynomial"):
  - 'tps'  thin-plate spline: smooth, interpolates the control points exactly.
  - 'poly' bivariate polynomial of degree 2 or 3: stiffer, least-squares fit.

All fitting is done in a normalised pixel frame for numerical stability.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Polynomial basis
# ---------------------------------------------------------------------------
def _poly_terms(degree: int) -> list[tuple[int, int]]:
    """Exponent pairs (i, j) for all x^i y^j with i + j <= degree."""
    return [(i, j) for d in range(degree + 1)
            for i in range(d + 1) for j in [d - i]]


def _poly_basis(xy: np.ndarray, degree: int) -> np.ndarray:
    """N x T design matrix for points xy (N x 2) and the given degree."""
    x, y = xy[:, 0], xy[:, 1]
    cols = [(x ** i) * (y ** j) for (i, j) in _poly_terms(degree)]
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Thin-plate spline
# ---------------------------------------------------------------------------
def _tps_U(r2: np.ndarray) -> np.ndarray:
    """TPS radial basis U(r) = r^2 log(r^2), evaluated from squared distance r2."""
    out = np.zeros_like(r2)
    nz = r2 > 1e-12
    out[nz] = r2[nz] * np.log(r2[nz])
    return out


def _pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix between rows of a (M x 2) and b (N x 2)."""
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)


class FloorWarp:
    """Fit image pixels -> world metres with a distortion-tolerant model.

    Usage:
        w = FloorWarp(model='tps').fit(img_pts, world_pts)
        world = w(img_pts)               # forward map
        err = w.residuals(img_pts, world_pts)   # per-point + rms (metres)
    """

    def __init__(self, model: str = "tps", poly_degree: int = 2, reg: float = 0.0):
        if model not in ("tps", "poly"):
            raise ValueError("model must be 'tps' or 'poly'")
        self.model = model
        self.poly_degree = int(poly_degree)
        self.reg = float(reg)
        self._fitted = False

    # ---- normalisation (for numerical stability) ------------------------
    def _normalise(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self._mu) / self._sc

    def fit(self, img_pts, world_pts) -> "FloorWarp":
        img_pts = np.asarray(img_pts, dtype=np.float64)
        world_pts = np.asarray(world_pts, dtype=np.float64)
        if img_pts.shape != world_pts.shape or img_pts.ndim != 2 or img_pts.shape[1] != 2:
            raise ValueError("img_pts and world_pts must both be N x 2 and equal length")
        n = img_pts.shape[0]

        self._mu = img_pts.mean(axis=0)
        sc = img_pts.std(axis=0)
        self._sc = np.where(sc < 1e-9, 1.0, sc)
        P = self._normalise(img_pts)

        if self.model == "poly":
            need = len(_poly_terms(self.poly_degree))
            if n < need:
                raise ValueError(f"poly degree {self.poly_degree} needs >= {need} points, got {n}")
            B = _poly_basis(P, self.poly_degree)
            coeffs, *_ = np.linalg.lstsq(B, world_pts, rcond=None)
            self._coeffs = coeffs                      # T x 2
        else:  # tps
            if n < 3:
                raise ValueError("TPS needs >= 3 points")
            K = _tps_U(_pairwise_sq(P, P))
            K[np.diag_indices_from(K)] += self.reg
            Pm = np.hstack([np.ones((n, 1)), P])       # n x 3
            L = np.zeros((n + 3, n + 3))
            L[:n, :n] = K
            L[:n, n:] = Pm
            L[n:, :n] = Pm.T
            Y = np.vstack([world_pts, np.zeros((3, 2))])
            params = np.linalg.solve(L, Y)
            self._w = params[:n]                       # n x 2
            self._a = params[n:]                       # 3 x 2
            self._ctrl = P

        self._fitted = True
        res = self.residuals(img_pts, world_pts)
        self.fit_rms = float(res["rms"])
        self.fit_max = float(res["max"])
        self.n_points = n
        return self

    def __call__(self, img_pts) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call .fit() first")
        single = np.asarray(img_pts).ndim == 1
        pts = np.atleast_2d(np.asarray(img_pts, dtype=np.float64))
        P = self._normalise(pts)
        if self.model == "poly":
            out = _poly_basis(P, self.poly_degree) @ self._coeffs
        else:
            U = _tps_U(_pairwise_sq(P, self._ctrl))    # M x n
            Pm = np.hstack([np.ones((P.shape[0], 1)), P])
            out = U @ self._w + Pm @ self._a
        return out[0] if single else out

    def residuals(self, img_pts, world_pts) -> dict:
        pred = self(img_pts)
        pred = np.atleast_2d(pred)
        world_pts = np.atleast_2d(np.asarray(world_pts, dtype=np.float64))
        d = np.linalg.norm(pred - world_pts, axis=1)
        return {"per_point": d, "rms": float(np.sqrt((d ** 2).mean())),
                "max": float(d.max()), "mean": float(d.mean())}

    # ---- serialisation --------------------------------------------------
    def to_dict(self) -> dict:
        if not self._fitted:
            raise RuntimeError("call .fit() first")
        d = {"model": self.model, "mu": self._mu.tolist(), "sc": self._sc.tolist(),
             "fit_rms": self.fit_rms, "fit_max": self.fit_max, "n_points": self.n_points}
        if self.model == "poly":
            d.update({"poly_degree": self.poly_degree, "coeffs": self._coeffs.tolist()})
        else:
            d.update({"reg": self.reg, "w": self._w.tolist(),
                      "a": self._a.tolist(), "ctrl": self._ctrl.tolist()})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FloorWarp":
        w = cls(model=d["model"], poly_degree=d.get("poly_degree", 2), reg=d.get("reg", 0.0))
        w._mu = np.array(d["mu"]); w._sc = np.array(d["sc"])
        w.fit_rms = d.get("fit_rms", float("nan")); w.fit_max = d.get("fit_max", float("nan"))
        w.n_points = d.get("n_points", 0)
        if d["model"] == "poly":
            w._coeffs = np.array(d["coeffs"])
        else:
            w._w = np.array(d["w"]); w._a = np.array(d["a"]); w._ctrl = np.array(d["ctrl"])
        w._fitted = True
        return w
