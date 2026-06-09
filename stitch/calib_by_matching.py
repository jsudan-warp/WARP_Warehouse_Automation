"""Estimate the barrel distortion WITHOUT a checkerboard, by stitch quality.

Correct undistortion should make genuinely-overlapping floor seams match better
(more geometrically-consistent SIFT inliers). So we sweep the radial coefficient
k1 (Brown model, K pinned from HFOV) and, for each value, undistort every still
and re-run edge-region SIFT on the seams that showed any real signal. The k1 that
maximises total inliers across those seams is our empirical distortion estimate.

Run:  ./venv/bin/python stitch/calib_by_matching.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from edge_match import (GRID, load_stills, detect_edge, match_pair, OUT)  # type: ignore
import undistort as ud

# seams that showed real or partial signal in the raw run (worth optimising on)
SIGNAL_SEAMS = [("07", "08", "H"), ("06", "11", "V"), ("12", "13", "H"),
                ("11", "16", "V"), ("03", "08", "V"), ("01", "02", "H")]

K1_GRID = [0.0, -0.02, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10, -0.12, -0.15]


def total_inliers_for(stills_u):
    rows, total = [], 0
    for a, b, kind in SIGNAL_SEAMS:
        if a not in stills_u or b not in stills_u:
            continue
        sa, sb = ("R", "L") if kind == "H" else ("B", "T")
        kpa, desa, _ = detect_edge(stills_u[a], sa)
        kpb, desb, _ = detect_edge(stills_u[b], sb)
        _, inl, _, _ = match_pair(kpa, desa, kpb, desb)
        rows.append((f"{a}-{b}", inl))
        total += inl
    return total, rows


def main():
    stills = load_stills(GRID)
    h, w = next(iter(stills.values())).shape[:2]
    K = ud.build_K(w, h, hfov_deg=107.0)
    print(f"\nMatch-maximisation distortion estimate (K from HFOV: f={K[0,0]:.1f}px @ {w}x{h})")
    print(f"  optimising total inliers over seams: {[f'{a}-{b}' for a,b,_ in SIGNAL_SEAMS]}\n")
    print(f"  {'k1':>6}  {'total_inliers':>13}   per-seam")
    print("  " + "-" * 60)

    best = (None, -1, None)
    sweep = []
    for k1 in K1_GRID:
        if k1 == 0.0:
            stills_u = stills
        else:
            mapx, mapy, _, _ = ud.undistort_maps(w, h, K, k1, new_camera="same")
            stills_u = {c: ud.apply(im, mapx, mapy) for c, im in stills.items()}
        total, rows = total_inliers_for(stills_u)
        sweep.append({"k1": k1, "total": total, "per_seam": rows})
        per = " ".join(f"{name}:{inl}" for name, inl in rows)
        print(f"  {k1:6.2f}  {total:13d}   {per}")
        if total > best[1]:
            best = (k1, total, rows)

    print(f"\n  BEST k1 = {best[0]:.2f}  (total inliers {best[1]})")
    (OUT / "calib_by_matching.json").write_text(
        json.dumps({"best_k1": best[0], "sweep": sweep, "f_px": float(K[0, 0])}, indent=2))
    print(f"  saved -> stitch/out/calib_by_matching.json")
    return best


if __name__ == "__main__":
    main()
