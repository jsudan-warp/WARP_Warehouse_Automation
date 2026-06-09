"""Line-based (plumb-line) distortion estimation from the floor/box straight edges.

Principle (Alemán-Flores / classic plumb-line): a straight 3D line projects to a
straight 2D line under a pinhole camera; the lens bends it. So the distortion that
makes the image's long straight edges *straightest* is the right one. We don't need
to isolate floor joints — every long structural edge (rack, box, joint) is valid.

Metric: with FastLineDetector(do_merge=True), curved lines fragment into short
segments; at the correct undistortion they straighten and MERGE into long segments,
so the total length of long (>=100px) merged segments PEAKS. We sweep Brown k1
(K pinned from HFOV, same as the matching estimate) and find that peak — an estimate
that needs NO overlap, so it works for all 14 cameras.

Run:  ./venv/bin/python stitch/line_undistort.py
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from edge_match import OUT  # type: ignore
import undistort as ud

STILLS = OUT / "stills"
# cameras with plenty of long straight structure (floor joints + rack/box edges)
CAMS = ["06", "07", "08", "09", "11", "12", "13", "14"]
K1_GRID = [0.0, -0.02, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.12, -0.15]
LONG = 100.0      # a segment counts as "long structural line" above this length (px)

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def fld():
    return cv2.ximgproc.createFastLineDetector(
        length_threshold=40, distance_threshold=1.414,
        canny_th1=40, canny_th2=120, canny_aperture_size=3, do_merge=True)


def long_line_evidence(gray, center, radial_drop_deg=15.0):
    """Total length of long, NON-radial merged segments.

    A segment pointing toward the image centre is radial; radial lines stay
    straight under radial distortion regardless of k1, so they carry no signal
    and only dilute the metric (panel insight). Drop them."""
    det = fld()
    segs = det.detect(gray)
    if segs is None:
        return 0.0, 0
    segs = segs.reshape(-1, 4)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    mid = np.stack([(segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2], 1)
    seg_dir = np.stack([segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]], 1)
    seg_dir /= (np.linalg.norm(seg_dir, axis=1, keepdims=True) + 1e-9)
    rad = mid - np.array(center)
    rad /= (np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9)
    cos = np.abs((seg_dir * rad).sum(1))
    tangential = cos < np.cos(np.deg2rad(radial_drop_deg))   # drop near-radial
    keep = (L >= LONG) & tangential
    return float(L[keep].sum()), int(keep.sum())


def main():
    grays = {}
    for c in CAMS:
        im = cv2.imread(str(STILLS / f"{c}.png"))
        if im is not None:
            grays[c] = _CLAHE.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    h, w = next(iter(grays.values())).shape[:2]
    K = ud.build_K(w, h, hfov_deg=107.0)

    print(f"\nLine-straightness (plumb-line) distortion estimate over {len(grays)} cameras")
    print(f"  metric = total length of merged straight segments >= {LONG:.0f}px (higher = straighter)\n")
    print(f"  {'k1':>6}  {'total_long_len':>14}  {'#long_segs':>10}")
    print("  " + "-" * 40)

    best, sweep = (None, -1), []
    for k1 in K1_GRID:
        if k1 == 0.0:
            grays_u = grays
        else:
            mapx, mapy, _, _ = ud.undistort_maps(w, h, K, k1, new_camera="same")
            grays_u = {c: ud.apply(g, mapx, mapy) for c, g in grays.items()}
        tot, cnt = 0.0, 0
        center = (w / 2.0, h / 2.0)
        for c in grays_u:
            t, n = long_line_evidence(grays_u[c], center)
            tot += t; cnt += n
        sweep.append({"k1": k1, "total_len": tot, "n_long": cnt})
        print(f"  {k1:6.2f}  {tot:14.0f}  {cnt:10d}")
        if tot > best[1]:
            best = (k1, tot)

    print(f"\n  line-based best k1 = {best[0]:.2f}")
    print(f"  (match-based estimate was k1 = -0.08)")
    (OUT / "line_undistort.json").write_text(
        json.dumps({"best_k1": best[0], "sweep": sweep}, indent=2))

    # before/after line visualisation at the line-based best k1
    cam = "08"
    g = grays[cam]
    mapx, mapy, _, _ = ud.undistort_maps(w, h, K, best[0], new_camera="same")
    gu = ud.apply(g, mapx, mapy)
    det = fld()
    out = []
    for tag, gg in (("raw", g), (f"undist_k1={best[0]}", gu)):
        vis = cv2.cvtColor(gg, cv2.COLOR_GRAY2BGR)
        segs = det.detect(gg)
        if segs is not None:
            for s in segs.reshape(-1, 4):
                if np.hypot(s[2] - s[0], s[3] - s[1]) >= LONG:
                    cv2.line(vis, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 0, 255), 2)
        cv2.putText(vis, tag, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        out.append(vis)
    cv2.imwrite(str(OUT / "line_undistort_check.png"), np.hstack(out))
    print(f"  before/after long lines -> stitch/out/line_undistort_check.png")
    return best


if __name__ == "__main__":
    main()
