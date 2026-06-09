"""Re-estimate the barrel k1 on FULL-FOV undistorted plates.

The earlier estimate used newK=K (cropped), which clipped the corrected edges off
-frame — so the very region where barrel is strongest never moved, biasing k1 low.
Now we undistort into the expanded canvas (undistort_maps_full) at each k1 and
maximize total inliers across the real seams. Denoised median plates are used.

Run:  ./venv/bin/python stitch/recalibrate.py
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from edge_match import OUT  # type: ignore
from mosaic_full import detect  # type: ignore
import undistort as ud

SEAMS = [("07", "08"), ("06", "11"), ("12", "13"), ("08", "13"), ("01", "06"),
         ("07", "12"), ("11", "12"), ("09", "14"), ("03", "04"), ("07", "13")]
K1S = [-0.05, -0.07, -0.08, -0.09, -0.10, -0.11, -0.12, -0.14]
NEEDED = sorted({t for ab in SEAMS for t in ab})


def inliers(fa, fb):
    (kpa, desa), (kpb, desb) = fa, fb
    if desa is None or desb is None:
        return 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desa, desb, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 6:
        return 0
    pa = np.float32([kpa[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pb = np.float32([kpb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return int(inl.sum()) if inl is not None else 0


def main():
    plates = {}
    for t in NEEDED:
        im = cv2.imread(str(OUT / "plates" / f"{t}.png"))
        if im is not None:
            plates[t] = im
    h0, w0 = next(iter(plates.values())).shape[:2]
    K = ud.build_K(w0, h0, 107.0)

    print(f"\nRecalibrating k1 on FULL-FOV plates (maximize inliers over {len(SEAMS)} seams)\n")
    print(f"  {'k1':>6} {'canvas':>11} {'total_inliers':>13}   per-seam(>=12)")
    print("  " + "-" * 64)
    best, sweep = (None, -1), []
    for k1 in K1S:
        mapx, mapy, newK, (Wn, Hn), _ = ud.undistort_maps_full(w0, h0, K, k1)
        feats = {t: detect(cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR))  # SIFT once per tile
                 for t, im in plates.items()}
        per = []
        tot = 0
        for a, b in SEAMS:
            if a in feats and b in feats:
                n = inliers(feats[a], feats[b]); tot += n
                if n >= 12:
                    per.append(f"{a}-{b}:{n}")
        sweep.append({"k1": k1, "total": tot, "size": [Wn, Hn]})
        print(f"  {k1:6.2f} {Wn}x{Hn:<5} {tot:13d}   {' '.join(per)}")
        if tot > best[1]:
            best = (k1, tot)

    print(f"\n  BEST k1 = {best[0]:.2f}  (total inliers {best[1]})")
    json.dump({"best_k1": best[0], "sweep": sweep}, open(OUT / "recalibrate.json", "w"), indent=2)
    print("  saved -> stitch/out/recalibrate.json")
    return best


if __name__ == "__main__":
    main()
