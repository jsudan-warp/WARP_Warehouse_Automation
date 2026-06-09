"""Does the temporal median plate beat a single frame for seam matching?

Clips are static (motion_frac~0), so plates can't expose new floor — but denoising
might still yield more SIFT inliers. Compare inliers per candidate seam:
single sharp frame vs 48-frame median plate, both undistorted (k1=-0.06).

Run:  ./venv/bin/python stitch/compare_plates.py
"""
from __future__ import annotations

import cv2
import numpy as np

from edge_match import GRID, load_stills, neighbour_pairs, OUT  # type: ignore
from mosaic_full import seam_offset, K1  # type: ignore
import undistort as ud

I = np.eye(2)


def undist_all(imgs, K, mapx, mapy):
    return {t: ud.apply(im, mapx, mapy) for t, im in imgs.items()}


def main():
    stills = load_stills(GRID)
    plates = {}
    for t in stills:
        p = cv2.imread(str(OUT / "plates" / f"{t}.png"))
        if p is not None:
            plates[t] = p
    h, w = stills[next(iter(stills))].shape[:2]
    c = np.array([w / 2.0, h / 2.0])
    K = ud.build_K(w, h, 107.0)
    mapx, mapy, _, _ = ud.undistort_maps(w, h, K, K1, new_camera="same")
    S = undist_all(stills, K, mapx, mapy)
    P = undist_all(plates, K, mapx, mapy)

    print(f"\nSeam inliers: single frame vs 48-frame median plate (undistorted k1={K1})\n")
    print(f"  {'seam':10} {'dir':3} {'single':>7} {'plate':>7}   delta")
    print("  " + "-" * 44)
    tot_s = tot_p = 0
    real_s = real_p = 0
    for a, b, kind in neighbour_pairs(GRID):
        if a not in S or b not in S or a not in P or b not in P:
            continue
        offs, ins, _ = seam_offset(S[a], S[b], I, I, c)
        offp, inp, _ = seam_offset(P[a], P[b], I, I, c)
        ins = ins or 0; inp = inp or 0
        tot_s += ins; tot_p += inp

        def sane(off, inl):
            if off is None or inl < 12:
                return False
            if kind == "H":
                return off[0] < 0 and abs(off[1]) < 0.5 * 0.95 * h and 0.40 * 0.96 * w <= abs(off[0]) <= 1.15 * 0.96 * w
            return off[1] < 0 and abs(off[0]) < 0.5 * 0.96 * w and 0.40 * 0.95 * h <= abs(off[1]) <= 1.15 * 0.95 * h
        rs, rp = sane(offs, ins), sane(offp, inp)
        real_s += rs; real_p += rp
        mark = ("  REAL+" if rp and not rs else ("  REAL-" if rs and not rp else ""))
        print(f"  {a+'-'+b:10} {kind:3} {ins:7d} {inp:7d}   {inp-ins:+d}{mark}")
    print("  " + "-" * 44)
    print(f"  totals: single={tot_s}  plate={tot_p}  ({tot_p-tot_s:+d})")
    print(f"  sane real seams: single={real_s}  plate={real_p}")


if __name__ == "__main__":
    main()
