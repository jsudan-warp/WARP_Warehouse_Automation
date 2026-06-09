"""Re-map all 19 grid-neighbour seams AFTER undistortion (k1 from calib_by_matching).

Undistortion promoted weak seams into real ones in the calibration sweep, so re-test
the full neighbour set to learn the true post-undistortion connectivity graph.

Run:  ./venv/bin/python stitch/edge_match_undist.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from edge_match import (GRID, load_stills, neighbour_pairs, detect_edge,
                        match_pair, draw, OUT, MIN_INLIERS)  # type: ignore
import undistort as ud

K1 = -0.08      # from calib_by_matching.py
MATCHES_U = OUT / "matches_undist"
STILLS_U = OUT / "stills_undist"
for d in (MATCHES_U, STILLS_U):
    d.mkdir(parents=True, exist_ok=True)


def main():
    stills = load_stills(GRID)
    h, w = next(iter(stills.values())).shape[:2]
    K = ud.build_K(w, h, hfov_deg=107.0)
    mapx, mapy, newK, valid = ud.undistort_maps(w, h, K, K1, new_camera="same")
    stills_u = {}
    for c, im in stills.items():
        u = ud.apply(im, mapx, mapy)
        stills_u[c] = u
        cv2.imwrite(str(STILLS_U / f"{c}.png"), u)

    # before/after sample for the user (camera 07)
    sample = "07" if "07" in stills else next(iter(stills))
    cv2.imwrite(str(OUT / "undistort_before_after.png"),
                np.hstack([stills[sample], stills_u[sample]]))

    pairs = neighbour_pairs(GRID)
    print(f"\nPost-undistortion seam map (k1={K1}); {len(pairs)} seams, "
          f"threshold {MIN_INLIERS} inliers\n")
    print(f"  {'seam':12} {'dir':3} {'good':>5} {'inliers':>8} {'consistent':>11}   verdict")
    print("  " + "-" * 60)

    results, n_overlap = [], 0
    for a, b, kind in pairs:
        if a not in stills_u or b not in stills_u:
            continue
        sa, sb = ("R", "L") if kind == "H" else ("B", "T")
        kpa, desa, rectA = detect_edge(stills_u[a], sa)
        kpb, desb, rectB = detect_edge(stills_u[b], sb)
        n_good, inl, good, mask = match_pair(kpa, desa, kpb, desb)
        cons = inl / n_good if n_good else 0.0
        ov = inl >= MIN_INLIERS
        n_overlap += ov
        verdict = "OVERLAP" if ov else ("weak" if inl >= 10 else "no overlap")
        print(f"  {a+'-'+b:12} {kind:3} {n_good:5d} {inl:8d} {cons:10.0%}   {verdict}")
        draw(stills_u[a], kpa, rectA, stills_u[b], kpb, rectB, good, mask,
             MATCHES_U / f"{a}-{b}_{kind}.jpg")
        results.append({"a": a, "b": b, "dir": kind, "good": n_good,
                        "inliers": inl, "consistency": round(cons, 3), "overlap": bool(ov)})

    (OUT / "edge_match_undist.json").write_text(json.dumps(results, indent=2))
    print(f"\n  {n_overlap}/{len(results)} seams overlap after undistortion "
          f"(was 1/19 before).")
    print(f"  undistorted stills -> stitch/out/stills_undist/ ; "
          f"before/after -> stitch/out/undistort_before_after.png")
    return results


if __name__ == "__main__":
    main()
