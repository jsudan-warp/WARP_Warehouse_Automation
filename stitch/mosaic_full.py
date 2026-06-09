"""Full 14-tile 'honest collage' floor mosaic.

Governing reality (panel + measurements): ~9 of 14 tiles share NO floor, so this is
a coverage problem. Deliverable = a layout-faithful collage:
  * undistort every tile (shared k1, plumb-line/match cross-validated),
  * orient each tile from its own floor-line grid (rotation only; scale is shared
    since one lens at one height),
  * place tiles by a GLOBAL least-squares solve: weak grid 'springs' on every
    adjacent edge (lattice can't fold) + strong constraints from the few REAL
    seams (07-08, 12-13, 06-11, 08-13), weighted by inliers,
  * blend real overlaps, draw every tile boundary so approximate joins are honest,
  * office drawn as a labelled block.

Run:  ./venv/bin/python stitch/mosaic_full.py
"""
from __future__ import annotations

import json
from itertools import combinations

import cv2
import numpy as np

from edge_match import GRID, load_stills, neighbour_pairs, OUT  # type: ignore
import undistort as ud

K1 = -0.06                 # shared distortion (plumb-line & match-maximization agree ~-0.04..-0.08)
ROT_CLAMP_DEG = 18.0       # cameras are near-nadir on one building -> rotations are small
REAL_SEAM_MIN_INL = 10     # a grid seam becomes a strong constraint above this
W_GRID = 0.06              # weak spring weight (vs real-seam weight ~ inliers)
GAUGE = "07"

_SIFT = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.006,
                        edgeThreshold=16, sigma=1.6)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _fld():
    return cv2.ximgproc.createFastLineDetector(length_threshold=40, do_merge=True)


def line_grid_rotation(gray, center):
    """Dominant floor/structure line orientation -> small rotation that axis-aligns it.
    Returns (theta_deg in [-45,45] folded to small angle, confidence 0..1)."""
    segs = _fld().detect(gray)
    if segs is None:
        return 0.0, 0.0
    segs = segs.reshape(-1, 4)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    ang90 = np.mod(ang, 90.0)                       # collapse the 2 perpendicular families
    keep = L >= 60
    if keep.sum() < 5:
        return 0.0, 0.0
    a, wts = ang90[keep], L[keep]
    # weighted circular mean on the doubled angle (period 90 -> use 4x for 2pi)
    phi = np.deg2rad(a * 4.0)
    C = (wts * np.cos(phi)).sum(); S = (wts * np.sin(phi)).sum()
    mean = np.rad2deg(np.arctan2(S, C)) / 4.0
    theta = ((mean + 45) % 90) - 45                 # nearest axis, in [-45,45]
    conf = float(np.hypot(C, S) / wts.sum())        # angular concentration 0..1
    if abs(theta) > ROT_CLAMP_DEG or conf < 0.60:
        theta = 0.0                                  # implausible/uncertain -> trust the mount
    return float(theta), conf


def detect(img):
    return _SIFT.detectAndCompute(_CLAHE.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), None)


def seam_offset(imgA, imgB, Ra, Rb, c):
    """Measured (t_a - t_b) and inlier count for an overlapping seam, rotations applied."""
    kpa, desa = detect(imgA); kpb, desb = detect(imgB)
    if desa is None or desb is None:
        return None, 0, 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desa, desb, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 6:
        return None, 0, 0
    pa = np.float32([kpa[m.queryIdx].pt for m in good])
    pb = np.float32([kpb[m.trainIdx].pt for m in good])
    _, inl = cv2.estimateAffinePartial2D(pa.reshape(-1, 1, 2), pb.reshape(-1, 1, 2),
                                         method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if inl is None:
        return None, 0, 0
    keep = inl.ravel().astype(bool)
    if keep.sum() < 4:
        return None, int(keep.sum()), len(good)
    pa, pb = pa[keep], pb[keep]
    # t_a - t_b = R_b(q-c) - R_a(p-c)
    ga = (Ra @ (pa - c).T).T
    gb = (Rb @ (pb - c).T).T
    off = np.median(gb - ga, axis=0)
    return off, int(keep.sum()), len(good)


def solve_translations(tiles, offsets, grid_pairs, stepx, stepy, gauge):
    """Two decoupled least-squares (x, y) for tile translations."""
    idx = {t: i for i, t in enumerate(tiles)}
    N = len(tiles)

    def build(axis):           # axis 0=x,1=y
        rows, rhs = [], []
        # gauge: fix gauge tile at origin (strong)
        r = np.zeros(N); r[idx[gauge]] = 1.0
        rows.append(r * 1e3); rhs.append(0.0)
        # real-seam constraints
        for (a, b), (off, w) in offsets.items():
            r = np.zeros(N); r[idx[a]] = 1.0; r[idx[b]] = -1.0
            rows.append(r * w); rhs.append(off[axis] * w)
        # grid springs on every adjacent edge
        for (a, b, kind) in grid_pairs:
            nom = (-stepx, 0.0) if kind == "H" else (0.0, -stepy)
            r = np.zeros(N); r[idx[a]] = 1.0; r[idx[b]] = -1.0
            rows.append(r * W_GRID); rhs.append(nom[axis] * W_GRID)
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        return sol

    sx, sy = build(0), build(1)
    return {t: np.array([sx[idx[t]], sy[idx[t]]]) for t in tiles}


def main():
    raw = load_stills(GRID)
    tiles = sorted(raw.keys())
    h, w = raw[tiles[0]].shape[:2]
    c = np.array([w / 2.0, h / 2.0])
    K = ud.build_K(w, h, 107.0)
    mapx, mapy, _, _ = ud.undistort_maps(w, h, K, K1, new_camera="same")
    img = {t: ud.apply(raw[t], mapx, mapy) for t in tiles}
    mask0 = ud.apply(np.full((h, w), 255, np.uint8), mapx, mapy)

    # 1) per-tile rotation from the floor/structure line grid
    print(f"\nFull mosaic — undistort k1={K1}, line-grid orientation, global placement\n")
    R = {}; thetas = {}
    for t in tiles:
        g = _CLAHE.apply(cv2.cvtColor(img[t], cv2.COLOR_BGR2GRAY))
        th, conf = line_grid_rotation(g, c)
        thetas[t] = (th, conf)
        rad = np.deg2rad(th)
        R[t] = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    print("  line-grid rotation per tile (deg, conf): " +
          ", ".join(f"{t}:{thetas[t][0]:+.1f}/{thetas[t][1]:.2f}" for t in tiles))

    # 2) default butt-join lattice pitch (most tiles abut with little/no overlap);
    #    real seams only pull their specific pairs closer than this.
    stepx, stepy = 0.96 * w, 0.95 * h
    print(f"  lattice pitch (butt-join default): stepX={stepx:.0f}px, stepY={stepy:.0f}px")

    gp = neighbour_pairs(GRID)
    offsets = {}
    print("  seam screening (inl / consistency / offset -> verdict):")
    for a, b, kind in gp:
        if a not in img or b not in img:
            continue
        off, inl, good = seam_offset(img[a], img[b], R[a], R[b], c)
        if off is None:
            print(f"    {a}-{b} {kind}: no transform"); continue
        cons = inl / good if good else 0.0
        # offset-sanity gate: a real seam points the right way and is ~one step
        if kind == "H":
            ref, mag, ok_dir = stepx, abs(off[0]), (off[0] < 0 and abs(off[1]) < 0.5 * stepy)
        else:
            ref, mag, ok_dir = stepy, abs(off[1]), (off[1] < 0 and abs(off[0]) < 0.5 * stepx)
        ok_mag = 0.40 * ref <= mag <= 1.15 * ref
        # offset direction+magnitude is the real discriminator; full-frame
        # consistency is unreliable (denominator full of frame-wide false matches).
        accept = inl >= 12 and ok_dir and ok_mag
        if accept:
            offsets[(a, b)] = (off, float(inl))
        print(f"    {a}-{b} {kind}: {inl:3d}inl {cons:4.0%}  off=({off[0]:+.0f},{off[1]:+.0f})"
              f"  -> {'REAL' if accept else 'reject'}")
    print("  accepted real-seam constraints: " +
          (", ".join(f"{a}-{b}({int(w)})" for (a, b), (_, w) in offsets.items()) or "none"))

    # 3) global least-squares translations
    T = solve_translations(tiles, offsets, gp, stepx, stepy, GAUGE)

    # 4) canvas + render
    def place(t, pts):
        return (R[t] @ (pts - c).T).T + T[t]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    allc = np.vstack([place(t, corners) for t in tiles])
    # office nominal cell (row3, cols1-2): from gauge 07 at grid (r1,c1)
    base = T[GAUGE] - np.array([1 * stepx, 1 * stepy])
    office_c = base + np.array([1.5 * stepx, 3 * stepy])
    allc = np.vstack([allc, office_c + [[-stepx, -stepy / 2], [stepx, stepy / 2]]])
    minx, miny = allc.min(0); maxx, maxy = allc.max(0)
    pad = 20
    offv = np.array([-minx + pad, -miny + pad])
    W = int(maxx - minx + 2 * pad); H = int(maxy - miny + 2 * pad)

    acc = np.zeros((H, W, 3), np.float64); wsum = np.zeros((H, W), np.float64)
    # global gain: match each tile's median luminance
    med_ref = np.median([np.median(cv2.cvtColor(img[t], cv2.COLOR_BGR2GRAY)[mask0 > 0]) for t in tiles])
    polys = {}
    for t in tiles:
        M = np.hstack([R[t], (T[t] - R[t] @ c + offv).reshape(2, 1)])
        warp = cv2.warpAffine(img[t], M, (W, H))
        cover = cv2.warpAffine(mask0, M, (W, H), flags=cv2.INTER_NEAREST)
        med = np.median(cv2.cvtColor(img[t], cv2.COLOR_BGR2GRAY)[mask0 > 0])
        gain = float(np.clip(med_ref / max(med, 1), 0.6, 1.7))
        weight = cv2.distanceTransform((cover > 0).astype(np.uint8), cv2.DIST_L2, 3) + 1e-3
        acc += (warp.astype(np.float64) * gain).clip(0, 255) * weight[..., None]
        wsum += weight
        polys[t] = (place(t, corners) + offv).astype(np.int32)
    mosaic = (acc / np.maximum(wsum[..., None], 1e-6)).clip(0, 255).astype(np.uint8)

    # office block
    oc = (office_c + offv).astype(int)
    cv2.rectangle(mosaic, (int(oc[0] - stepx), int(oc[1] - stepy / 2)),
                  (int(oc[0] + stepx), int(oc[1] + stepy / 2)), (60, 95, 165), -1)
    cv2.putText(mosaic, "OFFICE", (int(oc[0] - 90), int(oc[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    # honest tile boundaries + labels; green = real-stitched, yellow = grid-placed
    real_tiles = set()
    for (a, b) in offsets:
        real_tiles.update((a, b))
    for t in tiles:
        col = (0, 220, 0) if t in real_tiles else (0, 215, 255)
        cv2.polylines(mosaic, [polys[t]], True, col, 2, cv2.LINE_AA)
        ctr = polys[t].mean(0).astype(int)
        cv2.putText(mosaic, t, tuple(ctr), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(mosaic, t, tuple(ctr), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)

    out = OUT / "mosaic_full.png"
    cv2.imwrite(str(out), mosaic)
    json.dump({"k1": K1, "thetas": {t: thetas[t] for t in tiles},
               "real_seams": {f"{a}-{b}": w for (a, b), (_, w) in offsets.items()},
               "stepx": stepx, "stepy": stepy,
               "poses": {t: T[t].tolist() for t in tiles}},
              open(OUT / "mosaic_full.json", "w"), indent=2)
    print(f"\n  canvas {W}x{H};  green=real-stitched ({sorted(real_tiles)}), "
          f"yellow=grid-placed (approx)")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
