"""Full mosaic — presentation render + floor-content boundary tightening.

Refinement 1 (clean render): multi-band blending on real overlaps, exposure-gain
equalisation, subtle tile outlines + a small legend (green = feature-stitched,
grey = grid-placed), office as a labelled block.

Refinement 2 (tighten grid-placed tiles with the floor's own lines/cracks):
the concrete shows irregular cracks rather than a perfectly regular saw-cut grid
at 848x478, so instead of spacing-snapping we do a BOUNDED, confidence-gated
boundary nudge: between adjacent grid-placed tiles that both show open floor,
cross-correlate the floor black-hat structure across the seam and add a soft
along-seam constraint (|shift| <= 50px) only when the correlation peak is strong.
Honest: it only fires where the floor is actually visible at the seam.

Run:  ./venv/bin/python stitch/mosaic_final.py
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from edge_match import GRID, load_stills, neighbour_pairs, OUT  # type: ignore
from mosaic_full import line_grid_rotation, detect, seam_offset, K1, GAUGE  # type: ignore
from joints import joint_response
import undistort as ud

W_GRID = 0.06
W_NUDGE = 0.5
NUDGE_MAX = 50
NUDGE_MIN_NCC = 0.45


def solve(tiles, real, nudges, gp, stepx, stepy, gauge):
    """LSQ tile translations: real-seam (2D) + nudge (1-axis) + grid springs."""
    idx = {t: i for i, t in enumerate(tiles)}; N = len(tiles)

    def build(axis):
        rows, rhs = [], []
        r = np.zeros(N); r[idx[gauge]] = 1.0; rows.append(r * 1e3); rhs.append(0.0)
        for (a, b), (off, w) in real.items():
            r = np.zeros(N); r[idx[a]] = 1; r[idx[b]] = -1
            rows.append(r * w); rhs.append(off[axis] * w)
        for (a, b, ax), (val, w) in nudges.items():
            if ax != axis:
                continue
            r = np.zeros(N); r[idx[a]] = 1; r[idx[b]] = -1
            rows.append(r * w); rhs.append(val * w)
        for (a, b, kind) in gp:
            nom = (-stepx, 0.0) if kind == "H" else (0.0, -stepy)
            r = np.zeros(N); r[idx[a]] = 1; r[idx[b]] = -1
            rows.append(r * W_GRID); rhs.append(nom[axis] * W_GRID)
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        return sol
    sx, sy = build(0), build(1)
    return {t: np.array([sx[idx[t]], sy[idx[t]]]) for t in tiles}


def ncc_shift(pa, pb, max_shift):
    """Best lag aligning 1-D profiles pa,pb (zero-mean NCC); returns (lag, peak)."""
    a = pa - pa.mean(); b = pb - pb.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0, 0.0
    best, bestv = 0, -1.0
    for d in range(-max_shift, max_shift + 1):
        bs = np.roll(b, d)
        v = float((a * bs).sum() / (na * nb))
        if v > bestv:
            bestv, best = v, d
    return best, bestv


def boundary_nudges(img, T, R, c, gp, real, w, h):
    """Floor-content along-seam nudges between adjacent grid-placed tiles."""
    resp = {t: joint_response(img[t])[0].astype(np.float64) for t in img}
    fmask = {t: joint_response(img[t])[1] for t in img}
    S = int(0.18 * w)
    nudges = {}
    for a, b, kind in gp:
        if (a, b) in real or a not in img or b not in img:
            continue
        if kind == "H":   # a left, b right -> align horizontal structure in y
            pa = resp[a][:, w - S:].sum(1); pb = resp[b][:, :S].sum(1)
            fa = (fmask[a][:, w - S:] > 0).mean(); fb = (fmask[b][:, :S] > 0).mean()
            axis = 1
        else:             # a top, b bottom -> align vertical structure in x
            pa = resp[a][h - S:, :].sum(0); pb = resp[b][:S, :].sum(0)
            fa = (fmask[a][h - S:, :] > 0).mean(); fb = (fmask[b][:S, :] > 0).mean()
            axis = 0
        if min(fa, fb) < 0.45:           # need open floor on BOTH sides
            continue
        lag, peak = ncc_shift(pa, pb, NUDGE_MAX)
        if peak >= NUDGE_MIN_NCC and abs(lag) <= NUDGE_MAX:
            nominal = (T[a] - T[b])[axis]
            nudges[(a, b, axis)] = (nominal + lag, W_NUDGE)
            print(f"    nudge {a}-{b} {kind}: floor={min(fa,fb):.0%} ncc={peak:.2f} shift={lag:+d}px")
    if not nudges:
        print("    (no confident floor-content nudges — open floor too sparse at seams)")
    return nudges


def load_plates():
    """Prefer the temporal median plates (denoised, full-clip); fall back to stills."""
    raw = {}
    for p in sorted((OUT / "plates").glob("*.png")):
        im = cv2.imread(str(p))
        if im is not None:
            raw[p.stem] = im
    if not raw:
        raw = load_stills(GRID)
    return raw


def main():
    raw = load_plates(); tiles = sorted(raw.keys())
    h0, w0 = raw[tiles[0]].shape[:2]
    K = ud.build_K(w0, h0, 107.0)
    # full-canvas undistort: keep the WHOLE corrected FOV (no cropped edges/overlaps)
    mapx, mapy, newK, (w, h), mask0 = ud.undistort_maps_full(w0, h0, K, K1)
    img = {t: cv2.remap(raw[t], mapx, mapy, cv2.INTER_LINEAR) for t in tiles}
    c = np.array([newK[0, 2], newK[1, 2]])     # principal point = rotation centre
    print(f"  undistort canvas: {w0}x{h0} -> {w}x{h} (no edges cropped)")

    print(f"\nFinal mosaic — k1={K1}, clean render + floor-content tightening\n")
    R = {}; thetas = {}
    for t in tiles:
        g = cv2.createCLAHE(3.0, (8, 8)).apply(cv2.cvtColor(img[t], cv2.COLOR_BGR2GRAY))
        th, conf = line_grid_rotation(g, c); thetas[t] = (th, conf)
        rad = np.deg2rad(th)
        R[t] = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    gp = neighbour_pairs(GRID)
    stepx, stepy = 0.96 * w, 0.95 * h
    real = {}
    for a, b, kind in gp:
        off, inl, good = seam_offset(img[a], img[b], R[a], R[b], c)
        if off is None:
            continue
        if kind == "H":
            ref, mag, okd = stepx, abs(off[0]), (off[0] < 0 and abs(off[1]) < 0.5 * stepy)
        else:
            ref, mag, okd = stepy, abs(off[1]), (off[1] < 0 and abs(off[0]) < 0.5 * stepx)
        if inl >= 12 and okd and 0.40 * ref <= mag <= 1.15 * ref:
            real[(a, b)] = (off, float(inl))
    print(f"  real seams: {', '.join(f'{a}-{b}' for a,b in real) or 'none'}")

    T = solve(tiles, real, {}, gp, stepx, stepy, GAUGE)
    print("  floor-content boundary nudges:")
    nudges = boundary_nudges(img, T, R, c, gp, real, w, h)
    T = solve(tiles, real, nudges, gp, stepx, stepy, GAUGE)

    # ---- render ----
    def place(t, pts):
        return (R[t] @ (pts - c).T).T + T[t]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    allc = np.vstack([place(t, corners) for t in tiles])
    base = T[GAUGE] - np.array([stepx, stepy])
    office_c = base + np.array([1.5 * stepx, 3 * stepy])
    allc = np.vstack([allc, office_c + [[-stepx, -stepy / 2], [stepx, stepy / 2]]])
    minx, miny = allc.min(0); maxx, maxy = allc.max(0)
    pad = 24; offv = np.array([-minx + pad, -miny + pad])
    W = int(maxx - minx + 2 * pad); H = int(maxy - miny + 2 * pad)

    med_ref = np.median([np.median(cv2.cvtColor(img[t], cv2.COLOR_BGR2GRAY)[mask0 > 0]) for t in tiles])
    warps, masks, polys = {}, {}, {}
    for t in tiles:
        M = np.hstack([R[t], (T[t] - R[t] @ c + offv).reshape(2, 1)])
        med = np.median(cv2.cvtColor(img[t], cv2.COLOR_BGR2GRAY)[mask0 > 0])
        gain = float(np.clip(med_ref / max(med, 1), 0.6, 1.7))
        warps[t] = np.clip(cv2.warpAffine(img[t], M, (W, H)).astype(np.float64) * gain,
                           0, 255).astype(np.uint8)
        masks[t] = cv2.warpAffine(mask0, M, (W, H), flags=cv2.INTER_NEAREST)
        # trim the heavily-stretched outer barrel ring (low quality + jagged edge)
        masks[t] = cv2.erode(masks[t], np.ones((9, 9), np.uint8), iterations=2)
        polys[t] = (place(t, corners) + offv).astype(np.int32)

    # feather (distance-weighted) blend — smooth across the curved full-FOV tiles;
    # overlapping tiles fade into each other and cover each other's curved borders.
    acc = np.zeros((H, W, 3), np.float64); wsum = np.zeros((H, W), np.float64)
    for t in tiles:
        wt = cv2.distanceTransform((masks[t] > 0).astype(np.uint8), cv2.DIST_L2, 5)
        wt = wt ** 1.5 + 1e-3                  # bias toward each tile's reliable centre
        acc += warps[t].astype(np.float64) * wt[..., None]; wsum += wt
    mosaic = (acc / np.maximum(wsum[..., None], 1e-6)).clip(0, 255).astype(np.uint8)

    # office block: fit its WIDTH to the actual gap between tile 16 (left) and
    # tile 19 (right), and its height to their vertical span (cols 1-2, row 3).
    p16, p19 = polys.get("16"), polys.get("19")
    oc = (office_c + offv).astype(int)
    if p16 is not None and p19 is not None and p19[:, 0].min() > p16[:, 0].max():
        x0, x1 = int(p16[:, 0].max()), int(p19[:, 0].min())
        y0 = int(min(p16[:, 1].min(), p19[:, 1].min()))
        y1 = int(max(p16[:, 1].max(), p19[:, 1].max()))
    else:                                   # fallback to nominal cell if no clean fit
        x0, x1 = int(oc[0] - stepx), int(oc[0] + stepx)
        y0, y1 = int(oc[1] - stepy / 2), int(oc[1] + stepy / 2)
    cv2.rectangle(mosaic, (x0, y0), (x1, y1), (66, 100, 168), -1)
    (tw, th), _ = cv2.getTextSize("OFFICE", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(mosaic, "OFFICE", ((x0 + x1) // 2 - tw // 2, (y0 + y1) // 2 + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    real_tiles = {t for ab in real for t in ab}
    clean = mosaic.copy()
    for t in tiles:                       # subtle outlines + small corner labels
        col = (90, 210, 90) if t in real_tiles else (170, 170, 170)
        cv2.polylines(clean, [polys[t]], True, col, 1, cv2.LINE_AA)
        p = polys[t].min(0) + [6, 20]
        cv2.putText(clean, t, tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(clean, t, tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    # crop the surrounding black border down to the content
    ys, xs = np.where(wsum > 0)
    if len(xs) and len(ys):
        x0, x1 = max(int(xs.min()) - 15, 0), min(int(xs.max()) + 15, W)
        y0, y1 = max(int(ys.min()) - 15, 0), min(int(ys.max()) + 15, H)
        mosaic = mosaic[y0:y1, x0:x1]; clean = clean[y0:y1, x0:x1]

    # legend
    cv2.rectangle(clean, (10, 10), (360, 70), (30, 30, 30), -1)
    cv2.putText(clean, "green = feature-stitched", (20, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (90, 210, 90), 2, cv2.LINE_AA)
    cv2.putText(clean, "grey  = grid-placed (approx)", (20, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (170, 170, 170), 2, cv2.LINE_AA)

    cv2.imwrite(str(OUT / "mosaic_final.png"), mosaic)
    cv2.imwrite(str(OUT / "mosaic_final_labeled.png"), clean)
    print(f"\n  canvas {W}x{H}; real-stitched={sorted(real_tiles)}")
    print(f"  saved -> stitch/out/mosaic_final.png (clean) + mosaic_final_labeled.png (annotated)")


if __name__ == "__main__":
    main()
