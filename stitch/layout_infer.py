"""Infer the TRUE camera adjacency from the data (not from an assumed 4x4/7x2).

For every camera PAIR (all 91) we match full-FOV undistorted plates; a geometrically
consistent match (enough inliers, scale~1, small rotation, spatially spread) means the
two cameras see the same floor => they are physically adjacent. The median match
offset gives the RELATIVE direction/translation, so a global least-squares over those
offsets reconstructs the real 2D arrangement of the connected cameras.

Run:  ./venv/bin/python stitch/layout_infer.py
"""
from __future__ import annotations

import json
from itertools import combinations

import cv2
import numpy as np

from edge_match import OUT  # type: ignore
import undistort as ud

K1 = -0.06             # full-FOV at the proven seam-finding scale (1128x636, manageable)
MATCH_W = 1400         # only downscale if wider than this
MIN_INLIERS = 15
NFEAT = 5000

_SIFT = cv2.SIFT_create(nfeatures=NFEAT, nOctaveLayers=4, contrastThreshold=0.008,
                        edgeThreshold=12, sigma=1.6)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def load_undist_plates():
    plates = {}
    for p in sorted((OUT / "plates").glob("*.png")):
        plates[p.stem] = cv2.imread(str(p))
    h0, w0 = next(iter(plates.values())).shape[:2]
    K = ud.build_K(w0, h0, 107.0)
    mapx, mapy, _, _, _ = ud.undistort_maps_full(w0, h0, K, K1)
    out = {}
    for t, im in plates.items():
        u = cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR)
        if u.shape[1] > MATCH_W:
            s = MATCH_W / u.shape[1]
            u = cv2.resize(u, (MATCH_W, int(u.shape[0] * s)), interpolation=cv2.INTER_AREA)
        out[t] = u
    return out


def detect(img):
    return _SIFT.detectAndCompute(_CLAHE.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), None)


def match(fa, fb):
    (kpa, da), (kpb, db) = fa, fb
    if da is None or db is None:
        return 0, None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 6:
        return 0, None
    pa = np.float32([kpa[m.queryIdx].pt for m in good])
    pb = np.float32([kpb[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(pa.reshape(-1, 1, 2), pb.reshape(-1, 1, 2),
                                         method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None or inl is None:
        return 0, None
    keep = inl.ravel().astype(bool)
    n = int(keep.sum())
    if n < MIN_INLIERS:
        return n, None
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    pak, pbk = pa[keep], pb[keep]
    if not (0.80 <= scale <= 1.20 and abs(rot) < 12):   # consistent rigid-ish overlap
        return n, None
    offset = np.median(pbk - pak, axis=0)        # pos_a - pos_b = offset
    return n, offset


def components(cams, edges):
    adj = {c: set() for c in cams}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    seen, comps = set(), []
    for c in cams:
        if c in seen:
            continue
        stack, comp = [c], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.append(x); stack.extend(adj[x] - seen)
        comps.append(sorted(comp))
    return comps


def solve_positions(comp, offsets):
    idx = {c: i for i, c in enumerate(comp)}
    n = len(comp)
    if n == 1:
        return {comp[0]: np.array([0.0, 0.0])}
    def build(axis):
        rows, rhs = [], []
        r = np.zeros(n); r[0] = 1.0; rows.append(r * 1e3); rhs.append(0.0)   # gauge
        for (a, b), off in offsets.items():
            if a in idx and b in idx:
                r = np.zeros(n); r[idx[a]] = 1; r[idx[b]] = -1
                rows.append(r); rhs.append(off[axis])
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        return sol
    sx, sy = build(0), build(1)
    return {c: np.array([sx[idx[c]], sy[idx[c]]]) for c in comp}


def direction(off):
    dx, dy = off
    h = "right" if dx < 0 else "left"      # pos_a - pos_b = off; off.x<0 => b is right of a
    v = "below" if dy < 0 else "above"
    return (h if abs(dx) > abs(dy) else v) + f" (dx={dx:+.0f},dy={dy:+.0f})"


def main():
    plates = load_undist_plates()
    cams = sorted(plates)
    feats = {c: detect(plates[c]) for c in cams}
    print(f"\nLayout inference: matching all {len(cams)*(len(cams)-1)//2} camera pairs "
          f"(full-FOV k1={K1})\n")
    edges, offsets = [], {}
    for a, b in combinations(cams, 2):
        n, off = match(feats[a], feats[b])
        if off is not None:
            edges.append((a, b)); offsets[(a, b)] = off
            print(f"  {a}-{b}: {n:3d} inliers  -> {b} is {direction(off)} of {a}")
    comps = components(cams, edges)
    print(f"\n  overlap edges: {len(edges)}   connected groups: {len(comps)}")
    for i, comp in enumerate(comps):
        print(f"    group {i} ({len(comp)}): {comp}")

    # global positions + plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    allpos = {}
    ox = 0.0
    for comp in comps:
        pos = solve_positions(comp, offsets)
        xs = [p[0] for p in pos.values()]
        shift = ox - (min(xs) if xs else 0)
        for c, p in pos.items():
            allpos[c] = (p[0] + shift, -p[1])     # flip y for image-like display
        ox = max(allpos[c][0] for c in comp) + 600
    for (a, b) in edges:
        if a in allpos and b in allpos:
            ax.plot([allpos[a][0], allpos[b][0]], [allpos[a][1], allpos[b][1]],
                    "-", color="0.7", zorder=1)
    for c, (x, y) in allpos.items():
        ax.scatter([x], [y], s=600, c="#cfe", edgecolors="k", zorder=2)
        ax.text(x, y, c, ha="center", va="center", fontsize=11, fontweight="bold", zorder=3)
    ax.set_title(f"Inferred camera adjacency from measured overlaps (k1={K1})")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "layout_inferred.png", dpi=130); plt.close(fig)

    json.dump({"edges": [list(e) for e in edges], "components": comps,
               "positions": {c: allpos[c] for c in allpos}},
              open(OUT / "layout_inferred.json", "w"), indent=2)
    print(f"\n  saved -> stitch/out/layout_inferred.png (+ .json)")


if __name__ == "__main__":
    main()
