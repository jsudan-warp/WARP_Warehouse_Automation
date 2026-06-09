"""Fresh stitch pipeline — Step 1: edge-region SIFT across grid neighbours.

The cameras are laid out (user-specified, 'OF' = office, spans 2 cells):
    01 02 03 04
    06 07 08 09
    11 12 13 14
    16 OF OF 19

We KNOW adjacency from this grid, so we don't discover neighbours — we test the
ones that should touch. For each adjacent pair we run SIFT *only on the touching
edge strips* (right strip of the left image vs left strip of the right image; or
bottom strip of the upper image vs top strip of the lower image), match, and
count RANSAC-consistent inliers. That tells us which seams actually share floor
(and can be stitched) vs. which are butt-joined with no overlap.

Run:  ./venv/bin/python stitch/edge_match.py
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VID = ROOT / "videos"
OUT = ROOT / "stitch" / "out"
STILLS = OUT / "stills"
MATCHES = OUT / "matches"
for d in (STILLS, MATCHES):
    d.mkdir(parents=True, exist_ok=True)

GRID = [["01", "02", "03", "04"],
        ["06", "07", "08", "09"],
        ["11", "12", "13", "14"],
        ["16", "OF", "OF", "19"]]

EDGE_FRAC = 0.40        # fraction of the frame used as the touching edge strip
RATIO = 0.80
RANSAC_PX = 4.0
MIN_INLIERS = 18        # >= this => the seam genuinely overlaps


# --------------------------------------------------------------------------- #
def neighbour_pairs(grid):
    """4-connected adjacency between real cameras; tag H (left|right) or V (top/bottom)."""
    R, C = len(grid), max(len(r) for r in grid)
    def cam(r, c):
        if r < 0 or c < 0 or r >= R or c >= len(grid[r]):
            return None
        v = grid[r][c]
        return None if v in ("OF", None) else v
    pairs = []
    for r, c in product(range(R), range(C)):
        a = cam(r, c)
        if a is None:
            continue
        right, down = cam(r, c + 1), cam(r + 1, c)
        if right:
            pairs.append((a, right, "H"))     # a is left, right is right
        if down:
            pairs.append((a, down, "V"))       # a is top, down is bottom
    return pairs


def sharpest_still(video_path, n=14):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    best, best_score = None, -1.0
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        s = cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if s > best_score:
            best_score, best = s, fr
    cap.release()
    return best


def load_stills(grid):
    stills = {}
    for row in grid:
        for cell in row:
            if cell in ("OF", None):
                continue
            p = STILLS / f"{cell}.png"
            if p.exists():
                stills[cell] = cv2.imread(str(p))
            else:
                fr = sharpest_still(VID / f"{cell}.mp4")
                if fr is not None:
                    cv2.imwrite(str(p), fr)
                    stills[cell] = fr
    return stills


_SIFT = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.006,
                        edgeThreshold=16, sigma=1.6)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def strip_mask(shape_hw, side, frac=EDGE_FRAC):
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    if side == "R":
        m[:, int(w * (1 - frac)):] = 255
    elif side == "L":
        m[:, :int(w * frac)] = 255
    elif side == "B":
        m[int(h * (1 - frac)):, :] = 255
    elif side == "T":
        m[:int(h * frac), :] = 255
    rect = {"R": (int(w * (1 - frac)), 0, w, h), "L": (0, 0, int(w * frac), h),
            "B": (0, int(h * (1 - frac)), w, h), "T": (0, 0, w, int(h * frac))}[side]
    return m, rect


def detect_edge(img, side):
    gray = _CLAHE.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    mask, rect = strip_mask(gray.shape, side)
    kp, des = _SIFT.detectAndCompute(gray, mask)
    return kp, des, rect


def find_h(pa, pb):
    try:
        return cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, RANSAC_PX)
    except Exception:
        return cv2.findHomography(pa, pb, cv2.RANSAC, RANSAC_PX)


def match_pair(kpa, desa, kpb, desb):
    if desa is None or desb is None or len(kpa) < 4 or len(kpb) < 4:
        return 0, 0, None, None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desa, desb, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < 4:
        return len(good), 0, good, None
    pa = np.float32([kpa[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pb = np.float32([kpb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = find_h(pa, pb)
    if H is None or mask is None:
        return len(good), 0, good, None
    return len(good), int(mask.sum()), good, mask


def draw(imgA, kpA, rectA, imgB, kpB, rectB, good, mask, path):
    a, b = imgA.copy(), imgB.copy()
    cv2.rectangle(a, rectA[:2], rectA[2:], (0, 255, 255), 2)
    cv2.rectangle(b, rectB[:2], rectB[2:], (0, 255, 255), 2)
    sel = [g for g, k in zip(good, (mask.ravel() if mask is not None else np.ones(len(good))))
           if k] if good else []
    vis = cv2.drawMatches(a, kpA, b, kpB, sel, None, matchColor=(0, 255, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(path), vis)


def main():
    stills = load_stills(GRID)
    pairs = neighbour_pairs(GRID)
    print(f"\nLoaded {len(stills)} stills; testing {len(pairs)} grid-neighbour seams "
          f"(edge strip = {int(EDGE_FRAC*100)}% of frame, neighbour threshold = {MIN_INLIERS} inliers)\n")
    print(f"  {'seam':12} {'dir':3} {'good':>5} {'inliers':>8} {'consistent':>11}   verdict")
    print("  " + "-" * 60)

    results = []
    n_overlap = 0
    for a, b, kind in pairs:
        if a not in stills or b not in stills:
            continue
        sa, sb = ("R", "L") if kind == "H" else ("B", "T")
        kpa, desa, rectA = detect_edge(stills[a], sa)
        kpb, desb, rectB = detect_edge(stills[b], sb)
        n_good, inl, good, mask = match_pair(kpa, desa, kpb, desb)
        cons = inl / n_good if n_good else 0.0
        overlap = inl >= MIN_INLIERS
        n_overlap += overlap
        verdict = "OVERLAP" if overlap else ("weak" if inl >= 6 else "no overlap")
        print(f"  {a+'-'+b:12} {kind:3} {n_good:5d} {inl:8d} {cons:10.0%}   {verdict}")
        draw(stills[a], kpa, rectA, stills[b], kpb, rectB, good, mask,
             MATCHES / f"{a}-{b}_{kind}.jpg")
        results.append({"a": a, "b": b, "dir": kind, "good": n_good,
                        "inliers": inl, "consistency": round(cons, 3), "overlap": overlap})

    (OUT / "edge_match.json").write_text(json.dumps(results, indent=2))
    print(f"\n  {n_overlap}/{len(results)} seams genuinely overlap (>= {MIN_INLIERS} inliers).")
    print(f"  details -> stitch/out/edge_match.json ; visualisations -> stitch/out/matches/")
    return results


if __name__ == "__main__":
    main()
