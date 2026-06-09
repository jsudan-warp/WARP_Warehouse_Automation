"""Phase 2 — camera identification & overlap graph.

Filenames are meaningless. We recover which cameras are neighbours by matching
the floor's own features between every pair of stills, counting geometrically
consistent (RANSAC-inlier) matches. Many consistent matches => the two cameras
see the same patch of floor => neighbours.

Outputs:
  artifacts/overlap_matrix.csv   inlier-count matrix
  artifacts/overlap_heatmap.png  visual of the matrix
  artifacts/matches/<a>__<b>.jpg match visualisations (confirm they land on FLOOR)
  artifacts/contact_sheet.png    labelled montage for human grid-cell confirmation
  artifacts/camera_layout.json   cam -> {grid, neighbours, component}
  artifacts/pairwise.json        reusable pairwise homographies + inliers (for Phase 4)

Acceptance: overlap_matrix.csv + overlap_heatmap.png produced; neighbour list
printed; number of connected groups reported.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from .config import Config, ensure_dirs
from .io_utils import list_videos, cam_id_for
from .results import append_section
from .undistort import load_intrinsics, Undistorter


def _load_chosen_stills(cfg: Config) -> dict[str, np.ndarray]:
    """Load the representative still per camera (from Phase 1 manifest)."""
    frames_dir = cfg.path("frames")
    manifest_path = frames_dir / "frames_manifest.json"
    stills: dict[str, np.ndarray] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for cam, m in manifest.items():
            img = cv2.imread(str(frames_dir / m["chosen_file"]))
            if img is not None:
                stills[cam] = img
        return stills
    # fallback: re-derive from medians if no manifest
    for p in list_videos(cfg.path("videos")):
        cam = cam_id_for(p)
        f = frames_dir / f"{cam}_median.png"
        if f.exists():
            stills[cam] = cv2.imread(str(f))
    return stills


def _resize_work(img: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img, scale


def _roi_mask(shape_hw, frac_box):
    """Build a uint8 detection mask (255 inside ROI) from a fractional box
    [fx0, fy0, fx1, fy1] (0..1). Returns (mask, (x0,y0,x1,y1)) in pixels."""
    h, w = shape_hw
    fx0, fy0, fx1, fy1 = frac_box
    x0, x1 = sorted((int(round(fx0 * w)), int(round(fx1 * w))))
    y0, y1 = sorted((int(round(fy0 * h)), int(round(fy1 * h))))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask, (x0, y0, x1, y1)


def _detect(stills, max_dim, sift_params, roi_map=None):
    """Detect SIFT keypoints/descriptors, tuned to pull MANY features from faint
    floor texture (low contrast threshold, extra octave layers, CLAHE pre-filter).

    roi_map: optional {cam_id: [fx0,fy0,fx1,fy1]} restricting detection to a
    fractional sub-region of each frame (e.g. right half of one camera vs left
    half of its neighbour). Cameras absent from the map use the full frame.
    """
    roi_map = roi_map or {}
    sift = cv2.SIFT_create(
        nfeatures=int(sift_params.get("max_features", 0)),
        nOctaveLayers=int(sift_params.get("n_octave_layers", 3)),
        contrastThreshold=float(sift_params.get("contrast_threshold", 0.04)),
        edgeThreshold=float(sift_params.get("edge_threshold", 10)),
        sigma=float(sift_params.get("sigma", 1.6)),
    )
    clahe = cv2.createCLAHE(clipLimit=float(sift_params.get("clahe_clip", 3.0)),
                            tileGridSize=(8, 8))
    feats = {}
    for cam, img in stills.items():
        work, scale = _resize_work(img, max_dim)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)                 # local-contrast boost on faint floor texture
        mask, roi_px = (None, None)
        if cam in roi_map:
            mask, roi_px = _roi_mask(gray.shape[:2], roi_map[cam])
        kp, des = sift.detectAndCompute(gray, mask)
        feats[cam] = {"kp": kp, "des": des, "scale": scale, "work": work,
                      "img": img, "roi_px": roi_px}
    return feats


def _find_homography(pa, pb, ransac_px):
    """MAGSAC++ if available (recovers more true inliers), else plain RANSAC."""
    try:
        return cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, ransac_px)
    except Exception:  # noqa: BLE001 (older OpenCV without USAC_MAGSAC)
        return cv2.findHomography(pa, pb, cv2.RANSAC, ransac_px)


def _match_pair(fa, fb, ratio, ransac_px):
    """Match two feature sets.

    Returns (n_good, inliers, good, mm, H_work, mask) where
      n_good  = ratio-test survivors (raw match density, PRE-geometry)
      inliers = matches consistent with a single homography (the overlap metric)
      good    = the list of ratio-good DMatch (for the all-matches visualisation)
    """
    if fa["des"] is None or fb["des"] is None or len(fa["kp"]) < 4 or len(fb["kp"]) < 4:
        return 0, 0, None, None, None, None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(fa["des"], fb["des"], k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    n_good = len(good)
    if n_good < 4:
        return n_good, 0, good, None, None, None
    pa = np.float32([fa["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pb = np.float32([fb["kp"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = _find_homography(pa, pb, ransac_px)
    if H is None or mask is None:
        return n_good, 0, good, None, None, None
    inliers = int(mask.sum())
    return n_good, inliers, good, (pa, pb, mask), H, mask


def _vis_matches(feats, a, b, match_list, path):
    """Draw match lines between two camera work-images (with ROI rectangles)."""
    imgA, imgB = feats[a]["work"].copy(), feats[b]["work"].copy()
    for im, cam in ((imgA, a), (imgB, b)):
        rp = feats[cam]["roi_px"]
        if rp:
            cv2.rectangle(im, (rp[0], rp[1]), (rp[2], rp[3]), (0, 255, 255), 2)
    vis = cv2.drawMatches(
        imgA, feats[a]["kp"], imgB, feats[b]["kp"], match_list, None,
        matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(path), vis)


def _rescale_homography(H_work, scale_a, scale_b):
    """Convert a homography computed at working res back to full-res pixels.
    H_work maps a_work -> b_work. Full-res: b_full = Sb^-1 H_work Sa a_full."""
    Sa = np.diag([scale_a, scale_a, 1.0])
    Sb = np.diag([scale_b, scale_b, 1.0])
    return np.linalg.inv(Sb) @ H_work @ Sa


def _connected_components(cams, edges):
    adj = {c: set() for c in cams}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = set()
    comps = []
    for c in cams:
        if c in seen:
            continue
        stack = [c]
        comp = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        comps.append(sorted(comp))
    return comps, adj


def _save_heatmap(cams, M, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.2 + 0.6 * len(cams), 1.0 + 0.6 * len(cams)))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(cams)))
    ax.set_yticks(range(len(cams)))
    ax.set_xticklabels(cams, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cams, fontsize=8)
    for i in range(len(cams)):
        for j in range(len(cams)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] < M.max() / 2 else "black", fontsize=7)
    ax.set_title("Pairwise RANSAC inliers (overlap)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_contact_sheet(feats, path: Path, ncols=7):
    cams = list(feats.keys())
    n = len(cams)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    thumbs = []
    tw, th = 320, 180
    for cam in cams:
        img = cv2.resize(feats[cam]["img"], (tw, th))
        cv2.rectangle(img, (0, 0), (tw - 1, 24), (0, 0, 0), -1)
        cv2.putText(img, cam, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        thumbs.append(img)
    while len(thumbs) < nrows * ncols:
        thumbs.append(np.zeros((th, tw, 3), np.uint8))
    grid = np.vstack([np.hstack(thumbs[r * ncols:(r + 1) * ncols]) for r in range(nrows)])
    cv2.imwrite(str(path), grid)


def run(cfg: Config, args=None) -> dict:
    ensure_dirs(cfg)
    matches_dir = cfg.path("matches")
    matches_dir.mkdir(parents=True, exist_ok=True)

    mcfg = cfg.raw["matching"]
    max_dim = int(mcfg["work_max_dim"])
    ratio = float(mcfg["ratio_test"])
    ransac_px = float(mcfg["ransac_reproj_px"])
    min_inliers = int(mcfg["min_inliers"])

    stills = _load_chosen_stills(cfg)
    cams = sorted(stills.keys())

    # Phase-3 hook: undistort BEFORE matching if a lens model exists and is enabled.
    intr = load_intrinsics(cfg.path("intrinsics"))
    use_undist = bool(cfg.get("calibration", "use_for_matching", default=True))
    if intr is not None and use_undist and stills:
        alpha = float(cfg.get("calibration", "alpha", default=0.0))
        for cam, img in list(stills.items()):
            h, w = img.shape[:2]
            stills[cam] = Undistorter(intr, (w, h), alpha)(img)
        print(f"  undistortion: APPLIED before matching "
              f"(intrinsics.npz, model={intr.get('model', '?')}, alpha={alpha})")
    else:
        print("  undistortion: not applied "
              f"({'disabled' if (intr is not None and not use_undist) else 'no intrinsics.npz'}) "
              "— matching on raw frames (reduced edge accuracy at the 2.8mm barrel corners)")

    print(f"\nPhase 2 — overlap graph  ({len(cams)} camera still(s): {', '.join(cams)})")
    if len(cams) < 2:
        print("  Need >= 2 stills to build an overlap graph. Run `frames` first / add clips.")
        # still emit a trivial layout so downstream doesn't crash
        layout = {c: {"grid": None, "neighbours": [], "component": i}
                  for i, c in enumerate(cams)}
        cfg.path("camera_layout").write_text(json.dumps(layout, indent=2))
        return {"cams": cams, "n_components": len(cams), "edges": []}

    roi_map = mcfg.get("roi", {}) or {}
    feats = _detect(stills, max_dim, mcfg, roi_map)
    print(f"    SIFT: contrastThr={mcfg.get('contrast_threshold', 0.04)}, "
          f"edgeThr={mcfg.get('edge_threshold', 10)}, "
          f"octaveLayers={mcfg.get('n_octave_layers', 3)}, "
          f"nfeatures_cap={mcfg.get('max_features', 0) or 'all'}, ratio={ratio}")
    if roi_map:
        print(f"    ROI restriction active for: {', '.join(sorted(roi_map))}")
    for cam in cams:
        nkp = len(feats[cam]["kp"]) if feats[cam]["kp"] else 0
        roi = feats[cam]["roi_px"]
        roi_txt = f" within ROI px {roi}" if roi else " (full frame)"
        print(f"    {cam}: {nkp} SIFT keypoints (work scale {feats[cam]['scale']:.3f}){roi_txt}")

    n = len(cams)
    idx = {c: i for i, c in enumerate(cams)}
    M = np.zeros((n, n), dtype=int)
    G = np.zeros((n, n), dtype=int)          # raw ratio-good match counts (density)
    edges = []
    pairwise = {}

    for a, b in combinations(cams, 2):
        n_good, inl, good, mm, H_work, mask = _match_pair(feats[a], feats[b], ratio, ransac_px)
        M[idx[a], idx[b]] = M[idx[b], idx[a]] = inl
        G[idx[a], idx[b]] = G[idx[b], idx[a]] = n_good
        is_edge = inl >= min_inliers
        consistency = (inl / n_good) if n_good else 0.0
        tag = "NEIGHBOUR" if is_edge else "—"
        print(f"    {a} <-> {b}: {n_good:5d} good matches -> {inl:4d} RANSAC inliers "
              f"({consistency:4.0%} consistent)  {tag}")
        # ALL ratio-good matches: shows raw density + whether lines are parallel
        if good:
            _vis_matches(feats, a, b, good, matches_dir / f"{a}__{b}_all.jpg")
        if H_work is not None:
            H_full = _rescale_homography(H_work, feats[a]["scale"], feats[b]["scale"])
            pairwise[f"{a}__{b}"] = {
                "good_matches": n_good,
                "inliers": inl,
                "consistency": round(consistency, 3),
                "edge": bool(is_edge),
                "H_a_to_b": H_full.tolist(),
            }
            # geometrically-consistent inliers only
            if inl > 0:
                _, _, msk = mm
                inlier_matches = [g for g, k in zip(good, msk.ravel()) if k]
                _vis_matches(feats, a, b, inlier_matches,
                             matches_dir / f"{a}__{b}_inliers.jpg")
        if is_edge:
            edges.append((a, b))

    # overlap_matrix.csv
    csv_path = cfg.path("overlap_matrix")
    with open(csv_path, "w") as f:
        f.write("," + ",".join(cams) + "\n")
        for i, c in enumerate(cams):
            f.write(c + "," + ",".join(str(int(M[i, j])) for j in range(n)) + "\n")

    _save_heatmap(cams, M, cfg.path("overlap_heatmap"))
    _save_contact_sheet(feats, cfg.path("artifacts") / "contact_sheet.png",
                        ncols=cfg.get("cameras", "grid", default=[7, 2])[0])
    (cfg.path("artifacts") / "pairwise.json").write_text(json.dumps(pairwise, indent=2))

    comps, adj = _connected_components(cams, edges)
    comp_of = {c: i for i, comp in enumerate(comps) for c in comp}

    layout = {}
    for c in cams:
        layout[c] = {
            "grid": None,                              # human-confirmed in Phase 2 gate
            "neighbours": sorted(adj[c]),
            "component": comp_of[c],
            "video": stills and None,
        }
    cfg.path("camera_layout").write_text(json.dumps(layout, indent=2))

    print(f"\n  neighbour graph:")
    for c in cams:
        print(f"    {c}: neighbours = {sorted(adj[c]) or '(none)'}")
    print(f"\n  connected groups: {len(comps)}")
    for i, comp in enumerate(comps):
        print(f"    group {i}: {comp}")
    if len(comps) > 1:
        print("  NOTE: >1 group — cameras in separate groups don't share enough floor "
              "to be tied by features alone. They get anchored in Phase 5 (measured points).")

    print(f"\n  artifacts: {csv_path.name}, {cfg.path('overlap_heatmap').name}, "
          f"contact_sheet.png, camera_layout.json, matches/*.jpg")
    print("  ACTION (human gate): open artifacts/contact_sheet.png and use the office + "
          "yellow columns to confirm each camera's (row,col) on the 7x2 grid; "
          "fill `grid` in camera_layout.json.")

    # Log to RESULTS.md
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pair_lines = "\n".join(
        f"- {a} ↔ {b}: {G[idx[a], idx[b]]} good matches → **{M[idx[a], idx[b]]}** "
        f"RANSAC inliers"
        f"{'  (neighbour)' if (a, b) in edges or (b, a) in edges else ''}"
        for a, b in combinations(cams, 2)
    )
    body = (
        f"_Run: {ts}_\n\n"
        f"- Cameras with stills: **{len(cams)}** ({', '.join(cams)})\n"
        f"- Matcher: SIFT + Lowe ratio {ratio} + RANSAC ({ransac_px}px); "
        f"neighbour threshold = {min_inliers} inliers.\n"
        f"- **Connected groups: {len(comps)}** — "
        + "; ".join(f"group {i}={comp}" for i, comp in enumerate(comps)) + "\n\n"
        f"Pairwise inliers:\n{pair_lines}\n\n"
        f"Artifacts: `overlap_matrix.csv`, `overlap_heatmap.png`, `contact_sheet.png`, "
        f"`camera_layout.json`, `matches/*.jpg`.\n"
        + ("\n> NOTE: >1 connected group — unconnected cameras flagged for Phase-5 anchors.\n"
           if len(comps) > 1 else "")
    )
    append_section(cfg.path("results"), "Phase 2 — Overlap graph", body)

    return {
        "cams": cams, "matrix": M, "edges": edges,
        "n_components": len(comps), "components": comps,
    }
