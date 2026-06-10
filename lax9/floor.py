"""Phase 7 — pallet floor map (2D bird's-eye canvas).

Detect pallets per camera, map each footprint to a floor position, de-duplicate across
the data-confirmed neighbour seams, and render a metric 2D floor canvas: the floor
rectangle, the 4x4 camera grid, the office obstacle, and every pallet at its position
with a stable ID.

Position model (HONEST): with no measured anchors yet, each camera's detections are
placed inside that camera's floor CELL (relative positions preserved). The render is
metric in shape (51.73 x 26.3 m, origin = floor centre, +X length, +Y depth) and becomes
metric-EXACT with no display change once Phase-5 anchors / the per-camera TPS warp land.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import cv2
import numpy as np

from .config import Config, ensure_dirs
from .detect import make_detector
from .io_utils import cam_id_for, list_videos

# Data-confirmed 4x4 layout (see stitch/layout_infer.py); 'OF' = office (2 cells).
LAYOUT = [["01", "02", "03", "04"],
          ["06", "07", "08", "09"],
          ["11", "12", "13", "14"],
          ["16", "OF", "OF", "19"]]
# Camera pairs whose overlap was geometrically confirmed (only de-dup across these).
CONFIRMED_NEIGHBOURS = {("07", "08"), ("12", "13"), ("06", "11"), ("08", "13")}
MERGE_DIST_M = 1.2          # merge same-pallet detections seen by two neighbour cameras


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _contains_center(a, b):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return a[0] <= cx <= a[2] and a[1] <= cy <= a[3]


def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def group_detections(dets, frame_wh, iou_thr=0.15, group_frac=0.0):
    """Merge duplicate/over-split detections into whole-pallet units (union-find).

    Always merges clearly-duplicate boxes (IoU>iou_thr or one box's centre inside the
    other). If group_frac>0, also merges boxes whose FOOTPRINTS are within
    group_frac*frame_diag (groups split boxes of one pallet load; set 0 to keep distinct
    stacks separate). Merged box = union; footprint = bottom-centre of the union."""
    from .detect import Detection
    n = len(dets)
    if n <= 1:
        return list(dets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    diag = float(np.hypot(*frame_wh))
    fps = [d.footprint for d in dets]
    have_masks = all(d.mask is not None for d in dets)
    for i in range(n):
        for j in range(i + 1, n):
            if have_masks:
                # distinct instance masks barely overlap; only merge near-duplicate masks
                dup = _mask_iou(dets[i].mask, dets[j].mask) > 0.5
            else:
                bi, bj = dets[i].box_xyxy, dets[j].box_xyxy
                dup = (_iou(bi, bj) > iou_thr or _contains_center(bi, bj)
                       or _contains_center(bj, bi))
            grp = group_frac > 0 and np.hypot(fps[i][0] - fps[j][0],
                                              fps[i][1] - fps[j][1]) < group_frac * diag
            if dup or grp:
                parent[find(i)] = find(j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    out = []
    for idxs in clusters.values():
        members = [dets[i] for i in idxs]
        xs1 = min(m.box_xyxy[0] for m in members); ys1 = min(m.box_xyxy[1] for m in members)
        xs2 = max(m.box_xyxy[2] for m in members); ys2 = max(m.box_xyxy[3] for m in members)
        best = max(members, key=lambda m: m.score)
        masks = [m.mask for m in members if m.mask is not None]
        merged_mask = np.logical_or.reduce(masks) if masks else None
        out.append(Detection((xs1, ys1, xs2, ys2), best.score, best.label, mask=merged_mask))
    return out


def camera_cells(layout, L, D):
    """cam_id -> (x0, x1, y0, y1) metres; cols span +X length, rows span +Y depth."""
    rows, cols = len(layout), max(len(r) for r in layout)
    cw, ch = L / cols, D / rows
    cells = {}
    office = None
    for r, row in enumerate(layout):
        for c, cell in enumerate(row):
            x0, y0 = -L / 2 + c * cw, -D / 2 + r * ch
            if cell == "OF":
                office = (x0, y0, x0 + cw, y0 + ch) if office is None else \
                    (min(office[0], x0), min(office[1], y0),
                     max(office[2], x0 + cw), max(office[3], y0 + ch))
            elif cell:
                cells[cell] = (x0, x0 + cw, y0, y0 + ch)
    return cells, office, cw, ch


def footprint_to_floor(cam, fp_px, wh, cells):
    """Map a footprint pixel (u,v) into the camera's floor cell (metres)."""
    x0, x1, y0, y1 = cells[cam]
    u, v = fp_px
    W, H = wh
    return (x0 + (u / W) * (x1 - x0), y0 + (v / H) * (y1 - y0))


def footprint_rect(cam, det, wh, cells, depth_ratio=0.8, depth_clamp=(0.4, 2.5)):
    """Tight floor-contact footprint rectangle (X1,Y1,X2,Y2) metres, anchored at the base.

    Uses the bottom band of the mask (the actual floor contact) for width — NOT the full
    stack box — so tall stacks don't render as deep overlapping rectangles. Depth is a
    modest fraction of width (top-down view can't measure true depth), clamped sane."""
    W, H = wh
    x0c, x1c, y0c, y1c = cells[cam]
    sx = (x1c - x0c) / W
    if det.mask is not None and det.mask.any():
        ys, xs = np.nonzero(det.mask)
        yb = int(ys.max())
        band = ys >= yb - max(3, int(0.12 * (yb - int(ys.min()) + 1)))
        xs_b = xs[band]
        xc, wpx, ybottom = float(xs_b.mean()), float(xs_b.max() - xs_b.min() + 1), float(yb)
    else:
        bx1, by1, bx2, by2 = det.box_xyxy
        xc, wpx, ybottom = (bx1 + bx2) / 2, (bx2 - bx1), by2
    X, Yb = footprint_to_floor(cam, (xc, ybottom), wh, cells)
    w_m = max(0.3, wpx * sx)
    d_m = float(np.clip(depth_ratio * w_m, *depth_clamp))
    return (X - w_m / 2, Yb - d_m, X + w_m / 2, Yb)


def filter_detections(dets, wh, min_area_frac, max_area_frac):
    """Drop implausible detections: huge boxes (false 'whole-rack' hits) + tiny noise."""
    fa = wh[0] * wh[1]
    out = []
    for d in dets:
        x1, y1, x2, y2 = d.box_xyxy
        a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if min_area_frac * fa <= a <= max_area_frac * fa:
            out.append(d)
    return out


def dedup(pallets):
    """Union-find merge of detections from CONFIRMED neighbour cameras within MERGE_DIST.
    pallets: list of dict(cam, xy, score, label). Returns merged list with 'id' + 'cams'."""
    n = len(pallets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = pallets[i]["cam"], pallets[j]["cam"]
            if ci == cj:
                continue
            if (ci, cj) in CONFIRMED_NEIGHBOURS or (cj, ci) in CONFIRMED_NEIGHBOURS:
                d = np.hypot(*(np.array(pallets[i]["xy"]) - np.array(pallets[j]["xy"])))
                if d < MERGE_DIST_M:
                    parent[find(i)] = find(j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    merged = []
    for k, (_, idxs) in enumerate(sorted(clusters.items())):
        members = [pallets[i] for i in idxs]
        best = max(members, key=lambda m: m["score"])   # representative = highest score
        merged.append({"id": k + 1, "xy": best["xy"], "rect": best["rect"],
                       "score": best["score"], "label": best["label"],
                       "cams": sorted({m["cam"] for m in members})})
    return merged


# --------------------------------------------------------------------------- #
def _w2p(X, Y, L, D, ppm, margin):
    return int(margin + (X + L / 2) * ppm), int(margin + (Y + D / 2) * ppm)


def render_floor_map(merged, cells, office, L, D, ppm=20, margin=60):
    W = int(L * ppm) + 2 * margin
    H = int(D * ppm) + 2 * margin
    canvas = np.full((H, W, 3), 245, np.uint8)
    p = lambda X, Y: _w2p(X, Y, L, D, ppm, margin)

    # floor border
    cv2.rectangle(canvas, p(-L / 2, -D / 2), p(L / 2, D / 2), (40, 40, 40), 2)
    # metric grid every 5 m
    x = -L / 2
    while x <= L / 2 + 1e-6:
        cv2.line(canvas, p(x, -D / 2), p(x, D / 2), (225, 225, 225), 1)
        cv2.putText(canvas, f"{x:+.0f}", (p(x, -D / 2)[0] - 8, margin - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        x += 5
    y = -D / 2
    while y <= D / 2 + 1e-6:
        cv2.line(canvas, p(-L / 2, y), p(L / 2, y), (225, 225, 225), 1)
        cv2.putText(canvas, f"{y:+.0f}", (12, p(-L / 2, y)[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        y += 5
    # camera cells (4x4) + labels
    for cam, (x0, x1, y0, y1) in cells.items():
        cv2.rectangle(canvas, p(x0, y0), p(x1, y1), (180, 200, 180), 1)
        cx, cy = p((x0 + x1) / 2, y0)
        cv2.putText(canvas, f"cam {cam}", (cx - 26, cy + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (90, 150, 90), 1, cv2.LINE_AA)
    # office obstacle
    if office:
        cv2.rectangle(canvas, p(office[0], office[1]), p(office[2], office[3]),
                      (66, 100, 168), -1)
        ox, oy = p((office[0] + office[2]) / 2, (office[1] + office[3]) / 2)
        cv2.putText(canvas, "OFFICE", (ox - 44, oy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # pallets as footprint BOXES (digitised layout) — semi-transparent fill + outline + id
    overlay = canvas.copy()
    for m in merged:
        x1, y1, x2, y2 = m["rect"]
        a, b = p(x1, y1), p(x2, y2)
        col = (0, 150, 255) if len(m["cams"]) > 1 else (190, 120, 70)
        cv2.rectangle(overlay, a, b, col, -1)
    canvas = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)
    for m in merged:
        x1, y1, x2, y2 = m["rect"]
        a, b = p(x1, y1), p(x2, y2)
        edge = (0, 110, 220) if len(m["cams"]) > 1 else (120, 70, 30)
        cv2.rectangle(canvas, a, b, edge, 1, cv2.LINE_AA)
        cx, cy = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        cv2.putText(canvas, f"P{m['id']}", (cx - 9, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (255, 255, 255), 1, cv2.LINE_AA)
    # title / legend
    cv2.putText(canvas, f"Pallet floor map  ({L:.2f} x {D:.1f} m)  -  {len(merged)} pallets",
                (margin, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "blue boxes = pallet footprints   orange = seen by 2 cameras (merged)",
                (margin, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 70, 30), 1, cv2.LINE_AA)
    return canvas


def render_detection_grid(frames, dets_by_cam, layout, tw=420, th=236, pad=6):
    """Verification view: each camera frame in the 4x4 layout with detection boxes +
    footprint dots + a per-camera count, so recall/duplicates can be eyeballed."""
    rows, cols = len(layout), max(len(r) for r in layout)
    canvas = np.full((rows * th + (rows + 1) * pad, cols * tw + (cols + 1) * pad, 3),
                     245, np.uint8)
    for r, row in enumerate(layout):
        for c, cell in enumerate(row):
            x, y = pad + c * (tw + pad), pad + r * (th + pad)
            if cell in ("OF", None) or cell not in frames:
                if cell == "OF":
                    canvas[y:y + th, x:x + tw] = (66, 100, 168)
                continue
            fr = frames[cell]
            H, W = fr.shape[:2]
            sx, sy = tw / W, th / H
            tile = cv2.resize(fr, (tw, th))
            for d in dets_by_cam.get(cell, []):
                x1, y1, x2, y2 = [int(v) for v in d.box_xyxy]
                cv2.rectangle(tile, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)),
                              (0, 230, 0), 2)
                fu, fv = d.footprint
                cv2.circle(tile, (int(fu * sx), int(fv * sy)), 4, (0, 0, 255), -1)
            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (255, 255, 255), 2)
            cv2.rectangle(tile, (0, 0), (118, 22), (0, 0, 0), -1)
            cv2.putText(tile, f"cam {cell}: {len(dets_by_cam.get(cell, []))}", (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            canvas[y:y + th, x:x + tw] = tile
    return canvas


def _load_frames(cfg):
    """One frame per camera: prefer stitch median plates, else sharpest video frame."""
    frames = {}
    plates = cfg.root / "stitch" / "out" / "plates"
    if plates.exists():
        for f in sorted(plates.glob("*.png")):
            frames[f.stem] = cv2.imread(str(f))
    if not frames:
        from .io_utils import sample_frames
        for vp in list_videos(cfg.path("videos")):
            fr = sample_frames(vp, 8)
            if fr:
                frames[cam_id_for(vp).replace("cam_", "")] = fr[len(fr) // 2]
    return frames


def run(cfg: Config, args=None) -> dict:
    ensure_dirs(cfg)
    L = cfg.floor_length_m
    D = cfg.floor_depth_m
    cells, office, cw, ch = camera_cells(LAYOUT, L, D)

    frames = _load_frames(cfg)
    print(f"\nPhase 7 — pallet floor map  ({len(frames)} cameras)")
    if not frames:
        print("  no frames found (need stitch/out/plates or videos/)"); return {}

    det = make_detector(cfg)
    pallets = []
    dets_by_cam = {}
    for cam in sorted(frames):
        if cam not in cells:
            continue
        fr = frames[cam]
        H, W = fr.shape[:2]
        dets = det.detect(fr, cam)
        n_raw = len(dets)
        dets = filter_detections(
            dets, (W, H),
            min_area_frac=float(cfg.get("detect", "min_area_frac", default=0.0015)),
            max_area_frac=float(cfg.get("detect", "max_area_frac", default=0.18)))
        dets = group_detections(
            dets, (W, H),
            iou_thr=float(cfg.get("detect", "group_iou", default=0.15)),
            group_frac=float(cfg.get("detect", "group_frac", default=0.0)))
        dets_by_cam[cam] = dets
        if len(dets) != n_raw:
            print(f"  cam {cam}: {n_raw} -> {len(dets)} after filter + de-dup")
        for d in dets:
            xy = footprint_to_floor(cam, d.footprint, (W, H), cells)
            rect = footprint_rect(cam, d, (W, H), cells)
            pallets.append({"cam": cam, "xy": xy, "rect": rect,
                            "score": d.score, "label": d.label})
        print(f"  cam {cam}: {len(dets)} detections")

    grid = render_detection_grid(frames, dets_by_cam, LAYOUT)
    cv2.imwrite(str(cfg.path("artifacts") / "pallet_detections_grid.png"), grid)
    print(f"  per-camera verification grid -> artifacts/pallet_detections_grid.png")

    merged = dedup(pallets)
    n_dup = sum(1 for m in merged if len(m["cams"]) > 1)
    print(f"\n  raw detections: {len(pallets)}  ->  unique pallets: {len(merged)} "
          f"({n_dup} merged across overlaps)")

    canvas = render_floor_map(merged, cells, office, L, D)
    out = cfg.path("artifacts") / "pallet_floor_map.png"
    cv2.imwrite(str(out), canvas)
    (cfg.path("artifacts") / "pallets.json").write_text(json.dumps(
        {"pallets": [{"id": m["id"], "x_m": round(m["xy"][0], 2), "y_m": round(m["xy"][1], 2),
                      "w_m": round(m["rect"][2] - m["rect"][0], 2),
                      "l_m": round(m["rect"][3] - m["rect"][1], 2),
                      "cams": m["cams"], "score": round(m["score"], 3)} for m in merged]},
        indent=2))
    print(f"  saved -> {out}  (+ artifacts/pallets.json)")
    print("  NOTE: positions are cell-relative (no metric anchors yet) — exact once "
          "Phase-5 anchors land.")
    return {"n_raw": len(pallets), "n_unique": len(merged), "n_merged": n_dup}
