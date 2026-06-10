"""Phase: video — run detection across the clips and write two videos.

  artifacts/detections_video.mp4  — the 14-camera grid with detection boxes per frame.
  artifacts/floor_map_video.mp4   — the 2D digital-twin floor map (pallet footprints) per frame.

We sample N frames uniformly across each clip (clips are ~static and not time-synced,
so frame k of every camera is treated as one timestep). Detector/filter/de-dup are the
same as `python -m lax9 detect`.

Run:  python -m lax9 video
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config, ensure_dirs
from .detect import make_detector
from .floor import (LAYOUT, camera_cells, footprint_to_floor, footprint_rect,
                    filter_detections, group_detections, dedup,
                    render_detection_grid, render_floor_map)
from .io_utils import list_videos


def _video_paths(cfg):
    return {p.stem: p for p in list_videos(cfg.path("videos"))}


def run(cfg: Config, args=None) -> dict:
    ensure_dirs(cfg)
    N = int(cfg.get("video", "n_frames", default=40))
    fps = float(cfg.get("video", "fps", default=5))
    L, D = cfg.floor_length_m, cfg.floor_depth_m
    cells, office, _, _ = camera_cells(LAYOUT, L, D)
    minA = float(cfg.get("detect", "min_area_frac", default=0.0015))
    maxA = float(cfg.get("detect", "max_area_frac", default=0.18))
    giou = float(cfg.get("detect", "group_iou", default=0.15))
    gfrac = float(cfg.get("detect", "group_frac", default=0.0))

    vids = _video_paths(cfg)
    cams = [c for row in LAYOUT for c in row if c not in ("OF", None) and c in vids and c in cells]
    caps = {c: cv2.VideoCapture(str(vids[c])) for c in cams}
    idxs = {c: np.linspace(0, max(int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 0), N).astype(int)
            for c in cams}

    print(f"\nPhase: video — {len(cams)} cameras x {N} frames @ {fps} fps")
    det = make_detector(cfg)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_grid = cfg.path("artifacts") / "detections_video.mp4"
    out_map = cfg.path("artifacts") / "floor_map_video.mp4"
    gw = mw = None

    for k in range(N):
        frames_k = {}
        for c in cams:
            caps[c].set(cv2.CAP_PROP_POS_FRAMES, int(idxs[c][k]))
            ok, fr = caps[c].read()
            if ok:
                frames_k[c] = fr
        dets_by_cam, pallets = {}, []
        for c, fr in frames_k.items():
            H, W = fr.shape[:2]
            dd = det.detect(fr, c)
            dd = filter_detections(dd, (W, H), minA, maxA)
            dd = group_detections(dd, (W, H), iou_thr=giou, group_frac=gfrac)
            dets_by_cam[c] = dd
            for d in dd:
                pallets.append({"cam": c, "xy": footprint_to_floor(c, d.footprint, (W, H), cells),
                                "rect": footprint_rect(c, d, (W, H), cells),
                                "score": d.score, "label": d.label})
        merged = dedup(pallets)
        g = render_detection_grid(frames_k, dets_by_cam, LAYOUT)
        m = render_floor_map(merged, cells, office, L, D)
        if gw is None:
            gw = cv2.VideoWriter(str(out_grid), fourcc, fps, (g.shape[1], g.shape[0]))
            mw = cv2.VideoWriter(str(out_map), fourcc, fps, (m.shape[1], m.shape[0]))
        gw.write(g)
        mw.write(m)
        if k % 5 == 0 or k == N - 1:
            print(f"  frame {k + 1}/{N}: {sum(len(v) for v in dets_by_cam.values())} dets "
                  f"-> {len(merged)} pallets")

    gw.release(); mw.release()
    for c in caps:
        caps[c].release()
    print(f"\n  saved -> {out_grid}")
    print(f"  saved -> {out_map}")
    return {"frames": N, "fps": fps}
