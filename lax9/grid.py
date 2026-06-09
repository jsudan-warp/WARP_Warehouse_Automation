"""Camera grid montage — one representative frame per video, in a fixed layout.

Default layout (user-specified); 'OFFICE' spans 2 cells:
    01  02  03  04
    06  07  08  09
    11  12  13  14
    16  [ office ]  19

This is a display arrangement (how the cameras are laid out for review), not the
metric floor map. Cameras with no video get a labelled placeholder.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config, ensure_dirs
from .io_utils import list_videos, sample_frames

FONT = cv2.FONT_HERSHEY_SIMPLEX

DEFAULT_LAYOUT = [
    ["01", "02", "03", "04"],
    ["06", "07", "08", "09"],
    ["11", "12", "13", "14"],
    ["16", "OFFICE", "OFFICE", "19"],
]

TW, TH = 480, 270        # tile size (16:9)
PAD = 8                  # gap between tiles
OFFICE_BGR = (60, 95, 165)   # warm brown/orange (BGR)


def _representative_frame(video_path, n_sample=10):
    """Sharpest of a few sampled frames (a clean, in-focus still)."""
    frames = sample_frames(video_path, n_sample)
    if not frames:
        return None
    return max(frames, key=lambda f: cv2.Laplacian(
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _label_chip(img, text, color=(0, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    cv2.rectangle(img, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(img, text, (8, th + 8), FONT, 0.8, color, 2, cv2.LINE_AA)


def _camera_tile(frame, cam_id):
    if frame is None:
        tile = np.full((TH, TW, 3), 70, np.uint8)
        cv2.putText(tile, f"cam {cam_id}", (TW // 2 - 70, TH // 2 - 8),
                    FONT, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(tile, "(no video)", (TW // 2 - 70, TH // 2 + 28),
                    FONT, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
    else:
        tile = cv2.resize(frame, (TW, TH), interpolation=cv2.INTER_AREA)
    _label_chip(tile, f"CAM {cam_id}")
    cv2.rectangle(tile, (0, 0), (TW - 1, TH - 1), (255, 255, 255), 2)
    return tile


def _office_tile(width):
    tile = np.full((TH, width, 3), OFFICE_BGR, np.uint8)
    cv2.putText(tile, "OFFICE", (width // 2 - 110, TH // 2 - 4),
                FONT, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(tile, "two-story block (obstacle)", (width // 2 - 145, TH // 2 + 40),
                FONT, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (width - 1, TH - 1), (255, 255, 255), 2)
    return tile


def run(cfg: Config, args=None) -> dict:
    ensure_dirs(cfg)
    layout = cfg.get("grid_layout", default=None) or DEFAULT_LAYOUT
    vids = {p.stem: p for p in list_videos(cfg.path("videos"))}

    rows, cols = len(layout), max(len(r) for r in layout)
    canvas_w = cols * TW + (cols + 1) * PAD
    canvas_h = rows * TH + (rows + 1) * PAD
    canvas = np.full((canvas_h, canvas_w, 3), 245, np.uint8)

    print(f"\nCamera grid ({rows}x{cols}); videos found: {len(vids)}")
    placed = []
    for r, row in enumerate(layout):
        c = 0
        while c < len(row):
            cell = row[c]
            y = PAD + r * (TH + PAD)
            x = PAD + c * (TW + PAD)
            if cell == "OFFICE":
                span = 1
                while c + span < len(row) and row[c + span] == "OFFICE":
                    span += 1
                width = span * TW + (span - 1) * PAD
                canvas[y:y + TH, x:x + width] = _office_tile(width)
                print(f"  ({r},{c}) OFFICE  [spans {span} cells]")
                c += span
                continue
            frame = _representative_frame(vids[cell]) if cell in vids else None
            canvas[y:y + TH, x:x + TW] = _camera_tile(frame, cell)
            status = "ok" if frame is not None else "NO VIDEO"
            print(f"  ({r},{c}) cam {cell}: {status}")
            placed.append(cell)
            c += 1

    out = cfg.path("artifacts") / "camera_grid.png"
    cv2.imwrite(str(out), canvas)
    print(f"\n  {len(placed)} camera tiles placed -> {out}")
    return {"path": str(out), "placed": placed}
