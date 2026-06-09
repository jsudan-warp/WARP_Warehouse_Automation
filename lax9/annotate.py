"""Visual-prompt annotation for YOLOE — mark one example pallet per camera.

Robust mouse+key tool (replaces the finicky cv2.selectROIs). Run on YOUR machine.

  drag a box around a clear pallet  ->  ENTER / SPACE / s : save & next camera
                                        u : undo last box   n : skip this camera
                                        q / ESC : quit
TIP: click the image window first so it has keyboard focus.

If the GUI keys don't respond at all (some macOS OpenCV builds), use the no-GUI
fallback: this tool also writes artifacts/annotate_ref/<cam>.png with a pixel grid;
read box coords off it and put them in artifacts/visual_prompts.json as
  { "03": [[x1,y1,x2,y2]], ... }
Then set detect.mode: visual and run `python -m lax9 detect`.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from .config import Config, ensure_dirs
from .floor import _load_frames

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _grid_ref(img, step=50):
    g = img.copy()
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(g, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(g, str(x), (x + 2, 12), FONT, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(g, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(g, str(y), (2, y + 12), FONT, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    return g


def _gui_available(frame):
    try:
        cv2.namedWindow("__gui_check__", cv2.WINDOW_NORMAL)
        cv2.imshow("__gui_check__", frame)
        cv2.waitKey(1)
        cv2.destroyWindow("__gui_check__")
        return True
    except Exception:
        return False


def _annotate_one(cam, img):
    """Return list of [x1,y1,x2,y2] boxes, or None if user pressed quit."""
    st = {"drawing": False, "p0": None, "p1": None}
    boxes = []

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            st["drawing"], st["p0"], st["p1"] = True, (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and st["drawing"]:
            st["p1"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and st["p0"]:
            st["drawing"] = False
            (x0, y0), (x1, y1) = st["p0"], (x, y)
            b = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            if b[2] - b[0] > 5 and b[3] - b[1] > 5:
                boxes.append(b)

    win = f"cam {cam}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, cb)
    while True:
        disp = img.copy()
        for b in boxes:
            cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), (0, 230, 0), 2)
        if st["drawing"] and st["p0"]:
            cv2.rectangle(disp, st["p0"], st["p1"], (0, 200, 255), 1)
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(disp, f"cam {cam}: drag a pallet box | ENTER/s=save&next  u=undo  "
                          f"n=skip  q=quit   ({len(boxes)} marked)", (6, 19),
                    FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10, 32, ord("s")):           # enter / return / space / s
            cv2.destroyWindow(win); return boxes
        if k == ord("u") and boxes:
            boxes.pop()
        elif k == ord("n"):
            cv2.destroyWindow(win); return []
        elif k in (27, ord("q")):
            cv2.destroyWindow(win); return None


def run(cfg: Config, args=None) -> dict:
    ensure_dirs(cfg)
    frames = _load_frames(cfg)
    if not frames:
        print("no frames found (need stitch/out/plates or videos/)"); return {}

    # always write gridded reference images (the no-GUI fallback)
    refdir = cfg.path("artifacts") / "annotate_ref"
    refdir.mkdir(parents=True, exist_ok=True)
    for cam, im in frames.items():
        cv2.imwrite(str(refdir / f"{cam}.png"), _grid_ref(im))
    print(f"gridded reference images -> {refdir}  (manual fallback)")

    vp_path = cfg.path("artifacts") / "visual_prompts.json"
    prompts = json.loads(vp_path.read_text()) if vp_path.exists() else {}

    if not _gui_available(next(iter(frames.values()))):
        print("\nNo GUI window available here. Use the manual fallback:")
        print(f"  1) open the gridded images in {refdir}")
        print(f"  2) read pixel coords of one clear pallet per camera")
        print(f"  3) edit {vp_path}:  {{ \"03\": [[x1,y1,x2,y2]], ... }}")
        print("  4) set detect.mode: visual  and run  python -m lax9 detect")
        return {"gui": False}

    only = getattr(args, "cam", None)
    cams = [only] if only else sorted(frames)
    print("Click the window to focus it, then drag a box around a CLEAR, representative "
          "pallet.\n  ENTER/SPACE/s = save & next   u = undo   n = skip   q/ESC = quit")
    for cam in cams:
        if cam not in frames:
            print(f"  cam {cam}: no frame"); continue
        res = _annotate_one(cam, frames[cam])
        if res is None:
            print("  quit."); break
        if res:
            prompts[cam] = res
            print(f"  cam {cam}: saved {len(res)} box(es)")
        else:
            print(f"  cam {cam}: skipped")

    vp_path.write_text(json.dumps(prompts, indent=2))
    print(f"\nsaved -> {vp_path}\nNow set detect.mode: visual in config.yaml, then: "
          f"python -m lax9 detect")
    return {"gui": True, "cams_with_prompts": sorted(prompts)}
