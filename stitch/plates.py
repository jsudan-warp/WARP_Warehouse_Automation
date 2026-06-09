"""Temporal floor plates from the full ~30 s clips (not a single frame).

Per camera we sample many frames across the clip and take the per-pixel MEDIAN:
  * moving forklifts / people / carts are removed (median = the value seen most
    of the time = the static floor), exposing floor a single frame may occlude;
  * sensor noise is averaged down -> cleaner concrete texture -> more stable SIFT
    features and clearer line evidence for undistortion.
Parked pallets that never move stay (median can't remove a static occluder) — that
is fine, we match the floor, not the pallets.

Outputs stitch/out/plates/<cam>.png and reports per-camera motion fraction.

Run:  ./venv/bin/python stitch/plates.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VID = ROOT / "videos"
OUT = ROOT / "stitch" / "out"
PLATES = OUT / "plates"
PLATES.mkdir(parents=True, exist_ok=True)

CAMS = ["01", "02", "03", "04", "06", "07", "08", "09", "11", "12", "13", "14", "16", "19"]
N_SAMPLE = 48
MOTION_STD = 12.0


def median_plate(video_path, n=N_SAMPLE):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None, 0.0, 0.0
    idxs = np.unique(np.linspace(0, total - 1, n).astype(int))
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        return None, 0.0, 0.0
    stack = np.stack(frames)
    median = np.median(stack, axis=0).astype(np.uint8)
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    motion_frac = float((grays.std(0) > MOTION_STD).mean())
    sharp = float(cv2.Laplacian(cv2.cvtColor(median, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    return median, motion_frac, sharp


def main():
    print(f"\nTemporal median plates ({N_SAMPLE} frames/clip)\n")
    print(f"  {'cam':4} {'motion_frac':>11} {'plate_sharpness':>15}")
    for cam in CAMS:
        vp = VID / f"{cam}.mp4"
        if not vp.exists():
            continue
        plate, mf, sharp = median_plate(vp)
        if plate is None:
            print(f"  {cam}: unreadable"); continue
        cv2.imwrite(str(PLATES / f"{cam}.png"), plate)
        flag = "  <- moving occluders removed" if mf > 0.02 else ""
        print(f"  {cam:4} {mf:11.3f} {sharp:15.0f}{flag}")
    print(f"\n  saved -> stitch/out/plates/")


if __name__ == "__main__":
    main()
