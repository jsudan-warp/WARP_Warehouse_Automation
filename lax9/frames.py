"""Phase 1 — clean stills. One floor-dominant still per camera.

Short clips (~25-30 s): we sample many frames and take the per-pixel median.
The median only erases things that MOVE during the clip; a parked forklift/pallet
stays. So we also keep the single sharpest frame and, per camera, choose a
representative still:

  - high motion in the clip  -> the median is actively removing transient
    occluders and is the better plate -> choose median.
  - mostly static clip       -> the median just adds a little blur with no
    occluder benefit -> choose the sharpest single frame (crisper floor texture).

We save BOTH (`<cam>_median.png`, `<cam>_sharp.png`) and log the choice + the
numbers that drove it. We only need the floor's marks visible for matching, not
a perfectly empty floor.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import cv2
import numpy as np

from .config import Config, ensure_dirs
from .io_utils import list_videos, cam_id_for, sample_frames
from .results import append_section

MOTION_STD_THRESH = 12.0     # per-pixel temporal std (0-255) above which a pixel "moved"
MOTION_FRAC_HIGH = 0.05      # >5% of pixels moving -> median is doing real work


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _process_clip(path, n_samples):
    frames = sample_frames(path, n_samples)
    if not frames:
        return None
    stack = np.stack(frames).astype(np.float32)          # (T,H,W,3)
    median = np.median(stack, axis=0).astype(np.uint8)   # (H,W,3)

    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    temporal_std = grays.std(axis=0)                     # (H,W)
    motion_frac = float((temporal_std > MOTION_STD_THRESH).mean())

    sharp_scores = [_laplacian_var(g.astype(np.uint8)) for g in grays]
    sharp_idx = int(np.argmax(sharp_scores))
    sharp = frames[sharp_idx]

    median_gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    median_sharp = _laplacian_var(median_gray)
    sharp_sharp = float(sharp_scores[sharp_idx])

    if motion_frac >= MOTION_FRAC_HIGH:
        choice = "median"
        reason = f"motion_frac={motion_frac:.3f} >= {MOTION_FRAC_HIGH}: median removes transient occluders"
    else:
        choice = "sharp" if sharp_sharp > median_sharp else "median"
        reason = (f"motion_frac={motion_frac:.3f} (static): "
                  f"chose the crisper plate (sharp_lap={sharp_sharp:.0f} vs median_lap={median_sharp:.0f})")

    return {
        "n_frames_used": len(frames),
        "median": median, "sharp": sharp,
        "motion_frac": motion_frac,
        "median_lap": median_sharp, "sharp_lap": sharp_sharp,
        "sharp_frame_idx": sharp_idx,
        "choice": choice, "reason": reason,
    }


def run(cfg: Config, args=None) -> dict:
    ensure_dirs(cfg)
    frames_dir = cfg.path("frames")
    n_samples = int(cfg.get("frames", "n_samples", default=32))
    paths = list_videos(cfg.path("videos"))

    manifest = {}
    rows = []
    print(f"\nPhase 1 — clean stills  (sampling {n_samples} frames/clip)")
    for p in paths:
        cam = cam_id_for(p)
        res = _process_clip(p, n_samples)
        if res is None:
            print(f"  {cam}: unreadable, skipped")
            continue
        med_path = frames_dir / f"{cam}_median.png"
        sharp_path = frames_dir / f"{cam}_sharp.png"
        cv2.imwrite(str(med_path), res["median"])
        cv2.imwrite(str(sharp_path), res["sharp"])
        chosen_path = med_path if res["choice"] == "median" else sharp_path
        manifest[cam] = {
            "video": p.name,
            "median": med_path.name,
            "sharp": sharp_path.name,
            "chosen": res["choice"],
            "chosen_file": chosen_path.name,
            "motion_frac": round(res["motion_frac"], 4),
            "median_lap": round(res["median_lap"], 1),
            "sharp_lap": round(res["sharp_lap"], 1),
            "n_frames_used": res["n_frames_used"],
        }
        rows.append((cam, res["choice"], res["motion_frac"], res["median_lap"],
                     res["sharp_lap"], res["reason"]))
        print(f"  {cam}: chose {res['choice'].upper():6} | {res['reason']}")

    manifest_path = frames_dir / "frames_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  saved {len(manifest)} still pair(s) -> {frames_dir}")
    print(f"  manifest -> {manifest_path}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body_rows = "\n".join(
        f"- **{cam}**: chose `{choice}` (motion_frac={mf:.3f}, "
        f"median_lap={ml:.0f}, sharp_lap={sl:.0f})"
        for cam, choice, mf, ml, sl, _ in rows
    )
    body = (
        f"_Run: {ts}_\n\n"
        f"Sampled {n_samples} frames/clip. Saved `<cam>_median.png` + `<cam>_sharp.png` "
        f"per camera; chose a representative still per camera (median if the clip has "
        f"motion to remove, else the sharper single frame).\n\n"
        f"{body_rows}\n"
    )
    append_section(cfg.path("results"), "Phase 1 — Clean stills", body)

    return {"manifest": manifest, "n": len(manifest)}
