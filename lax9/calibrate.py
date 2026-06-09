"""Phase 3 — lens intrinsics. One calibration covers all 14 (same lens/model).

Two routes, auto-selected by what's available:
  (A) checkerboard calibration  — if calib/ has board photos. Standard
      cv2.calibrateCamera with CALIB_RATIONAL_MODEL (handles strong barrel).
  (B) no data                   — print the gate options and exit cleanly
      (later phases run undistorted; edge accuracy reduced, noted in RESULTS).

Saves artifacts/intrinsics.npz (K, dist, image_size, rms, model). Acceptance:
RMS reprojection error < ~0.5 px; undistorted straight edges look straight.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import Config, ensure_dirs
from .results import append_section
from .undistort import undistort_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

GATE_MSG = """\
  Phase 3 GATE — no calibration data found in calib/.
  Pick a lens-model source (one calibration covers all 14; same lens):
    (A) Spare/bench unit  — calibrate an identical DS-2CD2083G2-I on a desk with a
        printed checkerboard (best corner coverage; never touches mounted cameras).
        Drop the photos in calib/ and re-run `python -m lax9 calibrate`.
    (B) Floor checkerboard — lay a board, capture from a ceiling cam while sliding/
        tilting it. Workable but corners (worst distortion) are undersampled.
    (C) No physical access — self-calibrate radial distortion from the existing
        footage (straighten rack/wall/floor edges). Respects 'cameras not moved /
        ops can't stop'. -> `python -m lax9 calibrate --selfcal` (when enabled).
  Until then, Phases 2/4 run UNDISTORTED (note the reduced edge accuracy)."""


def _find_calib_images(calib_dir: Path) -> list[Path]:
    if not calib_dir.exists():
        return []
    return sorted(p for p in calib_dir.iterdir()
                  if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))


def _checkerboard_calibrate(images, board, square_size, model):
    cols, rows = int(board[0]), int(board[1])
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_size)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    objpoints, imgpoints, used = [], [], []
    image_size = None
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if not ok:
            print(f"    {p.name}: board NOT found")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(p.name)
        print(f"    {p.name}: board found ({cols}x{rows})")

    if len(objpoints) < 3:
        return None, f"only {len(objpoints)} usable board view(s); need >= 3 (ideally 10-20)"

    calib_flags = cv2.CALIB_RATIONAL_MODEL if model == "rational" else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=calib_flags)

    # per-view reprojection error
    errs = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        errs.append(cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj))
    return {
        "K": K, "dist": dist, "image_size": image_size, "rms": float(rms),
        "model": model, "n_images": len(objpoints), "used": used,
        "per_view_err": errs, "square_size_m": float(square_size),
    }, None


def run(cfg: Config, args=None) -> dict:
    ensure_dirs(cfg)
    ccfg = cfg.raw.get("calibration", {})
    calib_dir = cfg.path("calib")

    print(f"\nPhase 3 — lens intrinsics  (calib dir: {calib_dir})")

    # DECISION: skip lens calibration; lens bend absorbed by the Phase 4/5 warp.
    if str(ccfg.get("mode", "checkerboard")).lower() == "skip" and not getattr(args, "selfcal", False):
        mcfg = cfg.raw.get("mapping", {})
        print("  mode = SKIP (by decision). No checkerboard / no undistortion.")
        print(f"  The 2.8 mm barrel is absorbed by the per-camera image->floor warp "
              f"(model='{mcfg.get('model', 'tps')}'), fit from measured floor points at Phase 5.")
        print(f"  Needs ~{mcfg.get('recommend_points', 18)} measured (pixel <-> world X,Y) "
              f"correspondences per camera, well spread incl. near the frame corners.")
        print("  To calibrate instead: set calibration.mode=checkerboard and add photos to calib/.")
        append_section(cfg.path("results"), "Phase 3 — Lens intrinsics",
                       f"_Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n"
                       "- **Decision: SKIP lens calibration** (no undistortion). The 2.8 mm barrel "
                       "is absorbed by the per-camera distortion-tolerant image->floor warp "
                       f"(`{mcfg.get('model', 'tps')}`), fit directly from measured floor points at "
                       "Phase 5. This also ties in non-overlapping cameras (each anchored "
                       "independently). Warp machinery (`lax9/warp.py`) built + unit-tested; "
                       "**now gated on the Phase-5 measured floor points** "
                       f"(~{mcfg.get('recommend_points', 18)} pixel↔(X,Y) per camera).\n")
        return {"calibrated": False, "mode": "skip"}

    images = _find_calib_images(calib_dir)
    if not images:
        print(GATE_MSG)
        append_section(cfg.path("results"), "Phase 3 — Lens intrinsics",
                       f"_Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n"
                       "- No calibration data in `calib/`. **Gate open.** Phases 2/4 run "
                       "UNDISTORTED for now (reduced edge accuracy, worst at the 2.8 mm "
                       "barrel corners). Awaiting a lens-model source "
                       "(spare unit / floor board / self-cal).\n")
        return {"calibrated": False, "reason": "no calib data"}

    print(f"  found {len(images)} calibration image(s)")
    res, err = _checkerboard_calibrate(
        images, ccfg.get("checkerboard", [9, 6]),
        ccfg.get("square_size_m", 0.025), ccfg.get("model", "rational"))
    if res is None:
        print(f"  calibration FAILED: {err}")
        return {"calibrated": False, "reason": err}

    out_path = cfg.path("intrinsics")
    np.savez(out_path, K=res["K"], dist=res["dist"],
             image_size=np.array(res["image_size"]), rms=res["rms"],
             model=res["model"], n_images=res["n_images"],
             square_size_m=res["square_size_m"])

    rms_target = float(ccfg.get("rms_target_px", 0.5))
    status = "PASS" if res["rms"] < rms_target else "HIGH (try fisheye model / more views)"
    print(f"\n  RMS reprojection error: {res['rms']:.4f} px  (target < {rms_target}) -> {status}")
    print(f"  K=\n{np.array2string(res['K'], precision=2)}")
    print(f"  saved -> {out_path}")

    # visual proof: undistort the first calib image
    sample = cv2.imread(str(images[0]))
    if sample is not None:
        und = undistort_image(sample, res, alpha=float(ccfg.get("alpha", 0.0)))
        cmp_path = cfg.path("artifacts") / "undistort_check.png"
        cv2.imwrite(str(cmp_path), np.hstack([
            cv2.resize(sample, (und.shape[1], und.shape[0])), und]))
        print(f"  undistort check (left=raw, right=undistorted) -> {cmp_path}")

    append_section(cfg.path("results"), "Phase 3 — Lens intrinsics",
                   f"_Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n"
                   f"- Checkerboard calibration, {res['n_images']} views, "
                   f"model=`{res['model']}`.\n"
                   f"- **RMS reprojection error: {res['rms']:.4f} px** "
                   f"(target < {rms_target}) — {status}.\n"
                   f"- Calibrated at {res['image_size'][0]}x{res['image_size'][1]}; "
                   f"K auto-scales to the processing resolution at undistort time.\n"
                   f"- Saved `intrinsics.npz`; `undistort_check.png` shows straightened edges.\n")
    return {"calibrated": True, **res}
