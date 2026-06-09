"""Video discovery, probing, and frame sampling."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".dav", ".m4v"}


@dataclass
class VideoInfo:
    path: Path
    cam_id: str           # stable id derived from filename stem
    ok: bool
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    codec: str
    error: str = ""


def list_videos(videos_dir: Path) -> list[Path]:
    """All video files in a directory, sorted by name (filenames are arbitrary)."""
    if not videos_dir.exists():
        return []
    vids = [p for p in videos_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")]
    return sorted(vids, key=lambda p: p.name.lower())


def cam_id_for(path: Path) -> str:
    """Stable camera id from a filename. Filenames are meaningless for layout,
    but we need a deterministic handle: use the stem (e.g. '01.mp4' -> 'cam_01')."""
    stem = path.stem
    return f"cam_{stem}"


def convert_dav(path: Path) -> Path:
    """Convert a Hikvision/Dahua .dav clip to .mp4 (stream copy). Returns new path."""
    out = path.with_suffix(".mp4")
    if out.exists():
        return out
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-c", "copy", str(out)],
        check=True, capture_output=True,
    )
    return out


def probe_video(path: Path) -> VideoInfo:
    """Read resolution / fps / frame count / duration via OpenCV (+ ffprobe fallback)."""
    cam = cam_id_for(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return VideoInfo(path, cam, False, 0, 0, 0.0, 0, 0.0, "?", "cannot open")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")

    # sanity: try to actually grab a frame
    ok, _ = cap.read()
    cap.release()

    duration = (n / fps) if (fps and n) else 0.0
    if not ok or w == 0 or h == 0:
        return VideoInfo(path, cam, False, w, h, fps, n, duration, codec or "?",
                         "opened but no readable frame")
    return VideoInfo(path, cam, True, w, h, fps, n, duration, codec or "?")


def sample_frames(path: Path, n_samples: int) -> list[np.ndarray]:
    """Evenly sample up to n_samples frames (BGR) across a clip."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    if total <= 0:
        # stream without a known length: read sequentially
        while len(frames) < n_samples:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        return frames

    idxs = np.linspace(0, total - 1, num=min(n_samples, total)).astype(int)
    idxs = np.unique(idxs)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)
    cap.release()
    return frames
