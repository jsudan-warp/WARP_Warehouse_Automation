"""Phase 0 — data intake. Probe every clip; confirm the feed count.

Acceptance: prints a table of all videos with resolution, fps, length; flags
anything unreadable; confirms 14 feeds or reports the real count. Short clips
(~25-30 s) are expected and fine.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .config import Config, ensure_dirs
from .io_utils import VideoInfo, convert_dav, list_videos, probe_video
from .results import append_section


def _fmt_table(infos: list[VideoInfo]) -> str:
    cols = ["file", "cam_id", "ok", "WxH", "fps", "frames", "len(s)", "codec", "note"]
    widths = [len(c) for c in cols]
    rows = []
    for vi in infos:
        row = [
            vi.path.name,
            vi.cam_id,
            "yes" if vi.ok else "NO",
            f"{vi.width}x{vi.height}",
            f"{vi.fps:.2f}",
            str(vi.n_frames),
            f"{vi.duration_s:.1f}",
            vi.codec,
            vi.error or "",
        ]
        rows.append(row)
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))
    out = [line(cols), line(["-" * w for w in widths])]
    out += [line(r) for r in rows]
    return "\n".join(out)


def run(cfg: Config, args=None) -> dict:
    ensure_dirs(cfg)
    videos_dir = cfg.path("videos")
    expected = int(cfg.get("cameras", "expected_count", default=14))

    paths = list_videos(videos_dir)

    # Convert any .dav up front (Hikvision/Dahua native container).
    converted = []
    for p in list(paths):
        if p.suffix.lower() == ".dav":
            try:
                newp = convert_dav(p)
                converted.append((p.name, newp.name))
                paths[paths.index(p)] = newp
            except Exception as e:  # noqa: BLE001
                print(f"  ! failed to convert {p.name}: {e}")

    infos = [probe_video(p) for p in paths]

    print(f"\nIntake — videos in {videos_dir}")
    if converted:
        for old, new in converted:
            print(f"  converted .dav: {old} -> {new}")
    if not infos:
        print("  (no video files found)")
    else:
        print(_fmt_table(infos))

    n_total = len(infos)
    n_ok = sum(1 for vi in infos if vi.ok)
    n_bad = n_total - n_ok

    # Resolution / spec sanity
    native = cfg.get("cameras", "native_resolution", default=[3840, 2160])
    res_set = sorted({(vi.width, vi.height) for vi in infos if vi.ok})
    downscaled = [r for r in res_set if r[0] < native[0]]

    print()
    print(f"  feeds found      : {n_total}  (expected {expected})")
    print(f"  readable         : {n_ok}")
    if n_bad:
        print(f"  UNREADABLE       : {n_bad}  <-- flagged above")
    if res_set:
        print(f"  resolution(s)    : {', '.join(f'{w}x{h}' for w, h in res_set)}")
    if downscaled:
        print(f"  NOTE: clip resolution is below the 4K Hikvision native "
              f"{native[0]}x{native[1]} spec — these look like downscaled preview "
              f"clips. Edge accuracy will be reduced vs. full-res feeds.")
    if n_total != expected:
        print(f"  NOTE: {n_total} feeds present, not the full {expected}. "
              f"Pipeline runs on whatever is here; layout/fusion scale with the real count.")

    # Log to RESULTS.md
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        f"_Run: {ts}_\n\n"
        f"- Feeds found: **{n_total}** (expected {expected})\n"
        f"- Readable: **{n_ok}**, unreadable: **{n_bad}**\n"
        f"- Resolution(s): {', '.join(f'{w}x{h}' for w, h in res_set) or 'n/a'}"
        f"{'  (below 4K native — downscaled preview clips)' if downscaled else ''}\n"
        f"- Clip length(s): "
        f"{', '.join(f'{vi.duration_s:.1f}s' for vi in infos) or 'n/a'}\n\n"
        "```\n" + _fmt_table(infos) + "\n```\n"
    )
    append_section(cfg.path("results"), "Phase 0 — Intake", body)

    return {
        "n_total": n_total, "n_ok": n_ok, "n_bad": n_bad,
        "expected": expected, "resolutions": res_set,
        "infos": infos,
    }
