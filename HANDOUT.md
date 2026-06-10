# WARP Warehouse Automation — Project Handout

_Last updated 2026-06-09. A one-read orientation for the team. Deep detail lives in
two companion docs: **`WARP_EXISTING_SYSTEM_REPORT.md`** (the production system) and
**`STITCHING_REPORT.md`** (the camera-stitching R&D)._

---

## 1. What this project is

Find every pallet on the LAX-9 warehouse floor from the ceiling cameras, place each on
one metric floor map, and route an autonomous forklift to move a chosen pallet A→B. Two
stages:

- **Stage 1 — perception:** ceiling cameras → pallet detection → cross-camera fusion →
  pallet floor map.
- **Stage 2 — execution:** an autonomous **Zowell EFORK-CPD20-Y** forklift consumes a
  goal and moves the pallet.

```
   ceiling cameras (NVR, 4K)            Zowell EFORK-CPD20-Y forklift
        │ RTSP                                   ▲ SEER RoboKit TCP (motion + fork)
        ▼                                        │
   warehouse-cv  ──MongoDB / WebSocket──▶  [ pallet selector ]  ──WS──▶  robot-hub
   (detect+track+merge)                    (the missing bridge)        (motion + fork)
```

---

## 2. ⭐ What already exists in production (the key finding)

A full system is **already deployed** on the LAX-9 Ubuntu box (`/home/warp/projects/`).
We did a read-only analysis of a snapshot of it — full detail in
`WARP_EXISTING_SYSTEM_REPORT.md`.

**`warehouse-cv`** (Stage-1, Python 3.12, `src/main.py --computer-vision`):
- **Trained model** `warp-260403-yolo26m-seg.pt` — YOLO26-m **instance segmentation**,
  classes: **pallet, gaylord, forklift, pallet_jack, human**. It's good (≈250 pallets on
  a live 4K snapshot, conf 0.5).
- Per-camera tracking → **cross-camera merge/de-dup** → MongoDB Atlas (`warp-tracking`),
  AWS SNS/SQS/S3, WebRTC + WebSocket via `gw.wearewarp.com`.
- **World mapping = per-camera pixel offset only — NO metric calibration / homography.**

**`robot-hub`** (Stage-2, Python): drives the forklift — motion **and fork height** —
over **SEER RoboKit TCP** (`192.168.1.126`, ports 19204–19210); fork height is a
RequestID 3051 `ROBOT_TASK_GOTARGET` task with `operation="ForkHeight"` (`end_height`
in metres: 0.075 load-ready, 0.25 carrying), not a separate link. Takes
`move_to_pallet_parallel {x,y,angle}` → generates pallet-approach sub-tasks + fork
height control → streams status back. (Supersedes an earlier Node prototype,
`augment-projects/robot-control-felix-2`.)

**The gap:** the piece that **selects a detected pallet and issues the move** is an
**external system not present in either codebase** — almost certainly where the
remaining "edge cases" live.

---

## 3. Live access (confirmed working)

- **NVR (Hikvision):** `192.168.1.174:1994`, must be on the warehouse LAN.
- **RTSP:** `rtsp://remoteplayer:Dev20250918@192.168.1.174:1994/Streaming/channels/<CH>`
  where `<CH>` = `<cam>01` (4K main) or `<cam>02` (720p sub).
- **18 cameras are live in 4K** (3840×2160): cams 1–16, 19, 20 — plus cams **5/10/15/20**
  are wide **panoramic 5120×1440**. (Cams 17/18 don't exist; that slot is the office.)
- **Production only runs detection on 6 of them** (`601/701/1101/1201/1601/1901`) — so
  **~12 cameras of coverage, including all panoramic units, are live but unused.**
- The forklift (`192.168.1.126`) was **offline** during this session.
- NVR streams **HEVC/H.265** — re-encode to H.264 to view (`h264_videotoolbox` = Mac
  hardware, low CPU/heat).

---

## 4. What we built/prototyped (this repo, `lax9/` + `stitch/`)

A from-scratch Stage-1 prototype, useful as reference and for the calibration ideas:
- **Camera stitching / floor mosaic** (`stitch/`): edge-SIFT seam detection, no-checkerboard
  barrel estimation (match-maximization **and** plumb-line, agree on k1≈−0.10), **full-FOV
  undistortion**, similarity-bundle mosaic. Honest result: cameras tile with minimal
  overlap, so only the connected block stitches — see `STITCHING_REPORT.md`.
- **Detection + digital twin** (`lax9/`): pluggable detector (YOLOE-seg zero-shot **or**
  WARP's `userseg` model), footprint→floor-cell mapping, de-dup, a 2D **digital-twin
  floor map** with sized pallet footprints, a per-camera detection grid, and demo videos
  with a live **FPS/CPU/MEM** HUD. CLI: `python -m lax9 {intake,frames,overlaps,detect,
  map,grid,annotate,video,bench}`.
- **Benchmark** (`python -m lax9 bench`): on an M3 (no CUDA), single-camera detection is
  ~10–15 fps; **14 cameras serially ≈ 0.4 fps each → not real-time without a GPU.**

We also ran **WARP's own model on the live 4K** (`live_detect_grid.py`) — clean, dense
detection (251 objects, 244 pallets/gaylords on the floor map).

---

## 5. Key findings & opportunities

1. **Metric calibration is unsolved — even in production.** Pallet positions are
   per-camera pixel offsets, not metres. A few **measured floor anchors** (or a homography
   /TPS warp — `lax9/warp.py` is built for this) would make the map truly metric. **This is
   the highest-leverage improvement.**
2. **~12 of 18 live cameras are unused for detection** (incl. the wide panoramic ones) —
   free coverage to add.
3. **The CV→forklift bridge is missing** from the codebases — define it explicitly
   (pallet-select → world goal → `robot-hub` WS command) and that's where edge-case
   handling belongs (no pallet-ID persistence, no collision avoidance, no mid-task
   recovery today).
4. **Real-time multi-camera needs a GPU** (or a lighter model / lower fps; pallets move
   slowly so 1–3 fps/cam is plenty).
5. **WARP's trained model >> our zero-shot demo** — for any real detection, use
   `warp-260403-yolo26m-seg.pt` via the `userseg` backend.

---

## 6. Artifacts in this repo

- **Reports:** `WARP_EXISTING_SYSTEM_REPORT.md`, `STITCHING_REPORT.md`, this handout.
- **Live data:** `live_grabs/` — 18× 4K frames, the camera grids, and `clips_30s/`
  (14× 30 s 4K HEVC = analysis data) + `clips_30s_h264/` (H.264, viewable).
- **Results:** `artifacts/live_detection_grid.png`, `artifacts/live_floor_map.png`
  (WARP model on live 4K), plus the digital-twin + benchmark outputs.
- **Model:** `models/warp-260403-yolo26m-seg.pt` (copied from the drive; git-ignored).
- **Code:** `lax9/` (detection + floor map + CLI), `stitch/` (mosaic R&D).
- Inputs (`videos/`, `live_grabs/*.mp4`) and model weights are git-ignored.

---

## 7. Open questions / next steps

- [ ] Get a handful of **measured floor anchor points** → metric floor map (biggest win).
- [ ] Find / write the **pallet-selection → robot-goal bridge**; enumerate the edge cases.
- [ ] Bring the **forklift online** and test an end-to-end move on a low-risk pallet.
- [ ] Decide detection compute: **GPU box** for live multi-feed, or batch/low-fps on CPU.
- [ ] Consider adding the **unused cameras** (esp. panoramic) to the detection config.

> Run instructions for the production services (start.sh, env, ports, MongoDB) and the
> full protocol/command catalogs are in `WARP_EXISTING_SYSTEM_REPORT.md`.
