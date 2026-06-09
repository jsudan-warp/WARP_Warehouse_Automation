# LAX-9 Pallet Localization — Brief for the Coding Agent

You are building this project mostly autonomously. Read this file and `PLAN.md`,
then work through `PLAN.md` phase by phase. After each phase, RUN its acceptance
check on the real data in this repo, show the result, fix until it passes, and
commit. Pause only for the human-input gates listed below.

## What we're building (and what we're NOT)
A system that finds every pallet on the warehouse floor from the existing ceiling
cameras, places each one on a single **metric floor map** in real time, and plans
collision-free routes for a forklift to move a chosen pallet from A to B.

- We are **not** building a pretty stitched panorama. The mosaic is only a visual
  check; the real deliverable is, per camera, a homography that maps its pixels
  onto one shared floor map, so detections from all 14 cameras share coordinates.
- We locate each pallet by its **floor-contact footprint (its base)**, never its
  top or centroid — a tall stack seen at an angle is offset on the floor
  (parallax), and the base is the only point guaranteed to lie on the floor plane.

## The two stages (THIS REPO = STAGE 1)
- **Stage 1 (this repo):** cameras → metric pallet map → route planning. Output is
  live pallet positions plus two routes (forklift→pallet, pallet→drop-off) as
  waypoints. Everything is shown on screen; nothing physical moves.
- **Stage 2 (separate):** an autonomous forklift consumes the waypoints and does
  the physical move. Out of scope here. Just emit clean waypoints for it.

## Hard constraints
- **No fiducial / QR-style markers** on the floor, and **cameras are not moved.**
  Operations cannot stop. We line cameras up using the floor's own features
  (cracks, stains, dark spots). The cameras look almost straight down, so the
  floor is the dominant flat surface and a single homography per camera is valid.
- Be honest about accuracy. Report measured residuals at every phase. Never claim
  a number you did not measure. If overlap or data is insufficient, say so and
  fall back to the metric-anchor path (Phase 5).

## The warehouse (measured facts — use these)
- Floor: **51.73 m × 26.3 m** (169.7 ft × 86.3 ft), clear height ~25 ft.
- Office: two-story block, ~**22.77 m × 9.2 m**, against the front wall, slightly
  right of center. It is an obstacle (cannot drive through it).
- Several **yellow structural columns** on the floor — obstacles.
- Tallest pallet stacks ≈ **2.1 m** (~7 ft). Most are shorter.

## Cameras
- **14 × Hikvision DS-2CD2083G2-I**, 4K (**3840×2160**), lens ≈ **2.8 mm**
  (HFOV ~107°, noticeable barrel distortion), mounted ~**24.5 ft** near-nadir.
- They tile the floor as a **7×2 grid** (7 across the length, 2 across the depth);
  one camera's view = one tile of that grid. Several pallets can fall in one tile.
- Input is **one short clip per camera (~25-30 s; assume ~25 s usable)**, recorded
  manually from the app, dropped in `videos/`. Filenames may be arbitrary / out of
  order. Clip length, start time, and time-sync between clips do **not** need to
  match — for calibration the floor is static, so timing is irrelevant. Best
  practice: record all 14 in one session so the pallet layout is consistent for the
  Phase 6/7 map; for the homographies alone, even that doesn't matter.

## World coordinate frame (use exactly this)
Right-handed, metres. **Origin = center of the floor. +X along the 51.73 m length,
+Y across the 26.3 m depth, +Z up. Floor = Z = 0.** Write this down in code as the
single source of truth and keep every camera and measured point in it.

## Accuracy target
**~10–20 cm** for a pallet's footprint on the map is good enough — the forklift's
own sensors refine the final approach. Aim for < 5 cm near measured anchor points.

## Inputs needed from the human (GATES — ask once, clearly, then continue)
These are the only things you cannot produce yourself. List what you need, and
keep working on any phase that is not blocked while you wait.
1. **The 14 clips** in `videos/` (~25-30 s each, recorded from the app). Any names;
   order, length, and timing between clips do not need to match. If any are `.dav`,
   convert with ffmpeg first.
2. **Camera world coordinates** (the (X,Y) of each camera) exported from the WARP
   viewer, if available — these are authoritative camera positions and help
   anchor/label the layout. If unavailable, proceed without them.
3. **A handful of measured floor points/distances** (laser distance meter): e.g.
   distances between recognizable floor spots, or 3–4 points with known (X,Y).
   Needed for Phase 5 (real-world scale). Without them the map is correct in shape
   but not in metres.
4. **The pallet-detection model** (the existing instance-segmentation model):
   weights or an inference endpoint, plus how to call it (input size, output
   format: masks/boxes/classes). Needed for Phase 6. If absent, stub the interface
   and proceed; allow a manual-annotation fallback for testing projection/fusion.
5. **(Optional) Checkerboard photos** for lens calibration (Phase 3). One
   calibration covers all 14 (same lens). If absent, proceed undistorted and note
   reduced edge accuracy.
6. Confirm the **lens focal length** (assume 2.8 mm until told otherwise).

## Repo layout & conventions
- Refactor the seed scripts `00_extract_frames.py`, `01_find_overlaps.py`,
  `02_calibrate_intrinsics.py`, `03_stitch_floor.py` into a package `lax9/` with a
  single CLI: `python -m lax9 <phase>` (e.g. `intake`, `frames`, `overlaps`,
  `calibrate`, `stitch`, `scale`, `detect`, `map`, `plan`). Keep their logic.
- Config in `config.yaml` (paths, world-frame constants above, thresholds,
  clearance). Artifacts in `artifacts/` (frames, matches, mosaic, homographies,
  metric transform, maps, routes). Log to stdout + `RESULTS.md`.
- Python 3.10+. Deps: `opencv-contrib-python numpy matplotlib pyyaml` (+ detector
  stack when provided). Pin versions in `requirements.txt`. Set random seeds.
- Tests with `pytest` for pure geometry (homography round-trip, footprint
  projection on a known geometry, similarity/scale fit, A* on a toy grid). Run them.

## How you (the agent) should work
- Go top to bottom through `PLAN.md`. For each phase: build → run on the data in
  this repo → check the acceptance criterion → iterate until it passes → record
  the metric in `RESULTS.md` → commit (`git commit -m "phase N: ..."`).
- Prefer small, runnable steps and frequent commits over large rewrites.
- When you hit a human-input gate, state exactly what you need (and why), then skip
  to any unblocked phase rather than stalling.
- Quantify everything: pixel/cm residuals, inlier counts, connectivity, de-dup
  rate, route clearance. Put the numbers in `RESULTS.md`.
- Robust matchers welcome: start with SIFT; if floor texture is weak, upgrade to a
  learned matcher (e.g. SuperPoint+LightGlue / LoFTR) behind the same interface.

## Definition of done (Stage 1)
Given the 14 clips and the measured anchors, running `python -m lax9 map` shows
every pallet on the metric floor map (footprint positions, IDs) from the recorded
clips; and `python -m lax9 plan --pallet <id> --to X Y` returns two
clearance-respecting routes as waypoint lists (JSON) drawn on the map. All phase
acceptance checks in `PLAN.md` pass and their numbers are logged in `RESULTS.md`.
(A true real-time replay with moving pallets is deferred until live feeds are
available — see Phase 7.)

## Glossary
- **Homography (H):** 3×3 matrix mapping one camera's image pixels onto the shared
  floor plane. The core artifact, one per camera.
- **Footprint / base:** the pallet's floor-contact point (bottom of its outline);
  the only point that obeys the floor-plane assumption.
- **Clearance:** safety margin (≈ forklift half-width + buffer, ~1.0 m) by which
  every obstacle is inflated for route planning so the forklift never clips it.
- **Anchor point:** a floor location with a known measured (X,Y); sets real metres.
