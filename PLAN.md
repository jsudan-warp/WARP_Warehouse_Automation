# Build Plan — LAX-9 Stage 1

Work top to bottom. For each phase: build, RUN it on the data in this repo, meet
the **Acceptance** check, log the number in `RESULTS.md`, tick the box, and commit.
Read `CLAUDE.md` first (goal, constraints, world frame, human-input gates).

Legend: `[ ]` todo · `[~]` blocked on a human-input gate · `[x]` done & verified.

---

## [x] Phase 0 — Project setup & data intake
**Goal:** runnable skeleton + confirm the input videos.
**Do:** create `lax9/` package, `config.yaml` (world-frame constants from CLAUDE.md,
paths, thresholds, clearance=1.0), `requirements.txt`, `RESULTS.md`; wire a CLI
`python -m lax9 <phase>`. Fold in the seed scripts' logic. Detect videos in
`videos/`; if any `.dav`, convert via ffmpeg.
**Deliverable:** `python -m lax9 intake`.
**Acceptance:** `intake` prints a table of all videos with resolution, fps, and
length; flags anything unreadable. Confirms 14 feeds (or reports the real count).
Short (~25-30 s) clips are expected and fine — just report each length, don't error.

## [x] Phase 1 — Clean stills (short clips, ~25 s)
**Goal:** one floor-dominant still per camera.
**Do:** the clips are short (~25-30 s), so sample many frames across each clip
(e.g. 25-40, or every Nth frame) for the per-pixel median. A short clip CANNOT
remove a *stationary* forklift/pallet — the median only erases things that move
during the clip — so: if the median still shows a parked occluder, fall back to the
single sharpest frame, and in all cases keep whichever still shows the MOST floor.
We only need the floor's marks visible (especially in the overlap with neighbours),
not a perfectly empty floor. Save `artifacts/frames/<cam>_median.png` (+ a sharp
single frame).
**Acceptance:** a usable, floor-dominant still per camera (median if it cleans up,
else a sharp frame); floor texture clearly visible. Log which one was chosen per cam.

## [x] Phase 2 — Camera identification & overlap graph
**Goal:** recover which cameras are neighbours (filenames are meaningless) and lay
them on the 7×2 grid.
**Do:** pairwise floor-feature matching across stills → inlier-count matrix →
neighbour graph → connected components. Generate a labeled contact sheet and ask
the human to confirm each file's grid cell using landmarks (office, columns); save
`artifacts/camera_layout.json` (cam → {grid:(row,col), neighbours:[...]}).
**Acceptance:** `overlap_matrix.csv` + `overlap_heatmap.png` produced; neighbour
list printed; **report the number of connected groups.** If >1 group, mark the
unconnected cameras as needing Phase-5 anchors and continue. Save `camera_layout.json`.

## [x] Phase 3 — Lens intrinsics (one calibration for all 14) — DECISION: SKIPPED
> Resolved by decision: skip lens calibration. The 2.8 mm barrel is absorbed by a
> per-camera distortion-tolerant image→floor warp (TPS/polynomial, `lax9/warp.py`,
> unit-tested), fit from measured floor points at Phase 5 — which also anchors the
> non-overlapping cameras. Checkerboard path remains available via
> `calibration.mode=checkerboard` + photos in `calib/`.
**Goal:** straighten barrel distortion before stitching.
**Do:** if `calib/` has checkerboard photos, calibrate (rational model) → save
`artifacts/intrinsics.npz`. If absent, proceed undistorted and note the limitation.
**Acceptance:** if calibrated, RMS reprojection error < ~0.5 px and undistorted
straight edges look straight. Log RMS. (Non-blocking: later phases run without it.)

## [ ] Phase 4 — Per-camera floor homography (the stitch)
**Goal:** map every camera onto ONE shared floor frame.
**Do:** undistort (if Phase 3 done); pick a reference camera (most overlap or a
central one); chain pairwise homographies (BFS over the neighbour graph) so every
camera maps into the reference frame; warp all into a mosaic. Save
`artifacts/mosaic.png` and `artifacts/homographies.json` (cam → 3×3).
**Acceptance:** mosaic floor lines up with **no doubled cracks/edges**; compute an
overlap-consistency residual (reproject matched floor points across cameras) and
report **median < ~3 px** at working resolution (note: this becomes cm after
Phase 5). List any camera not reachable from the reference.

## [~] Phase 5 — Metric scale & world frame — GATE: measured floor points
**Goal:** turn the relative map into real metres in the CLAUDE.md world frame.
**Do:** take the human's measured points/distances; fit a similarity (rotation +
uniform scale + translation) from map pixels → metres; set origin at floor center,
+X along length, +Y across depth. Use anchors to also tie in any camera that was
unreachable in Phase 4. Save `artifacts/metric_transform.json` and a metric map.
**Acceptance:** held-out check distance(s) reproduced within **< 5 cm** near
anchors; report the error for each check and the worst case. A point clicked on the
map returns plausible metres; floor dimensions come out ≈ 51.73 × 26.3 m.

## [~] Phase 6 — Detector integration & footprint projection — GATE: pallet model
**Goal:** put real pallets on the metric map.
**Do:** wrap the pallet model behind `lax9/detect.py` (input frame → list of
masks/boxes). For each detection, compute the **base/footprint contact point**,
project it through that camera's homography + metric transform → (X,Y) in metres.
Fuse across cameras: de-duplicate pallets seen by two cameras (merge within ~30 cm,
prefer the most-nadir view), assign stable IDs. If the model isn't available yet,
keep a manual-annotation fallback to test projection/fusion.
**Acceptance:** on a test frame set, pallets plot at plausible positions; a pallet
of known footprint projects to ≈ its real size; duplicates in overlap regions merge
to one. Output a per-frame list `[{id, x_m, y_m, w, l, theta?}]`. Report de-dup rate.
**Note:** cross-camera de-dup assumes the pallet layout is the same in every clip.
Because the clips are short and from one recording session, that should hold; flag
any pallet that looks like it moved between clips (it will project to two spots).

## [~] Phase 7 — Pallet map — GATE: needs Phase 6
**Goal:** the metric pallet map from the plan.
**Important:** the clips are short (~25 s) and NOT time-synchronised across cameras,
so a true multi-camera *live replay* is not possible from this data. Build the
version this data supports:
**Do:** a **single-timestamp snapshot map** — pick one representative frame per
camera, detect + project + fuse footprints, and render one metric floor map with the
7×2 camera-grid overlay and pallet IDs. `python -m lax9 map`.
**Acceptance:** the snapshot map shows each pallet once at a plausible metric
position, with duplicates merged in the overlaps.
**Deferred (needs simultaneous feeds):** a real-time / temporal multi-camera replay
with moving pallets requires the live RTSP feeds or synchronised recording; revisit
once that access is granted. Note this in `RESULTS.md` rather than faking it.

## [ ] Phase 8 — Route planning (Stage 1 output, the Stage-2 handoff)
**Goal:** given a chosen pallet + destination, produce two clearance-safe routes.
**Do:** build a metric occupancy grid (obstacles = other pallets, columns, office,
walls), inflate every obstacle by `clearance` (~1.0 m). A* for Route 1
(forklift→pallet approach) and Route 2 (pallet→drop-off); export waypoints.
`python -m lax9 plan --pallet <id> --to X Y`.
**Acceptance:** both routes are returned, drawn on the metric map, and stay ≥
clearance from every obstacle (verify by checking each waypoint's distance to
inflated obstacles). Waypoints saved as JSON for Stage 2. If no path exists, report
why (blocked/clearance too large).

---

## Tests to write and keep green (run with `pytest`)
- Homography round-trip: warp known points forward then inverse → recover within ε.
- Footprint projection: a synthetic box of known size at a known floor spot
  projects back to that spot/size.
- Metric fit: given synthetic anchors, the similarity recovers the known scale.
- A*: on a toy occupancy grid, returns a path that respects inflation; returns
  None when fully blocked.

## Notes
- Keep the seed scripts working as a fallback; the package just organizes them.
- Prefer SIFT first; if Phase 2/4 inliers are weak on real floor texture, swap in a
  learned matcher behind the same function signature and re-run.
- Commit after every phase. Append every measured number to `RESULTS.md`.
