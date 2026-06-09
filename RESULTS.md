# RESULTS — LAX-9 Stage 1

Measured numbers per phase (see PLAN.md acceptance checks).

## Phase 0 — Intake

_Run: 2026-06-08 23:45 UTC_

- Feeds found: **2** (expected 14)
- Readable: **2**, unreadable: **0**
- Resolution(s): 848x478  (below 4K native — downscaled preview clips)
- Clip length(s): 30.9s, 30.9s

```
file    cam_id  ok   WxH      fps    frames  len(s)  codec  note
------  ------  ---  -------  -----  ------  ------  -----  ----
01.mp4  cam_01  yes  848x478  20.01  619     30.9    h264       
02.mp4  cam_02  yes  848x478  20.01  619     30.9    h264       
```

## Phase 1 — Clean stills

_Run: 2026-06-08 23:45 UTC_

Sampled 32 frames/clip. Saved `<cam>_median.png` + `<cam>_sharp.png` per camera; chose a representative still per camera (median if the clip has motion to remove, else the sharper single frame).

- **cam_01**: chose `sharp` (motion_frac=0.002, median_lap=2821, sharp_lap=3064)
- **cam_02**: chose `sharp` (motion_frac=0.002, median_lap=1794, sharp_lap=2104)

## Phase 2 — Overlap graph

_Run: 2026-06-08 23:52 UTC_

- Cameras with stills: **2** (cam_01, cam_02)
- Matcher: SIFT + Lowe ratio 0.8 + RANSAC (4.0px); neighbour threshold = 18 inliers.
- **Connected groups: 2** — group 0=['cam_01']; group 1=['cam_02']

Pairwise inliers:
- cam_01 ↔ cam_02: **6** inliers

Artifacts: `overlap_matrix.csv`, `overlap_heatmap.png`, `contact_sheet.png`, `camera_layout.json`, `matches/*.jpg`.

> NOTE: >1 connected group — unconnected cameras flagged for Phase-5 anchors.

**Descriptor-density check (SIFT tuning).** To rule out a matcher-sensitivity
artifact, SIFT was retuned to extract far more descriptors (contrastThreshold
0.04→0.008, edgeThreshold 10→14, octaveLayers 3→5, no top-N cap, CLAHE pre-filter,
ratio 0.75→0.8):

| run | keypoints cam_01 / cam_02 | RANSAC inliers | groups |
|-----|---------------------------|----------------|--------|
| default SIFT | 2079 / 2966 | 5 | 2 |
| dense SIFT   | 7145 / 7173 | 6 | 2 |

~3× more descriptors moved inliers 5→6 (both far below the 18 threshold). The
match visualisation (`matches/cam_01__cam_02.jpg`) shows these are scattered,
geometrically-inconsistent matches on box texture, not a shared floor patch. The
non-overlap is **real**, not a matcher limitation — cam_01 and cam_02 are
different parts of the warehouse and must be tied together via Phase-5 anchors.

## Phase 2 — Overlap graph

_Run: 2026-06-09 00:55 UTC_

- Cameras with stills: **2** (cam_01, cam_02)
- Matcher: SIFT + Lowe ratio 0.8 + RANSAC (4.0px); neighbour threshold = 18 inliers.
- **Connected groups: 2** — group 0=['cam_01']; group 1=['cam_02']

Pairwise inliers:
- cam_01 ↔ cam_02: **6** inliers

Artifacts: `overlap_matrix.csv`, `overlap_heatmap.png`, `contact_sheet.png`, `camera_layout.json`, `matches/*.jpg`.

> NOTE: >1 connected group — unconnected cameras flagged for Phase-5 anchors.

## Phase 2 — Overlap graph

_Run: 2026-06-09 00:59 UTC_

- Cameras with stills: **2** (cam_01, cam_02)
- Matcher: SIFT + Lowe ratio 0.8 + RANSAC (4.0px); neighbour threshold = 18 inliers.
- **Connected groups: 2** — group 0=['cam_01']; group 1=['cam_02']

Pairwise inliers:
- cam_01 ↔ cam_02: **5** inliers

Artifacts: `overlap_matrix.csv`, `overlap_heatmap.png`, `contact_sheet.png`, `camera_layout.json`, `matches/*.jpg`.

> NOTE: >1 connected group — unconnected cameras flagged for Phase-5 anchors.

## Phase 2 — Overlap graph

_Run: 2026-06-09 01:04 UTC_

- Cameras with stills: **2** (cam_01, cam_02)
- Matcher: SIFT + Lowe ratio 0.8 + RANSAC (4.0px); neighbour threshold = 18 inliers.
- **Connected groups: 2** — group 0=['cam_01']; group 1=['cam_02']

Pairwise inliers:
- cam_01 ↔ cam_02: 59 good matches → **6** RANSAC inliers

Artifacts: `overlap_matrix.csv`, `overlap_heatmap.png`, `contact_sheet.png`, `camera_layout.json`, `matches/*.jpg`.

> NOTE: >1 connected group — unconnected cameras flagged for Phase-5 anchors.

## Phase 3 — Lens intrinsics

_Run: 2026-06-09 01:23 UTC_

- No calibration data in `calib/`. **Gate open.** Phases 2/4 run UNDISTORTED for now (reduced edge accuracy, worst at the 2.8 mm barrel corners). Awaiting a lens-model source (spare unit / floor board / self-cal).

## Phase 2 — Overlap graph

_Run: 2026-06-09 01:23 UTC_

- Cameras with stills: **2** (cam_01, cam_02)
- Matcher: SIFT + Lowe ratio 0.8 + RANSAC (4.0px); neighbour threshold = 18 inliers.
- **Connected groups: 2** — group 0=['cam_01']; group 1=['cam_02']

Pairwise inliers:
- cam_01 ↔ cam_02: 59 good matches → **6** RANSAC inliers

Artifacts: `overlap_matrix.csv`, `overlap_heatmap.png`, `contact_sheet.png`, `camera_layout.json`, `matches/*.jpg`.

> NOTE: >1 connected group — unconnected cameras flagged for Phase-5 anchors.

## Phase 3 — Lens intrinsics

_Run: 2026-06-09 01:29 UTC_

- **Decision: SKIP lens calibration** (no undistortion). The 2.8 mm barrel is absorbed by the per-camera distortion-tolerant image->floor warp (`tps`), fit directly from measured floor points at Phase 5. This also ties in non-overlapping cameras (each anchored independently). Warp machinery (`lax9/warp.py`) built + unit-tested; **now gated on the Phase-5 measured floor points** (~18 pixel↔(X,Y) per camera).
