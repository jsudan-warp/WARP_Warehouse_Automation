# LAX-9 camera stitch — getting started

**Goal:** turn the 14 ceiling feeds into ONE shared floor map by lining the
cameras up on the floor's own features (cracks, stains, marks). No printed
markers, no moving cameras. This is the "lining up the cameras" step from the
plan. Real-world metres come right after, from a few measured distances.

**What you have:** one short clip per camera (~25-30 s; assume ~25 s usable),
recorded from the app. That is enough — we only need one clean frame per camera, and
the clips do not need to line up in time (the floor is static, so timing is
irrelevant for calibration). We sample frames across each clip and take the
per-pixel median to get a clean floor plate. One caveat of short clips: the median
can only erase things that *move* during the clip, so a parked forklift/pallet will
remain — that is fine, because we match the floor's marks, not the pallets, and the
code falls back to the sharpest single frame and keeps whichever view shows the most
floor. Tip: record all 14 in one session so the pallet layout is consistent for the
later map step.

**Why a single homography per camera works here:** the cameras look almost
straight down, so the floor is the dominant flat surface. One 3x3 homography maps
each camera's image onto a single common floor plane. Those 14 homographies are
the real deliverable — the mosaic is the visual proof they line up.

## Folder layout
```
camera_stitch/
  videos/   <- drop all 14 exported feeds here (ANY filenames; order doesn't matter)
  calib/    <- (optional) checkerboard photos for lens calibration
  frames/   <- created by step 0: one clean still per camera
  out/      <- created: overlap matrix, mosaic, homographies
  00_extract_frames.py
  01_find_overlaps.py
  02_calibrate_intrinsics.py   (optional, run once — same lens covers all 14)
  03_stitch_floor.py
```

## Setup
```
python3 -m venv venv && source venv/bin/activate
pip install opencv-contrib-python numpy matplotlib
```
If a video won't open and it's a Hikvision/Dahua `.dav` file, convert it first:
```
ffmpeg -i input.dav -c copy output.mp4
```

## Run in order
```
1) python3 00_extract_frames.py      -> frames/*_median.png  (+ prints resolution/fps/length)
2) python3 01_find_overlaps.py       -> which cameras are neighbours (handles "not in order")
(3) python3 02_calibrate_intrinsics.py  -> intrinsics.npz  (do once; optional for a first pass)
4) python3 03_stitch_floor.py        -> out/mosaic.png + out/homographies.{npz,json}
```

## About "files may not be in order"
Filenames don't matter. **Step 1** matches the floor between every pair of cameras
and counts how many matches survive a geometric check; lots of consistent matches
means the two cameras see the same patch of floor, i.e. they are neighbours. It
prints a neighbour list and the connected groups, and saves `out/matches/*.jpg`
so you can confirm the matches land on the **floor**, not on tall pallets. Then
use landmarks — the office and the yellow columns — to drop each file onto the
7x2 camera grid.

## The one thing that decides if this works
Neighbours must actually share some floor. Near-nadir cameras can have thin
overlaps. If step 1 reports **more than one group** (some cameras don't connect),
those cameras can't be tied in by features alone — they get joined in the metric
step using a few measured floor points, which we do next anyway for real-world
coordinates.

## What "done" looks like
Open `out/mosaic.png`: the floor should line up across cameras with no doubled
cracks or edges. `out/homographies.json` gives, per camera, the 3x3 matrix that
maps that camera's pixels onto the shared map.

## Next steps (after the stitch looks right)
- **Real-world scale:** measure a handful of floor distances/points and fit a
  similarity (rotate/scale/shift) so the map reads in metres and matches the
  floor. (This is also where any non-overlapping cameras get anchored.)
- **Plug in the detector:** run the pallet model per camera, take each pallet's
  *base* (footprint), and push it through that camera's homography onto the map.
  That is the live pallet map from the plan.

## Notes
- For accuracy at the image edges, do the lens calibration (step 2). The 2.8 mm
  lens has noticeable barrel distortion; one calibration covers all 14 (same lens).
- This is the demo path. For production we'd add surveyed anchor points plus
  held-out check points to actually measure accuracy.
