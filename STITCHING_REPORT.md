# LAX-9 Floor Mosaic — Investigation Report & Handoff

**Purpose.** This is a working log + brainstorm brief for the next Claude (or human)
picking up the camera-stitching effort. It records *what we tried, what we measured,
what we learned, and where to go next*. Read `CLAUDE.md` and `PLAN.md` for the
original Stage-1 spec; this report covers the **fresh stitching pipeline** under
`stitch/` (which deliberately set aside the phased `lax9/` plan to focus on
"undistort + stitch the 14 ceiling tiles as close as possible").

---

## 1. TL;DR

- **Goal:** turn 14 ceiling-camera clips into one flat floor mosaic.
- **The dominant reality:** this is a **coverage problem, not an algorithm problem.**
  The cameras tile the 51.73 × 26.3 m floor with **little-to-no overlap** between most
  neighbours, so most tiles share no floor features and cannot be feature-stitched.
- **Best result so far:** **9 of 14 tiles genuinely feature-stitched** (01, 06, 07,
  08, 09, 11, 12, 13, 14) into a connected block; the other 5 (02, 03, 04, 16, 19) are
  **grid-placed (approximate)**. Office drawn as a labelled block.
- **Three things that moved the needle the most, in order:**
  1. **Full-FOV undistortion** (don't crop the corrected edges) → real seams 4 → 8.
  2. **Correct barrel coefficient** `k1 ≈ −0.10` (recalibrated on un-cropped images;
     the cropped estimate `−0.06` was biased low).
  3. **Temporal median plates** (use the whole 30 s clip, not one frame) → +63% inliers.
- **Hard blockers remaining:** the clips are **848×478 downscaled previews** (not the
  native 4K), there is **no lens calibration** and **no measured floor anchors**, so
  there is **no metric scale** and the non-overlapping tiles cannot be placed precisely.

---

## 2. The data we actually have

| property | value |
|---|---|
| clips | 14 `.mp4`, one per camera (`videos/01,02,03,04,06,07,08,09,11,12,13,14,16,19.mp4`) |
| resolution | **848 × 478** (downscaled preview — NOT the 3840×2160 native 4K in the spec) |
| fps / length | 20 fps, ~30.9 s, 619 frames each, H.264 |
| **motion** | **~0** in every clip (motion_frac ≈ 0.000–0.003): nothing moves — no forklifts/people during any recording |
| camera | Hikvision DS-2CD2083G2-I, 2.8 mm lens, HFOV ~107° (strong barrel), near-nadir ~24.5 ft |
| floor | 51.73 m × 26.3 m concrete; office block ~22.77 × 9.2 m; yellow structural columns |

**User-specified display layout** (`OF` = office, spans 2 cells):
```
01 02 03 04
06 07 08 09
11 12 13 14
16 OF OF 19
```
Grid adjacency from this layout defines the 19 candidate neighbour "seams" we test.

**Key consequence of the downscale:** barrel distortion is *mild* at 848 px (the lens
property is the same, but fewer pixels = smaller pixel displacement), floor texture is
faint, and overlaps are thin. **The native 4K frames would very likely change the
results substantially** (more features, stronger correctable distortion, more overlap).

---

## 3. Chronological log — what we tried and what happened

### 3.0  `lax9/` package (original phased plan, Phases 0–2)
Built a CLI pipeline (`python -m lax9 <phase>`): `intake`, `frames`, `overlaps`,
plus geometry (`warp.py`, `geometry.py`) with 17 passing pytest tests.
- **Intake:** confirmed 2 clips initially (later 14), 848×478, static.
- **Frames:** per-pixel median vs sharpest frame — chose sharpest (clips static).
- **Overlaps (cam_01 ↔ cam_02):** SIFT → **only 5–6 RANSAC inliers** → **no overlap**,
  2 connected groups. Tuning SIFT from ~2k to **~7k keypoints** (lower
  `contrastThreshold`, CLAHE) did **not** help (still 6 inliers) → the non-overlap is
  *real*, not a matcher-sensitivity artifact. ROI edge-strip matching confirmed it.
- **Phase 3 decision (GATE):** user chose **skip lens calibration** → built a
  distortion-tolerant **image→floor warp** (`lax9/warp.py`, TPS + 2nd/3rd-order
  polynomial), unit-tested to <5 cm on synthetic barrel data. *(This path is dormant —
  it needs measured floor anchors that we don't have yet.)*

### 3.1  Fresh `stitch/` pipeline — edge-region matching
`edge_match.py`: SIFT on the **touching edge strips** of each grid-neighbour pair.
- **Result:** of 19 seams, **only 07-08 genuinely overlaps** (58 inliers, 65%
  consistent — parallel match lines). A few weak (06-11:13, 12-13:11, 11-16:8, 03-08:7);
  the rest sit at the ~5-inlier noise floor. → **The cameras butt-join with ~no overlap.**

### 3.2  Undistortion without a checkerboard
Two independent estimators, both pointing at mild barrel:
- `calib_by_matching.py` — **match-maximization**: sweep `k1`, undistort, maximize
  total seam inliers. On *cropped* frames → peak **k1 = −0.08**, total inliers
  **103 → 272** (07-08: 58 → 162). *(Early bug: `getOptimalNewCameraMatrix(alpha=1)`
  produced degenerate frames at some k1 → fixed by `newK = K`.)*
- `line_undistort.py` — **plumb-line / line-straightness** (the floor's saw-cut joints
  + rack/box edges are straight 3-D lines; find the k1 that straightens them).
  Researched lightweight methods: **Alemán-Flores one-parameter division model
  (IPOL 2014)**, Bukhari–Dailey, minimal-Hough-entropy. Implemented a
  FastLineDetector "long-line evidence" metric; **dropping near-radial segments**
  (they stay straight under radial distortion → no signal) sharpened it. Cross-checked
  **k1 ≈ −0.04 to −0.08** (noisier than match-max, same ballpark).

### 3.3  Re-test overlap after undistortion
`edge_match_undist.py` (k1 = −0.08): real seams **1 → 3** (07-08:162, 12-13:66,
06-11:24; 08-13:14 weak). **Undistortion promotes weak seams into real ones.**

### 3.4  Stitching the connected block
- `stitch_block.py`: similarity-chain the {07,08,12,13} block (similarity, NOT
  homography — far more stable for thin overlaps).
- `stitch_block_polished.py`: **global linear similarity bundle adjustment** (the
  "matched points land in the same place" constraint is linear in similarity params →
  one least-squares; the strong horizontal seams constrain the weak vertical one).
  → **2.1 px median** reprojection across the block, + exposure gain + multi-band blend.

### 3.5  Full 14-tile "honest collage"
`mosaic_full.py`: undistort → **line-grid rotation** per tile (use only when confident)
→ global least-squares placement with **weak grid "springs" on every edge + real-seam
constraints** → render with green/grey outlines (stitched vs grid-placed) + office.
- Critical discriminator: the **offset-sanity gate** — a real seam's measured offset
  must point the right way and be ~one grid step. This cleanly rejected false seams
  (e.g. 03-08 `(−560,+168)` — huge sideways offset for a *vertical* seam) where the
  full-frame **consistency ratio was useless** (denominator full of frame-wide false
  matches). → 3 real seams, 6 tiles stitched.

### 3.6  Clean render + floor-content tightening
`mosaic_final.py`: multi-band/feather blend, exposure equalisation, subtle outlines +
legend, **office width fitted to the actual gap** between tiles 16 and 19. Floor-joint
study (`joints.py`): the concrete shows **irregular cracks/stains, NOT a regular
saw-cut grid** at 848 px (joint spacings came out uneven: 48, 57, 44, 81…), so
spacing-snapping isn't viable; instead added a **bounded, confidence-gated boundary
nudge** (cross-correlate floor structure across a seam, ≤50 px) that fires only where
both sides show open floor.

### 3.7  Use the whole clip (temporal median plates)
`plates.py` + `compare_plates.py`: 48-frame per-pixel median per camera.
- Clips are static → median can't *expose* occluded floor, but **denoising** helped:
  **total seam inliers 174 → 284 (+63%)**, 07-08: 52 → 110, and **08-13 became a real
  seam** (3 → 12) — connecting the center 2×2 block. It also **suppressed false matches**
  (03-08: 18 → 7). → 4 real seams.

### 3.8  ⭐ Full-FOV undistortion (the big fix — user-spotted)
The undistort used `newK = K` (fixed 848×478 frame), which **clipped the corrected
edges off-frame — exactly the overlap zones.** `undistort.undistort_maps_full()`
undistorts into an **expanded canvas** (no FOV lost; 848×478 → 1128×636 at k1=−0.06).
- **Result: real seams 4 → 8, feature-stitched tiles 6 → 9** (added 01-06, 07-12,
  09-14, 11-12; new stitched tiles 01, 09, 14). The crop had been deleting overlap.

### 3.9  Recalibrate k1 on un-cropped images
`recalibrate.py`: the old k1 was estimated on *cropped* frames where edge correction
was nullified → biased low. On **full-FOV plates**, clean unimodal peak:

| k1 | −0.05 | −0.07 | −0.08 | −0.09 | **−0.10** | −0.11 | −0.12 | −0.14 |
|---|---|---|---|---|---|---|---|---|
| total inliers | 257 | 410 | 497 | 582 | **721** | 561 | 399 | 173 |

**Optimal k1 = −0.10** (vs −0.06 cropped). *(Numerical note: Brown-model inversion
diverges for corner points at strong k1 → added a bounding-box clamp; the division
model would be more stable — see brainstorm.)*

**STATUS / immediate next step:** the final render (`mosaic_final.py`) currently still
uses **k1 = −0.06**. It should be re-run at **k1 = −0.10 full-FOV**, and **all 19 seams
re-screened** at that setting — more tiles may now connect. The clean-blend code
(feather + outer-ring trim + content crop) is staged and ready.

---

## 4. Key measured results (one place)

| metric | value | where |
|---|---|---|
| cam_01↔02 inliers (raw / 7k-kp tuned) | 5 / 6 → no overlap | §3.0 |
| real seams: raw single-frame | 1 / 19 (07-08 only) | §3.1 |
| match-max k1 (cropped) | −0.08 (inliers 103→272) | §3.2 |
| plumb-line k1 (line-straightness) | ≈ −0.04…−0.08 | §3.2 |
| real seams: undistorted single-frame | 3 / 19 | §3.3 |
| block BA reprojection error | **2.1 px median** | §3.4 |
| plates vs single-frame seam inliers | 174 → **284 (+63%)** | §3.7 |
| real seams: undistorted plates (cropped) | 4 / 19 | §3.7 |
| **real seams: full-FOV plates** | **8 / 19** | §3.8 |
| **feature-stitched tiles** | **9 / 14** | §3.8 |
| **recalibrated k1 (full-FOV)** | **−0.10** (peak 721 inliers) | §3.9 |

Connected (feature-stitched): **01, 06, 07, 08, 09, 11, 12, 13, 14**.
Grid-placed (approximate): **02, 03, 04, 16, 19**.

---

## 5. Root-cause insights (the "why")

1. **Coverage, not algorithm.** 14 cameras blanket a big floor → minimal overlap by
   design. No undistortion/matcher/BA can register pixels that don't physically overlap.
   More frames/descriptors sharpen the *real* seams; they can't manufacture overlap.
2. **Undistortion is a force multiplier for overlap** — and must be done **full-FOV**.
   Cropping the corrected edges deletes the exact regions where neighbours overlap.
3. **The barrel was under-corrected** because k1 was tuned on cropped frames. True
   `k1 ≈ −0.10`. Edges (worst distortion) only "move" when you keep the full FOV.
4. **Temporal median = free denoising** (even on static clips): +63% inliers, and it
   *both* strengthens true seams *and* suppresses false ones.
5. **Offset-sanity (direction + magnitude ≈ one grid step) is the reliable real/false
   seam discriminator.** Full-frame inlier *consistency ratio* is not (false matches
   inflate the denominator).
6. **Similarity, not homography**, for thin near-nadir seams; global **linear**
   similarity BA is well-posed and cheap.
7. **The floor lacks a regular joint grid** at this resolution (irregular cracks) → can't
   metric-snap by joint spacing; only a bounded floor-content nudge is justified.

---

## 6. Honest limitations / what is NOT trustworthy

- **No metric scale.** Everything is in pixels; floor metres need measured anchors.
- **5 tiles are grid-placed** (02, 03, 04, 16, 19) — positioned by layout, not registered.
- **Tall-stack parallax:** tiles over tall pallets (e.g. 01–04 region) violate the single
  floor-plane assumption; their box-tops are offset from their floor footprints.
- **Corners after full-FOV undistort are heavily stretched** and least reliable.
- **848×478 caps everything** — keypoint count, sub-pixel accuracy, overlap width.
- Black wedges between curved full-FOV tiles are genuine no-data regions.

---

## 7. Brainstorm — where to push next (ranked)

1. **Get the native 4K frames.** Biggest lever by far. ~20× the pixels →
   far more SIFT features, stronger (correctable) barrel, and likely **wider/more
   overlaps** that turn grid-placed tiles into stitched ones. Re-run the whole pipeline.
2. **Re-render at k1 = −0.10 full-FOV and re-screen all 19 seams.** Cheap, pending;
   may already connect more than 9 tiles.
3. **Measured floor anchors (laser).** Even ~8–15 points/camera unlock (a) **metric
   scale** and (b) anchoring the **non-overlapping** tiles — the `lax9/warp.py`
   TPS/polynomial path is already built and tested for exactly this.
4. **Division model (Fitzgibbon) instead of Brown** for the distortion. More numerically
   stable at strong k1 (Brown inversion diverged → needed a bbox clamp), single
   well-posed parameter, cleanly invertible. IPOL 2014 reference implementation exists.
5. **Learned matchers (SuperPoint + LightGlue, or LoFTR)** behind the same interface.
   Better recall on thin, low-texture floor overlaps than SIFT; could connect more seams.
6. **Joint global bundle adjustment** over *all* seams + grid priors at once (currently a
   spanning-tree-ish solve + springs); add k1 and principal-point as shared unknowns.
7. **Per-camera k1** (small differences from mounting/lens variation) once 4K is in.
8. **Don't oversell the collage.** Keep non-overlapping joins honestly drawn; a true
   metric map is a separate deliverable that needs anchors.

---

## 8. Code & artifact map

**`stitch/` modules** (run as `./venv/bin/python stitch/<file>.py`):
| file | role |
|---|---|
| `edge_match.py` | grid adjacency, sharp-still loader, edge-strip SIFT seam test |
| `undistort.py` | K from HFOV; `undistort_maps` (fixed frame) + **`undistort_maps_full`** (expanded canvas) |
| `calib_by_matching.py` | match-maximization k1 sweep (cropped) |
| `line_undistort.py` | plumb-line / line-straightness k1 estimate (+ radial-segment drop) |
| `edge_match_undist.py` | re-screen seams after undistortion |
| `stitch_block.py` / `stitch_block_polished.py` | 4-tile block stitch; global similarity BA |
| `joints.py` | floor saw-cut joint detection (black-hat + floor mask) |
| `plates.py` | temporal median plates from full clips |
| `compare_plates.py` | single-frame vs plate seam-inlier comparison |
| `mosaic_full.py` | 14-tile collage: rotation + global solve + sanity gate |
| `mosaic_final.py` | **main render**: plates + (full-FOV undistort) + clean blend + office fit |
| `recalibrate.py` | k1 sweep on full-FOV plates → **k1 = −0.10** |

**Key output images** (`stitch/out/`): `mosaic_final.png` / `mosaic_final_labeled.png`
(current best), `undistort_crop_vs_full.png` (the crop bug), `mosaic_block_polished.png`,
`joints_*.png`, `line_undistort_check.png`. JSON: `edge_match*.json`,
`calib_by_matching.json`, `line_undistort.json`, `recalibrate.json`, `mosaic_full.json`.

**Environment:** `./venv/` (Python 3.12, opencv-contrib-python 4.10, numpy, scipy,
matplotlib). All `stitch/` work is currently **uncommitted**.

---

## 9. The single most important next action

Re-run `mosaic_final.py` with **k1 = −0.10** and **full-FOV undistort** (both already in
the code; just bump the constant), **re-screen all 19 seams at that setting**, then
render. Then decide between (a) chasing the native 4K footage, or (b) collecting
measured floor anchors to go metric and place the remaining tiles. Both are gated on the
human; everything algorithmic that can be done with this footage has largely been done.
